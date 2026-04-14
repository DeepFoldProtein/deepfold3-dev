#!/usr/bin/env python3
"""Profile DeepFold3 training to identify bottlenecks and estimate per-iteration time.

Runs the real Trainer.run() code path with log_microsteps=True for per-step
wall-clock timing. Optionally wraps with torch.profiler for kernel-level detail.

Usage:
    uv run python scripts/profile_training.py \
        --data-dir data/patchr_training --config configs/initial.json

    # With Chrome trace for kernel breakdown:
    uv run python scripts/profile_training.py \
        --data-dir data/patchr_training --config configs/initial.json \
        --trace /tmp/profile_trace.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Profile DeepFold3 training bottlenecks")
    p.add_argument("--data-dir", required=True, help="Training data directory")
    p.add_argument("--config", required=True, help="Training config JSON")
    p.add_argument("--params", default="", help="Pretrained weights (optional)")
    p.add_argument("--steps", type=int, default=5, help="Total training steps (incl. warmup)")
    p.add_argument("--trace", default=None, help="Chrome trace output path")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--diffusion-batch-size", type=int, default=None)
    p.add_argument("--diffusion-chunk-size", type=int, default=None)
    p.add_argument("--max-recycles", type=int, default=None)
    p.add_argument("--seed", type=int, default=99)
    return p.parse_args()


def build_config(args):
    from deepfold3.model.config import ModelConfig, TrainingConfig
    from deepfold3.scripts.train import _apply_overrides

    with open(args.config) as f:
        cfg_json = json.load(f)

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if "model" in cfg_json:
        _apply_overrides(model_cfg, cfg_json["model"])
    if "training" in cfg_json:
        _apply_overrides(train_cfg, cfg_json["training"])

    train_cfg.data_dir = args.data_dir
    train_cfg.output_dir = "/tmp/profile_run"
    train_cfg.seed = args.seed
    train_cfg.max_steps = args.steps
    train_cfg.accumulate_grad_batches = 1  # each microstep = optimizer step
    train_cfg.log_interval = 1
    train_cfg.log_microsteps = True
    train_cfg.wandb_enabled = False
    train_cfg.ema_decay = -1
    train_cfg.checkpoint_interval = 999999
    train_cfg.eval_interval = 999999
    train_cfg.num_workers = 0  # in-process loading avoids DataLoader crash on bad records

    if args.max_tokens is not None:
        train_cfg.max_tokens = args.max_tokens
    if args.diffusion_batch_size is not None:
        train_cfg.diffusion_batch_size = args.diffusion_batch_size
    if args.diffusion_chunk_size is not None:
        train_cfg.diffusion_chunk_size = args.diffusion_chunk_size
    if args.max_recycles is not None:
        train_cfg.max_recycles = args.max_recycles
    if args.params:
        train_cfg.params_path = args.params

    return model_cfg, train_cfg


def main():
    args = parse_args()

    # Must run under torchrun for DDP
    if "RANK" not in os.environ:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            __file__,
        ] + sys.argv[1:]
        logger.info(f"Re-launching under torchrun: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    if not torch.cuda.is_available():
        logger.error("CUDA required")
        return

    model_cfg, train_cfg = build_config(args)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)

    logger.info(f"GPU: {gpu_name}, {gpu_mem_gb:.1f} GB")
    logger.info(
        f"Config: max_tokens={train_cfg.max_tokens}, diffusion_batch={train_cfg.diffusion_batch_size}, "
        f"chunk={train_cfg.diffusion_chunk_size}, max_recycles={train_cfg.max_recycles}, "
        f"mini_rollout={train_cfg.mini_rollout_steps}, dtype={train_cfg.dtype}"
    )

    from deepfold3.data.prepare.manifest import discover_records
    from deepfold3.training.trainer import Trainer

    records = discover_records(Path(args.data_dir))
    logger.info(f"Found {len(records)} records")

    trainer = Trainer(model_cfg, train_cfg, records=records)

    # If trace requested, monkey-patch _run_inner to wrap with torch.profiler
    if args.trace:
        _orig_run_inner = trainer._run_inner

        def _profiled_run_inner(resume_from):
            # We need to intercept the training loop.  Instead of wrapping the
            # entire _run_inner (complex), we use the CUDA profiler as a
            # background context that captures everything.
            warmup = min(2, args.steps - 1)
            args.steps - warmup

            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            prof.__enter__()
            try:
                _orig_run_inner(resume_from)
            finally:
                prof.__exit__(None, None, None)
                prof.export_chrome_trace(args.trace)
                logger.info(f"Chrome trace saved to {args.trace}")

                print("\n--- Top 25 CUDA kernels by total GPU time ---")
                print(
                    prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=25,
                        top_level_events_only=False,
                    )
                )

        trainer._run_inner = _profiled_run_inner

    # Run the real training loop
    trainer.run()

    # Print summary
    n_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print(
        f"\nConfig: max_tokens={train_cfg.max_tokens}, diffusion_batch={train_cfg.diffusion_batch_size}, "
        f"chunk={train_cfg.diffusion_chunk_size}, max_recycles={train_cfg.max_recycles}, "
        f"mini_rollout={train_cfg.mini_rollout_steps}"
    )
    print(f"Model: {n_params:.1f}M params, dtype={train_cfg.dtype}")
    print(f"GPU: {gpu_name}, {gpu_mem_gb:.1f} GB")

    # Production estimates
    print("\n--- Note ---")
    print("  Per-microstep timings are in the log above (dt_micro=...s)")
    print("  Production config: accumulate=32, 8×GPU")
    print("  Optimizer step time = dt_micro × 32")
    print("  Use --trace for kernel-level breakdown")
    print()


if __name__ == "__main__":
    main()
