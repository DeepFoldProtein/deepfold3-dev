"""``deepfold3 train`` -- training entry point.

Examples:
    Single-GPU::

        deepfold3 train --data-dir /data/training --output output/run1

    Multi-GPU (DDP via torchrun)::

        torchrun --nproc_per_node=4 -m deepfold3.scripts.train \\
            --data-dir /data/training --output output/run1

    Resume::

        deepfold3 train --data-dir /data/training --output output/run1 \\
            --resume output/run1/checkpoint_1000.pt
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from deepfold3.scripts import cli


@cli.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True), help="Training data directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for checkpoints and logs.")
@click.option("--params", default="", help="Pretrained weights (.safetensors or .pt).")
@click.option("--resume", default=None, type=click.Path(exists=True), help="Resume from checkpoint.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="Training config JSON.")
@click.option("--lr", default=None, type=float, help="Override learning rate.")
@click.option("--max-steps", default=None, type=int, help="Override max training steps.")
@click.option("--max-tokens", default=None, type=int, help="Override max tokens per crop.")
@click.option("--grad-accum", default=None, type=int, help="Override gradient accumulation steps.")
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option("--dtype", default=None, type=click.Choice(["bf16", "fp32", "fp16"]), help="Override precision.")
def train(
    data_dir: str,
    output: str,
    params: str,
    resume: str | None,
    config_path: str | None,
    lr: float | None,
    max_steps: int | None,
    max_tokens: int | None,
    grad_accum: int | None,
    seed: int | None,
    dtype: str | None,
) -> None:
    """Train DeepFold3 model."""
    from deepfold3.model.config import ModelConfig, TrainingConfig
    from deepfold3.training.trainer import Trainer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Build config from defaults + JSON override + CLI overrides
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)

        # Apply training config overrides
        for key, val in overrides.get("training", {}).items():
            if hasattr(train_cfg, key):
                setattr(train_cfg, key, val)

        # Apply model config overrides (flat or nested)
        for key, val in overrides.get("model", {}).items():
            if hasattr(model_cfg, key):
                sub_cfg = getattr(model_cfg, key)
                if isinstance(val, dict) and hasattr(sub_cfg, "__dataclass_fields__"):
                    for sk, sv in val.items():
                        if hasattr(sub_cfg, sk):
                            setattr(sub_cfg, sk, sv)
                else:
                    setattr(model_cfg, key, val)

    # Apply CLI overrides
    train_cfg.data_dir = data_dir
    train_cfg.output_dir = output
    train_cfg.params_path = params
    if lr is not None:
        train_cfg.lr = lr
    if max_steps is not None:
        train_cfg.max_steps = max_steps
    if max_tokens is not None:
        train_cfg.max_tokens = max_tokens
    if grad_accum is not None:
        train_cfg.accumulate_grad_batches = grad_accum
    if seed is not None:
        train_cfg.seed = seed
    if dtype is not None:
        train_cfg.dtype = dtype

    # Load records from manifest or by scanning data_dir
    from deepfold3.data.prepare.manifest import discover_records

    all_records = discover_records(Path(data_dir))
    if not all_records:
        logging.getLogger(__name__).warning("No records found in %s", data_dir)

    # Shuffle with fixed seed for reproducible train/eval split
    import random

    rng = random.Random(42)
    rng.shuffle(all_records)

    # Split: last 5% for eval (at least 1 if available)
    n_eval = max(1, len(all_records) // 20)
    records = all_records[:-n_eval] if len(all_records) > n_eval else all_records
    eval_records = all_records[-n_eval:] if len(all_records) > n_eval else []

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        records=records,
        eval_records=eval_records,
    )
    trainer.run(resume_from=resume)


# Allow running as module for torchrun: python -m deepfold3.scripts.train
if __name__ == "__main__":
    train()
