"""Benchmark the optional fused Triton kernels at the production crop (N=384).

Measures end-to-end (fwd + bwd, with gradient checkpointing) training step time and
peak memory for a stack of ``PairFormerBlock``s -- the trunk component that hosts the
four candidate kernels (triangle_mul, transition_pair/swiglu, pair_bias) plus the
default fused dropout-add. Also runs a forward-only inference sweep across buckets.

Unlike ``docs/optimizations.md`` (forward-only, RTX 5090, no checkpointing), this
reflects the real training regime, so it is the basis for finalizing the flags.

Usage:
    uv run python scripts/bench_kernels.py --blocks 8 --iters 20
    uv run python scripts/bench_kernels.py --regime inference
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from deepfold3.model.config import OptimConfig
from deepfold3.model.evoformer import PairFormerBlock

C_PAIR = 128
C_SINGLE = 384
DTYPE = torch.bfloat16

# Candidate flags (all default-off in training) we want to finalize.
CANDIDATES = ["use_fused_swiglu", "use_fused_transition_pair", "use_fused_triangle_mul", "use_fused_pair_bias"]


def make_optim(**overrides) -> OptimConfig:
    """Training baseline: fused evo-attn + fused dropout on, candidates off."""
    cfg = OptimConfig()
    cfg.use_fused_evo_attention = True
    cfg.use_fused_dropout_add = True
    for k in CANDIDATES:
        setattr(cfg, k, False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_stack(optim: OptimConfig, n_blocks: int) -> torch.nn.Module:
    blocks = torch.nn.ModuleList(
        [PairFormerBlock(c_pair=C_PAIR, c_single=C_SINGLE, optim=optim) for _ in range(n_blocks)]
    )
    return blocks.cuda().to(DTYPE)


def make_inputs(n: int, requires_grad: bool):
    g = torch.Generator(device="cuda").manual_seed(0)
    pair = torch.randn(1, n, n, C_PAIR, device="cuda", dtype=DTYPE, generator=g)
    single = torch.randn(1, n, C_SINGLE, device="cuda", dtype=DTYPE, generator=g)
    pair_mask = torch.ones(1, n, n, device="cuda", dtype=DTYPE)
    seq_mask = torch.ones(1, n, device="cuda", dtype=DTYPE)
    if requires_grad:
        pair.requires_grad_(True)
        single.requires_grad_(True)
    return pair, single, pair_mask, seq_mask


def run_stack(blocks, pair, single, pair_mask, seq_mask, checkpoint: bool):
    from torch.utils.checkpoint import checkpoint as ckpt

    for blk in blocks:
        if checkpoint and torch.is_grad_enabled():
            pair, single = ckpt(blk, pair, single, pair_mask, seq_mask, use_reentrant=False)
        else:
            pair, single = blk(pair, single, pair_mask, seq_mask)
    return pair, single


def bench_training(blocks, n: int, iters: int, warmup: int) -> dict:
    blocks.train()
    pair, single, pair_mask, seq_mask = make_inputs(n, requires_grad=True)

    def step():
        for p in blocks.parameters():
            p.grad = None
        pair.grad = None
        single.grad = None
        with torch.autocast("cuda", dtype=DTYPE):
            out_pair, out_single = run_stack(blocks, pair, single, pair_mask, seq_mask, checkpoint=True)
            loss = out_pair.float().pow(2).mean() + out_single.float().pow(2).mean()
        loss.backward()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return {"ms_median": statistics.median(times), "ms_p10": min(times), "peak_gb": peak_gb}


def bench_inference(blocks, n: int, iters: int, warmup: int) -> dict:
    blocks.eval()
    pair, single, pair_mask, seq_mask = make_inputs(n, requires_grad=False)

    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
            run_stack(blocks, pair, single, pair_mask, seq_mask, checkpoint=False)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return {"ms_median": statistics.median(times), "ms_p10": min(times), "peak_gb": peak_gb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--regime", choices=["training", "inference", "both"], default="both")
    ap.add_argument("--buckets", type=int, nargs="+", default=[256, 384, 512])
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}  blocks={args.blocks}  N={args.n}  dtype={DTYPE}")
    print(f"candidates: {CANDIDATES}\n")

    configs = {"baseline (evo_attn+dropout)": {}}
    for flag in CANDIDATES:
        configs[f"+{flag.replace('use_fused_', '')}"] = {flag: True}
    # The training-safe trio (excludes pair_bias, the bf16 grad-blowup hazard).
    configs["+3 safe (no pair_bias)"] = {
        "use_fused_swiglu": True,
        "use_fused_transition_pair": True,
        "use_fused_triangle_mul": True,
    }
    configs["+ALL four"] = {f: True for f in CANDIDATES}

    if args.regime in ("training", "both"):
        print(f"=== TRAINING (fwd+bwd, gradient checkpointing, N={args.n}) ===")
        base = None
        for name, ov in configs.items():
            torch.manual_seed(0)
            blocks = build_stack(make_optim(**ov), args.blocks)
            r = bench_training(blocks, args.n, args.iters, args.warmup)
            if base is None:
                base = r["ms_median"]
                base_mem = r["peak_gb"]
            spd = base / r["ms_median"]
            dmem = r["peak_gb"] - base_mem
            print(
                f"  {name:32s} {r['ms_median']:7.2f} ms  ({spd:4.2f}x)   "
                f"peak {r['peak_gb']:6.2f} GB ({dmem:+.2f})"
            )
            del blocks
            torch.cuda.empty_cache()
        print()

    if args.regime in ("inference", "both"):
        print("=== INFERENCE (fwd only, no checkpointing) ===")
        for name, ov in configs.items():
            line = f"  {name:32s}"
            torch.manual_seed(0)
            blocks = build_stack(make_optim(**ov), args.blocks)
            for nb in args.buckets:
                r = bench_inference(blocks, nb, args.iters, args.warmup)
                line += f"  N={nb}:{r['ms_median']:6.2f}ms/{r['peak_gb']:4.1f}GB"
            print(line)
            del blocks
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
