"""Old vs new inference MSA transition: peak GPU memory, wall time and numerics.

Timing is warm for both paths: the first call of each path and shape pays Triton
compilation and, on the old path, autotune. It is reported as ``cold`` and excluded from
``warm``. Run each mode in its own process so one path's cache state cannot flatter the other.

Usage:
    python benchmarks/bench_msa_transition.py transition --path old --S 16000 --N 566 --C 64
    python benchmarks/bench_msa_transition.py block --path new --S 15986 --N 566 --C 128
    python benchmarks/bench_msa_transition.py numerics --S 4000 --N 566 --C 64
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from deepfold3.model.config import OptimConfig
from deepfold3.model.evoformer import EvoformerBlock
from deepfold3.model.primitives import TransitionBlock
from deepfold3.model.triton.fused_transition import FusedTransitionBlock
from deepfold3.model.triton.msa_transition import fused_msa_transition, msa_transition_residual_

GIB = 2**30


def _autocast(dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16")


def _measure(fn: Callable[[], None], repeats: int) -> None:
    """Print cold and warm wall time and the peak allocation above the starting level."""
    timings, peaks = [], []
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        peaks.append((torch.cuda.max_memory_allocated() - base) / GIB)
    print(
        f"  cold {timings[0]:.3f}s  warm min {min(timings[1:]):.3f}s  warm mean "
        f"{sum(timings[1:]) / repeats:.3f}s  peak above input {max(peaks[1:]):.2f} GiB",
        flush=True,
    )


def bench_transition(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    block = FusedTransitionBlock(args.C).cuda().eval()
    torch.nn.init.normal_(block.linear2.weight, std=0.02)
    x = torch.randn(args.S, args.N, args.C, device="cuda")
    if args.dtype == "bf16":
        x = x.bfloat16()
    work = x.clone()

    def old() -> None:
        _ = x + 1.0 * block(x)

    def new() -> None:
        work.copy_(x)  # no allocation; the kernel overwrites its input
        msa_transition_residual_(block, work, 1.0)

    print(f"transition {args.path} S={args.S} N={args.N} C={args.C} {args.dtype} {args.matmul}", flush=True)
    with torch.inference_mode(), _autocast(args.dtype):
        _measure(old if args.path == "old" else new, args.repeats)


def bench_block(args: argparse.Namespace) -> None:
    """Whole EvoformerBlock, the unit that has to fit for the run to survive."""
    torch.manual_seed(0)
    optim = OptimConfig.inference()
    optim.use_fused_msa_transition_inference = args.path == "new"
    block = EvoformerBlock(args.C, 128, optim=optim).cuda().eval()
    msa = torch.randn(1, args.S, args.N, args.C, device="cuda")
    pair = torch.randn(1, args.N, args.N, 128, device="cuda")
    msa_mask = torch.ones(1, args.S, args.N, device="cuda")
    pair_mask = torch.ones(1, args.N, args.N, device="cuda")

    print(f"block {args.path} S={args.S} N={args.N} C={args.C} {args.dtype} {args.matmul}", flush=True)
    with torch.inference_mode(), _autocast(args.dtype):
        _measure(lambda: block(msa, pair, msa_mask, pair_mask), args.repeats)


def _report(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (got.double() - ref.double()).abs()
    rel = diff / ref.double().abs().clamp_min(1e-30)
    scale = ref.double().abs().max()
    print(
        f"  {name}: max abs {diff.max():.3e}  max abs / max|ref| {diff.max() / scale:.3e}  "
        f"rel median {rel.median():.3e}  rel p99.9 {torch.quantile(rel[::97].flatten(), 0.999):.3e}  "
        f"rel max {rel.max():.3e}",
        flush=True,
    )


def numerics(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    fused = FusedTransitionBlock(args.C).cuda().eval()
    torch.nn.init.normal_(fused.linear2.weight, std=0.02)
    stock = TransitionBlock(args.C).cuda().eval()
    stock.load_state_dict(fused.state_dict())
    x = torch.randn(args.S, args.N, args.C, device="cuda")
    weights = (fused.norm.weight, fused.norm.bias, fused.linear1.weight, fused.linear2.weight)

    with torch.inference_mode():
        truth = TransitionBlock(args.C).cuda().double().eval()
        truth.load_state_dict(fused.state_dict())
        ref64 = truth(x.double())
        for dtype in ("bf16", "fp32"):
            with _autocast(dtype):
                new = fused_msa_transition(x, *weights)
                old = fused(x)
                plain = stock(x)
                print(
                    f"S={args.S} N={args.N} C={args.C} {dtype} {args.matmul}; update, |ref| max {old.abs().max():.3f}"
                )
                _report("new vs old fused", new, old)
                _report("new vs TransitionBlock", new, plain)
                _report("new vs fp64 truth", new, ref64)
                _report("old fused vs fp64 truth", old, ref64)
                _report("TransitionBlock vs fp64", plain, ref64)
                residual = fused_msa_transition(x, *weights, residual_scale=1.0)
                _report("x + update, new vs old", residual, x + old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["transition", "block", "numerics"])
    parser.add_argument("--path", choices=["old", "new"], default="new")
    parser.add_argument("--S", type=int, default=16000)
    parser.add_argument("--N", type=int, default=566)
    parser.add_argument("--C", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--matmul", choices=["highest", "high"], default="high")  # 'high' is the CLI default
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_float32_matmul_precision(args.matmul)
    {"transition": bench_transition, "block": bench_block, "numerics": numerics}[args.mode](args)


if __name__ == "__main__":
    main()
