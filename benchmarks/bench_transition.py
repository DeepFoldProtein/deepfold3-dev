"""Benchmark TransitionBlock and AdaTransitionBlock.

Compares PyTorch baseline vs torch.compile vs Triton FusedTransitionBlock.

Note: Liger SwiGLUMLP benchmarks removed — not API-compatible and consistently
slower than Triton FusedTransitionBlock at beneficial shapes (C>=256).

Usage:
    uv run python benchmarks/bench_transition.py
    uv run python benchmarks/bench_transition.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

from benchmarks.utils import (
    bench_fwd_full,
    check_cuda,
    dry_run_shapes,
    save_results,
    try_compile,
    try_import,
)

SHAPES = [
    {"N": 256, "C": 64},
    {"N": 256, "C": 128},
    {"N": 256, "C": 384},
    {"N": 512, "C": 64},
    {"N": 512, "C": 128},
    {"N": 512, "C": 384},
    {"N": 1024, "C": 128},
    {"N": 1024, "C": 384},
    {"N": 2048, "C": 128},
    {"N": 2048, "C": 384},
]

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def _bench_module(op_name, provider, shape, module, x, do, warmup, rep):
    """Benchmark a module's forward and full (fwd+bwd) pass."""

    def fwd():
        return module(x)

    def full():
        o = module(x)
        o.backward(do, retain_graph=True)

    return bench_fwd_full(op_name, provider, shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])


def _bench_ada_module(op_name, provider, shape, module, x, cond, do, warmup, rep):
    def fwd():
        return module(x, cond)

    def full():
        o = module(x, cond)
        o.backward(do, retain_graph=True)

    return bench_fwd_full(op_name, provider, shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])


def bench_transition_block(shapes: list[dict], warmup: int, rep: int) -> list:
    from deepfold3.model.primitives import TransitionBlock

    FusedTransitionBlock = try_import(  # noqa: N806
        "deepfold3.model.triton.fused_transition", "FusedTransitionBlock"
    )

    results = []
    for shape in shapes:
        N, C = shape["N"], shape["C"]
        _free_gpu()

        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

        tb = TransitionBlock(C).to(DEVICE, DTYPE).eval()
        results.extend(_bench_module("TransitionBlock", "pytorch", shape, tb, x, do, warmup, rep))

        # torch.compile
        tb_c = try_compile(TransitionBlock(C).to(DEVICE, DTYPE).eval())
        if tb_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            results.extend(_bench_module("TransitionBlock", "compile", shape, tb_c, x_c, do, warmup, rep))
            del tb_c, x_c

        if FusedTransitionBlock is not None:
            ft = FusedTransitionBlock(C).to(DEVICE, DTYPE).eval()
            ft.load_state_dict(tb.state_dict())
            x2 = x.detach().clone().requires_grad_(True)
            results.extend(_bench_module("TransitionBlock", "triton", shape, ft, x2, do, warmup, rep))
            del ft, x2
        else:
            print(f"  [skip] Triton FusedTransitionBlock not available (N={N}, C={C})")

        del tb, x, do
        print(f"  TransitionBlock N={N} C={C} done")

    return results


def bench_ada_transition_block(shapes: list[dict], warmup: int, rep: int) -> list:
    from deepfold3.model.primitives import AdaTransitionBlock

    results = []
    for shape in shapes:
        N, C = shape["N"], shape["C"]
        _free_gpu()

        cond_dim = C
        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        cond = torch.randn(N, cond_dim, device=DEVICE, dtype=DTYPE)
        do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

        atb = AdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval()
        results.extend(_bench_ada_module("AdaTransitionBlock", "pytorch", shape, atb, x, cond, do, warmup, rep))

        # torch.compile
        atb_c = try_compile(AdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval())
        if atb_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            results.extend(_bench_ada_module("AdaTransitionBlock", "compile", shape, atb_c, x_c, cond, do, warmup, rep))
            del atb_c, x_c

        del atb, x, cond, do
        print(f"  AdaTransitionBlock N={N} C={C} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransitionBlock")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with tiny shapes")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    shapes = dry_run_shapes(SHAPES, args.dry_run)
    warmup = 5 if args.dry_run else args.warmup
    rep = 5 if args.dry_run else args.rep

    all_results = []
    print("=== TransitionBlock ===")
    all_results.extend(bench_transition_block(shapes, warmup, rep))
    print("=== AdaTransitionBlock ===")
    all_results.extend(bench_ada_transition_block(shapes, warmup, rep))

    save_results(all_results, "transition.json")
    print(f"Done. {len(all_results)} measurements.")


if __name__ == "__main__":
    main()
