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
import sys

import torch

from benchmarks.utils import (
    bench_fwd_full,
    check_cuda,
    dry_run_shapes,
    free_gpu,
    save_results,
    try_compile,
    try_import,
)

# Realistic shapes: pair transitions use M=N² rows, single/diffusion use M=N rows.
# C=64: MSA transition (M=S*N, S~16-32 — small, similar to single)
# C=128: pair transition (M=N² — the big one)
# C=384: single transition in PairFormer (M=N)
SHAPES_PAIR = [
    {"M": 384 * 384, "C": 128, "label": "pair N=384"},
    {"M": 512 * 512, "C": 128, "label": "pair N=512"},
    {"M": 640 * 640, "C": 128, "label": "pair N=640"},
    {"M": 768 * 768, "C": 128, "label": "pair N=768"},
]
SHAPES_SINGLE = [
    {"M": 384, "C": 384, "label": "single N=384"},
    {"M": 768, "C": 384, "label": "single N=768"},
    {"M": 512 * 16, "C": 64, "label": "MSA N=512 S=16"},
    {"M": 384 * 32, "C": 64, "label": "MSA N=384 S=32"},
]
SHAPES_ADA = [
    {"M": 384, "C": 384, "label": "diff N=384"},
    {"M": 768, "C": 384, "label": "diff N=768"},
]

DEVICE = "cuda"
DTYPE = torch.bfloat16


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

    all_shapes = SHAPES_PAIR + SHAPES_SINGLE
    if len(shapes) == 1 and shapes[0].get("N", 0) <= 4:
        all_shapes = [{"M": 16, "C": 128, "label": "dry"}]

    results = []
    for shape in all_shapes:
        M, C = shape["M"], shape["C"]
        label = shape.get("label", f"M={M}")
        free_gpu()

        bench_shape = {"M": M, "C": C}
        # Use label prefix as op name so pair/single/MSA get separate plot rows
        kind = label.split()[0] if " " in label else "pair"
        op_name = f"Transition ({kind})"

        x = torch.randn(M, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(M, C, device=DEVICE, dtype=DTYPE)

        tb = TransitionBlock(C).to(DEVICE, DTYPE).eval()
        results.extend(_bench_module(op_name, "pytorch", bench_shape, tb, x, do, warmup, rep))

        # torch.compile
        tb_c = try_compile(TransitionBlock(C).to(DEVICE, DTYPE).eval())
        if tb_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            results.extend(_bench_module(op_name, "compile", bench_shape, tb_c, x_c, do, warmup, rep))
            del tb_c, x_c

        if FusedTransitionBlock is not None:
            ft = FusedTransitionBlock(C).to(DEVICE, DTYPE).eval()
            ft.load_state_dict(tb.state_dict())
            x2 = x.detach().clone().requires_grad_(True)
            results.extend(_bench_module(op_name, "triton", bench_shape, ft, x2, do, warmup, rep))
            del ft, x2
        else:
            print(f"  [skip] Triton FusedTransitionBlock not available ({label})")

        del tb, x, do
        print(f"  TransitionBlock {label} C={C} done")

    return results


def bench_ada_transition_block(shapes: list[dict], warmup: int, rep: int) -> list:
    from deepfold3.model.primitives import AdaTransitionBlock

    FusedAdaTransitionBlock = try_import(  # noqa: N806
        "deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock"
    )

    ada_shapes = SHAPES_ADA
    if len(shapes) == 1 and shapes[0].get("N", 0) <= 4:
        ada_shapes = [{"M": 16, "C": 64, "label": "dry"}]

    results = []
    for shape in ada_shapes:
        M, C = shape["M"], shape["C"]
        label = shape.get("label", f"M={M}")
        free_gpu()

        cond_dim = C
        bench_shape = {"M": M, "C": C}
        x = torch.randn(M, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        cond = torch.randn(M, cond_dim, device=DEVICE, dtype=DTYPE)
        do = torch.randn(M, C, device=DEVICE, dtype=DTYPE)

        atb = AdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval()
        results.extend(_bench_ada_module("AdaTransitionBlock", "pytorch", bench_shape, atb, x, cond, do, warmup, rep))

        # torch.compile
        atb_c = try_compile(AdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval())
        if atb_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            results.extend(_bench_ada_module("AdaTransitionBlock", "compile", bench_shape, atb_c, x_c, cond, do, warmup, rep))
            del atb_c, x_c

        # Triton fused
        if FusedAdaTransitionBlock is not None:
            ft = FusedAdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval()
            ft.load_state_dict(atb.state_dict(), strict=False)
            x2 = x.detach().clone().requires_grad_(True)
            results.extend(_bench_ada_module("AdaTransitionBlock", "triton", bench_shape, ft, x2, cond, do, warmup, rep))
            del ft, x2
        else:
            print(f"  [skip] Triton FusedAdaTransitionBlock not available ({label})")

        del atb, x, cond, do
        print(f"  AdaTransitionBlock {label} C={C} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransitionBlock")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with tiny shapes")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    # shapes is passed through so functions can detect dry-run mode
    shapes = dry_run_shapes(SHAPES_PAIR, args.dry_run)
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
