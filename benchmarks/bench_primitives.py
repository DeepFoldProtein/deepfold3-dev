"""Benchmark primitives: LayerNorm, SwiGLU, LinearNoBias, LayerNormLinear.

Compares PyTorch baseline vs torch.compile vs Triton equivalents.

Note: Liger LayerNorm/SwiGLU benchmarks removed — they are consistently slower
than PyTorch's fused CUDA kernels at DeepFold3's channel dimensions (64-384).

Usage:
    uv run python benchmarks/bench_primitives.py
    uv run python benchmarks/bench_primitives.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.utils import (
    bench_fwd_full,
    check_cuda,
    dry_run_shapes,
    save_results,
    try_compile,
    try_import,
)

SHAPES = [
    {"N": 256, "C": 128},
    {"N": 512, "C": 128},
    {"N": 1024, "C": 128},
    {"N": 2048, "C": 128},
    {"N": 256, "C": 384},
    {"N": 512, "C": 384},
    {"N": 1024, "C": 384},
    {"N": 2048, "C": 384},
]

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _make_fwd_full(module, x, do):
    """Create fwd/full closures with bound variables."""

    def fwd():
        return module(x)

    def full():
        o = module(x)
        o.backward(do, retain_graph=True)

    return fwd, full


def bench_layernorm(shapes: list[dict], warmup: int, rep: int) -> list:
    results = []
    for shape in shapes:
        N, C = shape["N"], shape["C"]
        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn_like(x)

        ln = nn.LayerNorm(C).to(DEVICE, DTYPE)
        fwd, full = _make_fwd_full(ln, x, do)
        results.extend(
            bench_fwd_full("LayerNorm", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])
        )

        # torch.compile (known to be slower at these sizes, included for reference)
        ln_c = try_compile(nn.LayerNorm(C).to(DEVICE, DTYPE))
        if ln_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            fwd_c, full_c = _make_fwd_full(ln_c, x_c, do)
            results.extend(
                bench_fwd_full("LayerNorm", "compile", shape, fwd_c, full_c, warmup=warmup, rep=rep, grad_to_none=[x_c])
            )

        print(f"  LayerNorm N={N} C={C} done")

    return results


# SwiGLU realistic shapes: pair transitions (M=N², C=128) and single transitions (M=N, C=384).
# Pair transitions dominate runtime; single transitions are too small for Triton to win.
SWIGLU_SHAPES = [
    {"M": 384 * 384, "C": 128, "label": "pair N=384"},
    {"M": 512 * 512, "C": 128, "label": "pair N=512"},
    {"M": 640 * 640, "C": 128, "label": "pair N=640"},
    {"M": 768 * 768, "C": 128, "label": "pair N=768"},
    {"M": 384, "C": 384, "label": "single N=384"},
    {"M": 768, "C": 384, "label": "single N=768"},
]


def bench_swiglu(shapes: list[dict], warmup: int, rep: int) -> list:
    import gc

    results = []
    FusedSwiGLU = try_import("deepfold3.model.triton.swiglu", "FusedSwiGLU")  # noqa: N806

    swiglu_shapes = dry_run_shapes(SWIGLU_SHAPES, len(shapes) == 1 and shapes[0].get("N", 0) <= 4)

    for shape in swiglu_shapes:
        M, C = shape["M"], shape["C"]
        gc.collect()
        torch.cuda.empty_cache()
        x = torch.randn(M, C * 2, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(M, C, device=DEVICE, dtype=DTYPE)

        def fwd(x=x):
            a, b = x.chunk(2, dim=-1)
            return F.silu(a) * b

        def full(x=x, do=do):
            a, b = x.chunk(2, dim=-1)
            o = F.silu(a) * b
            o.backward(do, retain_graph=True)

        bench_shape = {"M": M, "C": C}
        label = shape.get("label", f"M={M}")
        kind = label.split()[0] if " " in label else "pair"
        op_name = f"SwiGLU ({kind})"
        results.extend(
            bench_fwd_full(op_name, "pytorch", bench_shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])
        )

        if FusedSwiGLU is not None:
            swiglu = FusedSwiGLU()
            x2 = x.detach().clone().requires_grad_(True)

            def fwd_t(x2=x2, swiglu=swiglu):
                return swiglu(x2)

            def full_t(x2=x2, swiglu=swiglu, do=do):
                o = swiglu(x2)
                o.backward(do, retain_graph=True)

            results.extend(
                bench_fwd_full(op_name, "triton", bench_shape, fwd_t, full_t, warmup=warmup, rep=rep, grad_to_none=[x2])
            )
            del swiglu, x2
        else:
            print("  [skip] Triton FusedSwiGLU not available")

        label = shape.get("label", f"M={M}")
        print(f"  SwiGLU {label} C={C} done")
        del x, do

    return results


def bench_layernorm_linear(shapes: list[dict], warmup: int, rep: int) -> list:
    """Benchmark fused LayerNorm+Linear vs sequential PyTorch baseline."""
    FusedLayerNormLinear = try_import("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear")  # noqa: N806

    LNL_SHAPES = [
        {"N": 256, "C": 128, "C_out": 1024},
        {"N": 512, "C": 128, "C_out": 1024},
        {"N": 1024, "C": 128, "C_out": 1024},
        {"N": 2048, "C": 128, "C_out": 1024},
        {"N": 256, "C": 384, "C_out": 3072},
        {"N": 512, "C": 384, "C_out": 3072},
        {"N": 1024, "C": 384, "C_out": 3072},
    ]
    lnl_shapes = dry_run_shapes(LNL_SHAPES, len(shapes) == 1 and shapes[0].get("N", 0) <= 4)

    results = []
    for shape in lnl_shapes:
        N, C, C_out = shape["N"], shape["C"], shape["C_out"]
        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(N, C_out, device=DEVICE, dtype=DTYPE)

        # Baseline: LN + Linear (no bias)
        ln = nn.LayerNorm(C).to(DEVICE, DTYPE)
        linear = nn.Linear(C, C_out, bias=False).to(DEVICE, DTYPE)

        def fwd_base(x=x, ln=ln, linear=linear):
            return linear(ln(x))

        def full_base(x=x, ln=ln, linear=linear, do=do):
            o = linear(ln(x))
            o.backward(do, retain_graph=True)

        results.extend(
            bench_fwd_full(
                "LayerNormLinear", "pytorch", shape, fwd_base, full_base, warmup=warmup, rep=rep, grad_to_none=[x]
            )
        )

        # torch.compile
        ln_c = nn.LayerNorm(C).to(DEVICE, DTYPE)
        linear_c = nn.Linear(C, C_out, bias=False).to(DEVICE, DTYPE)
        with torch.no_grad():
            ln_c.weight.copy_(ln.weight)
            ln_c.bias.copy_(ln.bias)
            linear_c.weight.copy_(linear.weight)

        @torch.compile
        def _lnl_compiled(x, ln_c=ln_c, linear_c=linear_c):
            return linear_c(ln_c(x))

        x_c = x.detach().clone().requires_grad_(True)

        def fwd_comp(x_c=x_c):
            return _lnl_compiled(x_c)

        def full_comp(x_c=x_c, do=do):
            o = _lnl_compiled(x_c)
            o.backward(do, retain_graph=True)

        results.extend(
            bench_fwd_full(
                "LayerNormLinear", "compile", shape, fwd_comp, full_comp, warmup=warmup, rep=rep, grad_to_none=[x_c]
            )
        )

        if FusedLayerNormLinear is not None:
            fused = FusedLayerNormLinear(C, C_out, has_linear_bias=False).to(DEVICE, DTYPE)
            with torch.no_grad():
                fused.weight.copy_(ln.weight)
                fused.bias.copy_(ln.bias)
                fused.linear_weight.copy_(linear.weight.T)
            x2 = x.detach().clone().requires_grad_(True)

            def fwd_fused(x2=x2, fused=fused):
                return fused(x2)

            def full_fused(x2=x2, fused=fused, do=do):
                o = fused(x2)
                o.backward(do, retain_graph=True)

            try:
                fused(x2)  # smoke test
                results.extend(
                    bench_fwd_full(
                        "LayerNormLinear",
                        "triton",
                        shape,
                        fwd_fused,
                        full_fused,
                        warmup=warmup,
                        rep=rep,
                        grad_to_none=[x2],
                    )
                )
            except Exception as e:
                print(f"  [skip] Triton FusedLayerNormLinear failed (N={N}, C={C}->{C_out}): {e}")
        else:
            print(f"  [skip] Triton FusedLayerNormLinear not available (N={N}, C={C})")

        print(f"  LayerNormLinear N={N} C={C}->{C_out} done")

    return results


def bench_linear(shapes: list[dict], warmup: int, rep: int) -> list:
    results = []

    for shape in shapes:
        N, C = shape["N"], shape["C"]
        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
        linear = nn.Linear(C, C, bias=False).to(DEVICE, DTYPE)

        fwd, full = _make_fwd_full(linear, x, do)
        results.extend(
            bench_fwd_full("LinearNoBias", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])
        )

        # torch.compile (known to be slower — cuBLAS already optimal)
        linear_c = try_compile(nn.Linear(C, C, bias=False).to(DEVICE, DTYPE))
        if linear_c is not None:
            x_c = x.detach().clone().requires_grad_(True)
            fwd_c, full_c = _make_fwd_full(linear_c, x_c, do)
            results.extend(
                bench_fwd_full(
                    "LinearNoBias", "compile", shape, fwd_c, full_c, warmup=warmup, rep=rep, grad_to_none=[x_c]
                )
            )

        print(f"  LinearNoBias N={N} C={C} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark primitives")
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
    print("=== LayerNorm ===")
    all_results.extend(bench_layernorm(shapes, warmup, rep))
    print("=== SwiGLU ===")
    all_results.extend(bench_swiglu(shapes, warmup, rep))
    print("=== LayerNormLinear ===")
    all_results.extend(bench_layernorm_linear(shapes, warmup, rep))
    print("=== LinearNoBias ===")
    all_results.extend(bench_linear(shapes, warmup, rep))

    save_results(all_results, "primitives.json")
    print(f"Done. {len(all_results)} measurements.")


if __name__ == "__main__":
    main()
