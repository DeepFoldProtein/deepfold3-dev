"""Benchmark pair operations: TriangleMulOutgoing, TriangleMulIncoming, OuterProductMean.

Compares PyTorch baseline vs torch.compile (both standalone and integrated via OptimConfig).

Usage:
    uv run python benchmarks/bench_pair_ops.py
    uv run python benchmarks/bench_pair_ops.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

import torch

from benchmarks.utils import (
    bench_fwd_full,
    check_cuda,
    dry_run_shapes,
    save_results,
    try_compile,
)

TRI_SHAPES = [
    {"N": 128, "C": 128},
    {"N": 256, "C": 128},
    {"N": 384, "C": 128},
    {"N": 512, "C": 128},
    {"N": 768, "C": 128},
]

OPM_SHAPES = [
    {"N": 128, "S": 32, "C_msa": 64, "C_pair": 128},
    {"N": 256, "S": 32, "C_msa": 64, "C_pair": 128},
    {"N": 384, "S": 32, "C_msa": 64, "C_pair": 128},
    {"N": 512, "S": 16, "C_msa": 64, "C_pair": 128},
]

DEVICE = "cuda"
DTYPE = torch.bfloat16


def bench_triangle_mul(shapes, warmup, rep):
    from deepfold3.model.config import OptimConfig
    from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing

    results = []
    for shape in shapes:
        N, C = shape["N"], shape["C"]

        for cls, name in [(TriangleMulOutgoing, "TriangleMulOutgoing"), (TriangleMulIncoming, "TriangleMulIncoming")]:
            # --- Baseline ---
            mod = cls(C).to(DEVICE, DTYPE).eval()
            z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
            mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)
            do = torch.randn_like(z)

            def fwd(mod=mod, z=z, mask=mask):
                return mod(z, mask)

            def full(mod=mod, z=z, mask=mask, do=do):
                o = mod(z, mask)
                o.backward(do, retain_graph=True)

            results.extend(bench_fwd_full(name, "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[z]))

            # --- Standalone torch.compile (wraps entire module externally) ---
            mod_c = try_compile(cls(C).to(DEVICE, DTYPE).eval())
            if mod_c is not None:
                z_c = z.detach().clone().requires_grad_(True)

                def fwd_c(mod_c=mod_c, z_c=z_c, mask=mask):
                    return mod_c(z_c, mask)

                def full_c(mod_c=mod_c, z_c=z_c, mask=mask, do=do):
                    o = mod_c(z_c, mask)
                    o.backward(do, retain_graph=True)

                try:
                    results.extend(bench_fwd_full(name, "compile", shape, fwd_c, full_c, warmup=warmup, rep=rep, grad_to_none=[z_c]))
                except Exception as e:
                    print(f"  [skip] torch.compile {name} failed (N={N}): {e}")
                del mod_c, z_c

            # --- Integrated: TriangleMul(optim=OptimConfig(use_torch_compile=True)) ---
            try:
                optim = OptimConfig(use_torch_compile=True)
                mod_int = cls(C, optim=optim).to(DEVICE, DTYPE).eval()
                mod_int.load_state_dict(mod.state_dict())
                z_int = z.detach().clone().requires_grad_(True)

                def fwd_int(mod_int=mod_int, z_int=z_int, mask=mask):
                    return mod_int(z_int, mask)

                def full_int(mod_int=mod_int, z_int=z_int, mask=mask, do=do):
                    o = mod_int(z_int, mask)
                    o.backward(do, retain_graph=True)

                results.extend(bench_fwd_full(name, "compile_integrated", shape, fwd_int, full_int, warmup=warmup, rep=rep, grad_to_none=[z_int]))
                del mod_int, z_int
            except Exception as e:
                print(f"  [skip] Integrated compile {name} failed (N={N}): {e}")

            print(f"  {name} N={N} C={C} done")

    return results


def bench_outer_product_mean(shapes, warmup, rep):
    from deepfold3.model.ops import OuterProductMean

    results = []
    for shape in shapes:
        N, S = shape["N"], shape["S"]
        C_msa, C_pair = shape["C_msa"], shape["C_pair"]

        mod = OuterProductMean(C_msa, C_pair).to(DEVICE, DTYPE).eval()
        msa = torch.randn(S, N, C_msa, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(S, N, device=DEVICE, dtype=DTYPE)
        do = torch.randn(N, N, C_pair, device=DEVICE, dtype=DTYPE)

        def fwd(mod=mod, msa=msa, mask=mask):
            return mod(msa, mask)

        def full(mod=mod, msa=msa, mask=mask, do=do):
            o = mod(msa, mask)
            o.backward(do, retain_graph=True)

        results.extend(bench_fwd_full("OuterProductMean", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[msa]))

        # torch.compile
        mod_c = try_compile(OuterProductMean(C_msa, C_pair).to(DEVICE, DTYPE).eval())
        if mod_c is not None:
            msa_c = msa.detach().clone().requires_grad_(True)

            def fwd_c(mod_c=mod_c, msa_c=msa_c, mask=mask):
                return mod_c(msa_c, mask)

            def full_c(mod_c=mod_c, msa_c=msa_c, mask=mask, do=do):
                o = mod_c(msa_c, mask)
                o.backward(do, retain_graph=True)

            try:
                results.extend(bench_fwd_full("OuterProductMean", "compile", shape, fwd_c, full_c, warmup=warmup, rep=rep, grad_to_none=[msa_c]))
            except Exception as e:
                print(f"  [skip] torch.compile OuterProductMean failed (N={N}): {e}")
            del mod_c, msa_c

        print(f"  OuterProductMean N={N} S={S} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark pair operations")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with tiny shapes")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    warmup = 5 if args.dry_run else args.warmup
    rep = 5 if args.dry_run else args.rep

    all_results = []

    print("=== Triangle Multiplication ===")
    shapes = dry_run_shapes(TRI_SHAPES, args.dry_run)
    all_results.extend(bench_triangle_mul(shapes, warmup, rep))

    print("=== Outer Product Mean ===")
    shapes = dry_run_shapes(OPM_SHAPES, args.dry_run)
    all_results.extend(bench_outer_product_mean(shapes, warmup, rep))

    save_results(all_results, "pair_ops.json")
    print(f"Done. {len(all_results)} measurements.")


if __name__ == "__main__":
    main()
