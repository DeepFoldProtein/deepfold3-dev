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
    free_gpu,
    save_results,
    try_compile,
    try_import,
)

TRI_SHAPES = [
    {"N": 384, "C": 128},
    {"N": 512, "C": 128},
    {"N": 640, "C": 128},
    {"N": 768, "C": 128},
]

OPM_SHAPES = [
    {"N": 384, "S": 32, "C_msa": 64, "C_pair": 128},
    {"N": 512, "S": 16, "C_msa": 64, "C_pair": 128},
    {"N": 640, "S": 16, "C_msa": 64, "C_pair": 128},
    {"N": 768, "S": 16, "C_msa": 64, "C_pair": 128},
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
            free_gpu()

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
                    results.extend(
                        bench_fwd_full(
                            name, "compile", shape, fwd_c, full_c, warmup=warmup, rep=rep, grad_to_none=[z_c]
                        )
                    )
                except Exception as e:
                    print(f"  [skip] torch.compile {name} failed (N={N}): {e}")
                del mod_c, z_c
            free_gpu()

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

                results.extend(
                    bench_fwd_full(
                        name,
                        "compile_integrated",
                        shape,
                        fwd_int,
                        full_int,
                        warmup=warmup,
                        rep=rep,
                        grad_to_none=[z_int],
                    )
                )
                del mod_int, z_int
            except Exception as e:
                print(f"  [skip] Integrated compile {name} failed (N={N}): {e}")
            free_gpu()

            # --- Triton fused triangle mul (Level 1+2: gate + center_proj) ---
            try:
                optim_tri = OptimConfig(use_fused_triangle_mul=True)
                mod_tri = cls(C, optim=optim_tri).to(DEVICE, DTYPE).eval()
                mod_tri.load_state_dict(mod.state_dict(), strict=False)
                # Disable dual GEMM to isolate Level 1+2 performance
                mod_tri._fused_dual_gemm = None
                z_tri = z.detach().clone().requires_grad_(True)

                def fwd_tri(mod_tri=mod_tri, z_tri=z_tri, mask=mask):
                    return mod_tri(z_tri, mask)

                def full_tri(mod_tri=mod_tri, z_tri=z_tri, mask=mask, do=do):
                    o = mod_tri(z_tri, mask)
                    o.backward(do, retain_graph=True)

                results.extend(
                    bench_fwd_full(
                        name, "triton_l1+l2", shape, fwd_tri, full_tri, warmup=warmup, rep=rep, grad_to_none=[z_tri]
                    )
                )
                del mod_tri, z_tri
            except Exception as e:
                print(f"  [skip] Triton L1+L2 {name} failed (N={N}): {e}")
            free_gpu()

            # --- Triton fused triangle mul (Level 3: dual GEMM + center_proj) ---
            try:
                optim_full = OptimConfig(use_fused_triangle_mul=True)
                mod_full = cls(C, optim=optim_full).to(DEVICE, DTYPE).eval()
                mod_full.load_state_dict(mod.state_dict(), strict=False)
                z_full = z.detach().clone().requires_grad_(True)

                def fwd_full(mod_full=mod_full, z_full=z_full, mask=mask):
                    return mod_full(z_full, mask)

                def full_full(mod_full=mod_full, z_full=z_full, mask=mask, do=do):
                    o = mod_full(z_full, mask)
                    o.backward(do, retain_graph=True)

                results.extend(
                    bench_fwd_full(
                        name, "triton_l3", shape, fwd_full, full_full, warmup=warmup, rep=rep, grad_to_none=[z_full]
                    )
                )
                del mod_full, z_full
            except Exception as e:
                print(f"  [skip] Triton L3 {name} failed (N={N}): {e}")

            print(f"  {name} N={N} C={C} done")

    return results


def bench_outer_product_mean(shapes, warmup, rep):
    from deepfold3.model.ops import OuterProductMean

    results = []
    for shape in shapes:
        N, S = shape["N"], shape["S"]
        C_msa, C_pair = shape["C_msa"], shape["C_pair"]
        free_gpu()

        mod = OuterProductMean(C_msa, C_pair).to(DEVICE, DTYPE).eval()
        msa = torch.randn(S, N, C_msa, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(S, N, device=DEVICE, dtype=DTYPE)
        do = torch.randn(N, N, C_pair, device=DEVICE, dtype=DTYPE)

        def fwd(mod=mod, msa=msa, mask=mask):
            return mod(msa, mask)

        def full(mod=mod, msa=msa, mask=mask, do=do):
            o = mod(msa, mask)
            o.backward(do, retain_graph=True)

        results.extend(
            bench_fwd_full("OuterProductMean", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[msa])
        )

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
                results.extend(
                    bench_fwd_full(
                        "OuterProductMean",
                        "compile",
                        shape,
                        fwd_c,
                        full_c,
                        warmup=warmup,
                        rep=rep,
                        grad_to_none=[msa_c],
                    )
                )
            except Exception as e:
                print(f"  [skip] torch.compile OuterProductMean failed (N={N}): {e}")
            del mod_c, msa_c

        print(f"  OuterProductMean N={N} S={S} done")

    return results


LN_TRANSPOSE_SHAPES = [
    {"N": 384, "C": 128},
    {"N": 512, "C": 128},
    {"N": 640, "C": 128},
    {"N": 768, "C": 128},
]

GRID_COL_SHAPES = [
    {"N": 384, "C": 128, "H": 4},
    {"N": 512, "C": 128, "H": 4},
    {"N": 640, "C": 128, "H": 4},
]


def bench_layernorm_transpose(shapes, warmup, rep):
    """Benchmark fused LN+transpose vs sequential LN + permute."""
    import torch.nn as nn
    import torch.nn.functional as F

    FusedLayerNormTranspose = try_import(  # noqa: N806
        "deepfold3.model.triton.layernorm_transpose", "FusedLayerNormTranspose"
    )

    results = []
    for shape in shapes:
        N, C = shape["N"], shape["C"]
        free_gpu()

        x = torch.randn(1, N, N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        do = torch.randn(1, N, N, C, device=DEVICE, dtype=DTYPE)

        # Baseline: LN + permute
        ln = nn.LayerNorm(C).to(DEVICE, DTYPE)

        def fwd_base(x=x, ln=ln):
            return F.layer_norm(x, (C,), ln.weight, ln.bias).permute(0, 2, 1, 3).contiguous()

        def full_base(x=x, ln=ln, do=do):
            o = F.layer_norm(x, (C,), ln.weight, ln.bias).permute(0, 2, 1, 3).contiguous()
            o.backward(do, retain_graph=True)

        results.extend(
            bench_fwd_full("LNTranspose", "pytorch", shape, fwd_base, full_base, warmup=warmup, rep=rep, grad_to_none=[x])
        )

        # Triton fused
        if FusedLayerNormTranspose is not None:
            fused = FusedLayerNormTranspose(C, do_transpose=True).to(DEVICE, DTYPE)
            with torch.no_grad():
                fused.weight.copy_(ln.weight)
                fused.bias.copy_(ln.bias)
            x2 = x.detach().clone().requires_grad_(True)

            def fwd_fused(x2=x2, fused=fused):
                return fused(x2)

            def full_fused(x2=x2, fused=fused, do=do):
                o = fused(x2)
                o.backward(do, retain_graph=True)

            try:
                results.extend(
                    bench_fwd_full(
                        "LNTranspose", "triton", shape, fwd_fused, full_fused, warmup=warmup, rep=rep, grad_to_none=[x2]
                    )
                )
            except Exception as e:
                print(f"  [skip] Triton LNTranspose failed (N={N}): {e}")
            del fused, x2
        else:
            print(f"  [skip] Triton FusedLayerNormTranspose not available (N={N})")

        print(f"  LNTranspose N={N} C={C} done")

    return results


def bench_grid_col_attention(shapes, warmup, rep):
    """Benchmark column GridSelfAttention: baseline vs fused LN+transpose."""

    from deepfold3.model.attention import GridSelfAttention
    from deepfold3.model.config import OptimConfig

    results = []
    for shape in shapes:
        N, C, H = shape["N"], shape["C"], shape["H"]
        free_gpu()

        z = torch.randn(1, N, N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(1, N, N, device=DEVICE, dtype=DTYPE)
        do = torch.randn(1, N, N, C, device=DEVICE, dtype=DTYPE)

        # Baseline column attention (no optim)
        base = GridSelfAttention(C, num_heads=H, transpose=True).to(DEVICE, DTYPE).eval()

        def fwd_base(z=z, base=base, mask=mask):
            return base(z, mask)

        def full_base(z=z, base=base, mask=mask, do=do):
            o = base(z, mask)
            o.backward(do, retain_graph=True)

        results.extend(
            bench_fwd_full(
                "GridColAttn", "pytorch", shape, fwd_base, full_base, warmup=warmup, rep=rep, grad_to_none=[z]
            )
        )

        # With fused LN+transpose
        try:
            optim = OptimConfig(use_fused_ln_transpose=True)
            fused = GridSelfAttention(C, num_heads=H, transpose=True, optim=optim).to(DEVICE, DTYPE).eval()
            fused.load_state_dict(base.state_dict(), strict=False)
            z2 = z.detach().clone().requires_grad_(True)

            def fwd_fused(z2=z2, fused=fused, mask=mask):
                return fused(z2, mask)

            def full_fused(z2=z2, fused=fused, mask=mask, do=do):
                o = fused(z2, mask)
                o.backward(do, retain_graph=True)

            results.extend(
                bench_fwd_full(
                    "GridColAttn", "triton_ln_transpose", shape, fwd_fused, full_fused,
                    warmup=warmup, rep=rep, grad_to_none=[z2]
                )
            )
            del fused, z2
        except Exception as e:
            print(f"  [skip] Fused LN+transpose GridColAttn failed (N={N}): {e}")

        # With fused evo attention (for comparison)
        try:
            optim_evo = OptimConfig(use_fused_evo_attention=True)
            evo = GridSelfAttention(C, num_heads=H, transpose=True, optim=optim_evo).to(DEVICE, DTYPE).eval()
            evo.load_state_dict(base.state_dict(), strict=False)
            z3 = z.detach().clone().requires_grad_(True)

            def fwd_evo(z3=z3, evo=evo, mask=mask):
                return evo(z3, mask)

            def full_evo(z3=z3, evo=evo, mask=mask, do=do):
                o = evo(z3, mask)
                o.backward(do, retain_graph=True)

            results.extend(
                bench_fwd_full(
                    "GridColAttn", "triton_evo", shape, fwd_evo, full_evo,
                    warmup=warmup, rep=rep, grad_to_none=[z3]
                )
            )
            del evo, z3
        except Exception as e:
            print(f"  [skip] EvoAttn GridColAttn failed (N={N}): {e}")

        # With fused evo + pair_bias (training: fwd+bwd)
        try:
            optim_pb = OptimConfig(use_fused_evo_attention=True, use_fused_pair_bias=True)
            fused_pb = GridSelfAttention(C, num_heads=H, transpose=True, optim=optim_pb).to(DEVICE, DTYPE).eval()
            fused_pb.load_state_dict(base.state_dict(), strict=False)
            z4 = z.detach().clone().requires_grad_(True)

            def fwd_pb(z4=z4, fused_pb=fused_pb, mask=mask):
                return fused_pb(z4, mask)

            def full_pb(z4=z4, fused_pb=fused_pb, mask=mask, do=do):
                o = fused_pb(z4, mask)
                o.backward(do, retain_graph=True)

            results.extend(
                bench_fwd_full(
                    "GridColAttn", "triton_evo+pb", shape, fwd_pb, full_pb,
                    warmup=warmup, rep=rep, grad_to_none=[z4]
                )
            )
            del fused_pb, z4
        except Exception as e:
            print(f"  [skip] Fused pair bias GridColAttn failed (N={N}): {e}")

        # Note: SDPA+compile benchmark removed — dtype mismatch between
        # pair_bias (bf16) and SDPA's internal fp32 logits needs a separate fix.

        del base, z, do
        print(f"  GridColAttn N={N} C={C} H={H} done")

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

    print("=== LayerNorm + Transpose ===")
    shapes = dry_run_shapes(LN_TRANSPOSE_SHAPES, args.dry_run)
    all_results.extend(bench_layernorm_transpose(shapes, warmup, rep))

    print("=== Grid Column Attention ===")
    shapes = dry_run_shapes(GRID_COL_SHAPES, args.dry_run)
    all_results.extend(bench_grid_col_attention(shapes, warmup, rep))

    save_results(all_results, "pair_ops.json")
    print(f"Done. {len(all_results)} measurements.")


if __name__ == "__main__":
    main()
