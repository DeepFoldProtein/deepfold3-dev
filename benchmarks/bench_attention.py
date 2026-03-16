"""Benchmark attention modules: GridSelfAttention, MSAAttention, SingleAttention.

Compares PyTorch baseline vs torch.compile vs Triton FusedEvoAttention.

Tests both raw kernel performance and integrated module paths (via OptimConfig).

Usage:
    uv run python benchmarks/bench_attention.py
    uv run python benchmarks/bench_attention.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

from benchmarks.utils import (
    BenchResult,
    bench_fwd_full,
    check_cuda,
    dry_run_shapes,
    save_results,
    try_import,
)

# DeepFold3 pair attention shapes (pair_channel=128, num_heads=4)
GRID_SHAPES = [
    {"N": 128, "C": 128, "H": 4},
    {"N": 256, "C": 128, "H": 4},
    {"N": 384, "C": 128, "H": 4},
    {"N": 512, "C": 128, "H": 4},
    {"N": 768, "C": 128, "H": 4},
]

# DeepFold3 MSA attention shapes (msa_channel=64, pair_channel=128, 8 heads)
MSA_SHAPES = [
    {"N": 128, "S": 32, "C_msa": 64, "C_pair": 128, "H": 8},
    {"N": 256, "S": 32, "C_msa": 64, "C_pair": 128, "H": 8},
    {"N": 384, "S": 32, "C_msa": 64, "C_pair": 128, "H": 8},
    {"N": 512, "S": 16, "C_msa": 64, "C_pair": 128, "H": 8},
]

# DeepFold3 single attention shapes (seq_channel=384, 16 heads)
SINGLE_SHAPES = [
    {"N": 128, "C": 384, "H": 16},
    {"N": 256, "C": 384, "H": 16},
    {"N": 384, "C": 384, "H": 16},
    {"N": 512, "C": 384, "H": 16},
    {"N": 768, "C": 384, "H": 16},
]

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def _bench_grid_pytorch(attn, z, mask, do, shape, warmup, rep):
    def fwd():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return attn(z, mask)

    def full():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = attn(z, mask)
        o.backward(do, retain_graph=True)

    return bench_fwd_full("GridSelfAttention", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[z])


def _bench_grid_triton_raw(triton_fn, q, k, v, res_mask, pair_bias, do_triton, shape, warmup, rep):
    """Benchmark raw Triton kernel (5D tensors directly)."""

    def fwd():
        return triton_fn(q, k, v, res_mask, pair_bias)

    def full():
        o = triton_fn(q, k, v, res_mask, pair_bias)
        o.backward(do_triton, retain_graph=True)

    return bench_fwd_full(
        "GridSelfAttention", "triton_raw", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[q, k, v]
    )


def bench_grid_attention(shapes: list[dict], warmup: int, rep: int) -> list[BenchResult]:
    from deepfold3.model.attention import GridSelfAttention
    from deepfold3.model.config import OptimConfig

    _EvoformerAttention = try_import(  # noqa: N806
        "deepfold3.model.triton.evo_attention", "_EvoformerAttention"
    )

    results = []
    for shape in shapes:
        N, C, H = shape["N"], shape["C"], shape["H"]
        _free_gpu()

        # --- Baseline PyTorch ---
        attn = GridSelfAttention(C, num_heads=H, transpose=False).to(DEVICE, DTYPE).eval()
        z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)
        do = torch.randn_like(z)

        try:
            results.extend(_bench_grid_pytorch(attn, z, mask, do, shape, warmup, rep))
        except torch.OutOfMemoryError:
            print(f"  [skip] PyTorch GridSelfAttention OOM (N={N})")
            _free_gpu()

        # --- Integrated: GridSelfAttention(optim=OptimConfig(use_torch_compile=True)) ---
        try:
            optim_compile = OptimConfig(use_torch_compile=True)
            attn_comp = GridSelfAttention(C, num_heads=H, transpose=False, optim=optim_compile).to(DEVICE, DTYPE).eval()
            z_comp = z.detach().clone().requires_grad_(True)
            compile_results = _bench_grid_pytorch(attn_comp, z_comp, mask, do, shape, warmup, rep)
            for r in compile_results:
                r.provider = "compile"
            results.extend(compile_results)
            del attn_comp, z_comp
        except Exception as e:
            print(f"  [skip] torch.compile GridSelfAttention failed (N={N}): {e}")
            _free_gpu()

        # --- Integrated: GridSelfAttention(optim=OptimConfig(use_fused_evo_attention=True)) ---
        if _EvoformerAttention is not None:
            try:
                optim_evo = OptimConfig(use_fused_evo_attention=True)
                attn_fused = (
                    GridSelfAttention(C, num_heads=H, transpose=False, optim=optim_evo).to(DEVICE, DTYPE).eval()
                )
                attn_fused.load_state_dict(attn.state_dict())
                z_fused = z.detach().clone().requires_grad_(True)
                # Smoke test
                attn_fused(z_fused.detach(), mask)
                fused_results = _bench_grid_pytorch(attn_fused, z_fused, mask, do, shape, warmup, rep)
                for r in fused_results:
                    r.provider = "triton"
                results.extend(fused_results)
                del attn_fused, z_fused
            except Exception as e:
                print(f"  [skip] Triton GridSelfAttention failed (N={N}): {e}")
                _free_gpu()

        del attn, z, mask, do
        _free_gpu()

        # --- Raw kernel benchmark (for kernel-level comparison) ---
        if _EvoformerAttention is not None:
            head_dim = max(C // H, 16)
            q = torch.randn(1, N, N, H, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
            k = torch.randn(1, N, N, H, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
            v = torch.randn(1, N, N, H, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
            rm = torch.ones(1, N, 1, 1, N, device=DEVICE, dtype=torch.bool)
            pb = torch.randn(1, 1, H, N, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
            do_t = torch.randn(1, N, N, H, head_dim, device=DEVICE, dtype=DTYPE)

            try:
                _EvoformerAttention.apply(q, k, v, rm, pb)  # smoke test
                results.extend(
                    _bench_grid_triton_raw(_EvoformerAttention.apply, q, k, v, rm, pb, do_t, shape, warmup, rep)
                )
            except Exception as e:
                print(f"  [skip] Triton raw FusedEvoAttention failed (N={N}): {e}")
                _free_gpu()
            del q, k, v, rm, pb, do_t

        _free_gpu()
        print(f"  GridSelfAttention N={N} C={C} done")

    return results


def _bench_msa_pytorch(attn, msa, mask, pair, do, shape, warmup, rep):
    def fwd():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return attn(msa, mask, pair)

    def full():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = attn(msa, mask, pair)
        o.backward(do, retain_graph=True)

    return bench_fwd_full("MSAAttention", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[msa])


def bench_msa_attention(shapes: list[dict], warmup: int, rep: int) -> list[BenchResult]:
    from deepfold3.model.attention import MSAAttention
    from deepfold3.model.config import OptimConfig

    results = []
    for shape in shapes:
        N, S = shape["N"], shape["S"]
        C_msa, C_pair, H = shape["C_msa"], shape["C_pair"], shape["H"]
        _free_gpu()

        # --- Baseline ---
        attn = MSAAttention(C_msa, C_pair, num_heads=H).to(DEVICE, DTYPE).eval()
        msa = torch.randn(S, N, C_msa, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(S, N, device=DEVICE, dtype=DTYPE)
        pair = torch.randn(N, N, C_pair, device=DEVICE, dtype=DTYPE)
        do = torch.randn_like(msa)

        results.extend(_bench_msa_pytorch(attn, msa, mask, pair, do, shape, warmup, rep))

        # --- Integrated: MSAAttention(optim=OptimConfig(use_torch_compile=True)) ---
        try:
            optim_compile = OptimConfig(use_torch_compile=True)
            attn_c = MSAAttention(C_msa, C_pair, num_heads=H, optim=optim_compile).to(DEVICE, DTYPE).eval()
            attn_c.load_state_dict(attn.state_dict())
            msa_c = msa.detach().clone().requires_grad_(True)
            compile_results = _bench_msa_pytorch(attn_c, msa_c, mask, pair, do, shape, warmup, rep)
            for r in compile_results:
                r.provider = "compile"
            results.extend(compile_results)
            del attn_c, msa_c
        except Exception as e:
            print(f"  [skip] torch.compile MSAAttention failed (N={N}): {e}")

        del attn, msa, mask, pair, do
        print(f"  MSAAttention N={N} S={S} done")

    return results


def _bench_single_pytorch(attn, x, mask, pair_logits, do, shape, warmup, rep):
    def fwd():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return attn(x, mask, pair_logits)

    def full():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = attn(x, mask, pair_logits)
        o.backward(do, retain_graph=True)

    return bench_fwd_full("SingleAttention", "pytorch", shape, fwd, full, warmup=warmup, rep=rep, grad_to_none=[x])


def bench_single_attention(shapes: list[dict], warmup: int, rep: int) -> list[BenchResult]:
    from deepfold3.model.attention import SingleAttention
    from deepfold3.model.config import OptimConfig

    results = []
    for shape in shapes:
        N, C, H = shape["N"], shape["C"], shape["H"]
        _free_gpu()

        # --- Baseline ---
        attn = SingleAttention(C, num_heads=H).to(DEVICE, DTYPE).eval()
        x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        mask = torch.ones(N, device=DEVICE, dtype=DTYPE)
        pair_logits = torch.randn(H, N, N, device=DEVICE, dtype=DTYPE)
        do = torch.randn_like(x)

        results.extend(_bench_single_pytorch(attn, x, mask, pair_logits, do, shape, warmup, rep))

        # --- Integrated: SingleAttention(optim=OptimConfig(use_torch_compile=True)) ---
        # Note: Triton EvoAttention is NOT benchmarked for SingleAttention — it's slower
        # than baseline (0.87-1.03x) due to N_seq=1 kernel overhead. torch.compile is better.
        try:
            optim_compile = OptimConfig(use_torch_compile=True)
            attn_c = SingleAttention(C, num_heads=H, optim=optim_compile).to(DEVICE, DTYPE).eval()
            attn_c.load_state_dict(attn.state_dict())
            x_c = x.detach().clone().requires_grad_(True)
            compile_results = _bench_single_pytorch(attn_c, x_c, mask, pair_logits, do, shape, warmup, rep)
            for r in compile_results:
                r.provider = "compile"
            results.extend(compile_results)
            del attn_c, x_c
        except Exception as e:
            print(f"  [skip] torch.compile SingleAttention failed (N={N}): {e}")

        del attn, x, mask, pair_logits, do
        _free_gpu()
        print(f"  SingleAttention N={N} C={C} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention modules")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with tiny shapes")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    warmup = 5 if args.dry_run else args.warmup
    rep = 5 if args.dry_run else args.rep

    all_results = []

    print("=== GridSelfAttention ===")
    shapes = dry_run_shapes(GRID_SHAPES, args.dry_run)
    all_results.extend(bench_grid_attention(shapes, warmup, rep))

    print("=== MSAAttention ===")
    shapes = dry_run_shapes(MSA_SHAPES, args.dry_run)
    all_results.extend(bench_msa_attention(shapes, warmup, rep))

    print("=== SingleAttention ===")
    shapes = dry_run_shapes(SINGLE_SHAPES, args.dry_run)
    all_results.extend(bench_single_attention(shapes, warmup, rep))

    save_results(all_results, "attention.json")
    print(f"Done. {len(all_results)} measurements.")


if __name__ == "__main__":
    main()
