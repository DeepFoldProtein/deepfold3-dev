"""Auto-tuner: finds the best optimization config for a given workload.

Sweeps combinations of enabled/disabled optimizations and picks the fastest.

Usage:
    uv run python benchmarks/tuner.py --op transition_block --N 512 --C 384
    uv run python benchmarks/tuner.py --op transition_block --N 512 --C 384 --mode full
    uv run python benchmarks/tuner.py --op attention --N 256 --C 128
    uv run python benchmarks/tuner.py --dry-run
    uv run python benchmarks/tuner.py --triton-tune
"""

from __future__ import annotations

import argparse
import gc
import json
import sys

import torch

from benchmarks.optim_config import OptimConfig, get_optimized_module, is_available, print_availability
from benchmarks.utils import RESULTS_DIR, BenchResult, check_cuda, gpu_timer, peak_memory_mb

DEVICE = "cuda"
DTYPE = torch.bfloat16

# DeepFold3 common shapes for autotuning (N_tokens -> transition shapes)
DEEPFOLD3_SHAPES = {
    "pair_128": (128 * 128, 128, 512),
    "pair_256": (256 * 256, 128, 512),
    "pair_384": (384 * 384, 128, 512),
    "pair_512": (512 * 512, 128, 512),
    "single_128": (128, 384, 1536),
    "single_256": (256, 384, 1536),
    "single_384": (384, 384, 1536),
    "single_512": (512, 384, 1536),
    "msa_128": (1024 * 128, 64, 256),
    "msa_256": (1024 * 256, 64, 256),
}

# DeepFold3 attention shapes for tuning
ATTENTION_SHAPES = {
    "grid_128": (128, 128, 4),  # (N, C, H) — pair attention
    "grid_256": (256, 128, 4),
    "grid_384": (384, 128, 4),
    "grid_512": (512, 128, 4),
    "single_128": (128, 384, 16),  # (N, C, H) — single attention
    "single_256": (256, 384, 16),
    "single_384": (384, 384, 16),
    "single_512": (512, 384, 16),
}


# Define which config flags are relevant per operation
OP_FLAGS = {
    "transition_block": ["use_fused_transition"],
    "attention": ["use_fused_evo_attention", "use_torch_compile"],
}


def _configs_for_op(op: str) -> list[tuple[str, OptimConfig]]:
    """Generate all valid OptimConfig combos for an operation."""
    flags = OP_FLAGS.get(op, [])
    configs = []

    # Baseline (all off)
    configs.append(("baseline", OptimConfig()))

    # Each flag individually
    backend_map = {
        "use_fused_transition": "triton_transition",
        "use_fused_evo_attention": "triton_evo_attention",
        "use_torch_compile": "torch_compile",
    }
    for flag in flags:
        backend = backend_map.get(flag)
        if backend and not is_available(backend):
            continue
        cfg = OptimConfig()
        setattr(cfg, flag, True)
        configs.append((flag, cfg))

    # All relevant flags on (if more than 1 flag)
    if len(flags) > 1:
        available_flags = []
        for flag in flags:
            backend = backend_map.get(flag)
            if backend is None or is_available(backend):
                available_flags.append(flag)

        if len(available_flags) > 1:
            cfg = OptimConfig()
            for flag in available_flags:
                setattr(cfg, flag, True)
            configs.append(("all", cfg))

    return configs


def _free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def _bench_transition(config: OptimConfig, N: int, C: int, mode: str, warmup: int, rep: int) -> BenchResult:
    """Build transition module from config and benchmark it."""
    shape = {"N": N, "C": C}
    mod = get_optimized_module("transition_block", config, channels=C)
    x = torch.randn(N, C, device=DEVICE, dtype=DTYPE, requires_grad=(mode != "fwd"))
    do = torch.randn(N, C, device=DEVICE, dtype=DTYPE) if mode != "fwd" else None

    def fwd():
        return mod(x)

    if mode == "fwd":
        time_ms = gpu_timer(fwd, warmup=warmup, rep=rep)
        mem, _ = peak_memory_mb(fwd)
    else:
        def full():
            o = mod(x)
            o.backward(do, retain_graph=True)
        time_ms = gpu_timer(full, warmup=warmup, rep=rep, grad_to_none=[x])
        mem, _ = peak_memory_mb(full)

    return BenchResult("transition_block", "tuner", shape, mode, time_ms, mem, extra={"config": {k: v for k, v in config.__dict__.items() if isinstance(v, bool)}})


def _bench_attention(config: OptimConfig, N: int, C: int, H: int, mode: str, warmup: int, rep: int) -> BenchResult:
    """Build GridSelfAttention from config and benchmark it."""
    from deepfold3.model.attention import GridSelfAttention

    shape = {"N": N, "C": C, "H": H}
    attn = GridSelfAttention(C, num_heads=H, transpose=False, optim=config).to(DEVICE, DTYPE).eval()
    z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE, requires_grad=(mode != "fwd"))
    mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)
    do = torch.randn_like(z) if mode != "fwd" else None

    def fwd():
        return attn(z, mask)

    if mode == "fwd":
        time_ms = gpu_timer(fwd, warmup=warmup, rep=rep)
        mem, _ = peak_memory_mb(fwd)
    else:
        def full():
            o = attn(z, mask)
            o.backward(do, retain_graph=True)
        time_ms = gpu_timer(full, warmup=warmup, rep=rep, grad_to_none=[z])
        mem, _ = peak_memory_mb(full)

    return BenchResult("attention", "tuner", shape, mode, time_ms, mem, extra={"config": {k: v for k, v in config.__dict__.items() if isinstance(v, bool)}})


def tune(op: str, N: int, C: int, mode: str = "full", warmup: int = 50, rep: int = 200, H: int = 4) -> dict:
    """Tune an operation and return best config."""
    configs = _configs_for_op(op)
    print(f"\nTuning {op} (N={N}, C={C}, mode={mode}) -- {len(configs)} configs")

    results = []
    for name, cfg in configs:
        _free_gpu()
        try:
            if op == "attention":
                result = _bench_attention(cfg, N, C, H, mode, warmup, rep)
            else:
                result = _bench_transition(cfg, N, C, mode, warmup, rep)
            results.append((name, cfg, result))
            print(f"  {name:30s}  {result.time_ms:8.3f} ms  {result.peak_memory_mb:8.1f} MB")
        except Exception as e:
            print(f"  {name:30s}  FAILED: {e}")

    if not results:
        print("  No valid configs found!")
        return {}

    # Pick best by time
    best_name, best_cfg, best_result = min(results, key=lambda x: x[2].time_ms)
    print(f"\n  Best: {best_name} ({best_result.time_ms:.3f} ms)")

    # Speedup over baseline
    baseline = next((r for n, _, r in results if n == "baseline"), None)
    if baseline and best_name != "baseline":
        speedup = baseline.time_ms / best_result.time_ms
        mem_ratio = baseline.peak_memory_mb / max(best_result.peak_memory_mb, 0.1)
        print(f"  Speedup: {speedup:.2f}x time, {mem_ratio:.2f}x memory over baseline")

    return {
        "op": op,
        "shape": {"N": N, "C": C} | ({"H": H} if op == "attention" else {}),
        "mode": mode,
        "best": best_name,
        "best_config": {k: v for k, v in best_cfg.__dict__.items() if isinstance(v, bool)},
        "best_time_ms": best_result.time_ms,
        "best_memory_mb": best_result.peak_memory_mb,
        "all_results": [(n, r.time_ms, r.peak_memory_mb) for n, _, r in results],
    }


def triton_tune_shapes():
    """Tune Triton block sizes for DeepFold3 common shapes."""
    print("\n=== Triton Tuning: DeepFold3 Transition Shapes ===")
    all_results = []
    for name, (M, K, N) in DEEPFOLD3_SHAPES.items():
        print(f"\n--- {name}: M={M}, K={K}, N={N} ---")
        result = tune("transition_block", M, K, mode="full", warmup=20, rep=50)
        if result:
            result["shape_name"] = name
            result["triton_shape"] = {"M": M, "K": K, "N": N}
            all_results.append(result)

    print("\n=== Triton Tuning: DeepFold3 Attention Shapes ===")
    for name, (N, C, H) in ATTENTION_SHAPES.items():
        print(f"\n--- {name}: N={N}, C={C}, H={H} ---")
        result = tune("attention", N, C, mode="fwd", warmup=20, rep=50, H=H)
        if result:
            result["shape_name"] = name
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Auto-tune optimization config")
    parser.add_argument("--op", type=str, default="transition_block", choices=list(OP_FLAGS.keys()))
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--C", type=int, default=384)
    parser.add_argument("--H", type=int, default=4, help="Number of attention heads (for --op attention)")
    parser.add_argument("--mode", type=str, default="full", choices=["fwd", "full"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep multiple shapes")
    parser.add_argument("--triton-tune", action="store_true", help="Tune Triton for DeepFold3 shapes")
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    print_availability()

    if args.triton_tune:
        results = triton_tune_shapes()
        out_path = RESULTS_DIR / "triton_tune_deepfold3.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved tuning results to {out_path}")
        return

    warmup = 5 if args.dry_run else args.warmup
    rep = 5 if args.dry_run else args.rep

    if args.sweep:
        if args.op == "attention":
            sweep_shapes = [(128, 128, 4), (256, 128, 4), (384, 128, 4), (512, 128, 4)]
            if args.dry_run:
                sweep_shapes = [(4, 8, 4)]
            all_tune_results = []
            for n, c, h in sweep_shapes:
                result = tune(args.op, n, c, args.mode, warmup, rep, H=h)
                if result:
                    all_tune_results.append(result)
        else:
            sweep_shapes = [
                (256, 128),
                (512, 128),
                (1024, 128),
                (256, 384),
                (512, 384),
                (1024, 384),
            ]
            if args.dry_run:
                sweep_shapes = [(4, 8)]
            all_tune_results = []
            for n, c in sweep_shapes:
                result = tune(args.op, n, c, args.mode, warmup, rep)
                if result:
                    all_tune_results.append(result)

        out_path = RESULTS_DIR / f"tune_{args.op}.json"
        out_path.write_text(json.dumps(all_tune_results, indent=2))
        print(f"\nSaved sweep results to {out_path}")
    else:
        N = 4 if args.dry_run else args.N
        C = 8 if args.dry_run else args.C
        H = args.H
        result = tune(args.op, N, C, args.mode, warmup, rep, H=H)
        if result:
            out_path = RESULTS_DIR / f"tune_{args.op}_{N}x{C}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
