#!/usr/bin/env python3
"""Benchmark DeepFold3 inference: time + peak memory across sequence lengths and optim presets.

Tracks:
  - First-run time (includes Triton autotuning / torch.compile tracing)
  - Steady-state time (mean + std over NUM_RUNS after warmup)
  - Peak GPU memory

Produces:
  benchmarks/inference_benchmark.csv
  benchmarks/inference_benchmark.png

Usage:
  python benchmarks/bench_inference.py
  python benchmarks/bench_inference.py --seq-lengths 384 512 768
  python benchmarks/bench_inference.py --presets recommended recommended_tiled
  python benchmarks/bench_inference.py --runs 5 --recycles 4 --samples 1
  python benchmarks/bench_inference.py --no-templates --msa-rows 512
  python benchmarks/bench_inference.py --params weights.safetensors --output results/
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch

MAX_ATOMS = 24  # MAX_ATOMS_PER_TOKEN

AVAILABLE_PRESETS = ["none", "recommended", "recommended_tiled", "inference", "triton_only", "all"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DeepFold3 inference")
    p.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 384, 512, 768, 1024],
        help="Sequence lengths to benchmark (default: 128 256 384 512 768 1024)",
    )
    p.add_argument(
        "--presets",
        nargs="+",
        default=["inference", "none"],
        help="Optimization presets to compare. Available: "
        + ", ".join(AVAILABLE_PRESETS)
        + ". Also supports recommended_tiled_cN for OPM chunk sweep.",
    )
    p.add_argument("--runs", type=int, default=3, help="Timed runs per config (default: 3)")
    p.add_argument("--recycles", type=int, default=10, help="Number of recycles (default: 10)")
    p.add_argument("--samples", type=int, default=1, help="Number of diffusion samples (default: 1)")
    p.add_argument("--msa-rows", type=int, default=1024, help="MSA depth (default: 1024)")
    p.add_argument("--templates", type=int, default=4, help="Number of templates (default: 4)")
    p.add_argument("--no-templates", action="store_true", help="Disable templates (0 templates)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--params",
        type=Path,
        default=Path("resources/alphafold3.safetensors"),
        help="Path to model weights (default: resources/alphafold3.safetensors)",
    )
    p.add_argument("--output", type=Path, default=Path("benchmarks"), help="Output directory (default: benchmarks/)")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    p.add_argument(
        "--skip",
        nargs="*",
        metavar="PRESET:LEN",
        default=None,
        help="Skip combos. With args: skip specific preset:seq_len or bare seq_len. "
        "Without args (--skip): skip all combos already in the CSV (run only new). "
        "Implies --update.",
    )
    p.add_argument(
        "--opm-chunks",
        type=int,
        nargs="+",
        default=None,
        help="OPM chunk sizes to sweep for recommended_tiled (e.g. --opm-chunks 32 64 128 256). "
        "Creates a preset per chunk size: recommended_tiled_c32, recommended_tiled_c64, etc.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Append to existing CSV (replace matching preset+seq_len rows, keep others)",
    )
    return p.parse_args()


def build_dummy_features(
    seq_len: int, device: torch.device, num_msa_rows: int = 1024, num_templates: int = 4
) -> dict[str, torch.Tensor]:
    """Build dummy features with realistic MSA depth and templates."""
    from deepfold3.data.module.dataset import InferenceInput
    from deepfold3.data.parse.templates import TemplateFeatures

    templates = []
    for _ in range(num_templates):
        templates.append(
            TemplateFeatures(
                aatype=np.zeros(seq_len, dtype=np.int64),
                atom_positions=np.random.randn(seq_len, MAX_ATOMS, 3).astype(np.float32) * 5.0,
                atom_mask=np.ones((seq_len, MAX_ATOMS), dtype=bool),
            )
        )

    feats = InferenceInput.from_sequence(
        "A" * seq_len, mol_type="protein", templates=templates if num_templates > 0 else None
    )

    if "msa" in feats and feats["msa"].shape[0] < num_msa_rows:
        current_rows = feats["msa"].shape[0]
        extra = num_msa_rows - current_rows
        extra_msa = torch.randint(0, 22, (extra, seq_len), dtype=feats["msa"].dtype)
        feats["msa"] = torch.cat([feats["msa"], extra_msa], dim=0)
        for key in ["msa_mask", "deletion_matrix"]:
            if key in feats and feats[key].dim() == 2:
                fill = 1 if key == "msa_mask" else 0
                pad = torch.full((extra, seq_len), fill, dtype=feats[key].dtype)
                feats[key] = torch.cat([feats[key], pad], dim=0)

    return {k: v.to(device) for k, v in feats.items()}


def load_model(preset: str, device: torch.device, params_path: Path):
    """Load model with given optimization preset."""
    from safetensors.torch import load_file

    from deepfold3.model.config import ModelConfig, OptimConfig

    state_dict = load_file(str(params_path))

    cfg = ModelConfig()
    preset_map = {
        "recommended": OptimConfig.recommended,
        "recommended_tiled": OptimConfig.recommended_tiled,
        "inference": OptimConfig.inference,
        "triton_only": OptimConfig.triton_only,
        "all": OptimConfig.all_enabled,
    }
    # Handle recommended_tiled_cXXX variants (OPM chunk sweep)
    if preset.startswith("recommended_tiled_c"):
        chunk = int(preset.split("_c")[-1])
        cfg.optim = OptimConfig.recommended_tiled()
        cfg.optim.tiled_opm_chunk = chunk
        torch.set_float32_matmul_precision("high")
    elif preset in preset_map:
        cfg.optim = preset_map[preset]()
        torch.set_float32_matmul_precision("high")
    # else "none": default (all False)

    from deepfold3.model.model import DeepFold3

    model = DeepFold3(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def _is_oom(e: Exception) -> bool:
    """Check if an exception is an OOM or Triton OutOfResources error."""
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(e).lower()
        or "OutOfResources" in type(e).__name__
    )


def run_once(model, feats, device, seed: int, num_recycles: int, num_samples: int) -> tuple[float, float]:
    """Run inference once. Returns (elapsed_seconds, peak_memory_gb)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        model(feats, seed=seed, num_recycles=num_recycles, num_samples=num_samples)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = (torch.cuda.max_memory_allocated(device) - base) / (1024**3)
    return elapsed, peak_gb


def main():
    args = parse_args()

    num_templates = 0 if args.no_templates else args.templates
    csv_path = args.output / "inference_benchmark.csv"
    plot_path = args.output / "inference_benchmark.png"

    # Expand --opm-chunks into additional presets
    if args.opm_chunks:
        for chunk in args.opm_chunks:
            name = f"recommended_tiled_c{chunk}"
            if name not in args.presets:
                args.presets.append(name)

    # Build skip set.
    # --skip (no args): skip all combos already in CSV → run only new ones.
    # --skip recommended:1024 768: skip specific combos.
    skip_set: set[tuple[str, int]] = set()
    if args.skip is not None:
        args.update = True  # --skip implies --update to preserve skipped entries
        for entry in args.skip:
            if ":" in entry:
                p, sl = entry.rsplit(":", 1)
                skip_set.add((p, int(sl)))
            else:
                skip_set.update((p, int(entry)) for p in args.presets)
        # Also skip combos already in the CSV (run only unknown)
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    skip_set.add((row["preset"], int(row["seq_len"])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({gpu_name})")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Presets: {args.presets}")
    print(f"Runs per config: {args.runs} (+ 1 warmup/compile run)")
    print(f"MSA rows: {args.msa_rows}, Templates: {num_templates}, Samples: {args.samples}")
    print(f"Recycles: {args.recycles}")

    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for preset in args.presets:
        print(f"\n{'=' * 70}")
        print(f"  PRESET: {preset}")
        print(f"{'=' * 70}")
        model = load_model(preset, device, args.params)

        for seq_len in args.seq_lengths:
            if (preset, seq_len) in skip_set:
                print(f"\n  seq_len={seq_len} — SKIPPED (--skip)")
                continue

            print(f"\n  seq_len={seq_len}")

            try:
                feats = build_dummy_features(seq_len, device, args.msa_rows, num_templates)
            except Exception as e:
                if _is_oom(e):
                    print("    SKIPPED (OOM during featurization — skipping longer lengths too)")
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                raise

            n_msa = feats["msa"].shape[0]
            n_tpl = feats.get("template_aatype", torch.zeros(0)).shape[0]
            print(f"    features: {feats['res_type'].shape[0]} tokens, {n_msa} MSA rows, {n_tpl} templates")

            # --- First run (compile / autotune) ---
            try:
                compile_time, compile_mem = run_once(model, feats, device, args.seed, args.recycles, args.samples)
            except Exception as e:
                if _is_oom(e):
                    print("    SKIPPED (OOM during inference — skipping longer lengths too)")
                    del feats
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                raise

            print(f"    compile/warmup: {compile_time:.2f}s, peak_mem={compile_mem:.2f} GB")

            # --- Timed runs ---
            times = []
            mems = []
            for r in range(args.runs):
                elapsed, mem = run_once(model, feats, device, args.seed, args.recycles, args.samples)
                times.append(elapsed)
                mems.append(mem)
                print(f"    run {r + 1}/{args.runs}: {elapsed:.2f}s, peak_mem={mem:.2f} GB")

            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            mean_m = statistics.mean(mems)
            std_m = statistics.stdev(mems) if len(mems) > 1 else 0.0

            print(f"    => time: {mean_t:.2f} +/- {std_t:.2f}s | mem: {mean_m:.2f} +/- {std_m:.2f} GB")

            results.append(
                {
                    "preset": preset,
                    "seq_len": seq_len,
                    "compile_time_s": round(compile_time, 3),
                    "compile_peak_mem_gb": round(compile_mem, 3),
                    "mean_time_s": round(mean_t, 3),
                    "std_time_s": round(std_t, 3),
                    "mean_peak_mem_gb": round(mean_m, 3),
                    "std_peak_mem_gb": round(std_m, 3),
                }
            )

            del feats
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    fieldnames = [
        "preset",
        "seq_len",
        "compile_time_s",
        "compile_peak_mem_gb",
        "mean_time_s",
        "std_time_s",
        "mean_peak_mem_gb",
        "std_peak_mem_gb",
    ]

    if args.update and csv_path.exists():
        # Load existing rows, replace matching (preset, seq_len), keep others
        existing = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
        # Build set of new (preset, seq_len) keys
        new_keys = {(r["preset"], str(r["seq_len"])) for r in results}
        # Keep existing rows that don't conflict
        merged = [row for row in existing if (row["preset"], row["seq_len"]) not in new_keys]
        # Add new results
        for r in results:
            merged.append({k: r[k] for k in fieldnames})
        # Sort by preset then seq_len
        merged.sort(key=lambda row: (row["preset"], int(row["seq_len"])))
        all_rows = merged
    else:
        all_rows = [{k: r[k] for k in fieldnames} for r in results]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nCSV saved to {csv_path}" + (" (updated)" if args.update else ""))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_time, ax_compile, ax_mem = axes

        palette = ["#FF5722", "#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#607D8B"]
        marker_set = ["s", "^", "o", "D", "v", "p"]

        # Plot all data in the CSV (includes existing rows when --update)
        plot_data = []
        for row in all_rows:
            plot_data.append({k: float(v) if k != "preset" else v for k, v in row.items()})
        plot_presets = sorted(set(r["preset"] for r in plot_data))

        for i, preset in enumerate(plot_presets):
            rows = [r for r in plot_data if r["preset"] == preset]
            if not rows:
                continue
            rows.sort(key=lambda r: r["seq_len"])
            lens = [int(r["seq_len"]) for r in rows]
            c = palette[i % len(palette)]
            m = marker_set[i % len(marker_set)]

            ax_time.errorbar(
                lens,
                [r["mean_time_s"] for r in rows],
                yerr=[r["std_time_s"] for r in rows],
                marker=m,
                capsize=4,
                label=preset,
                color=c,
                linewidth=2,
            )
            ax_compile.plot(
                lens,
                [r["compile_time_s"] for r in rows],
                marker=m,
                label=preset,
                color=c,
                linewidth=2,
                linestyle="--",
            )
            ax_mem.errorbar(
                lens,
                [r["mean_peak_mem_gb"] for r in rows],
                yerr=[r["std_peak_mem_gb"] for r in rows],
                marker=m,
                capsize=4,
                label=preset,
                color=c,
                linewidth=2,
            )

        for ax in axes:
            ax.set_xlabel("Sequence Length")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax_time.set_ylabel("Time (s)")
        ax_time.set_title(f"Steady-State Inference ({args.recycles} recycles)")
        ax_compile.set_ylabel("Time (s)")
        ax_compile.set_title("First Run (compile/autotune)")
        ax_mem.set_ylabel("Peak Memory (GB)")
        ax_mem.set_title("Peak GPU Memory")

        fig.suptitle(
            f"DeepFold3 Inference Benchmark — {gpu_name}\n"
            f"MSA={args.msa_rows}, Templates={num_templates}, Samples={args.samples}",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
