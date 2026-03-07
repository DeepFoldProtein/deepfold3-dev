"""Plot benchmark results from JSON files.

Generates grouped bar charts for time and memory comparisons.

Usage:
    uv run python benchmarks/plot_results.py
    uv run python benchmarks/plot_results.py --input primitives.json --output primitives.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(filename: str) -> list[dict]:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"Warning: {path} not found, skipping")
        return []
    return json.loads(path.read_text())


def _shape_label(shape: dict) -> str:
    """Compact shape label for x-axis showing key=value pairs."""
    # Primary dimension N first, then remaining keys sorted
    parts = []
    if "N" in shape:
        parts.append(f"N={shape['N']}")
    for k in sorted(shape):
        if k == "N":
            continue
        parts.append(f"{k}={shape[k]}")
    # e.g. "N=256/C=128" or "N=256/C=384/H=16"
    return "/".join(parts)


def plot_benchmark(results: list[dict], title: str, output_path: Path):
    """Plot grouped bar chart from benchmark results."""
    if not results:
        return

    # Group by (op_name, mode)
    groups = defaultdict(list)
    for r in results:
        key = (r["op_name"], r["mode"])
        groups[key].append(r)

    # Create subplots: one row per (op_name, mode) combo
    fig_rows = []
    for (op_name, mode), entries in sorted(groups.items()):
        # Group by shape, collect providers
        shape_data = defaultdict(dict)
        for e in entries:
            sl = _shape_label(e["shape"])
            shape_data[sl][e["provider"]] = e
        fig_rows.append((op_name, mode, shape_data))

    if not fig_rows:
        return

    # Build global provider → color mapping so colors are consistent across rows
    all_providers = sorted({p for _, _, sd in fig_rows for shapes in sd.values() for p in shapes})
    provider_colors = {p: plt.cm.Set2(i / max(len(all_providers) - 1, 1)) for i, p in enumerate(all_providers)}

    n_rows = len(fig_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    for row_idx, (op_name, mode, shape_data) in enumerate(fig_rows):
        shape_labels = list(shape_data.keys())
        providers = sorted({p for sd in shape_data.values() for p in sd})
        n_shapes = len(shape_labels)
        n_providers = len(providers)

        x = np.arange(n_shapes)
        width = 0.8 / max(n_providers, 1)

        # Time chart
        ax_time = axes[row_idx, 0]
        for i, prov in enumerate(providers):
            times = [shape_data[sl].get(prov, {}).get("time_ms", 0) for sl in shape_labels]
            p25 = [shape_data[sl].get(prov, {}).get("time_p25_ms", 0) for sl in shape_labels]
            p75 = [shape_data[sl].get(prov, {}).get("time_p75_ms", 0) for sl in shape_labels]
            yerr_lo = [max(t - lo, 0) for t, lo in zip(times, p25, strict=True)]
            yerr_hi = [max(hi - t, 0) for t, hi in zip(times, p75, strict=True)]
            has_err = any(lo > 0 or hi > 0 for lo, hi in zip(yerr_lo, yerr_hi, strict=True))
            ax_time.bar(
                x + i * width - 0.4 + width / 2, times, width, label=prov, color=provider_colors[prov],
                yerr=[yerr_lo, yerr_hi] if has_err else None, capsize=2, error_kw={"linewidth": 0.8},
            )
        ax_time.set_ylabel("Time (ms)")
        ax_time.set_title(f"{op_name} [{mode}] — Time")
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(shape_labels, fontsize=7, rotation=20, ha="right")
        ax_time.set_xlabel("")
        ax_time.legend(fontsize=8)

        # Memory chart
        ax_mem = axes[row_idx, 1]
        for i, prov in enumerate(providers):
            mems = [shape_data[sl].get(prov, {}).get("peak_memory_mb", 0) for sl in shape_labels]
            ax_mem.bar(x + i * width - 0.4 + width / 2, mems, width, label=prov, color=provider_colors[prov])
        ax_mem.set_ylabel("Peak Memory (MB)")
        ax_mem.set_title(f"{op_name} [{mode}] — Memory")
        ax_mem.set_xticks(x)
        ax_mem.set_xticklabels(shape_labels, fontsize=7, rotation=20, ha="right")
        ax_mem.set_xlabel("")
        ax_mem.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default=None, help="Specific JSON file to plot")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.input:
        results = load_results(args.input)
        out = Path(args.output) if args.output else RESULTS_DIR / args.input.replace(".json", ".png")
        plot_benchmark(results, args.input.replace(".json", "").replace("_", " ").title(), out)
        return

    # Plot all available result files
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        results = json.loads(json_file.read_text())
        out = json_file.with_suffix(".png")
        title = json_file.stem.replace("_", " ").title()
        plot_benchmark(results, title, out)


if __name__ == "__main__":
    main()
