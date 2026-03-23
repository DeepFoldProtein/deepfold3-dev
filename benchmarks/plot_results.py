"""Plot benchmark results from JSON files.

Generates clean grouped bar charts (time + memory) for each benchmark suite.
One PNG per JSON file, with consistent colors and readable labels.

Usage:
    uv run python -m benchmarks.plot_results
    uv run python -m benchmarks.plot_results --input pair_ops.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# Consistent palette across all plots
PROVIDER_COLORS = {
    "pytorch": "#78909C",
    "compile": "#42A5F5",
    "compile_integrated": "#90CAF9",
    "triton": "#66BB6A",
    "triton_raw": "#FDD835",
    "triton_l1+l2": "#A5D6A7",
    "triton_l3": "#2E7D32",
    "triton_evo": "#66BB6A",
    "triton_ln_transpose": "#81C784",
    "triton_evo+pb": "#1B5E20",
}


def _shape_label(shape: dict) -> str:
    parts = []
    if "N" in shape:
        parts.append(f"N={shape['N']}")
    for k in sorted(shape):
        if k != "N":
            parts.append(f"{k}={shape[k]}")
    return "/".join(parts)


def _color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "#BDBDBD")


def plot_benchmark(results: list[dict], title: str, output_path: Path):
    """Plot grouped bar chart from benchmark results."""
    if not results:
        return

    # Group by (op_name, mode)
    groups = defaultdict(list)
    for r in results:
        groups[(r["op_name"], r["mode"])].append(r)

    fig_rows = []
    for (op_name, mode), entries in sorted(groups.items()):
        shape_data = defaultdict(dict)
        for e in entries:
            sl = _shape_label(e["shape"])
            shape_data[sl][e["provider"]] = e
        fig_rows.append((op_name, mode, shape_data))

    if not fig_rows:
        return

    matplotlib.rcParams.update({"font.size": 9})

    n_rows = len(fig_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    for row_idx, (op_name, mode, shape_data) in enumerate(fig_rows):
        shape_labels = list(shape_data.keys())
        providers = sorted({p for sd in shape_data.values() for p in sd})
        n_shapes = len(shape_labels)
        n_provs = len(providers)

        x = np.arange(n_shapes)
        width = 0.8 / max(n_provs, 1)

        # Time chart
        ax_t = axes[row_idx, 0]
        for i, prov in enumerate(providers):
            times = [shape_data[sl].get(prov, {}).get("time_ms", 0) for sl in shape_labels]
            ax_t.bar(
                x + i * width - 0.4 + width / 2,
                times,
                width,
                label=prov,
                color=_color(prov),
                edgecolor="white",
                linewidth=0.3,
            )
        ax_t.set_ylabel("Time (ms)")
        ax_t.set_title(f"{op_name} [{mode}] — Time", fontsize=10)
        ax_t.set_xticks(x)
        ax_t.set_xticklabels(shape_labels, fontsize=7, rotation=20, ha="right")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        ax_t.legend(fontsize=7, loc="upper left")

        # Memory chart
        ax_m = axes[row_idx, 1]
        for i, prov in enumerate(providers):
            mems = [shape_data[sl].get(prov, {}).get("peak_memory_mb", 0) for sl in shape_labels]
            ax_m.bar(
                x + i * width - 0.4 + width / 2,
                mems,
                width,
                label=prov,
                color=_color(prov),
                edgecolor="white",
                linewidth=0.3,
            )
        ax_m.set_ylabel("Peak Memory (MB)")
        ax_m.set_title(f"{op_name} [{mode}] — Memory", fontsize=10)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(shape_labels, fontsize=7, rotation=20, ha="right")
        ax_m.spines["top"].set_visible(False)
        ax_m.spines["right"].set_visible(False)
        ax_m.legend(fontsize=7, loc="upper left")

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
        path = RESULTS_DIR / args.input
        if not path.exists():
            print(f"Not found: {path}")
            return
        results = json.loads(path.read_text())
        out = Path(args.output) if args.output else RESULTS_DIR / args.input.replace(".json", ".png")
        plot_benchmark(results, args.input.replace(".json", "").replace("_", " ").title(), out)
        return

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        results = json.loads(json_file.read_text())
        out = json_file.with_suffix(".png")
        title = json_file.stem.replace("_", " ").title()
        plot_benchmark(results, title, out)


if __name__ == "__main__":
    main()
