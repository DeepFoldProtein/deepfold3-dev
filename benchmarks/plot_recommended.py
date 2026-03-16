"""Generate recommended optimizations PDF report from benchmark results.

Reads all JSON benchmark results and produces a PDF summarizing which
optimization backend is recommended for each operation at each shape.

Usage:
    uv run python benchmarks/plot_recommended.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = Path(__file__).parent / "results"

# Provider display names and colors
PROVIDER_COLORS = {
    "pytorch": "#5B9BD5",
    "compile": "#ED7D31",
    "compile_integrated": "#A5A5A5",
    "triton": "#70AD47",
    "triton_raw": "#FFC000",
}
PROVIDER_LABELS = {
    "pytorch": "PyTorch",
    "compile": "torch.compile",
    "compile_integrated": "compile (integrated)",
    "triton": "Triton fused",
    "triton_raw": "Triton raw kernel",
}

# Op-specific provider label overrides
_OP_PROVIDER_LABELS = {
    "GridSelfAttention": {
        "triton": "EvoAttention",
        "triton_raw": "EvoAttention (raw)",
    },
}


def _provider_label(provider: str, op_name: str = "") -> str:
    """Get display label for a provider, with op-specific overrides."""
    if op_name in _OP_PROVIDER_LABELS and provider in _OP_PROVIDER_LABELS[op_name]:
        return _OP_PROVIDER_LABELS[op_name][provider]
    return PROVIDER_LABELS.get(provider, provider)


def _load_all_results() -> list[dict]:
    """Load all JSON result files."""
    results = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        results.extend(json.loads(json_file.read_text()))
    return results


def _shape_label(shape: dict) -> str:
    parts = []
    if "N" in shape:
        parts.append(f"N={shape['N']}")
    for k in sorted(shape):
        if k == "N":
            continue
        parts.append(f"{k}={shape[k]}")
    return "\n".join(parts)


def _find_recommended(results: list[dict]) -> dict:
    """For each (op_name, mode), find the best provider per shape.

    Excludes raw kernel providers (e.g. triton_raw) from recommendations since
    they bypass module-level work (projections, gating, LN) and aren't a fair
    comparison against full modules.

    Returns: {(op_name, mode): [(shape_label, best_provider, speedup_vs_pytorch, entries_dict), ...]}
    """
    # Providers that are raw kernels — not fair module-level comparisons
    _RAW_PROVIDERS = {"triton_raw"}

    groups = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["provider"] in _RAW_PROVIDERS:
            continue
        key = (r["op_name"], r["mode"])
        sl = _shape_label(r["shape"])
        groups[key][sl][r["provider"]] = r

    recommended = {}
    for key, shape_data in groups.items():
        recs = []
        for sl in shape_data:
            entries = shape_data[sl]
            # Find fastest provider, excluding raw kernels
            candidates = {p: e for p, e in entries.items() if p not in _RAW_PROVIDERS}
            if not candidates:
                candidates = entries  # fallback if only raw providers available
            best_prov = min(candidates, key=lambda p: candidates[p]["time_ms"])
            pytorch_time = entries.get("pytorch", {}).get("time_ms", entries[best_prov]["time_ms"])
            speedup = pytorch_time / entries[best_prov]["time_ms"] if entries[best_prov]["time_ms"] > 0 else 1.0
            recs.append((sl, best_prov, speedup, entries))
        recommended[key] = recs
    return recommended


def generate_pdf(output_path: Path | None = None):
    """Generate the recommended optimizations PDF report."""
    results = _load_all_results()
    if not results:
        print("No benchmark results found in results/")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "recommended.pdf"

    # Only use "full" mode (fwd+bwd) for recommendations
    full_results = [r for r in results if r["mode"] == "full"]
    if not full_results:
        full_results = results

    recommended = _find_recommended(full_results)

    # Group by op_name for plotting
    ops = defaultdict(dict)
    for (op_name, mode), recs in recommended.items():
        ops[op_name][mode] = recs

    matplotlib.rcParams.update({"font.size": 9})

    with PdfPages(output_path) as pdf:
        # --- Title page ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.5, 0.6, "DeepFold3 Kernel Benchmark Report", ha="center", va="center", fontsize=22, fontweight="bold"
        )
        fig.text(0.5, 0.50, "Recommended Optimizations", ha="center", va="center", fontsize=16, color="#555555")

        import datetime

        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        fig.text(
            0.5,
            0.38,
            f"GPU: {gpu_name}  |  dtype: bfloat16  |  {datetime.date.today().isoformat()}",
            ha="center",
            va="center",
            fontsize=11,
            color="#888888",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # --- Per-operation comparison charts ---
        for op_name in sorted(ops):
            modes = ops[op_name]
            # Use "full" mode if available, else first available
            mode = "full" if "full" in modes else next(iter(modes))
            recs = modes[mode]

            shape_labels = [r[0] for r in recs]
            entries_list = [r[3] for r in recs]

            # Collect all providers across shapes
            all_providers = []
            seen = set()
            for entries in entries_list:
                for p in entries:
                    if p not in seen:
                        all_providers.append(p)
                        seen.add(p)

            n_shapes = len(shape_labels)
            n_providers = len(all_providers)

            fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11, max(3.5, 0.8 * n_shapes + 0.8)))
            display_name = f"{op_name} (EvoformerAttention)" if op_name == "GridSelfAttention" else op_name
            fig.suptitle(f"{display_name} — {mode} (fwd+bwd)", fontsize=13, fontweight="bold")

            y = np.arange(n_shapes)
            bar_h = 0.8 / max(n_providers, 1)

            for i, prov in enumerate(all_providers):
                times = [entries_list[j].get(prov, {}).get("time_ms", 0) for j in range(n_shapes)]
                color = PROVIDER_COLORS.get(prov, "#CCCCCC")
                label = _provider_label(prov, op_name)

                ax_time.barh(
                    y + i * bar_h - 0.4 + bar_h / 2,
                    times,
                    bar_h,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )

                # Mark the best with speedup annotation
                for j, (sl, best_prov, speedup, _) in enumerate(recs):
                    if prov == best_prov:
                        t = entries_list[j][prov]["time_ms"]
                        ax_time.annotate(
                            f" {speedup:.2f}x",
                            xy=(t, y[j] + i * bar_h - 0.4 + bar_h / 2),
                            fontsize=7,
                            fontweight="bold",
                            color="#228B22",
                            va="center",
                        )

                mems = [entries_list[j].get(prov, {}).get("peak_memory_mb", 0) for j in range(n_shapes)]
                ax_mem.barh(
                    y + i * bar_h - 0.4 + bar_h / 2,
                    mems,
                    bar_h,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax_time.set_xlabel("Time (ms)")
            ax_time.set_yticks(y)
            ax_time.set_yticklabels(shape_labels, fontsize=7)
            ax_time.set_title("Latency (lower is better)")
            ax_time.invert_yaxis()

            ax_mem.set_xlabel("Peak Memory (MB)")
            ax_mem.set_yticks(y)
            ax_mem.set_yticklabels(shape_labels, fontsize=7)
            ax_mem.set_title("Memory")
            ax_mem.invert_yaxis()

            # Place legend outside the plot area to avoid overlapping bars
            ax_time.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=n_providers)
            ax_mem.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=n_providers)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Summary recommendation table (paginated) ---
        col_labels = ["Operation", "Shape", "Recommended", "Time (ms)", "Speedup"]
        table_data = []
        for op_name in sorted(ops):
            modes = ops[op_name]
            mode = "full" if "full" in modes else next(iter(modes))
            for sl, best_prov, speedup, entries in modes[mode]:
                best_time = entries[best_prov]["time_ms"]
                table_data.append(
                    [
                        op_name,
                        sl.replace("\n", " / "),
                        _provider_label(best_prov, op_name),
                        f"{best_time:.3f}",
                        f"{speedup:.2f}x" if speedup > 1.005 else "baseline",
                    ]
                )

        if table_data:
            rows_per_page = 35
            for page_start in range(0, len(table_data), rows_per_page):
                page_rows = table_data[page_start : page_start + rows_per_page]
                n_rows = len(page_rows)
                fig_h = max(4, 0.32 * n_rows + 1.5)
                fig = plt.figure(figsize=(11, fig_h))

                if page_start == 0:
                    fig.text(
                        0.5, 1 - 0.4 / fig_h, "Recommendation Summary", ha="center", fontsize=16, fontweight="bold"
                    )

                ax = fig.add_axes([0.05, 0.05, 0.9, 0.85 if page_start == 0 else 0.92])
                ax.axis("off")
                tbl = ax.table(
                    cellText=page_rows,
                    colLabels=col_labels,
                    loc="upper center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(7)
                tbl.auto_set_column_width(range(5))
                tbl.scale(1.0, 1.3)

                for i, row in enumerate(page_rows):
                    if row[2] != "PyTorch":
                        tbl[i + 1, 2].set_facecolor("#E2EFDA")
                    if row[4] != "baseline" and float(row[4].replace("x", "")) >= 1.10:
                        tbl[i + 1, 4].set_facecolor("#E2EFDA")

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved recommended report to {output_path}")


if __name__ == "__main__":
    generate_pdf()
