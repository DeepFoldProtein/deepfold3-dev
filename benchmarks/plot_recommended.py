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
    "triton_l1+l2": "#A9D18E",
    "triton_l3": "#2E7D32",
    "triton_evo": "#70AD47",
    "triton_ln_transpose": "#C5E0B4",
    "triton_evo+pb": "#1B5E20",
    "triton_evo+ln_t": "#4CAF50",
    "sdpa+compile+pb": "#FF9800",
}
PROVIDER_LABELS = {
    "pytorch": "PyTorch",
    "compile": "torch.compile",
    "compile_integrated": "compile (integrated)",
    "triton": "Triton fused",
    "triton_raw": "Triton raw kernel",
    "triton_l1+l2": "Triton L1+L2 (gate+center)",
    "triton_l3": "Triton L3 (dual GEMM)",
    "triton_evo": "EvoAttention",
    "triton_ln_transpose": "LN+Transpose",
    "triton_evo+pb": "EvoAttn + PairBias",
    "triton_evo+ln_t": "EvoAttn + LN+T",
    "sdpa+compile+pb": "SDPA+compile+PairBias",
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

        # --- Optimization Strategy Overview ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.94, "Optimization Strategy", ha="center", fontsize=18, fontweight="bold")
        strategy_text = (
            "DeepFold3 Triton Kernel Optimization Stack\n"
            "(adapted from NVIDIA cuEquivariance design patterns)\n"
            "\n"
            "ATTENTION (104 modules, 3.7-4.6x speedup)\n"
            "  1. Fused EvoformerAttention \u2014 Triton FlashAttention with pair bias\n"
            "     \u2022 Autotuned tile sizes (BLOCK_Q/KV x num_stages x num_warps)\n"
            "  2. Fused Pair Bias \u2014 LN + tl.dot GEMM(128\u21924) + permute + mask\n"
            "     \u2022 5-10x faster than old scalar loop; supports training backward\n"
            "  3. Deferred Col Transpose \u2014 skip N\u00b2C copy for column attention\n"
            "     \u2022 Col overhead: 52% \u2192 8% vs row attention\n"
            "\n"
            "TRIANGLE MULTIPLY (96 modules, 1.2-1.3x speedup)\n"
            "  Level 3: Fused Dual GEMM \u2014 sigmoid(x@W1+b1) * (x2@W2+b2) * mask\n"
            "     \u2022 TWO_INPUTS mode, bias, mask fusion, TF32 precision control\n"
            "     \u2022 Eliminates Incoming/Outgoing memory asymmetry (+1.5GB \u2192 0)\n"
            "     \u2022 Backward: pre-GEMM kernel + cuBLAS (split pattern from cuEQ)\n"
            "  Level 2: Fused LN+Linear for post-einsum norm_center + out_proj\n"
            "  Level 1: Fused LN+packed GEMM+sigmoid for pre-einsum gate\n"
            "  \u2022 einsum replaced with torch.matmul on permuted views (zero-copy)\n"
            "\n"
            "TRANSITIONS (context-sensitive)\n"
            "  Pair transitions (M=N\u00b2, C=128): Triton fused \u2192 1.2-1.3x\n"
            "  Single transitions (M=N, C=384): PyTorch baseline faster (M too small)\n"
            "\n"
            "INFRASTRUCTURE\n"
            "  \u2022 Precision enum: DEFAULT / TF32 / TF32x3 / IEEE\n"
            "  \u2022 Persistent autotuning cache: ~/.cache/deepfold3-triton/\n"
            "  \u2022 GPU fingerprinting with fuzzy matching (20% tolerance)"
        )
        fig.text(0.08, 0.85, strategy_text, va="top", fontsize=10, fontfamily="monospace", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Preset Configurations ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.94, "OptimConfig Presets", ha="center", fontsize=18, fontweight="bold")
        presets_text = (
            "OptimConfig.recommended()  \u2014  Best overall (training + inference)\n"
            "  \u2714 use_fused_evo_attention      2.9-3.3x attention speedup\n"
            "  \u2714 use_fused_pair_bias           +25-30% on top (training + inference)\n"
            "  \u2714 use_fused_triangle_mul        1.2-1.3x TriMul + fixes mem asymmetry\n"
            "  \u2714 use_fused_transition_pair      1.2-1.3x pair transitions (M=N\u00b2)\n"
            "  \u2714 use_torch_compile              1.3-2.3x for MSA/Single attention\n"
            "\n"
            "OptimConfig.inference()  \u2014  Maximum inference speed\n"
            "  \u2714 Same as recommended, minus torch.compile\n"
            "  \u2714 use_fused_ln_transpose         1.7-2.3x LN+transpose (forward only)\n"
            "\n"
            "OptimConfig.triton_only()  \u2014  All beneficial Triton kernels\n"
            "  \u2714 All fused kernels that demonstrably outperform PyTorch\n"
            "  \u2718 No torch.compile (avoids compilation overhead)\n"
            "\n"
            "OptimConfig.all_enabled()  \u2014  For testing only\n"
            "  \u2714 Everything including slow single/ada transitions\n"
            "\n"
            "Flags NOT in recommended (benchmarked slower than PyTorch):\n"
            "  \u2718 use_fused_transition_single    0.8x at M=N, C=384\n"
            "  \u2718 use_fused_ada_transition       0.8x at M=N, C=768"
        )
        fig.text(0.08, 0.85, presets_text, va="top", fontsize=10, fontfamily="monospace", linespacing=1.6)
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
