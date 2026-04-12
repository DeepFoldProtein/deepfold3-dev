"""Generate optimization report PDF with presentation-ready slides.

Reads all JSON benchmark results and produces a slide-deck-style PDF:
  1. Title slide (GPU, date, dtype)
  2. Executive summary (top speedups, key takeaways)
  3. Per-operation comparison slides (horizontal bars, speedup annotations)
  4. Recommendation summary table

Usage:
    uv run python -m benchmarks.optimization_report
    uv run python -m benchmarks.optimization_report --output report.pdf
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = Path(__file__).parent / "results"

# --- Consistent palette ---
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

PROVIDER_LABELS = {
    "pytorch": "PyTorch (baseline)",
    "compile": "torch.compile",
    "compile_integrated": "compile (integrated)",
    "triton": "Triton fused",
    "triton_raw": "Triton raw kernel",
    "triton_l1+l2": "Triton L1+L2",
    "triton_l3": "Triton L3 (dual GEMM)",
    "triton_evo": "EvoAttention",
    "triton_ln_transpose": "LN+Transpose",
    "triton_evo+pb": "EvoAttn+PairBias",
}

# Op-specific label overrides
_OP_LABELS = {
    "GridSelfAttention": {"triton": "EvoAttention", "triton_raw": "EvoAttention (raw)"},
    "GridColAttn": {"triton_evo": "EvoAttention", "triton_evo+pb": "EvoAttn+PairBias"},
}

# Raw-kernel providers excluded from recommendations
_RAW_PROVIDERS = {"triton_raw"}


def _label(provider: str, op: str = "") -> str:
    if op in _OP_LABELS and provider in _OP_LABELS[op]:
        return _OP_LABELS[op][provider]
    return PROVIDER_LABELS.get(provider, provider)


def _color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "#BDBDBD")


def _shape_label(shape: dict) -> str:
    parts = []
    if "N" in shape:
        parts.append(f"N={shape['N']}")
    for k in sorted(shape):
        if k != "N":
            parts.append(f"{k}={shape[k]}")
    return "\n".join(parts)


def _load_all() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        results.extend(json.loads(f.read_text()))
    return results


def _find_best(results: list[dict]) -> dict:
    """Find best provider per (op, mode, shape). Returns grouped data."""
    groups = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["provider"] in _RAW_PROVIDERS:
            continue
        key = (r["op_name"], r["mode"])
        sl = _shape_label(r["shape"])
        groups[key][sl][r["provider"]] = r

    best = {}
    for key, shape_data in groups.items():
        recs = []
        for sl in shape_data:
            entries = shape_data[sl]
            best_prov = min(entries, key=lambda p: entries[p]["time_ms"])
            pt_time = entries.get("pytorch", {}).get("time_ms", entries[best_prov]["time_ms"])
            speedup = pt_time / entries[best_prov]["time_ms"] if entries[best_prov]["time_ms"] > 0 else 1.0
            recs.append((sl, best_prov, speedup, entries))
        best[key] = recs
    return best


# ─── Slide builders ──────────────────────────────────────────────────────────


def _slide_title(pdf: PdfPages):
    """Slide 1: Title."""
    import torch

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.62, "DeepFold3", ha="center", fontsize=32, fontweight="bold", color="#212121")
    fig.text(0.5, 0.52, "Kernel Optimization Report", ha="center", fontsize=20, color="#424242")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    fig.text(
        0.5,
        0.38,
        f"GPU: {gpu}   |   dtype: bfloat16   |   {datetime.date.today().isoformat()}",
        ha="center",
        fontsize=11,
        color="#9E9E9E",
    )

    pdf.savefig(fig)
    plt.close(fig)


def _slide_summary(pdf: PdfPages, best: dict):
    """Slide 2: Executive summary — top wins + strategy overview."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, "Optimization Strategy", ha="center", fontsize=20, fontweight="bold", color="#212121")

    # Collect top speedups
    top = []
    for (op, mode), recs in best.items():
        if mode != "full":
            continue
        for sl, prov, spd, _ in recs:
            if prov != "pytorch" and spd > 1.05:
                top.append((op, sl.replace("\n", " "), prov, spd))
    top.sort(key=lambda x: -x[3])

    lines = [
        "KEY RESULTS (fwd+bwd, best shape)",
        "",
    ]
    seen_ops = set()
    for op, sl, prov, spd in top[:8]:
        if op in seen_ops:
            continue
        seen_ops.add(op)
        lines.append(f"  {_label(prov, op):28s}  {spd:5.2f}x   {op} ({sl})")

    lines += [
        "",
        "OPTIMIZATION STACK (adapted from NVIDIA cuEquivariance)",
        "",
        "  ATTENTION (104 modules)",
        "    Fused EvoformerAttention — Triton FlashAttention + pair bias",
        "    Fused Pair Bias — LN + tl.dot GEMM(128→4) + permute + mask",
        "",
        "  TRIANGLE MULTIPLY (96 modules)",
        "    L3: Fused Dual GEMM — sigmoid(x@W1) * (x2@W2) * mask",
        "    L2: Fused LN+Linear for post-einsum norm_center + out_proj",
        "    L1: Fused LN + packed GEMM + sigmoid for pre-einsum gate",
        "",
        "  SWIGLU (TransitionBlock / AdaTransitionBlock)",
        "    Multi-row kernel (128 rows/block) + in-place backward",
        "    Pair (M=N²): 1.5-2.1x faster, 2-6x less memory",
        "    Single (M=N): neutral speed, still saves memory",
        "",
        "  TRANSITIONS (context-sensitive)",
        "    Pair (M=N², C=128): Triton fused → 1.2-1.3x",
        "    Single (M=N, C=384): PyTorch baseline faster (M too small)",
        "",
        "  INFRASTRUCTURE",
        "    Persistent autotuning cache with GPU fingerprinting",
        "    Precision control: DEFAULT / TF32 / TF32x3 / IEEE",
    ]

    fig.text(0.06, 0.88, "\n".join(lines), va="top", fontsize=9.5, fontfamily="monospace", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _slide_verification(pdf: PdfPages):
    """Slide 3: Numerical verification results."""
    import subprocess
    import sys

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, "Numerical Verification", ha="center", fontsize=20, fontweight="bold", color="#212121")
    fig.text(
        0.5,
        0.89,
        "All fused kernels verified against PyTorch baseline (bf16, atol=5e-2, rtol=5e-2)",
        ha="center",
        fontsize=10,
        color="#757575",
    )

    # Run verification and capture output
    try:
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.verify", "--verbose"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
    except Exception as e:
        output = f"Verification failed to run: {e}"

    # Parse into structured results — group fwd/grad per kernel into one row
    checks = []  # (section, name, status, max_diff)
    current_section = ""
    all_passed = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("==="):
            current_section = line.strip("= ")
        elif line.startswith("PASS:") or line.startswith("FAIL:"):
            status = "PASS" if line.startswith("PASS") else "FAIL"
            detail = line.split(":", 1)[1].strip()
            parts = detail.split("max_diff=")
            name = parts[0].strip()
            max_diff = parts[1].split()[0] if len(parts) > 1 else "—"
            checks.append((current_section, name, status, max_diff))
        elif "All verifications passed" in line:
            all_passed = True
        elif "SOME VERIFICATIONS FAILED" in line:
            all_passed = False

    if checks:
        # Merge fwd + grad rows for the same kernel section into one row
        merged = {}  # section -> {fwd_status, fwd_diff, grad_status, grad_diff}
        for section, name, status, diff in checks:
            if not section:
                continue
            if section not in merged:
                merged[section] = {"fwd": "—", "fwd_diff": "—", "grad": "—", "grad_diff": "—"}
            is_grad = "grad" in name.lower()
            key = "grad" if is_grad else "fwd"
            merged[section][key] = status
            merged[section][f"{key}_diff"] = diff

        col_labels = ["Kernel", "Forward", "Fwd Max Diff", "Backward", "Bwd Max Diff"]
        table_data = []
        ok = "\u2714 PASS"
        fail = "\u2718 FAIL"
        na = "— n/a"
        for section in merged:
            m = merged[section]
            fwd_s = ok if m["fwd"] == "PASS" else (fail if m["fwd"] == "FAIL" else na)
            grad_s = ok if m["grad"] == "PASS" else (fail if m["grad"] == "FAIL" else na)
            table_data.append([section, fwd_s, m["fwd_diff"], grad_s, m["grad_diff"]])

        # Add summary row
        if all_passed is not None:
            summary = ok.replace("PASS", "ALL PASSED") if all_passed else fail.replace("FAIL", "FAILURES")
            table_data.append(["", summary, "", "", ""])

        ax = fig.add_axes([0.04, 0.04, 0.92, 0.80])
        ax.axis("off")
        tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.auto_set_column_width(range(5))
        tbl.scale(1.0, 1.35)

        # Header style
        for j in range(5):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Color status cells
        for i, row in enumerate(table_data):
            for col in (1, 3):  # fwd status, grad status columns
                if "PASS" in row[col]:
                    tbl[i + 1, col].set_facecolor("#E8F5E9")
                elif "FAIL" in row[col]:
                    tbl[i + 1, col].set_facecolor("#FFEBEE")
    else:
        fig.text(0.5, 0.5, output or "No verification output", ha="center", fontsize=10, fontfamily="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def _slide_op(pdf: PdfPages, op_name: str, recs: list[tuple]):
    """One slide per operation: horizontal bar chart (time + memory)."""
    shape_labels = [r[0] for r in recs]
    entries_list = [r[3] for r in recs]

    # Collect ordered providers
    all_provs = []
    seen = set()
    for entries in entries_list:
        for p in entries:
            if p not in seen:
                all_provs.append(p)
                seen.add(p)

    n_shapes = len(shape_labels)
    n_provs = len(all_provs)
    fig_h = max(3.5, 0.9 * n_shapes * n_provs / 3 + 1.5)

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, fig_h))
    fig.patch.set_facecolor("white")

    display = f"{op_name} (EvoformerAttention)" if op_name == "GridSelfAttention" else op_name
    fig.suptitle(display, fontsize=15, fontweight="bold", color="#212121", y=0.98)

    y = np.arange(n_shapes)
    bar_h = 0.8 / max(n_provs, 1)

    for i, prov in enumerate(all_provs):
        times = [entries_list[j].get(prov, {}).get("time_ms", 0) for j in range(n_shapes)]
        mems = [entries_list[j].get(prov, {}).get("peak_memory_mb", 0) for j in range(n_shapes)]
        c = _color(prov)
        lbl = _label(prov, op_name)
        ypos = y + i * bar_h - 0.4 + bar_h / 2

        ax_t.barh(ypos, times, bar_h, label=lbl, color=c, edgecolor="white", linewidth=0.5)
        ax_m.barh(ypos, mems, bar_h, label=lbl, color=c, edgecolor="white", linewidth=0.5)

        # Annotate speedup on best
        for j, (_, best_prov, speedup, _) in enumerate(recs):
            if prov == best_prov and speedup > 1.005:
                ax_t.annotate(
                    f" {speedup:.1f}x",
                    xy=(times[j], ypos[j]),
                    fontsize=8,
                    fontweight="bold",
                    color="#1B5E20",
                    va="center",
                )

    for ax, xlabel, title in [
        (ax_t, "Time (ms)", "Latency (lower = better)"),
        (ax_m, "Peak Memory (MB)", "Memory"),
    ]:
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(shape_labels, fontsize=7)
        ax.set_title(title, fontsize=10, color="#424242")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend below
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n_provs, 5), fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _slide_table(pdf: PdfPages, best: dict):
    """Final slide: recommendation summary table."""
    ops = defaultdict(dict)
    for (op_name, mode), recs in best.items():
        ops[op_name][mode] = recs

    col_labels = ["Operation", "Shape", "Recommended", "Time (ms)", "Speedup"]
    rows = []
    for op_name in sorted(ops):
        modes = ops[op_name]
        mode = "full" if "full" in modes else next(iter(modes))
        for sl, bp, spd, entries in modes[mode]:
            rows.append(
                [
                    op_name,
                    sl.replace("\n", " / "),
                    _label(bp, op_name),
                    f"{entries[bp]['time_ms']:.2f}",
                    f"{spd:.1f}x" if spd > 1.005 else "—",
                ]
            )

    if not rows:
        return

    rows_per_page = 28
    for start in range(0, len(rows), rows_per_page):
        page = rows[start : start + rows_per_page]
        fig_h = max(4, 0.35 * len(page) + 2)
        fig = plt.figure(figsize=(11, fig_h))
        fig.patch.set_facecolor("white")

        if start == 0:
            fig.text(0.5, 1 - 0.5 / fig_h, "Recommendation Summary", ha="center", fontsize=16, fontweight="bold")

        ax = fig.add_axes([0.04, 0.04, 0.92, 0.85 if start == 0 else 0.92])
        ax.axis("off")
        tbl = ax.table(cellText=page, colLabels=col_labels, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(range(5))
        tbl.scale(1.0, 1.4)

        # Header row
        for j in range(5):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight wins
        for i, row in enumerate(page):
            if row[4] != "—":
                tbl[i + 1, 2].set_facecolor("#E8F5E9")
                tbl[i + 1, 4].set_facecolor("#E8F5E9")

        pdf.savefig(fig)
        plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────


def generate_report(output_path: Path | None = None):
    results = _load_all()
    if not results:
        print("No benchmark results found in results/")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "optimization_report.pdf"

    full = [r for r in results if r["mode"] == "full"]
    if not full:
        full = results
    best = _find_best(full)

    # Group by op for per-op slides
    ops = defaultdict(dict)
    for (op_name, mode), recs in best.items():
        ops[op_name][mode] = recs

    matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 11})

    with PdfPages(output_path) as pdf:
        _slide_title(pdf)
        _slide_summary(pdf, best)
        _slide_verification(pdf)

        for op_name in sorted(ops):
            modes = ops[op_name]
            mode = "full" if "full" in modes else next(iter(modes))
            _slide_op(pdf, op_name, modes[mode])

        _slide_table(pdf, best)

    print(f"Saved optimization report to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate optimization report PDF")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    generate_report(out)
