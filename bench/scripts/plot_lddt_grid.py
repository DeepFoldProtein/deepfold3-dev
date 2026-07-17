"""Cross-category lDDT trajectory: mean + median per step, all categories on one plot.

Reads the per-sample OST dumps written by ``eval_ost.py``
(``runs/{series}/eval/{category}/step_{N}/raw/{category}_ost.csv``), aggregates
each target to a single lDDT (mean over that target's seeds x samples, matching
``eval_ost``'s ``mean_lddt`` definition), then plots the across-target **mean**
(solid) and **median** (dashed) versus training step, one colour per category.

Both monomer (``pdb_id,seed,sample,lddt,...``) and interface
(``pdb_id,ic1,ic2,seed,sample,...,lddt``) raw layouts are handled; the interface
dump repeats the whole-model lDDT across interface rows, so lDDT is collapsed to
one value per (pdb_id, seed, sample) before averaging.

Usage:
    python bench/scripts/plot_lddt_grid.py --series df3v2
    python bench/scripts/plot_lddt_grid.py --series df3v2 --category monomer_protein monomer_rna
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = Path(__file__).resolve().parents[1]

# Stable colour per category so the plot looks the same across regens.
CATEGORY_COLORS = {
    "monomer_protein": "tab:blue",
    "monomer_rna": "tab:green",
    "monomer_dna": "tab:olive",
    "interface_protein_protein": "tab:red",
    "interface_antibody_antigen": "tab:purple",
    "interface_protein_dna": "tab:orange",
    "interface_protein_rna": "tab:brown",
    "interface_protein_peptide": "tab:pink",
}

# The figure has 3 rows: monomer whole-model lDDT, interface complex (whole-model)
# lDDT, and interface inter-chain lDDT (``ilddt``). ilddt is collapse-sensitive —
# the whole-model number mixes fold quality with assembly and hides homomer
# chain-collapse (which DockQ actively inflates). ``ilddt`` was added to eval_ost.py
# partway through some series (df3v2: continuous from step 10k, plus one stray pre-10k
# recompute at step_2000 for interface_protein_peptide). ``_drop_isolated_early`` peels
# such lone leading points off the front so the row stays continuous — series-agnostic,
# so df3v3 (valid ilddt from step_1000, no gap) keeps every point.
INTERFACE_CATEGORIES = {
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_dna",
    "interface_protein_rna",
    "interface_protein_peptide",
}


def _to_float(x: str | None) -> float | None:
    """Parse a raw cell to float, or None if blank/non-numeric (OST failure)."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def per_target_lddt(raw_csv: Path, col: str = "lddt") -> list[float]:
    """One score per target = mean over its (seed, sample) predictions.

    Args:
        raw_csv: ``raw/{category}_ost.csv`` written by eval_ost.
        col: which metric column to read (``lddt`` whole-model, or ``ilddt``
            inter-chain for interface categories).

    Returns:
        List of per-target mean scores (targets with no valid sample dropped).
    """
    # pdb_id -> {(seed, sample): score} — the dict de-dups repeated interface rows.
    by_target: dict[str, dict[tuple[str, str], float]] = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            ld = _to_float(row.get(col))
            if ld is None:
                continue
            by_target.setdefault(row["pdb_id"], {})[(row["seed"], row["sample"])] = ld
    out = []
    for samples in by_target.values():
        if samples:
            out.append(statistics.fmean(samples.values()))
    return out


def _drop_isolated_early(points: list[dict], gap_factor: float = 3.0) -> list[dict]:
    """Peel isolated leading points separated from the main series by a big gap.

    ``ilddt`` was added to eval_ost mid-run for some series, leaving a lone stray
    point far ahead of where the metric became continuous (df3v2: a step_2000
    recompute while the real series starts at step 10k). Rather than hardcode a
    per-series min step, drop leading points whose gap to the next point is much
    larger than the typical spacing. Only the front is peeled, so a genuine gap in
    the middle of a run is preserved, and a series with no leading gap (df3v3) is
    returned untouched.
    """
    if len(points) < 3:
        return points
    steps = [p["step"] for p in points]
    gaps = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    typical = statistics.median(gaps)
    i = 0
    while i < len(gaps) and gaps[i] > gap_factor * typical:
        i += 1
    return points[i:]


def collect(series: str, categories: list[str], col: str) -> dict[str, list[dict]]:
    """Gather per-step (mean, median) across-target ``col`` for each category.

    Steps whose raw dump lacks ``col`` (e.g. ``ilddt`` before it was added mid-run)
    yield no valid target and are skipped.
    """
    eval_root = BENCH / "runs" / series / "eval"
    result: dict[str, list[dict]] = {}
    for cat in categories:
        cat_dir = eval_root / cat
        if not cat_dir.is_dir():
            continue
        points = []
        for step_dir in sorted(cat_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
            step = int(step_dir.name.split("_")[1])
            raw = step_dir / "raw" / f"{cat}_ost.csv"
            if not raw.exists():
                continue
            vals = per_target_lddt(raw, col)
            if not vals:
                continue
            points.append(
                {"step": step, "n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals)}
            )
        if points:
            result[cat] = points
    return result


def write_csv(panels: list[tuple[str, dict[str, list[dict]]]], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "metric", "step", "n_targets", "mean_lddt", "median_lddt"])
        for metric, data in panels:
            for cat, pts in data.items():
                for p in pts:
                    w.writerow([cat, metric, p["step"], p["n"], round(p["mean"], 4), round(p["median"], 4)])


def _panel(ax, data: dict[str, list[dict]], title: str, ylabel: str) -> None:
    """Draw one row: mean (solid) + median (dashed) per category."""
    for cat, pts in data.items():
        steps = [p["step"] for p in pts]
        color = CATEGORY_COLORS.get(cat, "tab:gray")
        ax.plot(steps, [p["mean"] for p in pts], "-o", color=color, lw=2, ms=4, label=cat)
        ax.plot(steps, [p["median"] for p in pts], "--x", color=color, lw=1.3, ms=5, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylim(-0.02, 1.0)  # shared [0,1] across rows so the three metrics compare directly
    ax.grid(True, alpha=0.3)
    if data:
        ax.legend(fontsize=8, ncol=2, loc="upper left")


def plot(mono_lddt, iface_lddt, iface_ilddt, series: str, out_png: Path) -> None:
    """3-row figure: monomer whole-model / interface complex whole-model /
    interface inter-chain ilddt. Shared training-step x-axis."""
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    _panel(ax0, mono_lddt, "Monomer — whole-model lDDT", "lDDT")
    _panel(ax1, iface_lddt, "Interface — complex (whole-model) lDDT", "lDDT")
    _panel(ax2, iface_ilddt, "Interface — inter-chain lDDT (iLDDT, collapse-sensitive)", "iLDDT")
    ax2.set_xlabel("training step")
    fig.suptitle(f"{series}: per-category lDDT vs step   (solid = mean, dashed = median)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", nargs="*", default=list(CATEGORY_COLORS), help="subset of categories")
    args = ap.parse_args()

    mono = [c for c in args.category if c not in INTERFACE_CATEGORIES]
    iface = [c for c in args.category if c in INTERFACE_CATEGORIES]
    mono_lddt = collect(args.series, mono, "lddt")  # row 1
    iface_lddt = collect(args.series, iface, "lddt")  # row 2: complex whole-model
    iface_ilddt = collect(args.series, iface, "ilddt")  # row 3: inter-chain
    iface_ilddt = {cat: _drop_isolated_early(pts) for cat, pts in iface_ilddt.items()}
    if not (mono_lddt or iface_lddt or iface_ilddt):
        print(f"[lddt-grid] no eval raw dumps found for series {args.series}")
        return

    out_root = BENCH / "runs" / args.series
    out_png = out_root / "lddt_by_category.png"
    out_csv = out_root / "lddt_by_category.csv"
    write_csv([("lddt", mono_lddt), ("lddt", iface_lddt), ("ilddt", iface_ilddt)], out_csv)
    plot(mono_lddt, iface_lddt, iface_ilddt, args.series, out_png)

    print(f"[lddt-grid] -> {out_csv}")
    for label, data in (("monomer lddt", mono_lddt), ("interface lddt", iface_lddt), ("interface ilddt", iface_ilddt)):
        for cat, pts in data.items():
            cells = "  ".join(f"{p['step']}:m={p['mean']:.3f}/med={p['median']:.3f}(n={p['n']})" for p in pts)
            print(f"  [{label}] {cat:28s} {cells}")
    print(f"[lddt-grid] plot -> {out_png}")


if __name__ == "__main__":
    main()
