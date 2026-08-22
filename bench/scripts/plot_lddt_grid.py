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

A measured reference series can be overlaid as a dotted horizontal baseline per
category (``--ref``, off by default) — computed the SAME way as the trajectory
(``per_target_lddt`` over the ref series' latest benched step), so it is a
like-for-like comparison on the same num-token-capped manifest. A baseline is
drawn only for categories that have BOTH a trajectory in ``--series`` and an eval
in the ref series (categories the ref never benched simply get no baseline).
Series in ``summarize.REF_SERIES_CAVEAT`` — currently the template-free
``af3_ref`` fold — are drawn with the caveat stamped into the legend and the
title, never as a bare "AF3" baseline; see the note there. The published paper
means stay on rows 1-2 alongside a measured baseline (dash-dot), so the figure
shows both the full-strength AF3 target and the handicapped measured fold.

Usage:
    python bench/scripts/plot_lddt_grid.py --series df3v2
    python bench/scripts/plot_lddt_grid.py --series df3v2 --category monomer_protein monomer_rna
    python bench/scripts/plot_lddt_grid.py --series df3v4r14 --ref some_measured_series
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single source of truth for the retirement list and the published AF3 numbers.
from summarize import (  # noqa: E402
    AF3_INTERFACE_LDDT,
    AF3_MONOMER,
    REF_SERIES_CAVEAT,
    REF_SERIES_EXCLUDE_CATEGORIES,
)

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


def _panel(
    ax,
    data: dict[str, list[dict]],
    title: str,
    ylabel: str,
    ref: dict[str, dict] | None = None,
    ref_label: str = "",
    paper: dict[str, float] | None = None,
) -> None:
    """Draw one row: mean (solid) + median (dashed) per category.

    ``ref`` overlays a measured baseline series as faint horizontal lines — mean
    (solid) and median (dotted), category-coloured — only for categories the
    trajectory itself shows, so the comparison is line-to-line.

    ``paper`` is the published AF3 mean per category (FoldBench Supp. Table 3),
    drawn dash-dot. It is a mean over the paper's FULL set, not our
    num-token-capped subset — a target marker, not a like-for-like value. It is
    kept even when ``ref`` is present: a caveated measured baseline (template-free
    af3_ref) sits BELOW the published number, and showing only one of the two
    invites reading the gap that remains as either larger or smaller than it is.
    """
    for cat, pts in data.items():
        steps = [p["step"] for p in pts]
        color = CATEGORY_COLORS.get(cat, "tab:gray")
        ax.plot(steps, [p["mean"] for p in pts], "-o", color=color, lw=2, ms=4, label=cat)
        ax.plot(steps, [p["median"] for p in pts], "--x", color=color, lw=1.3, ms=5, alpha=0.8)
    if ref:
        drawn = False
        for cat, r in ref.items():
            if cat not in data:  # only baseline the categories that have a trajectory here
                continue
            color = CATEGORY_COLORS.get(cat, "tab:gray")
            ax.axhline(r["mean"], color=color, ls="-", lw=1.1, alpha=0.35, zorder=1)
            ax.axhline(r["median"], color=color, ls=":", lw=1.1, alpha=0.35, zorder=1)
            drawn = True
        if drawn:  # grey proxies so the legend explains the faint baseline styling
            ax.plot([], [], color="0.5", ls="-", lw=1.1, alpha=0.6, label=f"{ref_label} mean")
            ax.plot([], [], color="0.5", ls=":", lw=1.1, alpha=0.6, label=f"{ref_label} median")
    if paper:
        drawn = False
        for cat, v in paper.items():
            if cat not in data:
                continue
            ax.axhline(v, color=CATEGORY_COLORS.get(cat, "tab:gray"), ls="-.", lw=1.1, alpha=0.4, zorder=1)
            drawn = True
        if drawn:
            ax.plot([], [], color="0.5", ls="-.", lw=1.1, alpha=0.6, label="AF3 paper mean (full set)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylim(-0.02, 1.0)  # shared [0,1] across rows so the three metrics compare directly
    ax.grid(True, alpha=0.3)
    if data or ref:
        ax.legend(fontsize=8, ncol=2, loc="upper left")


def ref_values(ref_series: str, categories: list[str], col: str) -> dict[str, dict]:
    """Baseline ``{cat: {mean, median, step, n}}`` from a fixed reference series.

    Computed via the same ``collect`` path as the trajectory (like-for-like), then
    reduced to the ref series' latest benched step per category (``af3_ref`` writes
    a single ``step_0``; latest-step is robust to a ref benched elsewhere).

    Categories the ref series cannot be compared on at all (``af3_ref``'s
    pre-water-fix nucleic monomers) are dropped rather than drawn.
    """
    skip = REF_SERIES_EXCLUDE_CATEGORIES.get(ref_series, set())
    for cat in sorted(skip & set(categories)):
        print(f"[lddt-grid] {ref_series}/{cat}: fold not comparable — baseline dropped")
    usable = [c for c in categories if c not in skip]
    return {cat: pts[-1] for cat, pts in collect(ref_series, usable, col).items() if pts}


def plot(mono_lddt, iface_lddt, iface_ilddt, series: str, out_png: Path, refs=None, ref_label: str = "") -> None:
    """3-row figure: monomer whole-model / interface complex whole-model /
    interface inter-chain ilddt. Shared training-step x-axis. ``refs`` is a
    ``(mono, iface_lddt, iface_ilddt)`` triple of baseline dicts (or None)."""
    rm, ri, rii = refs or ({}, {}, {})
    # Paper AF3 means, used only where no measured baseline is available. There is no
    # published inter-chain iLDDT, so row 3 keeps no marker.
    paper_mono = {c: v["lddt"] for c, v in AF3_MONOMER.items()}
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    _panel(ax0, mono_lddt, "Monomer — whole-model lDDT", "lDDT", ref=rm, ref_label=ref_label, paper=paper_mono)
    _panel(
        ax1,
        iface_lddt,
        "Interface — complex (whole-model) lDDT",
        "lDDT",
        ref=ri,
        ref_label=ref_label,
        paper=AF3_INTERFACE_LDDT,
    )
    _panel(
        ax2,
        iface_ilddt,
        "Interface — inter-chain lDDT (iLDDT, collapse-sensitive)",
        "iLDDT",
        ref=rii,
        ref_label=ref_label,
    )
    ax2.set_xlabel("training step")
    base = f"{series}: per-category lDDT vs step   (solid = mean, dashed = median)"
    if ref_label:
        base += (
            f"   —   faint = {ref_label} measured on THIS subset (solid mean, dotted median)"
            "; dash-dot = AF3 paper mean (full set)"
        )
    else:
        base += "   —   dash-dot = AF3 paper mean (FoldBench full set, not this subset)"
    fig.suptitle(base, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", nargs="*", default=list(CATEGORY_COLORS), help="subset of categories")
    ap.add_argument(
        "--ref",
        default="",
        help="fixed reference series overlaid as a faint per-category baseline "
        "(mean solid, median dotted). Off by default; series in REF_SERIES_CAVEAT "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(REF_SERIES_CAVEAT.items()))}) are "
        "drawn with that caveat in the legend.",
    )
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

    # Reference baseline overlay (skip when empty or the series itself). A
    # caveated series is drawn, with the caveat carried in the legend label.
    ref_series = args.ref.strip()
    caveat = REF_SERIES_CAVEAT.get(ref_series, "")
    if caveat:
        print(f"[lddt-grid] ref series '{ref_series}' is {caveat} — labelled as such, not a parity baseline")
    ref_label = f"{ref_series} ({caveat})" if caveat else ref_series
    refs = None
    if ref_series and ref_series != args.series:
        ref_mono = ref_values(ref_series, mono, "lddt")
        ref_iface = ref_values(ref_series, iface, "lddt")
        ref_iface_i = ref_values(ref_series, iface, "ilddt")
        refs = (ref_mono, ref_iface, ref_iface_i)
        if not (ref_mono or ref_iface or ref_iface_i):
            print(f"[lddt-grid] ref series '{ref_series}' has no eval dumps — baseline skipped")
            ref_label, refs = "", None
    else:
        ref_label = ""

    out_root = BENCH / "runs" / args.series
    out_png = out_root / "lddt_by_category.png"
    out_csv = out_root / "lddt_by_category.csv"
    write_csv([("lddt", mono_lddt), ("lddt", iface_lddt), ("ilddt", iface_ilddt)], out_csv)
    plot(mono_lddt, iface_lddt, iface_ilddt, args.series, out_png, refs=refs, ref_label=ref_label)

    print(f"[lddt-grid] -> {out_csv}")
    for label, data in (("monomer lddt", mono_lddt), ("interface lddt", iface_lddt), ("interface ilddt", iface_ilddt)):
        for cat, pts in data.items():
            cells = "  ".join(f"{p['step']}:m={p['mean']:.3f}/med={p['median']:.3f}(n={p['n']})" for p in pts)
            print(f"  [{label}] {cat:28s} {cells}")
    print(f"[lddt-grid] plot -> {out_png}")


if __name__ == "__main__":
    main()
