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


def _to_float(x: str | None) -> float | None:
    """Parse a raw cell to float, or None if blank/non-numeric (OST failure)."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def per_target_lddt(raw_csv: Path) -> list[float]:
    """One lDDT per target = mean over its (seed, sample) predictions.

    Args:
        raw_csv: ``raw/{category}_ost.csv`` written by eval_ost.

    Returns:
        List of per-target mean lDDTs (targets with no valid sample dropped).
    """
    # pdb_id -> {(seed, sample): lddt} — the dict de-dups repeated interface rows.
    by_target: dict[str, dict[tuple[str, str], float]] = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            ld = _to_float(row.get("lddt"))
            if ld is None:
                continue
            by_target.setdefault(row["pdb_id"], {})[(row["seed"], row["sample"])] = ld
    out = []
    for samples in by_target.values():
        if samples:
            out.append(statistics.fmean(samples.values()))
    return out


def collect(series: str, categories: list[str]) -> dict[str, list[dict]]:
    """Gather per-step (mean, median) across-target lDDT for each category."""
    eval_root = BENCH / "runs" / series / "eval"
    result: dict[str, list[dict]] = {}
    for cat in categories:
        cat_dir = eval_root / cat
        if not cat_dir.is_dir():
            continue
        points = []
        for step_dir in sorted(cat_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
            raw = step_dir / "raw" / f"{cat}_ost.csv"
            if not raw.exists():
                continue
            vals = per_target_lddt(raw)
            if not vals:
                continue
            step = int(step_dir.name.split("_")[1])
            points.append(
                {
                    "step": step,
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "median": statistics.median(vals),
                }
            )
        if points:
            result[cat] = points
    return result


def write_csv(data: dict[str, list[dict]], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "step", "n_targets", "mean_lddt", "median_lddt"])
        for cat, pts in data.items():
            for p in pts:
                w.writerow([cat, p["step"], p["n"], round(p["mean"], 4), round(p["median"], 4)])


def plot(data: dict[str, list[dict]], series: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat, pts in data.items():
        steps = [p["step"] for p in pts]
        color = CATEGORY_COLORS.get(cat, "tab:gray")
        ax.plot(steps, [p["mean"] for p in pts], "-o", color=color, lw=2, ms=5, label=cat)
        ax.plot(steps, [p["median"] for p in pts], "--x", color=color, lw=1.5, ms=6, alpha=0.85)

    ax.set_xlabel("training step")
    ax.set_ylabel("lDDT (per-target mean, averaged over seeds x samples)")
    ax.set_title(f"{series}: per-category lDDT vs step  (solid = mean, dashed = median)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Two legends: category (colour) is parked below the axes, spread across
    # columns so it never covers data; the small mean/median style key stays inside.
    cat_legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), borderaxespad=0, ncol=4, fontsize=9, title="category"
    )
    ax.add_artist(cat_legend)
    style_handles = [
        plt.Line2D([], [], color="0.3", ls="-", marker="o", label="mean"),
        plt.Line2D([], [], color="0.3", ls="--", marker="x", label="median"),
    ]
    ax.legend(handles=style_handles, loc="upper left", fontsize=9)

    fig.savefig(out_png, dpi=130, bbox_inches="tight", bbox_extra_artists=[cat_legend])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", nargs="*", default=list(CATEGORY_COLORS), help="subset of categories")
    args = ap.parse_args()

    data = collect(args.series, args.category)
    if not data:
        print(f"[lddt-grid] no eval raw dumps found for series {args.series}")
        return

    out_root = BENCH / "runs" / args.series
    out_png = out_root / "lddt_by_category.png"
    out_csv = out_root / "lddt_by_category.csv"
    write_csv(data, out_csv)
    plot(data, args.series, out_png)

    print(f"[lddt-grid] {len(data)} categories -> {out_csv}")
    for cat, pts in data.items():
        cells = "  ".join(f"{p['step']}:mean={p['mean']:.3f}/med={p['median']:.3f}(n={p['n']})" for p in pts)
        print(f"  {cat:28s} {cells}")
    print(f"[lddt-grid] plot -> {out_png}")


if __name__ == "__main__":
    main()
