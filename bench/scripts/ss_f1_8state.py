"""DSSP 8-state per-class F1 plot from the ss_f1 trajectory CSV.

`ss_f1.py` already computes per-class F1 for all 8 DSSP states and writes them
to `ss_analysis/ss_f1_<category>.csv` (columns `f1_8_{H,E,G,I,T,S,B,P,-}`), but
its own plot only shows the 3-state collapse. This standalone reader renders the
full 8-state per-class F1 trajectory, grouping the codes by SS family
(helix-like H/G/I/P, strand-like E/B, coil-like T/S/-) so the strand-vs-helix
learning gap is legible at 8-state resolution.

Run `ss_f1.py` first to (re)generate the CSV. This script only reads it.

Usage:
    python bench/scripts/ss_f1_8state.py --series df3v1 --category monomer_protein
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]

# DSSP-8 code -> (long label, SS family, line colour, line style).
# Families share a hue; styles disambiguate codes within a family.
DSSP_8 = ["H", "E", "G", "I", "T", "S", "B", "P", "-"]
META = {
    "H": ("H  α-helix",      "helix",  "tab:red",    "-"),
    "G": ("G  3_10-helix",   "helix",  "darkorange", "--"),
    "I": ("I  π-helix",      "helix",  "saddlebrown", ":"),
    "P": ("P  PPII",         "helix",  "magenta",    "-."),
    "E": ("E  β-strand",     "strand", "tab:blue",   "-"),
    "B": ("B  β-bridge",     "strand", "tab:cyan",   "--"),
    "T": ("T  turn",         "coil",   "tab:green",  "-"),
    "S": ("S  bend",         "coil",   "olive",      "--"),
    "-": ("-  coil/loop",    "coil",   "tab:gray",   ":"),
}


def load(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    series = {c: [float(r[f"f1_8_{c}"]) for r in rows] for c in DSSP_8}
    macro8 = [float(r["macro_f1_8"]) for r in rows]
    n_tgt = rows[-1]["n_targets"] if rows else "?"
    return steps, series, macro8, n_tgt


def plot(series_name, category, csv_path, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib unavailable")

    steps, series, macro8, n_tgt = load(csv_path)
    families = ["helix", "strand", "coil"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharex=True, sharey=True)

    for ax, fam in zip(axes, families):
        codes = [c for c in DSSP_8 if META[c][1] == fam]
        for c in codes:
            label, _, colour, ls = META[c]
            ax.plot(steps, series[c], ls, color=colour, lw=2.2,
                    marker="o", ms=4, label=label)
        # macro-F1 (8-state) as a shared dashed reference on every panel
        ax.plot(steps, macro8, color="k", lw=1.2, ls=(0, (1, 1)),
                alpha=0.5, label="macro-F1 (8)")
        ax.set_title(f"{fam}-like states")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("per-class F1")
    axes[0].set_ylim(-0.02, 1.02)  # sharey=True → applies to all panels

    fig.suptitle(
        f"{series_name} — FoldBench-small {category} "
        f"(n={n_tgt})   DSSP 8-state per-class F1 vs step",
        fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[ss_f1_8state] plot -> {out_png}")

    # one-line endpoint summary, sorted worst-first
    last = {c: series[c][-1] for c in DSSP_8}
    print(f"[ss_f1_8state] step {steps[-1]} F1 by class:")
    for c in sorted(DSSP_8, key=lambda x: last[x]):
        print(f"    {META[c][0]:<14} {last[c]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--csv", default=None, help="override CSV path")
    args = ap.parse_args()
    ss_dir = BENCH / "runs" / args.series / "ss_analysis"
    csv_path = Path(args.csv) if args.csv else ss_dir / f"ss_f1_{args.category}.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run ss_f1.py first")
    out_png = ss_dir / f"ss_f1_8state_{args.category}.png"
    plot(args.series, args.category, csv_path, out_png)


if __name__ == "__main__":
    main()
