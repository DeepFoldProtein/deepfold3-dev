"""Aggregate a df3 checkpoint series into a FoldBench-small trajectory.

Stacks every ``bench/runs/{series}/eval/step_*/summary.csv`` (written by
eval_ost.py) into a per-category trajectory CSV + plot:

  - monomer_*           : mean / oracle LDDT vs step, plus a per-DSSP-fold-class
                          LDDT breakdown (the alpha-vs-beta gap across training)
  - interface_protein_* : DockQ success-rate (@0.23) and mean DockQ vs step

df3v1 has no confidence head, so aggregates are confidence-free (mean over
samples + oracle best-of-N), already baked into each step's summary.

Usage::

    python bench/scripts/summarize.py --series df3v1 --category monomer_protein
    python bench/scripts/summarize.py --series df3v1 --category interface_protein_protein
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
_STEP = re.compile(r"step_(\d+)")


def is_interface(category: str) -> bool:
    return category.startswith("interface_")


def load_step_summaries(series: str, category: str) -> list[dict]:
    eval_root = BENCH / "runs" / series / "eval" / category
    out = []
    for sd in sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1))):
        sc = sd / "summary.csv"
        if sc.exists():
            rows = list(csv.DictReader(open(sc)))
            if rows:
                out.append(rows[0])
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- monomer fold-class breakdown (needs raw per-sample + DSSP) ---
def fold_breakdown(series: str, category: str):
    fold = {}
    dpath = BENCH / "inputs" / f"{category}_dssp.csv"
    if dpath.exists():
        fold = {r["pdb_id"]: r["fold_class"] for r in csv.DictReader(open(dpath))}
    classes = ["alpha", "beta", "alpha_beta", "coil"]
    rows = []
    eval_root = BENCH / "runs" / series / "eval" / category
    for sd in sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1))):
        raw = sd / "raw" / f"{category}_ost.csv"
        if not raw.exists():
            continue
        step = int(_STEP.search(sd.name).group(1))
        # mean lddt per target over samples
        per_t = defaultdict(list)
        for r in csv.DictReader(open(raw)):
            v = fnum(r.get("lddt"))
            if v is not None:
                per_t[r["pdb_id"]].append(v)
        by_cls = defaultdict(list)
        for pdb, vals in per_t.items():
            by_cls[fold.get(pdb, "unknown")].append(sum(vals) / len(vals))
        rec = {"step": step}
        for c in classes:
            xs = by_cls.get(c, [])
            rec[f"{c}_n"] = len(xs)
            rec[f"{c}_lddt"] = round(sum(xs) / len(xs), 4) if xs else None
        rows.append(rec)
    return classes, rows


def summarize(series: str, category: str, plot: bool) -> None:
    summaries = load_step_summaries_for(series, category)
    if not summaries:
        print(f"[summary] no per-step summaries yet for {series}/{category}")
        return
    out_root = BENCH / "runs" / series
    traj_csv = out_root / f"trajectory_{category}.csv"
    fields = list(summaries[0].keys())
    with open(traj_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summaries)
    print(f"[summary] {series}/{category}: {len(summaries)} steps -> {traj_csv}")

    if is_interface(category):
        print(f"{'step':>8} {'n':>4} {'succ_oracle%':>12} {'mean_dockq':>11} {'oracle_dockq':>12}")
        for s in summaries:
            print(f"{s['step']:>8} {s.get('n_interfaces',''):>4} "
                  f"{str(s.get('success_rate_oracle')):>12} {str(s.get('mean_dockq')):>11} "
                  f"{str(s.get('oracle_dockq')):>12}")
    else:
        print(f"{'step':>8} {'n':>4} {'mean_lddt':>10} {'oracle_lddt':>11} {'mean_tm':>8}")
        for s in summaries:
            print(f"{s['step']:>8} {s.get('n_scored',''):>4} {str(s.get('mean_lddt')):>10} "
                  f"{str(s.get('oracle_lddt')):>11} {str(s.get('mean_tm_score')):>8}")

    if not plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[summary] matplotlib unavailable; skipping plot")
        return
    steps = [int(s["step"]) for s in summaries]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if is_interface(category):
        ax.plot(steps, [fnum(s.get("success_rate_oracle")) for s in summaries], "-o",
                color="tab:purple", lw=2.5, label="DockQ success% (oracle@0.23)")
        ax2 = ax.twinx()
        ax2.plot(steps, [fnum(s.get("mean_dockq")) for s in summaries], "-s",
                 color="tab:orange", alpha=0.7, label="mean DockQ")
        ax.set_ylabel("success rate (%)"); ax2.set_ylabel("mean DockQ")
        ax.legend(loc="upper left"); ax2.legend(loc="lower right")
    else:
        ax.plot(steps, [fnum(s.get("mean_lddt")) for s in summaries], "-o", color="k", lw=2.5,
                label="overall mean", zorder=5)
        classes, rows = fold_breakdown(series, category)
        colors = {"alpha": "tab:red", "beta": "tab:blue", "alpha_beta": "tab:green", "coil": "tab:gray"}
        rstep = [r["step"] for r in rows]
        for c in classes:
            ys = [r.get(f"{c}_lddt") for r in rows]
            if any(y is not None for y in ys):
                nlast = rows[-1].get(f"{c}_n", 0) if rows else 0
                ax.plot(rstep, ys, "-o", color=colors[c], alpha=0.8, label=f"{c} (n={nlast})")
        ax.set_ylabel("mean LDDT (no-stereocheck)"); ax.legend()
        # write byclass csv
        if rows:
            bf = ["step"] + [f"{c}_{k}" for c in classes for k in ("n", "lddt")]
            with open(out_root / f"byclass_{category}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=bf); w.writeheader(); w.writerows(rows)
    ax.set_xlabel("training step")
    ax.set_title(f"{series} — FoldBench-small {category}")
    ax.grid(alpha=0.3)
    png = out_root / f"trajectory_{category}.png"
    fig.tight_layout(); fig.savefig(png, dpi=130)
    print(f"[summary] plot -> {png}")


def load_step_summaries_for(series: str, category: str) -> list[dict]:
    """Per-step summaries filtered to this category (the eval dir holds one
    category's summary per step, but guard in case of reuse)."""
    out = []
    for s in load_step_summaries(series, category):
        if s.get("category") in (None, "", category):
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    summarize(args.series, args.category, plot=not args.no_plot)


if __name__ == "__main__":
    main()
