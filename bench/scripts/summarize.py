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


def quantile(xs, q):
    """Linear-interpolated quantile (matches numpy's default), pure-python so
    the non-plot CSV path keeps working without numpy."""
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(xs):
        return xs[lo] + frac * (xs[lo + 1] - xs[lo])
    return xs[lo]


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
        metrics = {"lddt": "lddt", "tm": "tm_score", "gdt": "oligo_gdtts"}
        # per-target mean of each metric over samples
        per_t = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(raw)):
            for mk, col in metrics.items():
                v = fnum(r.get(col))
                if v is not None:
                    per_t[r["pdb_id"]][mk].append(v)
        # per-target mean over samples, then group by DSSP fold class
        tgt_mean: dict[str, dict[str, float]] = {}
        for pdb, md in per_t.items():
            tgt_mean[pdb] = {mk: sum(md[mk]) / len(md[mk]) for mk in metrics if md[mk]}
        by_cls = defaultdict(lambda: defaultdict(list))
        for pdb, mm in tgt_mean.items():
            cls = fold.get(pdb, "unknown")
            for mk, v in mm.items():
                by_cls[cls][mk].append(v)
        rec = {"step": step}
        # overall distribution across targets: mean + median + IQR (q25, q75)
        for mk in metrics:
            allvals = [mm[mk] for mm in tgt_mean.values() if mk in mm]
            rec[f"{mk}_n"] = len(allvals)
            rec[f"{mk}_mean"] = round(sum(allvals) / len(allvals), 4) if allvals else None
            rec[f"{mk}_med"] = round(quantile(allvals, 0.5), 4) if allvals else None
            rec[f"{mk}_q25"] = round(quantile(allvals, 0.25), 4) if allvals else None
            rec[f"{mk}_q75"] = round(quantile(allvals, 0.75), 4) if allvals else None
        for c in classes:
            rec[f"{c}_n"] = len(by_cls[c].get("lddt", []))
            for mk in metrics:
                xs = by_cls[c].get(mk, [])
                rec[f"{c}_{mk}"] = round(sum(xs) / len(xs), 4) if xs else None
        rows.append(rec)
    return classes, rows


def interface_dockq_dist(series: str, category: str) -> list[dict]:
    """Per step: across-interface distribution (median + IQR) of the
    mean-over-samples DockQ. Mirrors the monomer median/IQR band."""
    out = []
    eval_root = BENCH / "runs" / series / "eval" / category
    for sd in sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1))):
        raw = sd / "raw" / f"{category}_ost.csv"
        if not raw.exists():
            continue
        per_if = defaultdict(list)
        for r in csv.DictReader(open(raw)):
            v = fnum(r.get("dockq"))
            if v is not None:
                per_if[(r["pdb_id"], r.get("ic1"), r.get("ic2"))].append(v)
        means = [sum(v) / len(v) for v in per_if.values() if v]
        if not means:
            continue
        out.append({"step": int(_STEP.search(sd.name).group(1)),
                    "med": quantile(means, 0.5),
                    "q25": quantile(means, 0.25), "q75": quantile(means, 0.75)})
    return out


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
    png = out_root / f"trajectory_{category}.png"
    colors = {"alpha": "tab:red", "beta": "tab:blue", "alpha_beta": "tab:green", "coil": "tab:gray"}

    if is_interface(category):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(steps, [fnum(s.get("success_rate_oracle")) for s in summaries], "-o",
                color="tab:purple", lw=2.5, label="DockQ success% (oracle@0.23)")
        ax2 = ax.twinx()
        ax2.plot(steps, [fnum(s.get("mean_dockq")) for s in summaries], "-s",
                 color="tab:orange", alpha=0.7, label="mean DockQ")
        dist = interface_dockq_dist(series, category)
        if dist:
            ds = [d["step"] for d in dist]
            ax2.fill_between(ds, [d["q25"] for d in dist], [d["q75"] for d in dist],
                             color="tab:orange", alpha=0.15, lw=0, label="DockQ IQR (q25–q75)")
            ax2.plot(ds, [d["med"] for d in dist], ":", color="tab:orange", lw=1.6,
                     label="median DockQ")
        ax.set_xlabel("training step"); ax.set_ylabel("success rate (%)")
        ax2.set_ylabel("mean DockQ")
        ax.legend(loc="upper left"); ax2.legend(loc="lower right")
        ax.set_title(f"{series} — FoldBench-small {category}"); ax.grid(alpha=0.3)
    else:
        # monomer: one stacked panel per metric (LDDT / TM-score / GDT-TS).
        # Each panel folds in the tmp/bench-style filled band — here the IQR
        # (q25–q75 across targets) — plus the median + mean trajectory and the
        # per-DSSP fold-class breakdown.
        classes, rows = fold_breakdown(series, category)
        panels = [("lddt", "LDDT (no-stereocheck)"),
                  ("tm", "TM-score"),
                  ("gdt", "GDT-TS")]
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 13), sharex=True)
        for ax, (mk, label) in zip(axes, panels):
            pts = [(r["step"], r.get(f"{mk}_q25"), r.get(f"{mk}_q75"),
                    r.get(f"{mk}_med"), r.get(f"{mk}_mean")) for r in rows]
            pts = [p for p in pts if None not in p]
            if pts:
                xs, lo, hi, med, mn = (list(z) for z in zip(*pts))
                ax.fill_between(xs, lo, hi, color="k", alpha=0.12, lw=0, label="IQR (q25–q75)")
                ax.plot(xs, med, "-", color="k", lw=2.5, label="median", zorder=6)
                ax.plot(xs, mn, "--", color="0.35", lw=1.6, label="mean", zorder=5)
            for c in classes:
                ys = [(r["step"], r.get(f"{c}_{mk}")) for r in rows]
                ys = [p for p in ys if p[1] is not None]
                if ys:
                    nlast = rows[-1].get(f"{c}_n", 0) if rows else 0
                    ax.plot([p[0] for p in ys], [p[1] for p in ys], "-",
                            color=colors[c], alpha=0.7, lw=1.2, label=f"{c} (n={nlast})")
            ax.set_ylabel(label); ax.set_title(label); ax.grid(alpha=0.3)
        axes[-1].set_xlabel("training step")
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{series} — FoldBench-small {category}  (median + IQR across targets)")
        if rows:
            bf = (["step"]
                  + [f"{mk}_{k}" for mk in ("lddt", "tm", "gdt")
                     for k in ("n", "mean", "med", "q25", "q75")]
                  + [f"{c}_{k}" for c in classes for k in ("n", "lddt", "tm", "gdt")])
            with open(out_root / f"byclass_{category}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=bf, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
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
