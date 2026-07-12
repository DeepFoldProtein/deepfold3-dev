"""Aggregate a df3 checkpoint series into a FoldBench-small trajectory.

Stacks every ``bench/runs/{series}/eval/step_*/summary.csv`` (written by
eval_ost.py) into a per-category trajectory CSV + plot:

  - monomer_*           : mean / oracle LDDT vs step, plus a per-DSSP-fold-class
                          LDDT breakdown (the alpha-vs-beta gap across training)
  - interface_protein_* : DockQ CAPRI quality tiers (stacked bars) + whole-model
                          lDDT (median + IQR) vs step

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


# DockQ CAPRI quality tiers: (name, label, lo, hi, colour). Half-open [lo, hi).
# ``incorrect`` (<0.23) is intentionally omitted from the stack, so the bar
# height reads directly as the DockQ success rate (@0.23) and the gap above is
# the incorrect fraction.
DOCKQ_TIERS = [
    ("acceptable", "0.23–0.49", 0.23, 0.49, "#fdae61"),
    ("medium", "0.49–0.80", 0.49, 0.80, "#a6d96a"),
    ("high", "≥0.80", 0.80, float("inf"), "#1a9850"),
]

# AF3 interface success rate from the FoldBench paper Supplementary Table 3
# (AF3 column, full set — targets released after 2023-01). Interfaces use
# DockQ >= 0.23; protein_ligand uses LRMSD < 2 A and LDDT-PLI > 0.8 (and isn't
# swept here by default). Drawn as a reference line on the success-rate panel.
# Note: the paper uses the *full* FoldBench set, not our num_tokens-capped small
# subset, so it's a target marker, not a like-for-like comparison. (n = paper
# target count per category.)
AF3_SUCCESS_RATE = {
    "interface_protein_protein": 0.7293,   # n=279
    "interface_antibody_antigen": 0.4790,  # n=172
    "interface_protein_dna": 0.7918,       # n=330
    "interface_protein_rna": 0.6232,       # n=70
    "interface_protein_peptide": 0.8298,   # n=51
    "interface_protein_ligand": 0.6490,    # n=558
}

# AF3 monomer reference from the same table (full set), keyed by the metric
# keys used in the monomer panels below (lddt / tm-score / gdt-ts).
AF3_MONOMER = {
    "monomer_protein": {"lddt": 0.88, "tm": 0.94, "gdt": 0.92},  # n=334
    "monomer_dna": {"lddt": 0.53, "tm": 0.31, "gdt": 0.58},      # n=14
    "monomer_rna": {"lddt": 0.61, "tm": 0.41, "gdt": 0.53},      # n=15
}


def dockq_tier_fractions(series: str, category: str, mode: str = "oracle") -> list[dict]:
    """Per step: fraction of interfaces in each DockQ CAPRI tier.

    Per-interface DockQ is the best-of-N (``mode='oracle'``) or sample-mean
    (``mode='mean'``) over that interface's seeds x samples. ``oracle`` matches
    the ``success_rate_oracle`` headline (incorrect fraction = 1 - success@0.23).
    """
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
        vals = [max(lst) if mode == "oracle" else sum(lst) / len(lst) for lst in per_if.values() if lst]
        if not vals:
            continue
        n = len(vals)
        rec = {"step": int(_STEP.search(sd.name).group(1)), "n": n}
        for name, _lbl, lo, hi, _c in DOCKQ_TIERS:
            rec[name] = sum(1 for d in vals if lo <= d < hi) / n
        out.append(rec)
    return out


def interface_lddt_dist(series: str, category: str) -> list[dict]:
    """Per step: across-target distribution (mean/median/IQR) of whole-model
    lDDT (dedup the repeated interface rows, then mean over the target's samples)."""
    out = []
    eval_root = BENCH / "runs" / series / "eval" / category
    for sd in sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1))):
        raw = sd / "raw" / f"{category}_ost.csv"
        if not raw.exists():
            continue
        by_t: dict = defaultdict(dict)
        for r in csv.DictReader(open(raw)):
            v = fnum(r.get("lddt"))
            if v is not None:
                by_t[r["pdb_id"]][(r.get("seed"), r.get("sample"))] = v
        per_t = [sum(d.values()) / len(d) for d in by_t.values() if d]
        if not per_t:
            continue
        out.append({"step": int(_STEP.search(sd.name).group(1)),
                    "mean": sum(per_t) / len(per_t), "med": quantile(per_t, 0.5),
                    "q25": quantile(per_t, 0.25), "q75": quantile(per_t, 0.75)})
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
    png = out_root / f"trajectory_{category}.png"
    colors = {"alpha": "tab:red", "beta": "tab:blue", "alpha_beta": "tab:green", "coil": "tab:gray"}

    if is_interface(category):
        # Two panels: (1) DockQ CAPRI quality tiers stacked per benchmarked step,
        # (2) whole-model lDDT median + IQR. Categorical x (one bar/point per
        # step) keeps the panels aligned; both use the full [0, 1] y-range.
        tiers = dockq_tier_fractions(series, category, mode="oracle")
        ld = interface_lddt_dist(series, category)
        fig, (axd, axl) = plt.subplots(2, 1, figsize=(10, 9))

        # Steps where the model hasn't folded (whole-model median lDDT < 0.5) get
        # their success-rate column hatched over: early DockQ "successes" there
        # are collapsed-blob artefacts (inflated fnat, no fnonnat penalty), so the
        # bar height is not trustworthy. See memory/early_interface_dockq_blob_artifact.
        LDDT_FOLD_THRESH = 0.5
        ld_by_step = {d["step"]: d["med"] for d in ld}

        if tiers:
            xs = list(range(len(tiers)))
            bottom = [0.0] * len(xs)
            for name, lbl, _lo, _hi, col in DOCKQ_TIERS:
                vals = [t[name] for t in tiers]
                axd.bar(xs, vals, bottom=bottom, width=0.72, color=col,
                        edgecolor="white", linewidth=0.5, label=f"{name} ({lbl})")
                bottom = [b + v for b, v in zip(bottom, vals)]
            unfolded = [i for i, t in enumerate(tiers)
                        if (ld_by_step.get(t["step"]) is not None
                            and ld_by_step[t["step"]] < LDDT_FOLD_THRESH)]
            for j, i in enumerate(unfolded):
                axd.axvspan(i - 0.42, i + 0.42, facecolor="none", edgecolor="0.4",
                            hatch="////", linewidth=0.0, zorder=9,
                            label=f"lDDT < {LDDT_FOLD_THRESH:g} (unfolded — unreliable)"
                            if j == 0 else None)
            af3 = AF3_SUCCESS_RATE.get(category)
            if af3 is not None:
                # bar height (acceptable+medium+high) == success rate @0.23, so
                # this line is directly comparable to the top of each stack.
                axd.axhline(af3, color="#d7301f", ls="--", lw=1.8, zorder=8,
                            label=f"AF3 success @0.23 ({af3:.0%})")
            axd.set_xticks(xs); axd.set_xticklabels([str(t["step"]) for t in tiers])
            axd.set_ylim(0, 1.0); axd.set_ylabel("fraction of interfaces")
            axd.set_title(f"DockQ CAPRI quality tiers (best-of-N per interface)  n={tiers[-1]['n']}")
            axd.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="tier")
            axd.grid(axis="y", alpha=0.3)

        if ld:
            xs2 = list(range(len(ld)))
            axl.fill_between(xs2, [d["q25"] for d in ld], [d["q75"] for d in ld],
                             color="k", alpha=0.12, lw=0, label="IQR (q25–q75)")
            axl.plot(xs2, [d["med"] for d in ld], "-o", color="k", lw=2.2, ms=4, label="median", zorder=6)
            axl.plot(xs2, [d["mean"] for d in ld], "--o", color="0.35", lw=1.5, ms=3, label="mean", zorder=5)
            axl.set_xticks(xs2); axl.set_xticklabels([str(d["step"]) for d in ld])
            axl.legend(loc="upper left", fontsize=8)
        axl.set_ylim(-0.02, 1.02); axl.set_ylabel("lDDT (whole model)")
        axl.set_xlabel("training step"); axl.grid(alpha=0.3)
        axl.set_title("whole-model lDDT (median + IQR across targets)")
        fig.suptitle(f"{series} — FoldBench-small {category}")
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
                # markers on every line so a single benchmarked step still shows
                # up as a point (a bare line needs >=2 points to render).
                ax.plot(xs, med, "-o", color="k", lw=2.5, ms=4, label="median", zorder=6)
                ax.plot(xs, mn, "--o", color="0.35", lw=1.6, ms=3, label="mean", zorder=5)
            for c in classes:
                ys = [(r["step"], r.get(f"{c}_{mk}")) for r in rows]
                ys = [p for p in ys if p[1] is not None]
                if ys:
                    nlast = rows[-1].get(f"{c}_n", 0) if rows else 0
                    ax.plot([p[0] for p in ys], [p[1] for p in ys], "-o",
                            color=colors[c], alpha=0.7, lw=1.2, ms=3, label=f"{c} (n={nlast})")
            af3v = AF3_MONOMER.get(category, {}).get(mk)
            if af3v is not None:
                # AF3 reference (FoldBench full set), labelled at the right edge
                # since each panel's value differs and only axes[0] shows a legend.
                ax.axhline(af3v, color="#d7301f", ls="--", lw=1.8, zorder=7)
                ax.text(0.995, af3v, f"AF3 {af3v:.2f}", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8, color="#d7301f", clip_on=False)
            ax.set_ylabel(label); ax.set_title(label); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
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
    fig.tight_layout(); fig.savefig(png, dpi=130, bbox_inches="tight")
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
