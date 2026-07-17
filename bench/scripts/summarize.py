"""Aggregate a df3 checkpoint series into a FoldBench-small trajectory.

Stacks every ``bench/runs/{series}/eval/step_*/summary.csv`` (written by
eval_ost.py) into a per-category trajectory CSV + plot:

  - monomer_*           : mean / oracle LDDT vs step, plus a per-DSSP-fold-class
                          LDDT breakdown (the alpha-vs-beta gap across training)
  - interface_protein_* : 3 stacked panels — DockQ CAPRI quality tiers (bars) +
                          whole-model lDDT + whole-model TM-score (median + IQR)
                          vs step (GDT-TS is logged to CSV but not plotted)

Each step's summary carries three aggregates: mean (over samples), oracle
(best-of-N), and conf (confidence top-1 = the highest-ranking_score sample).
The confidence top-1 line is drawn whenever the raw CSV carries ranking_score
(runs that train a confidence head, df3v2+); it is simply absent for older,
confidence-head-free evals.

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
        # per-target metric values keyed by (seed, sample), plus ranking_score
        per_ts = defaultdict(lambda: defaultdict(dict))
        rank_ts: dict = defaultdict(dict)
        for r in csv.DictReader(open(raw)):
            sk = (r.get("seed"), r.get("sample"))
            for mk, col in metrics.items():
                v = fnum(r.get(col))
                if v is not None:
                    per_ts[r["pdb_id"]][mk][sk] = v
            rs = fnum(r.get("ranking_score"))
            if rs is not None:
                rank_ts[r["pdb_id"]][sk] = rs
        # per-target mean over samples, then group by DSSP fold class
        tgt_mean: dict[str, dict[str, float]] = {}
        # per-target confidence top-1 (ranking_score-selected sample); empty when
        # the raw CSV predates the ranking_score column (pre-confidence-head evals)
        tgt_conf: dict[str, dict[str, float]] = {}
        for pdb, md in per_ts.items():
            tgt_mean[pdb] = {mk: sum(md[mk].values()) / len(md[mk]) for mk in metrics if md[mk]}
            ranks = rank_ts.get(pdb, {})
            if ranks:
                best_sk = max(ranks, key=ranks.get)
                cvals = {mk: md[mk][best_sk] for mk in metrics if best_sk in md.get(mk, {})}
                if cvals:
                    tgt_conf[pdb] = cvals
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
            cvals = [mm[mk] for mm in tgt_conf.values() if mk in mm]
            rec[f"{mk}_conf"] = round(quantile(cvals, 0.5), 4) if cvals else None
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
    "interface_protein_protein": 0.7293,  # n=279
    "interface_antibody_antigen": 0.4790,  # n=172
    "interface_protein_dna": 0.7918,  # n=330
    "interface_protein_rna": 0.6232,  # n=70
    "interface_protein_peptide": 0.8298,  # n=51
    "interface_protein_ligand": 0.6490,  # n=558
}

# AF3 monomer reference from the same table (full set), keyed by the metric
# keys used in the monomer panels below (lddt / tm-score / gdt-ts).
AF3_MONOMER = {
    "monomer_protein": {"lddt": 0.88, "tm": 0.94, "gdt": 0.92},  # n=334
    "monomer_dna": {"lddt": 0.53, "tm": 0.31, "gdt": 0.58},  # n=14
    "monomer_rna": {"lddt": 0.61, "tm": 0.41, "gdt": 0.53},  # n=15
}

# AF3 whole-complex lDDT per interface category (same table, full set) — the
# reference for the interface panel's lower (whole-model lDDT) plot. Ligand is
# omitted (it reports LDDT-LP / LDDT-PLI, not whole-model lDDT, and isn't swept).
AF3_INTERFACE_LDDT = {
    "interface_protein_protein": 0.84,
    "interface_antibody_antigen": 0.88,
    "interface_protein_dna": 0.85,
    "interface_protein_rna": 0.78,
    "interface_protein_peptide": 0.84,
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


def interface_metric_dist(series: str, category: str, col: str = "lddt", select: str = "mean") -> list[dict]:
    """Per step: across-target distribution (mean/median/IQR) of a whole-model
    metric (``col`` = ``lddt`` or ``tm_score``). Dedup the repeated interface
    rows to per-(seed, sample) values, then reduce over the target's samples:

    - ``select='mean'`` : mean over samples (the confidence-free floor).
    - ``select='conf'`` : the confidence top-1 sample's value (highest
      ``ranking_score``). Yields nothing when the raw CSV predates the
      ranking_score column (pre-confidence-head evals), so the conf line is
      simply absent for those steps.
    """
    out = []
    eval_root = BENCH / "runs" / series / "eval" / category
    for sd in sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1))):
        raw = sd / "raw" / f"{category}_ost.csv"
        if not raw.exists():
            continue
        by_t: dict = defaultdict(dict)
        by_rank: dict = defaultdict(dict)
        for r in csv.DictReader(open(raw)):
            sk = (r.get("seed"), r.get("sample"))
            v = fnum(r.get(col))
            if v is not None:
                by_t[r["pdb_id"]][sk] = v
            rs = fnum(r.get("ranking_score"))
            if rs is not None:
                by_rank[r["pdb_id"]][sk] = rs
        per_t = []
        for pdb, d in by_t.items():
            if not d:
                continue
            if select == "conf":
                ranks = by_rank.get(pdb, {})
                cand = [(ranks[sk], sk) for sk in d if sk in ranks]
                if not cand:
                    continue
                per_t.append(d[max(cand)[1]])
            else:
                per_t.append(sum(d.values()) / len(d))
        if not per_t:
            continue
        out.append(
            {
                "step": int(_STEP.search(sd.name).group(1)),
                "mean": sum(per_t) / len(per_t),
                "med": quantile(per_t, 0.5),
                "q25": quantile(per_t, 0.25),
                "q75": quantile(per_t, 0.75),
            }
        )
    return out


def summarize(series: str, category: str, plot: bool) -> None:
    summaries = load_step_summaries_for(series, category)
    if not summaries:
        print(f"[summary] no per-step summaries yet for {series}/{category}")
        return
    out_root = BENCH / "runs" / series
    traj_csv = out_root / f"trajectory_{category}.csv"
    # Union of keys across steps (first-seen order): schemas can differ when the
    # grid mixes evals from before/after a metric was added (e.g. mean_tm_score).
    fields = list(dict.fromkeys(k for s in summaries for k in s))
    with open(traj_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)
    print(f"[summary] {series}/{category}: {len(summaries)} steps -> {traj_csv}")

    if is_interface(category):
        print(
            f"{'step':>8} {'n':>4} {'succ_orac%':>10} {'succ_conf%':>10} {'oracle_dq':>10} "
            f"{'conf_dq':>8} {'mean_dq':>8} {'mean_lddt':>10} {'mean_tm':>8}"
        )
        for s in summaries:
            print(
                f"{s['step']:>8} {s.get('n_interfaces', ''):>4} "
                f"{str(s.get('success_rate_oracle')):>10} {str(s.get('success_rate_conf')):>10} "
                f"{str(s.get('oracle_dockq')):>10} {str(s.get('conf_dockq')):>8} "
                f"{str(s.get('mean_dockq')):>8} {str(s.get('mean_lddt')):>10} "
                f"{str(s.get('mean_tm_score')):>8}"
            )
    else:
        print(f"{'step':>8} {'n':>4} {'mean_lddt':>10} {'conf_lddt':>10} {'oracle_lddt':>11} {'mean_tm':>8}")
        for s in summaries:
            print(
                f"{s['step']:>8} {s.get('n_scored', ''):>4} {str(s.get('mean_lddt')):>10} "
                f"{str(s.get('conf_lddt')):>10} {str(s.get('oracle_lddt')):>11} {str(s.get('mean_tm_score')):>8}"
            )

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
        # Three stacked panels sharing a real-step x-axis (numeric, gaps to scale):
        # (1) DockQ CAPRI quality tiers, (2) whole-model lDDT + inter-chain ilddt
        # overlaid, (3) whole-model TM-score — each median + IQR across targets, on
        # the [0, 1] range. ilddt (collapse-sensitive) is only present from the step
        # it was added to eval_ost (≥10k), so its overlay line starts there.
        tiers = dockq_tier_fractions(series, category, mode="oracle")
        ld = interface_metric_dist(series, category, "lddt")
        tm = interface_metric_dist(series, category, "tm_score")
        il = interface_metric_dist(series, category, "ilddt")
        ld_conf = interface_metric_dist(series, category, "lddt", select="conf")
        tm_conf = interface_metric_dist(series, category, "tm_score", select="conf")
        fig, (axd, axl, axt) = plt.subplots(3, 1, figsize=(10, 13), sharex=True)

        # Steps where the model hasn't folded get their success-rate column
        # hatched over: early DockQ "successes" there are collapsed-blob artefacts
        # (inflated fnat, no fnonnat penalty), so the bar height is not
        # trustworthy. See memory/early_interface_dockq_blob_artifact. The "folded"
        # bar is half of AF3's own whole-complex lDDT for this category (harder
        # interfaces like RNA get a lower bar than easier ones like Ab-Ag), so it
        # scales with achievable difficulty rather than a flat 0.5.
        af3ld_ref = AF3_INTERFACE_LDDT.get(category)
        LDDT_FOLD_THRESH = 0.5 * af3ld_ref if af3ld_ref is not None else 0.5
        ld_by_step = {d["step"]: d["med"] for d in ld}

        # Bar width in step units = 72% of the smallest gap between benchmarked
        # steps, so bars never overlap even when the grid is irregular (1k/2k/5k).
        all_steps = sorted({t["step"] for t in tiers} | {d["step"] for d in ld} | {d["step"] for d in tm})
        gaps = [b - a for a, b in zip(all_steps, all_steps[1:], strict=False)]
        bar_w = 0.72 * min(gaps) if gaps else 800.0

        if tiers:
            steps = [t["step"] for t in tiers]
            bottom = [0.0] * len(steps)
            for name, lbl, _lo, _hi, col in DOCKQ_TIERS:
                vals = [t[name] for t in tiers]
                axd.bar(
                    steps,
                    vals,
                    bottom=bottom,
                    width=bar_w,
                    color=col,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"{name} ({lbl})",
                )
                bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
            unfolded = [
                t["step"]
                for t in tiers
                if (ld_by_step.get(t["step"]) is not None and ld_by_step[t["step"]] < LDDT_FOLD_THRESH)
            ]
            for j, s in enumerate(unfolded):
                axd.axvspan(
                    s - bar_w / 2,
                    s + bar_w / 2,
                    facecolor="none",
                    edgecolor="0.4",
                    hatch="////",
                    linewidth=0.0,
                    zorder=9,
                    label=f"lDDT < ½·AF3 ({LDDT_FOLD_THRESH:.2f}) — unreliable" if j == 0 else None,
                )
            af3 = AF3_SUCCESS_RATE.get(category)
            if af3 is not None:
                # bar height (acceptable+medium+high) == success rate @0.23, so
                # this line is directly comparable to the top of each stack.
                axd.axhline(af3, color="#d7301f", ls="--", lw=1.8, zorder=8, label=f"AF3 success @0.23 ({af3:.0%})")
            # Confidence top-1 success @0.23 (ranking_score-selected sample) — the
            # deployed number; its gap below the oracle stack top measures how good
            # the confidence head's ranking is. Present only once a run trains the
            # confidence head (success_rate_conf), so the line starts there.
            conf_succ = [(int(s["step"]), fnum(s.get("success_rate_conf"))) for s in summaries]
            conf_succ = [(st, v / 100.0) for st, v in conf_succ if v is not None]
            if conf_succ:
                axd.plot(
                    [p[0] for p in conf_succ],
                    [p[1] for p in conf_succ],
                    "-s",
                    color="#2166ac",
                    lw=2,
                    ms=5,
                    zorder=11,
                    label="confidence top-1 success @0.23",
                )
            axd.set_ylim(0, 1.0)
            axd.set_ylabel("fraction of interfaces")
            axd.set_title(f"DockQ CAPRI quality tiers — oracle (best-of-N per interface)  n={tiers[-1]['n']}")
            # Secondary axis: oracle QS-score (collapse-sensitive interface quality)
            # over the DockQ tiers — makes the decoupling visible (blob-inflated DockQ
            # success vs real interface quality). QS is present only from the step it
            # was added to eval_ost (≥10k), so the line starts there.
            axq = axd.twinx()
            qs = [(int(s["step"]), fnum(s.get("oracle_qs_score"))) for s in summaries]
            qs = [(st, v) for st, v in qs if v is not None]
            qs_handle = None
            if qs:
                (qs_handle,) = axq.plot(
                    [p[0] for p in qs],
                    [p[1] for p in qs],
                    "-D",
                    color="black",
                    lw=2,
                    ms=4,
                    zorder=10,
                    label="oracle QS-score",
                )
                axq.set_ylim(0, 1.0)  # match the left axis so QS reads on the same [0,1] scale
                axq.set_ylabel("oracle QS-score", color="black")
                axq.tick_params(axis="y", labelcolor="black")
            else:
                axq.set_yticks([])
            # Legend moved right of the QS axis so the two don't overlap.
            handles, labels = axd.get_legend_handles_labels()
            if qs_handle is not None:
                handles.append(qs_handle)
                labels.append("oracle QS-score")
            axd.legend(handles, labels, loc="center left", bbox_to_anchor=(1.14, 0.5), fontsize=8, title="tier / QS")
            axd.grid(axis="y", alpha=0.3)

        def _dist_panel(ax, dist, ylabel, title, af3ref=None, overlay=None, conf=None):
            """median + mean + IQR band of a whole-model metric vs real step.

            ``overlay=(dist, label, color)`` draws a second metric (e.g. inter-chain
            ilddt) in colour on the same axes — median line + IQR band.
            ``conf`` is the confidence-top-1 distribution (same shape as ``dist``);
            its across-target median is drawn as a distinct line (deployed number).
            """
            if dist:
                sx = [d["step"] for d in dist]
                ax.fill_between(
                    sx,
                    [d["q25"] for d in dist],
                    [d["q75"] for d in dist],
                    color="k",
                    alpha=0.12,
                    lw=0,
                    label="IQR (q25–q75)",
                )
                ax.plot(sx, [d["med"] for d in dist], "-o", color="k", lw=2.2, ms=4, label="median", zorder=6)
                ax.plot(sx, [d["mean"] for d in dist], "--o", color="0.35", lw=1.5, ms=3, label="mean", zorder=5)
                if af3ref is not None:
                    ax.axhline(af3ref, color="#d7301f", ls="--", lw=1.8, zorder=7, label=f"AF3 ({af3ref:.2f})")
            if conf:
                cx = [d["step"] for d in conf]
                ax.plot(
                    cx,
                    [d["med"] for d in conf],
                    "-s",
                    color="#2166ac",
                    lw=2,
                    ms=4,
                    zorder=8,
                    label="confidence top-1 (median)",
                )
            if overlay is not None and overlay[0]:
                odist, olabel, ocol = overlay
                ox = [d["step"] for d in odist]
                ax.fill_between(ox, [d["q25"] for d in odist], [d["q75"] for d in odist], color=ocol, alpha=0.15, lw=0)
                ax.plot(
                    ox, [d["med"] for d in odist], "-D", color=ocol, lw=2, ms=4, label=f"{olabel} (median)", zorder=6
                )
            if dist or (overlay is not None and overlay[0]):
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_title(title)

        _dist_panel(
            axl,
            ld,
            "lDDT",
            "whole-model lDDT (black) + inter-chain ilddt (red, collapse-sensitive, ≥10k) — median + IQR",
            AF3_INTERFACE_LDDT.get(category),
            overlay=(il, "inter-chain ilddt", "tab:red"),
            conf=ld_conf,
        )
        _dist_panel(
            axt,
            tm,
            "TM-score (whole model)",
            "whole-model TM-score (median + IQR across targets)",
            conf=tm_conf,
        )
        if all_steps:
            # pad by a bar so edge bars aren't clipped. sharex applies to all panels.
            axt.set_xlim(all_steps[0] - bar_w, all_steps[-1] + bar_w)
            # tick each benchmarked step only when there are few; on a dense grid
            # let matplotlib pick round-number ticks so labels don't collide.
            if len(all_steps) <= 12:
                axt.set_xticks(all_steps)
        axt.set_xlabel("training step")
        fig.suptitle(f"{series} — FoldBench-small {category}")
    else:
        # monomer: one stacked panel per metric (LDDT / TM-score / GDT-TS).
        # Each panel folds in the tmp/bench-style filled band — here the IQR
        # (q25–q75 across targets) — plus the median + mean trajectory and the
        # per-DSSP fold-class breakdown.
        classes, rows = fold_breakdown(series, category)
        panels = [("lddt", "LDDT (no-stereocheck)"), ("tm", "TM-score"), ("gdt", "GDT-TS")]
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 13), sharex=True)
        for ax, (mk, label) in zip(axes, panels, strict=True):
            pts = [
                (r["step"], r.get(f"{mk}_q25"), r.get(f"{mk}_q75"), r.get(f"{mk}_med"), r.get(f"{mk}_mean"))
                for r in rows
            ]
            pts = [p for p in pts if None not in p]
            if pts:
                xs, lo, hi, med, mn = (list(z) for z in zip(*pts, strict=True))
                ax.fill_between(xs, lo, hi, color="k", alpha=0.12, lw=0, label="IQR (q25–q75)")
                # markers on every line so a single benchmarked step still shows
                # up as a point (a bare line needs >=2 points to render).
                ax.plot(xs, med, "-o", color="k", lw=2.5, ms=4, label="median", zorder=6)
                ax.plot(xs, mn, "--o", color="0.35", lw=1.6, ms=3, label="mean", zorder=5)
            # confidence top-1 (ranking_score-selected sample) across-target median —
            # the deployed number; absent for pre-confidence-head steps.
            cpts = [(r["step"], r.get(f"{mk}_conf")) for r in rows]
            cpts = [p for p in cpts if p[1] is not None]
            if cpts:
                ax.plot(
                    [p[0] for p in cpts],
                    [p[1] for p in cpts],
                    "-s",
                    color="#2166ac",
                    lw=2,
                    ms=4,
                    label="confidence top-1 (median)",
                    zorder=7,
                )
            for c in classes:
                ys = [(r["step"], r.get(f"{c}_{mk}")) for r in rows]
                ys = [p for p in ys if p[1] is not None]
                if ys:
                    nlast = rows[-1].get(f"{c}_n", 0) if rows else 0
                    ax.plot(
                        [p[0] for p in ys],
                        [p[1] for p in ys],
                        "-o",
                        color=colors[c],
                        alpha=0.7,
                        lw=1.2,
                        ms=3,
                        label=f"{c} (n={nlast})",
                    )
            af3v = AF3_MONOMER.get(category, {}).get(mk)
            if af3v is not None:
                # AF3 reference (FoldBench full set), labelled at the right edge
                # since each panel's value differs and only axes[0] shows a legend.
                ax.axhline(af3v, color="#d7301f", ls="--", lw=1.8, zorder=7)
                ax.text(
                    0.995,
                    af3v,
                    f"AF3 {af3v:.2f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#d7301f",
                    clip_on=False,
                )
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("training step")
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{series} — FoldBench-small {category}  (median + IQR across targets)")
        if rows:
            bf = (
                ["step"]
                + [f"{mk}_{k}" for mk in ("lddt", "tm", "gdt") for k in ("n", "mean", "med", "q25", "q75", "conf")]
                + [f"{c}_{k}" for c in classes for k in ("n", "lddt", "tm", "gdt")]
            )
            with open(out_root / f"byclass_{category}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=bf, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    fig.tight_layout()
    if is_interface(category):
        # Centre the suptitle over the plot axes, not the full canvas: the interface
        # panels' legends sit outside on the right, so bbox_inches="tight" widens the
        # saved image on that side and a figure-centred suptitle reads as shifted right.
        c = axd.get_position()
        fig._suptitle.set_x(0.5 * (c.x0 + c.x1))
    fig.savefig(png, dpi=130, bbox_inches="tight")
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
