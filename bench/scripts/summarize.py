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
#
# TWO caveats, both in the line's label so the panel can't be misread:
#  - the paper uses the *full* FoldBench set, not our num_tokens-capped subset, so
#    it's a target marker, not a like-for-like comparison (n = paper target count);
#  - it is a MEAN-over-samples rate, while the panel's bar stack is the ORACLE
#    (best-of-N). Cross-checked against FoldBench's own per-target rows on our
#    capped subset: antibody-antigen AF3 mean 46.7 / oracle 64.4 (n=90) against the
#    47.9 below — the paper number tracks the mean, and sits ~18 points under the
#    oracle it shares an axis with. Never read "we crossed the line" as parity.
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

# --- measured reference series -------------------------------------------------
# A measured reference series is a full sweep of another model through OUR bench
# pipeline (same num-token-capped manifest), so its overlay is like-for-like in a
# way the paper dicts above are not. When one is given and has an eval we prefer
# its measured value; otherwise we fall back to the paper dict (labelled "paper"
# so the two are never confused). ``--ref-series ""`` disables the overlay.
#
# A series listed here is HANDICAPPED: drawable, but every label it produces
# carries the caveat so the figure cannot be read as parity with a full-strength
# AlphaFold3. This replaces an earlier blanket refusal of `af3_ref`, which kept
# the only like-for-like AF3 sweep we own off every figure — including the iLDDT
# row, where there is no published number to fall back to at all.
#
# `af3_ref` was folded template-free (the bench attached no `templates` field
# until MAX_TEMPLATES landed in 949e5de), which is AlphaFold3's weakest supported
# mode: on identical targets it trails the published AF3 by 3.8 points on
# protein-protein and 9.9 on antibody-antigen. Every current df3 series folds
# WITH templates, so a df3-minus-af3_ref delta is biased toward df3 by that
# unquantified amount. Drop the entry once af3_ref is re-folded with templates.
REF_SERIES_CAVEAT = {"af3_ref": "template-free"}

# Categories where a ref series' fold is not comparable at all, so no line is
# better than a wrong one. af3_ref folded the nucleic monomers on 2026-07-24,
# before the build_inputs water fix (2026-08-12): its inputs padded chains with
# water-derived DN tokens (8d78: 46 residues against 22 after the fix), so its
# monomer_dna/monomer_rna scores are on different sequences than any current
# series. Verified by folded-length comparison that the six protein-containing
# categories are unaffected — identical inputs on both sides.
REF_SERIES_EXCLUDE_CATEGORIES = {"af3_ref": {"monomer_dna", "monomer_rna"}}

# Set by ``resolve_ref_series``; suffixes every measured-AF3 label drawn below.
# Safe as module state: this is a single-shot CLI (one ``summarize()`` per run).
_REF_CAVEAT_TAG = ""

_REF_ROW_CACHE: dict[tuple[str, str], dict | None] = {}


def af3_tag() -> str:
    """``"AF3"``, or ``"AF3 (template-free)"`` when the ref series is handicapped."""
    return f"AF3{_REF_CAVEAT_TAG}"


def resolve_ref_series(ref_series: str, series: str) -> str:
    """``ref_series`` with self-reference stripped; stamps the handicap caveat.

    A caveated series is drawn (not refused) and ``af3_tag()`` starts returning
    the annotated name, so the caveat rides along on every label rather than
    living only in a comment.
    """
    global _REF_CAVEAT_TAG
    _REF_CAVEAT_TAG = ""
    if ref_series == series:  # don't draw a series' reference line against itself
        return ""
    caveat = REF_SERIES_CAVEAT.get(ref_series)
    if caveat:
        _REF_CAVEAT_TAG = f" ({caveat})"
        print(f"[summary] reference series '{ref_series}' is {caveat} — labelled as such, not a parity baseline")
    return ref_series


def _ref_row(ref_series: str, category: str) -> dict | None:
    """Latest-step summary row of a reference series for ``category`` (cached).

    ``af3_ref`` writes a single ``step_0`` summary, but pick the latest step to
    be robust to a series benched at some other step number.
    """
    if not ref_series:
        return None
    key = (ref_series, category)
    if key not in _REF_ROW_CACHE:
        eval_root = BENCH / "runs" / ref_series / "eval" / category
        row = None
        step_dirs = sorted(eval_root.glob("step_*"), key=lambda p: int(_STEP.search(p.name).group(1)))
        for sd in reversed(step_dirs):
            f = sd / "summary.csv"
            if f.exists():
                row = next(csv.DictReader(open(f)), None)
                if row:
                    break
        _REF_ROW_CACHE[key] = row
    return _REF_ROW_CACHE[key]


def ref_count(ref_series: str, category: str) -> int:
    """Targets/interfaces behind the ref series' summary row (0 if unknown)."""
    row = _ref_row(ref_series, category)
    if not row:
        return 0
    for k in ("n_interfaces", "n_scored"):
        try:
            return int(row[k])
        except (KeyError, TypeError, ValueError):
            continue
    return 0


def ref_value(
    ref_series: str, category: str, column: str, fallback: float | None, min_n: int = 0
) -> tuple[float | None, bool]:
    """``(value, measured)`` for a reference line.

    Measured mean ``column`` from the reference series when present, else
    ``fallback`` (the paper dict). ``measured`` tells the caller which source
    won so a paper fallback can be labelled distinctly.

    ``min_n`` rejects a summary row built from far fewer comparisons than the
    series it baselines — a ref swept on a partial manifest would otherwise be
    drawn as authoritative. af3_ref does NOT trip it: its DockQ coverage matches
    df3v5r16's exactly (240 / 138 / 35 / 20 / 6 interfaces for pp / agab /
    peptide / dna / rna).

    Note what that last pair means. The nucleic-interface success panels rest on
    20 and 6 scored interfaces on BOTH sides — small in absolute terms, but
    like-for-like, which is why the guard is relative and not an absolute floor.
    Whole-model lDDT is unaffected: those panels re-aggregate from raw/ and cover
    all 170 / 36 benched targets.
    """
    row = _ref_row(ref_series, category)
    if row is not None and row.get(column) not in (None, ""):
        n = ref_count(ref_series, category)
        if min_n and n and n < min_n:
            print(
                f"[summary] {ref_series}/{category}: ref '{column}' covers {n} < {min_n} "
                "comparisons — too thin to baseline, falling back to the paper number"
            )
            return fallback, False
        try:
            return float(row[column]), True
        except ValueError:
            pass
    return fallback, False


def _spine_marker(ax, y, color, marker="<", ms=8, alpha=0.95, label=None):
    """A marker sitting on the right spine at data-value ``y`` — lets the AF3 ref be
    read straight off the right axis edge. ``label`` adds it to the legend."""
    ax.plot(
        1.0, y, marker=marker, color=color, ms=ms, alpha=alpha, label=label,
        transform=ax.get_yaxis_transform(), clip_on=False, zorder=6,
    )


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


def summarize(series: str, category: str, plot: bool, ref_series: str = "") -> None:
    ref_series = resolve_ref_series(ref_series, series)
    if category in REF_SERIES_EXCLUDE_CATEGORIES.get(ref_series, ()):
        print(
            f"[summary] {ref_series}/{category}: fold not comparable "
            "(see REF_SERIES_EXCLUDE_CATEGORIES) — overlay dropped, using paper AF3 numbers"
        )
        ref_series = ""
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
        # (1) DockQ CAPRI quality tiers, (2) whole-model lDDT + inter-chain iLDDT
        # overlaid, (3) whole-model TM-score — each median + IQR across targets, on
        # the [0, 1] range. iLDDT (collapse-sensitive) is only present from the step
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
        # trustworthy.
        TM_FOLD_THRESH = 0.5
        tm_by_step = {d["step"]: d["mean"] for d in tm}

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
                if (tm_by_step.get(t["step"]) is not None and tm_by_step[t["step"]] < TM_FOLD_THRESH)
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
                    label=f"mean TM < {TM_FOLD_THRESH:.2f} (unreliable)" if j == 0 else None,
                )
            # Success panel uses ORACLE (matches the oracle DockQ-tier stack top);
            # every other panel uses the mean aggregate. Measured af3_ref value is a
            # percentage (÷100 to the fraction axis); the paper fallback is a fraction.
            # Half this series' own interface count is the floor for trusting the
            # measured ref (see ref_value's min_n) — below that, use the paper number.
            _own_n = max((int(s.get("n_interfaces") or 0) for s in summaries), default=0)
            af3_pct, af3_measured = ref_value(ref_series, category, "success_rate_oracle", None, min_n=_own_n // 2)
            af3 = (af3_pct / 100.0) if af3_measured else AF3_SUCCESS_RATE.get(category)
            if af3 is not None:
                # bar height (acceptable+medium+high) == success rate @0.23, so
                # this line is directly comparable to the top of each stack.
                # The measured ref is an oracle like the stack; the paper fallback is a
                # mean-over-samples rate on the full set — say which, on the line.
                _sfx = "oracle" if af3_measured else "paper mean, full set"
                _who = af3_tag() if af3_measured else "AF3 paper"
                # Faint thin line in the SAME colour as the success score it references
                # (the confidence top-1 success line, "#2166ac") — not a bold red dash.
                axd.axhline(
                    af3, color="#2166ac", alpha=0.5, lw=1.1, zorder=3,
                    label=f"{_who} success @0.23 ({af3:.0%}, {_sfx})",
                )
                _spine_marker(axd, af3, "#2166ac")
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
            # opaque + above the hatched bars (zorder 9) so the hatch doesn't show through
            _leg = axd.legend(handles, labels, loc="upper left", fontsize=8, title="tier / QS", framealpha=1.0)
            _leg.set_zorder(20)
            axd.grid(axis="y", alpha=0.3)

        def _dist_panel(
            ax,
            dist,
            ylabel,
            title,
            af3ref_median=None,
            overlay=None,
            conf=None,
            af3ref_mean=None,
            overlay_af3ref=None,
            paper_ref=None,
        ):
            """median + mean + IQR band of a whole-model metric vs real step.

            ``overlay=(dist, label, color)`` draws a second metric (e.g. inter-chain
            iLDDT) in colour on the same axes — median line + IQR band.
            ``conf`` is the confidence-top-1 distribution (same shape as ``dist``);
            its across-target median is drawn as a distinct line (deployed number).

            A measured reference series is folded like-for-like and reduced from its raw
            per-target values (its summary stores no median): the whole-model **median**
            ref (``af3ref_median``, black — matches the black median line) as a faint line
            + spine marker, the **mean** ref (``af3ref_mean``, gray — matches the gray mean
            line) as a spine marker, and the overlay's **median** (``overlay_af3ref``, the
            overlay's own colour). Such a series is left-skewed so median ≫ mean here.

            ``paper_ref`` is the published AF3 number (FoldBench Supp. Table 3) and is
            drawn dotted ONLY when no measured ref is available. It is a mean over the
            paper's FULL set, not our num-token-capped subset — a target marker, never a
            like-for-like value.
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
                if af3ref_median is not None:
                    ax.axhline(af3ref_median, color="k", alpha=0.5, lw=1.1, zorder=3,
                               label=f"{af3_tag()} median ({af3ref_median:.2f})")
                    _spine_marker(ax, af3ref_median, "k")
                if af3ref_mean is not None:
                    _spine_marker(ax, af3ref_mean, "0.45", label=f"{af3_tag()} mean ({af3ref_mean:.2f})")
                if af3ref_median is None and paper_ref is not None:
                    ax.axhline(paper_ref, color="k", alpha=0.35, lw=1.1, ls=":", zorder=3,
                               label=f"AF3 paper mean, full set ({paper_ref:.2f})")
                    _spine_marker(ax, paper_ref, "k", alpha=0.55)
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
                if overlay_af3ref is not None:
                    ax.axhline(
                        overlay_af3ref, color=ocol, alpha=0.5, lw=1.1, zorder=3,
                        label=f"AF3 {olabel} median ({overlay_af3ref:.2f})",
                    )
                    _spine_marker(ax, overlay_af3ref, ocol)
            if dist or (overlay is not None and overlay[0]):
                ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_title(title)

        # AF3 refs computed like-for-like from the measured ref series' raw per-target
        # values (its summary has no median column); left-skewed → median ≫ mean. With
        # no ref series these stay None and the lDDT panel falls back to the paper mean.
        def _af3(col):
            d = interface_metric_dist(ref_series, category, col) if ref_series else []
            return (d[0]["med"], d[0]["mean"]) if d else (None, None)

        _ld_med, _ld_mean = _af3("lddt")
        _il_med, _ = _af3("ilddt")
        _tm_med, _tm_mean = _af3("tm_score")
        _dist_panel(
            axl,
            ld,
            "lDDT",
            "whole-model lDDT (black) + inter-chain iLDDT (red) — median + IQR",
            af3ref_median=_ld_med,
            overlay=(il, "inter-chain iLDDT", "tab:red"),
            conf=ld_conf,
            af3ref_mean=_ld_mean,
            overlay_af3ref=_il_med,
            paper_ref=AF3_INTERFACE_LDDT.get(category),
        )
        _dist_panel(
            axt,
            tm,
            "TM-score (whole model)",
            "whole-model TM-score (median + IQR across targets)",
            af3ref_median=_tm_med,
            conf=tm_conf,
            af3ref_mean=_tm_mean,
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
            # AF3 refs like-for-like from the measured ref series' raw per-target values:
            # median (black, matches the median line) + mean (gray); its summary has no
            # median column. With no ref series, fall back to the paper mean (dotted).
            _raw_col = {"lddt": "lddt", "tm": "tm_score", "gdt": "oligo_gdtts"}[mk]
            _af3d = interface_metric_dist(ref_series, category, _raw_col) if ref_series else []
            _paper = AF3_MONOMER.get(category, {}).get(mk)
            if not _af3d and _paper is not None:
                ax.axhline(_paper, color="k", alpha=0.35, lw=1.1, ls=":", zorder=3)
                _spine_marker(ax, _paper, "k", alpha=0.55)
                ax.text(
                    0.005, _paper, f"AF3 paper mean (full set) {_paper:.2f}",
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color="k", alpha=0.7, clip_on=False,
                )
            if _af3d:
                _med, _mean = _af3d[0]["med"], _af3d[0]["mean"]
                ax.axhline(_med, color="k", alpha=0.5, lw=1.1, zorder=3)
                _spine_marker(ax, _med, "k")
                _spine_marker(ax, _mean, "0.45")
                ax.text(
                    0.005, _med, f"{af3_tag()} median {_med:.2f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color="k", alpha=0.7, clip_on=False,
                )
                ax.text(
                    0.005, _mean, f"AF3 mean {_mean:.2f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="top", fontsize=8, color="0.45", alpha=0.9, clip_on=False,
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
    ap.add_argument(
        "--ref-series",
        default="",
        help="Series whose measured mean (oracle for the success panel) is drawn as a "
        "reference line; falls back to the built-in paper AF3 dicts when its eval is "
        "absent or no series is given. Series listed in REF_SERIES_CAVEAT "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(REF_SERIES_CAVEAT.items()))}) are "
        "drawn with that caveat stamped into every label.",
    )
    args = ap.parse_args()
    summarize(args.series, args.category, plot=not args.no_plot, ref_series=args.ref_series)


if __name__ == "__main__":
    main()
