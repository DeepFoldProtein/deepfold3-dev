"""FoldBench-small: rank-1 (deployed) vs oracle (ceiling) per-target DockQ, all 8 models.

Twin thin bars per model — top = rank-1 (sample picked by the model's own confidence),
bottom = oracle (best sample) — both CAPRI-tiered, with MEAN drawn as a single vertical
line so the N-unbiased aggregate stays visible without competing for bar space.

External values come from `foldbench_all_models_per_target.csv` (FoldBench's own per-target
summary: ranked_dockq / mean_dockq / oracle_dockq over n_samples=20). That source carries a
ranking for OpenFold3-p2, which the tarball parse did not, so OF3p2 appears here at rank-1;
ESMFold2 exports none and is oracle/mean-only. DF3r14 is recomputed per checkpoint from its
own benched raw CSVs and aggregated the same way: per sample, mean DockQ over the target's
annotated interfaces; then rank-1 / mean / oracle over samples.

  python plot_foldbench_ranked.py                      # full set, latest r14 step
  python plot_foldbench_ranked.py --after 2023-06-01   # Boltz-2 training-cutoff control
  python plot_foldbench_ranked.py --step 19000
"""
import argparse, csv, glob, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = "/home/jovyan/work/deepfold3-dev"
R14 = f"{REPO}/bench/runs/df3v4r14/eval"
TGT = f"{REPO}/bench/data/targets"
OUT = f"{REPO}/export/af3-vs-r14-foldbench"
EXT = f"{OUT}/foldbench_all_models_per_target.csv"
DATES = f"{OUT}/foldbench_target_dates.csv"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TIER = ["#bcd4ec", "#5a9bd4", "#1f5c9e"]
R14TIER = ["#f6c3a6", "#f0946a", "#c1490f"]
R14COL = "#eb6834"
MEANC = "#111111"
# CSV `code` → display label.  DF3r14 is injected separately.
ORDER = [("af3", "AF3"), ("b2", "BT2"), ("b1", "BT1"), ("ptx_v2", "PXv2"),
         ("ptx", "PXv1"), ("of3", "OF3p2"), ("ef2", "EF2")]
DAGGER = {"BT2", "PXv2", "EF2"}   # substantially larger parameter count
CATS = [("ppi", "interface_protein_protein", "protein–protein"),
        ("agab", "interface_antibody_antigen", "antibody–antigen")]

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=None)
ap.add_argument("--after", default=None,
                help="keep only targets released AFTER this date (YYYY-MM-DD), e.g. 2023-06-01")
A = ap.parse_args()
AFTER = A.after
SFX = f"_post{AFTER}" if AFTER else ""


def load_dates():
    d = {}
    if os.path.isfile(DATES):
        for r in csv.DictReader(open(DATES)):
            if r.get("released"):
                d[r["pdb_id"]] = r["released"]
    return d


DATED = load_dates() if AFTER else {}


def latest_step(cat_full):
    steps = [int(re.search(r"step_(\d+)", p).group(1))
             for p in glob.glob(f"{R14}/{cat_full}/step_*/summary.csv")]
    return max(steps) if steps else None


def annotated(cat_full):
    """target → set of annotated interface chain pairs (targets can have several)."""
    d = defaultdict(set)
    for r in csv.DictReader(open(f"{TGT}/{cat_full}.csv")):
        d[r["pdb_id"]].add(frozenset([r["interface_chain_id_1"], r["interface_chain_id_2"]]))
    return d


def ext_per_target(short):
    """code → {target: [rank1, mean, oracle]}; rank1 is None where the model has no ranking."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(EXT)):
        if r["benchmark"] != short:
            continue
        rk = r["ranked_dockq"]
        out[r["code"]][r["target_id"]] = [
            float(rk) if rk not in ("", "None") else None,
            float(r["mean_dockq"]), float(r["oracle_dockq"])]
    return out


def r14_per_target(cat_full, step, ann):
    """Match FoldBench's aggregation: per sample, mean DockQ over the target's annotated
    interfaces; then rank-1 (top ranking_score) / mean / oracle over samples."""
    per = defaultdict(lambda: defaultdict(list))   # target → sample key → [dockq per interface]
    rs = defaultdict(dict)                          # target → sample key → ranking_score
    for r in csv.DictReader(open(f"{R14}/{cat_full}/step_{step}/raw/{cat_full}_ost.csv")):
        pdb = r["pdb_id"]
        if frozenset([r["ic1"], r["ic2"]]) not in ann.get(pdb, ()):
            continue
        try:
            dq = float(r["dockq"])
        except (TypeError, ValueError):
            continue
        key = (r.get("seed", ""), r.get("sample", ""))
        per[pdb][key].append(dq)
        v = r.get("ranking_score")
        if v not in (None, "", "None"):
            rs[pdb][key] = float(v)
    out = {}
    for pdb, samples in per.items():
        keys = sorted(samples)
        vals = [float(np.mean(samples[k])) for k in keys]
        if not vals:
            continue
        scored = [(rs[pdb][k], v) for k, v in zip(keys, vals) if k in rs[pdb]]
        rank1 = max(scored)[1] if scored else None
        out[pdb] = [rank1, float(np.mean(vals)), float(np.max(vals))]
    return out


def n_samples(cat_full, step):
    """Predictions per target actually on disk (distinct seed x sample). Read, never
    assumed: the per-category seed budget changes over time (SEEDS_<CAT> in auto.sh),
    and `oracle` is monotone in it, so a stale hardcoded count mislabels the figure."""
    per = defaultdict(set)
    for r in csv.DictReader(open(f"{R14}/{cat_full}/step_{step}/raw/{cat_full}_ost.csv")):
        per[r["pdb_id"]].add((r["seed"], r["sample"]))
    return max((len(v) for v in per.values()), default=0)


def tiers(vals):
    n = len(vals)
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    a = sum(1 for d in vals if 0.23 <= d < 0.49) / n * 100
    m = sum(1 for d in vals if 0.49 <= d < 0.80) / n * 100
    h = sum(1 for d in vals if d >= 0.80) / n * 100
    return a, m, h, a + m + h


results, common, steps_used, NS = {}, {}, {}, {}
for short, cat_full, cat_name in CATS:
    step = A.step or latest_step(cat_full)
    steps_used[short] = step
    ann = annotated(cat_full)
    ext = ext_per_target(short)
    NS[short] = n_samples(cat_full, step)
    pm = {"DF3r14": (r14_per_target(cat_full, step, ann), True)}
    for code, lbl in ORDER:
        pm[lbl] = (ext[code], False)

    comm = set.intersection(*[set(d) for d, _ in pm.values()])
    if AFTER:
        n0 = len(comm)
        undated = sum(1 for p in comm if not DATED.get(p))
        comm = {p for p in comm if DATED.get(p, "") > AFTER}
        print(f"  [{short}] released>{AFTER}: kept {len(comm)}/{n0} (undated dropped: {undated})")
    common[short] = len(comm)

    rows = []
    for lbl, (d, ours) in pm.items():
        r1 = [d[p][0] for p in comm if d[p][0] is not None]
        rt = tiers(r1) if len(r1) == len(comm) else None   # partial ranking → treat as absent
        mn = tiers([d[p][1] for p in comm])[3]
        ot = tiers([d[p][2] for p in comm])
        rows.append((lbl, rt, mn, ot, ours))
    rows.sort(key=lambda r: (-(r[1][3] if r[1] else -1), -r[3][3]))
    results[short] = rows
    print(f"[{cat_name}] step {step}, n={len(comm)}")
    for lbl, rt, mn, ot, ours in rows:
        r1s = f"{rt[3]:5.1f}" if rt else "    –"
        print(f"    {lbl:8s} rank1={r1s}  mean={mn:5.1f}  oracle={ot[3]:5.1f}")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=150, gridspec_kw={"wspace": 0.40})
fig.patch.set_facecolor("#fcfcfb")
BH, OFF = 0.29, 0.16   # thin bar height, half-separation of the twin bars

for ax, (short, cat_full, cat_name) in zip(axes, CATS):
    rows = results[short]
    y = np.arange(len(rows))[::-1]
    ax.set_facecolor("#fcfcfb")
    for yy, (lbl, rt, mn, ot, ours) in zip(y, rows):
        cols = R14TIER if ours else TIER
        acc = R14COL if ours else INK2
        # top bar = rank-1 (deployed), bottom bar = oracle (ceiling)
        for t, yb, alpha in ((rt, yy + OFF, 1.0), (ot, yy - OFF, 0.55)):
            if t is None:
                ax.text(1.5, yb, "no confidence — rank-1 n/a", va="center", ha="left",
                        fontsize=7.2, color=INK2, style="italic")
                continue
            left = 0.0
            for frac, col in zip(t[:3], cols):
                ax.barh(yb, frac, BH, left=left, color=col, alpha=alpha,
                        edgecolor="white", lw=0.6, zorder=3)
                left += frac
        # mean: a single line across both bars
        ax.plot([mn, mn], [yy - OFF - BH / 2 - 0.03, yy + OFF + BH / 2 + 0.03],
                color=MEANC, lw=1.5, ls=(0, (2.5, 1.6)), zorder=6)
        r1txt = f"{rt[3]:.0f}" if rt else "–"
        ax.text(min(max(ot[3], rt[3] if rt else 0, mn) + 1.8, 86), yy,
                f"{r1txt}% → {ot[3]:.0f}%", va="center", ha="left", fontsize=8.8,
                color=acc, fontweight=("bold" if ours else "normal"))

    def _lbl(r):
        s = r[0] + " †" if r[0] in DAGGER else r[0]
        return "★ " + s if r[4] else s

    ax.set_yticks(y)
    ax.set_yticklabels([_lbl(r) for r in rows], fontsize=10, color=INK)
    for tk, r in zip(ax.get_yticklabels(), rows):
        if r[4]:
            tk.set_color(R14COL)
            tk.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel("per-target DockQ success rate %   (upper = rank-1, lower = oracle; CAPRI tiers)",
                  fontsize=9, color=INK)
    ax.set_title(f"{cat_name}   (n={common[short]}"
                 f"{', released>' + AFTER if AFTER else ''}, r14 @ step {steps_used[short]})",
                 fontsize=11.5, color=INK, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.5, zorder=0)

fig.legend(handles=[Patch(color=TIER[0], label="Acceptable"), Patch(color=TIER[1], label="Medium"),
                    Patch(color=TIER[2], label="High"),
                    Patch(facecolor="#5a9bd4", alpha=0.55, label="lower bar = oracle"),
                    Line2D([0], [0], color=MEANC, lw=1.5, ls=(0, (2.5, 1.6)), label="mean over samples")],
           loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=8.4,
           framealpha=0.96, columnspacing=1.7, handletextpad=0.6, frameon=False)
fig.suptitle("FoldBench-small — rank-1 (deployed) vs oracle (ceiling) per-target DockQ" +
             (f"   ·   targets released after {AFTER} (Boltz-2 cutoff)" if AFTER else ""),
             fontsize=13, color=INK, y=1.02)
cut = (f"Targets restricted to release date > {AFTER} (Boltz-2 training cutoff) — removes structures that may sit in a model's training set.  Release dates: RCSB.\n"
       if AFTER else "")
# Sample budget is read from disk, not assumed — `oracle` is monotone in it, so an
# N mismatch has to be stated wherever oracle is shown.
_ns = {short: NS[short] for short, _, _ in CATS}
if len(set(_ns.values())) == 1 and next(iter(_ns.values())) == 20:
    _budget = "Sample budget is matched: 20 predictions/target (4 seeds × 5) for every model including DF3r14."
else:
    _detail = ", ".join(f"{name} {_ns[short]}" for short, _, name in CATS)
    _budget = (f"SAMPLE BUDGET NOT MATCHED — external models fold 20 predictions/target (4 seeds × 5); DF3r14 folds {_detail}.  "
               "Oracle is monotone in sample count, so any DF3r14 panel below 20 understates its oracle.")
fig.text(0.5, -0.06,
         cut +
         "Upper bar = RANK-1: the single sample the model's own confidence picks (deployed number).  Lower (faded) bar = ORACLE: best sample per target (ceiling).  Dashed line = MEAN over samples.  Labels: rank-1%→oracle%.\n"
         "Per-target DockQ = per sample, mean over the target's annotated interfaces; then rank-1 / mean / oracle over samples.  "
         + _budget + "\n"
         "External values from FoldBench's per-target summary (`foldbench_all_models_per_target.csv`).  EF2 exports no confidence.  Common target set = ∩ all models.   † = substantially larger parameter count.",
         ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.5)
fig.savefig(f"{OUT}/foldbench_ranked_ppi_agab{SFX}.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"wrote foldbench_ranked_ppi_agab{SFX}.png")

with open(f"{OUT}/foldbench_ranked_data{SFX}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "r14_step", "model", "n", "rank1_A%", "rank1_M%", "rank1_H%",
                "rank1_succ%", "mean_succ%", "oracle_succ%", "oracle_gain_over_rank1%"])
    for short, cat_full, cat_name in CATS:
        for lbl, rt, mn, ot, ours in results[short]:
            w.writerow([cat_name, steps_used[short], lbl, common[short],
                        *( [round(x, 1) for x in rt[:3]] + [round(rt[3], 1)] if rt else ["", "", "", ""] ),
                        round(mn, 1), round(ot[3], 1),
                        round(ot[3] - rt[3], 1) if rt else ""])
print(f"wrote foldbench_ranked_data{SFX}.csv")
