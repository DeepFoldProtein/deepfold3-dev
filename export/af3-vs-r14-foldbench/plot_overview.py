"""RETIRED — not run by update_foldbench.sh; its report section is frozen at step 23k.

The baseline here is the `af3_ref` sweep, folded before the bench attached templates:
a handicapped AlphaFold3, 3.8 points under the published AF3 on protein-protein and
9.9 on antibody-antigen on identical targets. Kept runnable so it can be revived once
af3_ref is re-folded WITH templates, or once precompute_external.py covers all eight
categories and this can be re-based on FoldBench's own AF3 predictions (the cache holds
only ppi + agab). Re-add the line to update_foldbench.sh when either lands.

FoldBench-style overview: af3_ref vs df3v4r14 at its latest benched step.
Left  — monomers: TM-score, conf-top1 (solid) → oracle (faint cap).
Right — interfaces: DockQ stacked in CAPRI tiers (Acceptable/Medium/High) at
        conf-top1 selection; oracle success rate marked as a caret (ceiling).

  python plot_overview.py             # auto-detect latest r14 step per category
  python plot_overview.py --step 19000
"""
import argparse, csv, glob, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RUNS = "/home/jovyan/work/deepfold3-dev/bench/runs"
OUT = "/home/jovyan/work/deepfold3-dev/export/af3-vs-r14-foldbench"
AF3, R14 = "#8a8f98", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TIER = ["#bcd4ec", "#5a9bd4", "#1f5c9e"]   # Acceptable / Medium / High (light→dark)
TIER_LBL = ["Acceptable (0.23–0.49)", "Medium (0.49–0.80)", "High (≥0.80)"]

MONO = [("monomer_protein","protein"),("monomer_rna","RNA"),("monomer_dna","DNA")]
IFACE = [("interface_protein_protein","protein–protein"),
         ("interface_antibody_antigen","antibody–antigen"),
         ("interface_protein_peptide","protein–peptide"),
         ("interface_protein_dna","protein–DNA"),
         ("interface_protein_rna","protein–RNA")]

A = argparse.ArgumentParser(); A.add_argument("--step", type=int, default=None)
A = A.parse_args()

def latest_step(cat):
    """Latest r14 step whose eval completed (summary.csv is written last)."""
    steps = [int(re.search(r"step_(\d+)", p).group(1))
             for p in glob.glob(f"{RUNS}/df3v4r14/eval/{cat}/step_*/summary.csv")]
    if not steps:
        raise SystemExit(f"no benched step for {cat}")
    return max(steps)

STEP = {cat: (A.step or latest_step(cat)) for cat, _ in MONO + IFACE}
# Only tag a category with its step when the panel is not on a single uniform step.
UNIFORM = len(set(STEP.values())) == 1
def tag(cat):
    return "" if UNIFORM else f"  ({STEP[cat]//1000}k)"

def summ(series, cat, step):
    return list(csv.DictReader(open(f"{RUNS}/{series}/eval/{cat}/step_{step}/summary.csv")))[0]

def dockq_tiers(series, cat, step, sel):
    """Return (acc%, med%, high%, succ%) for conf/oracle-selected per-interface DockQ."""
    by = defaultdict(list)
    for r in csv.DictReader(open(f"{RUNS}/{series}/eval/{cat}/step_{step}/raw/{cat}_ost.csv")):
        try: dq = float(r["dockq"])
        except (ValueError, TypeError): continue
        rs = r.get("ranking_score"); rs = float(rs) if rs not in (None,"","None") else None
        by[(r["pdb_id"], r["ic1"], r["ic2"])].append((dq, rs))
    picked = []
    for v in by.values():
        if sel == "oracle":
            picked.append(max(x[0] for x in v))
        else:
            vr = [x for x in v if x[1] is not None]
            picked.append(max(vr, key=lambda x: x[1])[0] if vr else sum(x[0] for x in v)/len(v))
    n = len(picked)
    acc = sum(1 for d in picked if 0.23<=d<0.49)/n*100
    med = sum(1 for d in picked if 0.49<=d<0.80)/n*100
    hi  = sum(1 for d in picked if d>=0.80)/n*100
    return acc, med, hi, acc+med+hi

# ---- monomer data (TM conf→oracle) ----
ml, mac, mao, mrc, mro = [], [], [], [], []
for cat, lab in MONO:
    a, r = summ("af3_ref",cat,0), summ("df3v4r14",cat,STEP[cat])
    ml.append(lab + tag(cat))
    mac.append(float(a["conf_tm_score"])); mao.append(float(a["oracle_tm_score"]))
    mrc.append(float(r["conf_tm_score"])); mro.append(float(r["oracle_tm_score"]))

fig, (axm, axi) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=150,
                               gridspec_kw={"width_ratios":[3,5.2], "wspace":0.44})
fig.patch.set_facecolor("#fcfcfb")

# ---- monomer panel ----
y = np.arange(len(ml))[::-1]; h = 0.36
for c, o, off, col, bold in [(mac,mao,+h/2,AF3,False),(mrc,mro,-h/2,R14,True)]:
    axm.barh(y+off, o, h, color=col, alpha=0.28, zorder=2)
    axm.barh(y+off, c, h, color=col, zorder=3)
    for yy,cv,ov in zip(y+off,c,o):
        axm.text(ov+0.012, yy, f"{cv:.2f}→{ov:.2f}", va="center", fontsize=8.2,
                 color=(R14 if bold else INK2), fontweight=("bold" if bold else "normal"))
axm.set_yticks(y); axm.set_yticklabels(ml, fontsize=10.5, color=INK)
axm.set_xlim(0,1); axm.set_xlabel("TM-score   (rank-1 → oracle)", fontsize=10.5, color=INK)
axm.set_title("Monomers — TM-score", fontsize=12, color=INK, pad=8)
axm.legend(handles=[Patch(color=AF3,label="AF3"),Patch(color=R14,label="DF3 r14"),
                    Patch(color="#999",alpha=0.28,label="oracle cap")], loc="lower right", fontsize=8.5)

# ---- interface panel: stacked CAPRI tiers, af3 vs r14 (rank-1), oracle caret ----
yi = np.arange(len(IFACE))[::-1]; hh = 0.36
labs = []
for idx,(cat, lab) in enumerate(IFACE):
    ybase = yi[idx]
    labs.append(lab + tag(cat))
    for series, step, off, edge, name in [("af3_ref",0,+hh/2,AF3,"AF3"),("df3v4r14",STEP[cat],-hh/2,R14,"r14")]:
        acc,med,hi,succ = dockq_tiers(series,cat,step,"conf")   # conf = rank-1 (max ranking_score)
        left=0
        for frac,col in [(acc,TIER[0]),(med,TIER[1]),(hi,TIER[2])]:
            axi.barh(ybase+off, frac, hh, left=left, color=col, edgecolor=edge, lw=1.5, zorder=3)
            left+=frac
        osucc = dockq_tiers(series,cat,step,"oracle")[3]        # oracle ceiling caret
        axi.plot([osucc,osucc],[ybase+off-hh/2,ybase+off+hh/2],color=edge,lw=1.8,ls=(0,(2,1)),zorder=5)
        axi.text(min(max(left,osucc)+1.4, 88), ybase+off, f"{name} {succ:.0f}%", va="center", ha="left", fontsize=7.8,
                 color=(R14 if name=="r14" else INK2), fontweight=("bold" if name=="r14" else "normal"))
axi.set_yticks(yi); axi.set_yticklabels(labs, fontsize=10.5, color=INK)
axi.set_xlim(0,100); axi.set_xlabel("DockQ success rate %   (stacked CAPRI tiers, rank-1;  ┆ = oracle ceiling)", fontsize=10, color=INK)
axi.set_title("Interfaces — DockQ quality tiers", fontsize=12, color=INK, pad=8)
tier_handles=[Patch(color=TIER[i],label=l) for i,l in enumerate(["Acceptable","Medium","High"])]
model_handles=[Patch(facecolor="white",edgecolor=AF3,lw=1.5,label="AF3 outline"),
               Patch(facecolor="white",edgecolor=R14,lw=1.5,label="r14 outline")]
axi.legend(handles=tier_handles+model_handles, loc="upper right", bbox_to_anchor=(1.0,1.0),
           fontsize=7.8, framealpha=0.96, ncol=1, handlelength=1.3, labelspacing=0.35)

for ax in (axm,axi):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.5, zorder=0)

steplab = f"step {min(STEP.values())//1000}k" if UNIFORM else \
          f"steps {min(STEP.values())//1000}–{max(STEP.values())//1000}k"

def _nsamp(series, cat, step):
    """Predictions per target on disk. Read, never assumed — the per-category seed
    budget (SEEDS_<CAT> in auto.sh) changes over time and `oracle` is monotone in it."""
    raw = f"{RUNS}/{series}/eval/{cat}/step_{step}/raw/{cat}_ost.csv"
    if not os.path.isfile(raw):
        return None
    per = defaultdict(set)
    for r in csv.DictReader(open(raw)):
        per[r["pdb_id"]].add((r.get("seed"), r.get("sample")))
    return max((len(v) for v in per.values()), default=0)

_r14_ns = {lab: _nsamp("df3v4r14", cat, STEP[cat]) for cat, lab in IFACE}
_af3_ns = {lab: _nsamp("af3_ref", cat, 0) for cat, lab in IFACE}
_uniq = {v for v in list(_r14_ns.values()) + list(_af3_ns.values()) if v}
if len(_uniq) == 1:
    _budget = f"{_uniq.pop()} predictions/target for both models"
else:
    _mismatch = ", ".join(f"{lab} r14 {_r14_ns[lab]} vs AF3 {_af3_ns[lab]}"
                          for cat, lab in IFACE if _r14_ns[lab] != _af3_ns[lab])
    _budget = ("predictions/target: "
               + ", ".join(f"{lab} {_r14_ns[lab]}" for cat, lab in IFACE if _r14_ns[lab])
               + f" (r14) — SAMPLE BUDGET NOT MATCHED where noted [{_mismatch}]; oracle is monotone in it"
               if _mismatch else f"{max(_uniq)} predictions/target")
fig.suptitle(f"AF3 reference  vs  DeepFold3 r14 ({steplab}) — FoldBench-small", fontsize=13.5, color=INK, y=1.02)
fig.text(0.5,-0.05,
  f"Same bench harness: MMseqs2 / ColabFold MSA pipeline (unpaired UniRef30 + ColabFoldDB env), same targets, {_budget}, "
  "OST --lddt-no-stereochecks.  rank-1 = sample with top ranking_score (0.8·ipTM+0.2·pTM); oracle = best sample.  DockQ CAPRI tiers: "
  "Acceptable 0.23–0.49 / Medium 0.49–0.80 / High ≥0.80." +
  ("" if UNIFORM else "  Per-category r14 step in parentheses where the latest benched step differs."),
  ha="center", fontsize=7.8, color=INK2, wrap=True)
fig.savefig(f"{OUT}/af3_vs_r14_overview.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("wrote af3_vs_r14_overview.png")

# data dump
with open(f"{OUT}/overview_data.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["category","model","r14_step","selection","acceptable%","medium%","high%","success%"])
    for cat,lab in IFACE:
        for series,step,nm in [("af3_ref",0,"AF3"),("df3v4r14",STEP[cat],"r14")]:
            for sel in ("conf","oracle"):
                t=dockq_tiers(series,cat,step,sel)
                w.writerow([cat,nm,STEP[cat],sel,round(t[0],1),round(t[1],1),round(t[2],1),round(t[3],1)])
    for cat,lab in MONO:
        a,r=summ("af3_ref",cat,0),summ("df3v4r14",cat,STEP[cat])
        w.writerow([cat,"AF3",STEP[cat],"conf/oracle TM",a["conf_tm_score"],"",a["oracle_tm_score"],""])
        w.writerow([cat,"r14",STEP[cat],"conf/oracle TM",r["conf_tm_score"],"",r["oracle_tm_score"],""])
print(f"wrote overview_data.csv  (r14 @ {steplab})")
