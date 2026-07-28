"""Confidence (ranking_score) calibration for r14, per category.
Left: rank-1 / oracle selection band (headline metric, 0-1).
Right: WITHIN-TARGET r(ranking_score, per-sample quality) — calibration strength.

Why within-target: rank-1 selection happens *inside* one target's sample pool, so
the only correlation that predicts it is the one computed inside each target and
then averaged (Fisher-z). Pooling every prediction across targets mixes in the
between-target difficulty spread, which dominates the variance and buries the
selection signal — on antibody at step 23k the pooled r reads 0.056 while the
within-target r is 0.128 [95% CI +0.044, +0.211], i.e. pooling reported "no
calibration" for a category whose rank-1 was beating its own mean by 5.8 pts.
Both are dumped to the CSV; the figure and any calibration claim use within-target.

Also dumps confidence_data.csv."""
import csv, os, glob, math, re, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "/home/jovyan/work/deepfold3-dev/bench/runs"
OUT = "/home/jovyan/work/deepfold3-dev/export/af3-vs-r14-foldbench"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_MEAN, C_CONF, C_ORA = "#8a8f98", "#2a78d6", "#1baf7a"

ap=argparse.ArgumentParser(); ap.add_argument("--step",type=int,default=None); A=ap.parse_args()
def latest_step(cat):
    st=[int(re.search(r"step_(\d+)",p).group(1)) for p in glob.glob(f"{RUNS}/df3v4r14/eval/{cat}/step_*/summary.csv")]
    return max(st) if st else None

# (category, label, metric_key, raw_metric_col) — step auto-detected (latest benched) or --step
CATS = [
    ("monomer_protein","monomer protein","tm_score","tm_score"),
    ("monomer_rna","monomer RNA","tm_score","tm_score"),
    ("monomer_dna","monomer DNA","tm_score","tm_score"),
    ("interface_protein_protein","protein–protein","dockq","dockq"),
    ("interface_antibody_antigen","antibody–antigen","dockq","dockq"),
    ("interface_protein_peptide","protein–peptide","dockq","dockq"),
    ("interface_protein_dna","protein–DNA","dockq","dockq"),
    ("interface_protein_rna","protein–RNA","dockq","dockq"),
]

def pear(xs, ys):
    n=len(xs)
    if n<3: return float('nan')
    mx,my=sum(xs)/n,sum(ys)/n
    sx=sum((x-mx)**2 for x in xs)**.5; sy=sum((y-my)**2 for y in ys)**.5
    return float('nan') if sx==0 or sy==0 else sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sx*sy)


def within_target_r(groups, min_samples=5):
    """Mean within-target Pearson r via Fisher z, with a 95% CI.

    `groups` maps a target key -> [(ranking_score, quality), ...] for that target's
    samples. Targets with <min_samples usable points, or with a constant score or
    constant quality (r undefined), are dropped — those carry no selection signal.
    Returns (r, lo, hi, n_targets); (nan, nan, nan, 0) if nothing is usable.
    """
    zs=[]
    for v in groups.values():
        if len(v)<min_samples: continue
        r=pear([a for a,_ in v],[b for _,b in v])
        if r!=r: continue                       # NaN: constant score or quality
        zs.append(math.atanh(max(-0.999, min(0.999, r))))
    if len(zs)<2: return float('nan'), float('nan'), float('nan'), len(zs)
    zbar=sum(zs)/len(zs)
    sd=(sum((z-zbar)**2 for z in zs)/(len(zs)-1))**.5
    se=sd/len(zs)**.5
    return math.tanh(zbar), math.tanh(zbar-1.96*se), math.tanh(zbar+1.96*se), len(zs)

rows=[]
for cat,lab,mk,rc in CATS:
    step=A.step or latest_step(cat)
    if step is None: continue
    s=list(csv.DictReader(open(f"{RUNS}/df3v4r14/eval/{cat}/step_{step}/summary.csv")))[0]
    if mk=="tm_score":                     # monomers: TM-score band
        mean=float(s["mean_tm_score"]); conf=float(s["conf_tm_score"]); ora=float(s["oracle_tm_score"])
    else:                                  # interfaces: DockQ success-rate band (fraction)
        mean=float(s["success_rate_mean"])/100; conf=float(s["success_rate_conf"])/100; ora=float(s["success_rate_oracle"])/100
    raw=list(csv.DictReader(open(f"{RUNS}/df3v4r14/eval/{cat}/step_{step}/raw/{cat}_ost.csv")))
    xs,ys=[],[]
    # Group by what rank-1 actually selects within: one interface for interface
    # categories, one target for monomers.
    groups=defaultdict(list)
    for r in raw:
        rs=r.get("ranking_score"); q=r.get(rc)
        if rs in (None,"","None") or q in (None,"","None"): continue
        try: rs=float(rs); q=float(q)
        except ValueError: continue
        xs.append(rs); ys.append(q)
        key=(r["pdb_id"], r.get("ic1",""), r.get("ic2",""))
        groups[key].append((rs,q))
    wr,lo,hi,ntg=within_target_r(groups)
    rows.append(dict(cat=cat,lab=lab,step=step,mean=mean,conf=conf,oracle=ora,
                     r=wr,lo=lo,hi=hi,n_targets=ntg,r_pooled=pear(xs,ys)))

fig,(axb,axr)=plt.subplots(1,2,figsize=(13.2,5.4),dpi=150,gridspec_kw={"width_ratios":[5,3],"wspace":0.32})
fig.patch.set_facecolor("#fcfcfb")
y=np.arange(len(rows))[::-1]
axb.set_facecolor("#fcfcfb")
for yy,r in zip(y,rows):
    axb.plot([r["conf"],r["oracle"]],[yy,yy],color="#c9cdd2",lw=6,zorder=1,solid_capstyle="round")
    axb.scatter(r["conf"],yy,color=C_CONF,s=75,zorder=4,marker="D")
    axb.scatter(r["oracle"],yy,facecolors="none",edgecolors=C_ORA,s=95,lw=2,zorder=3)
axb.set_yticks(y); axb.set_yticklabels([r["lab"] for r in rows],fontsize=10.5,color=INK)
axb.set_xlim(0,1); axb.set_xlabel("TM (monomers) / DockQ success-rate (interfaces)",fontsize=10.5,color=INK)
axb.set_title("rank-1 (deployed) → oracle (ceiling): headroom rank-1 leaves",fontsize=11.5,color=INK,pad=8)
from matplotlib.lines import Line2D
axb.legend(handles=[Line2D([0],[0],marker='D',color='none',markerfacecolor=C_CONF,markersize=9,label="rank-1 (ranking_score)"),
                    Line2D([0],[0],marker='o',color='none',markeredgecolor=C_ORA,markerfacecolor='none',markersize=10,label="oracle (best sample)")],
           loc="center right",bbox_to_anchor=(0.995,0.42),fontsize=9,framealpha=0.95)
for s_ in ("top","right"): axb.spines[s_].set_visible(False)
for s_ in ("left","bottom"): axb.spines[s_].set_color(GRID)
axb.tick_params(colors=INK2); axb.grid(axis="x",color=GRID,lw=0.5,alpha=0.5)

# right: within-target correlation with 95% CI. A CI that spans 0 means the
# ranking is indistinguishable from picking at random inside a target.
axr.set_facecolor("#fcfcfb")
rv=[r["r"] for r in rows]
def _col(r):
    if r["r"]!=r["r"] or r["lo"]!=r["lo"]: return "#c9cdd2"
    if r["lo"]<=0: return "#e34948"          # CI spans 0 → no usable signal
    return C_CONF if r["r"]>=0.3 else "#e0a13c"   # significant but weak
cols=[_col(r) for r in rows]
axr.barh(y,[0 if v!=v else v for v in rv],0.6,color=cols,zorder=3)
for yy,r in zip(y,rows):
    if r["r"]!=r["r"]: continue
    axr.plot([r["lo"],r["hi"]],[yy,yy],color=INK,lw=1.4,zorder=5,solid_capstyle="butt")
    for e in (r["lo"],r["hi"]):
        axr.plot([e,e],[yy-0.13,yy+0.13],color=INK,lw=1.4,zorder=5)
    axr.text(max(r["hi"],r["r"])+0.025,yy,f"{r['r']:.2f}",va="center",fontsize=9,color=INK2)
axr.set_yticks(y); axr.set_yticklabels([]); axr.set_xlim(-0.25,1)
axr.axvline(0,color=INK2,lw=1)
axr.axvline(0.3,color=INK2,ls=":",lw=1)
axr.set_xlabel("within-target r(ranking_score, quality)  ± 95% CI",fontsize=10.5,color=INK)
axr.set_title("confidence calibration (within-target)",fontsize=12,color=INK,pad=8)
for s_ in ("top","right"): axr.spines[s_].set_visible(False)
for s_ in ("left","bottom"): axr.spines[s_].set_color(GRID)
axr.tick_params(colors=INK2); axr.grid(axis="x",color=GRID,lw=0.5,alpha=0.5)

steps_set=sorted(set(r["step"] for r in rows))
steptxt=f"step {steps_set[0]//1000}k" if len(steps_set)==1 else f"steps {min(steps_set)//1000}–{max(steps_set)//1000}k"
fig.suptitle(f"DeepFold3 r14 ({steptxt}) — rank-1 (ranking_score) sample selection",fontsize=13.5,color=INK,y=1.01)
fig.text(0.5,-0.06,
  "rank-1 = sample with top ranking_score (0.8·ipTM+0.2·pTM), chosen inside one target's sample pool — so calibration is measured WITHIN target\n"
  "(Pearson r per target/interface, averaged over targets in Fisher-z space; bars show the 95% CI of that mean).  Pooling all predictions across\n"
  "targets instead mixes in the between-target difficulty spread and buries the selection signal, so it is reported in the CSV but not used here.\n"
  "Red = CI spans 0 (no usable signal)   ·   amber = significant but < 0.3   ·   blue = r ≥ 0.3.",
  ha="center",va="top",fontsize=8.0,color=INK2,linespacing=1.5)
fig.savefig(f"{OUT}/confidence_calibration.png",facecolor=fig.get_facecolor(),bbox_inches="tight")
print("wrote confidence_calibration.png")

with open(f"{OUT}/confidence_data.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["category","step","mean","conf_top1","oracle",
                "r_within_target","r_within_ci_lo","r_within_ci_hi","n_targets","r_pooled"])
    rr=lambda v: "" if v!=v else round(v,3)
    for r in rows:
        w.writerow([r["cat"],r["step"],round(r["mean"],3),round(r["conf"],3),round(r["oracle"],3),
                    rr(r["r"]),rr(r["lo"]),rr(r["hi"]),r["n_targets"],rr(r["r_pooled"])])
print("wrote confidence_data.csv")
