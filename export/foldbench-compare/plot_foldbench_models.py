"""FoldBench-small mean-DockQ comparison: DF3r14 vs external models (AF3, Boltz-1/2,
Protenix v1/v2, OpenFold-3 preview-2 [OF3p2], ESMFold2 [EF2]) on ppi + agab.

RE-RUNNABLE PER CHECKPOINT. External per-target DockQ is read from the fixed cache
(foldbench_ext_cache.json, built once by precompute_external.py); only r14 is recomputed
from the latest benched step. Metric = per-target MEAN DockQ over samples on the
FoldBench-designated interface, CAPRI-tiered. mean = N-unbiased.

Usage:
  python plot_foldbench_models.py            # auto-detect latest r14 step per category
  python plot_foldbench_models.py --step 12000
"""
import csv, json, glob, os, argparse, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO="/home/jovyan/work/deepfold3-dev"
R14=f"{REPO}/bench/runs/df3v4r14/eval"
TGT=f"{REPO}/bench/data/targets"
OUT=f"{REPO}/export/af3-vs-r14-foldbench"
CACHE=f"{OUT}/foldbench_ext_cache.json"
DATES=f"{OUT}/foldbench_target_dates.csv"   # per-target release dates (RCSB); for --after cutoff
INK,INK2,GRID="#0b0b0b","#52514e","#d9d8d4"
TIER=["#bcd4ec","#5a9bd4","#1f5c9e"]; R14COL="#eb6834"; R14TIER=["#f6c3a6","#f0946a","#c1490f"]
ORDER=[("af3","AF3"),("b2","BT2"),("b1","BT1"),("ptx_v2","PXv2"),("ptx","PXv1"),("of3","OF3p2"),("ef2","EF2")]
DAGGER={"BT2","PXv2","EF2"}   # substantially larger parameter count than the others; flagged with †
CATS=[("ppi","interface_protein_protein","protein–protein"),
      ("agab","interface_antibody_antigen","antibody–antigen")]

ap=argparse.ArgumentParser(); ap.add_argument("--step",type=int,default=None)
ap.add_argument("--after",default=None,help="keep only targets released AFTER this date (YYYY-MM-DD), e.g. 2023-06-01 (Boltz-2 cutoff)")
A=ap.parse_args(); AFTER=A.after; SFX=f"_post{AFTER}" if AFTER else ""

def load_dates():
    d={}
    if os.path.isfile(DATES):
        for r in csv.DictReader(open(DATES)):
            if r.get("released"): d[r["pdb_id"]]=r["released"]
    return d
DATED=load_dates() if AFTER else {}

def latest_step(cat_full):
    steps=[int(re.search(r"step_(\d+)",p).group(1))
           for p in glob.glob(f"{R14}/{cat_full}/step_*/raw/{cat_full}_ost.csv")]
    return max(steps) if steps else None

def designated(cat_full):
    return {r["pdb_id"]:frozenset([r["interface_chain_id_1"],r["interface_chain_id_2"]])
            for r in csv.DictReader(open(f"{TGT}/{cat_full}.csv"))}

def r14_per_target(cat_full, step, desig):
    raw=list(csv.DictReader(open(f"{R14}/{cat_full}/step_{step}/raw/{cat_full}_ost.csv")))
    by=defaultdict(list)
    for r in raw:
        pdb=r["pdb_id"]
        if pdb not in desig or frozenset([r["ic1"],r["ic2"]])!=desig[pdb]: continue
        try: by[pdb].append(float(r["dockq"]))
        except: pass
    return {p:[float(np.mean(v)),float(np.max(v))] for p,v in by.items() if v}  # [mean, oracle]

def _mo(v):  # normalize a cache/r14 value to [mean, oracle] (tolerate old scalar caches)
    return v if isinstance(v,(list,tuple)) else [v,v]

def tiers(vals):
    n=len(vals)
    if not n: return 0,0,0,0
    a=sum(1 for d in vals if 0.23<=d<0.49)/n*100; m=sum(1 for d in vals if 0.49<=d<0.80)/n*100
    h=sum(1 for d in vals if d>=0.80)/n*100; return a,m,h,a+m+h

def succ(vals):
    return sum(1 for d in vals if d>=0.23)/len(vals)*100 if vals else 0.0

cache=json.load(open(CACHE))
results={}; common={}; steps_used={}
for short,cat_full,cat_name in CATS:
    step=A.step or latest_step(cat_full)
    steps_used[short]=step
    desig=designated(cat_full)
    pm={"DF3r14":(r14_per_target(cat_full,step,desig),True)}
    for stem,lbl in ORDER:
        pm[lbl]=({k:v for k,v in cache[short][stem].items()},False)
    comm0=set.intersection(*[set(d.keys()) for d,_ in pm.values()])
    if AFTER:
        comm=set(p for p in comm0 if DATED.get(p,"")>AFTER)
        ndrop_undated=sum(1 for p in comm0 if not DATED.get(p))
        print(f"  [{short}] released>{AFTER}: kept {len(comm)}/{len(comm0)} targets (undated dropped: {ndrop_undated})")
    else:
        comm=comm0
    common[short]=len(comm)
    rows=[]
    for lbl,(d,ours) in pm.items():
        mo=[_mo(d[p]) for p in comm]
        mt=tiers([x[0] for x in mo])            # mean tiers (a,m,h,total)
        osucc=succ([x[1] for x in mo])          # oracle success rate
        rows.append((lbl,mt,osucc,len(comm),ours))
    rows.sort(key=lambda r:-r[1][3]); results[short]=rows
    print(f"[{cat_name}] step {step}, common n={len(comm)}")
    for lbl,t,osucc,n,ours in rows:
        print(f"    {lbl:8s} mean={t[3]:5.1f}  oracle={osucc:5.1f} (+{osucc-t[3]:4.1f})  A{t[0]:.0f}/M{t[1]:.0f}/H{t[2]:.0f}")

fig,axes=plt.subplots(1,2,figsize=(14.5,5.6),dpi=150,gridspec_kw={"wspace":0.42})
fig.patch.set_facecolor("#fcfcfb")
for ax,(short,cat_full,cat_name) in zip(axes,CATS):
    rows=results[short]; y=np.arange(len(rows))[::-1]; ax.set_facecolor("#fcfcfb")
    for yy,(lbl,t,osucc,n,ours) in zip(y,rows):
        cols=R14TIER if ours else TIER; left=0
        for frac,col in zip(t[:3],cols):
            ax.barh(yy,frac,0.62,left=left,color=col,edgecolor="white",lw=0.6,zorder=3); left+=frac
        acc=(R14COL if ours else INK2)
        # oracle (best-of-samples) headroom: light continuation from mean-end to oracle
        if osucc>left+0.2:
            ax.barh(yy,osucc-left,0.62,left=left,color=("#f3d9c8" if ours else "#e6e5e1"),
                    edgecolor="white",lw=0.6,zorder=2)
        ax.plot([osucc],[yy],marker="|",ms=13,mec=acc,mew=1.7,zorder=4)   # oracle tick
        ax.text(min(osucc+1.6,86),yy,f"{t[3]:.0f}%→{osucc:.0f}%",va="center",ha="left",fontsize=8.7,
                color=acc,fontweight=("bold" if ours else "normal"))
    def _lbl(r):
        s=r[0]+" †" if r[0] in DAGGER else r[0]
        return "★ "+s if r[4] else s
    ax.set_yticks(y); ax.set_yticklabels([_lbl(r) for r in rows],fontsize=10,color=INK)
    for tk,r in zip(ax.get_yticklabels(),rows):
        if r[4]: tk.set_color(R14COL); tk.set_fontweight("bold")
    ax.set_xlim(0,100); ax.set_xlabel("per-target DockQ success rate %   (solid = mean over samples; light = oracle headroom; CAPRI tiers)",fontsize=9,color=INK)
    ax.set_title(f"{cat_name}   (n={common[short]}{', released>'+AFTER if AFTER else ''}, r14 @ step {steps_used[short]})",fontsize=11.5,color=INK,pad=8)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x",color=GRID,lw=0.5,alpha=0.5,zorder=0)
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(color=TIER[0],label="Acceptable"),Patch(color=TIER[1],label="Medium"),
                    Patch(color=TIER[2],label="High"),
                    Patch(facecolor="#e6e5e1",edgecolor="#c7c6c2",label="mean→oracle headroom"),
                    Line2D([0],[0],marker="|",ls="none",mec=INK2,mew=1.7,ms=13,label="oracle success (best sample)")],
           loc="upper center",bbox_to_anchor=(0.5,0.02),ncol=5,
           fontsize=8.4,framealpha=0.96,columnspacing=1.7,handletextpad=0.5,frameon=False)
fig.suptitle("FoldBench-small — per-target DockQ, mean vs oracle — DF3r14 vs external models"+
             (f"   ·   targets released after {AFTER} (Boltz-2 cutoff)" if AFTER else ""),fontsize=13,color=INK,y=1.02)
cut_line=(f"Targets restricted to release date > {AFTER} (Boltz-2 training cutoff) — removes structures that may be in a model's training set.  Release dates: RCSB.\n" if AFTER else "")
fig.text(0.5,-0.055,
         cut_line+
         "Solid bar = per-target MEAN DockQ over all seed×sample predictions (N-unbiased), CAPRI-tiered.  Light bar + tick = ORACLE (best sample per target).  Labels: mean%→oracle%.\n"
         "Interface = FoldBench-designated.  Common target set = ∩ all models.   † = substantially larger parameter count.\n"
         "AF3=AlphaFold3, BT=Boltz, PX=Protenix, OF3p2=OpenFold-3 preview-2, EF2=ESMFold2 (multimer-capable), DF3r14=ours.",
         ha="center",va="top",fontsize=7.6,color=INK2,linespacing=1.5)
fig.savefig(f"{OUT}/foldbench_models_ppi_agab{SFX}.png",facecolor=fig.get_facecolor(),bbox_inches="tight")
print(f"wrote foldbench_models_ppi_agab{SFX}.png  (ppi@{steps_used['ppi']}, agab@{steps_used['agab']})")
with open(f"{OUT}/foldbench_models_data{SFX}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["category","r14_step","model","n","acceptable%","medium%","high%","mean_success%","oracle_success%","oracle_gain%"])
    for short,cat_full,cat_name in CATS:
        for lbl,t,osucc,n,ours in results[short]:
            w.writerow([cat_name,steps_used[short],lbl,n,round(t[0],1),round(t[1],1),round(t[2],1),round(t[3],1),round(osucc,1),round(osucc-t[3],1)])
print(f"wrote foldbench_models_data{SFX}.csv")
