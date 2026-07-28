"""RETIRED — not run by update_foldbench.sh; its report section is frozen at step 23k.
Same reason as plot_overview.py: the AF3 side is the template-free `af3_ref` sweep, a
handicapped AlphaFold3. Kept runnable; see that script's header to revive.

protein-protein interfaces split by composition (homo/hetero) x size (dimer / N-mer≥3).
Left: 4 cross buckets. Right: marginals. Each row af3 vs r14, stacked CAPRI tiers
(rank-1) + oracle caret. Homo/hetero from entity grouping in the bench input JSONs.

  python plot_multimer_breakdown.py             # auto-detect latest r14 step
  python plot_multimer_breakdown.py --step 12000
"""
import argparse, csv, glob, json, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BENCH="/home/jovyan/work/deepfold3-dev/bench"
OUT="/home/jovyan/work/deepfold3-dev/export/af3-vs-r14-foldbench"
INP=f"{BENCH}/inputs/interface_protein_protein"
AF3,R14="#8a8f98","#2a78d6"; INK,INK2,GRID="#0b0b0b","#52514e","#d9d8d4"
TIER=["#bcd4ec","#5a9bd4","#1f5c9e"]

_c={}
def emap(pdb):
    if pdb in _c: return _c[pdb]
    d=json.load(open(f"{INP}/{pdb}.json")); m={}; n=0
    for i,s in enumerate(d["sequences"]):
        for k,v in s.items():
            ids=v.get("id"); ids=[ids] if isinstance(ids,str) else ids
            for c in ids: m[c]=i
            n+=len(ids)
    _c[pdb]=(m,n); return m,n
def bkt(pdb,ic1,ic2):
    m,n=emap(pdb); return ("homo" if m[ic1]==m[ic2] else "hetero"),("dimer" if n==2 else "N-mer")

def collect(series,step):
    raw=list(csv.DictReader(open(f"{BENCH}/runs/{series}/eval/interface_protein_protein/step_{step}/raw/interface_protein_protein_ost.csv")))
    by=defaultdict(list)
    for r in raw:
        try: dq=float(r["dockq"])
        except: continue
        rs=r.get("ranking_score"); rs=float(rs) if rs not in (None,"","None") else None
        by[(r["pdb_id"],r["ic1"],r["ic2"])].append((dq,rs))
    B=defaultdict(lambda:{"r1":[],"orc":[]})
    for (pdb,ic1,ic2),v in by.items():
        comp,size=bkt(pdb,ic1,ic2)
        vr=[x for x in v if x[1] is not None]
        r1=max(vr,key=lambda x:x[1])[0] if vr else sum(x[0] for x in v)/len(v)
        orc=max(x[0] for x in v)
        for key in [(comp,size),(comp,"all"),("all",size)]:
            B[key]["r1"].append(r1); B[key]["orc"].append(orc)
    return B
def tiers(lst):
    n=len(lst);
    if not n: return 0,0,0,0
    a=sum(1 for d in lst if 0.23<=d<0.49)/n*100; m=sum(1 for d in lst if 0.49<=d<0.80)/n*100
    h=sum(1 for d in lst if d>=0.80)/n*100; return a,m,h,a+m+h

_a=argparse.ArgumentParser(); _a.add_argument("--step",type=int,default=None); _a=_a.parse_args()
CAT="interface_protein_protein"
def latest_step():
    st=[int(re.search(r"step_(\d+)",p).group(1))
        for p in glob.glob(f"{BENCH}/runs/df3v4r14/eval/{CAT}/step_*/summary.csv")]
    if not st: raise SystemExit(f"no benched step for {CAT}")
    return max(st)
STEP=_a.step or latest_step()
def _nsamp(series, step):
    per={}
    for r in csv.DictReader(open(f"{BENCH}/runs/{series}/eval/{CAT}/step_{step}/raw/{CAT}_ost.csv")):
        per.setdefault(r["pdb_id"], set()).add((r["seed"], r["sample"]))
    return max((len(v) for v in per.values()), default=0)
_NS_AF3=_nsamp("af3_ref",0); _NS_R14=_nsamp("df3v4r14",STEP)
A=collect("af3_ref",0); R=collect("df3v4r14",STEP)

fig,(axc,axm)=plt.subplots(1,2,figsize=(14.5,5.6),dpi=150,gridspec_kw={"wspace":0.5})
fig.patch.set_facecolor("#fcfcfb")

def draw(ax,rows,title):
    y=np.arange(len(rows))[::-1]; hh=0.36
    labs=[]
    for idx,(key,lab) in enumerate(rows):
        yb=y[idx]; labs.append(f"{lab}\n(n={len(R[key]['r1'])})")
        for B,off,edge,nm in [(A,+hh/2,AF3,"AF3"),(R,-hh/2,R14,"r14")]:
            a,m,h,succ=tiers(B[key]["r1"]); _,_,_,osucc=tiers(B[key]["orc"])
            left=0
            for frac,col in [(a,TIER[0]),(m,TIER[1]),(h,TIER[2])]:
                ax.barh(yb+off,frac,hh,left=left,color=col,edgecolor=edge,lw=1.5,zorder=3); left+=frac
            ax.plot([osucc,osucc],[yb+off-hh/2,yb+off+hh/2],color=edge,lw=1.8,ls=(0,(2,1)),zorder=5)
            ax.text(min(max(left,osucc)+1.4,90),yb+off,f"{nm} {succ:.0f}%",va="center",ha="left",fontsize=7.8,
                    color=(R14 if nm=="r14" else INK2),fontweight=("bold" if nm=="r14" else "normal"))
    ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=10.5,color=INK)
    ax.set_xlim(0,100); ax.set_title(title,fontsize=12,color=INK,pad=8)
    ax.set_xlabel("DockQ success rate %  (stacked CAPRI tiers, rank-1;  ┆ = oracle)",fontsize=9.5,color=INK)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x",color=GRID,lw=0.5,alpha=0.5,zorder=0)

draw(axc,[(("homo","dimer"),"homo-dimer"),(("hetero","dimer"),"hetero-dimer"),
          (("homo","N-mer"),"homo-N-mer"),(("hetero","N-mer"),"hetero-N-mer")],
     "Composition × size")
draw(axm,[(("homo","all"),"homo (any size)"),(("hetero","all"),"hetero (any size)"),
          (("all","dimer"),"dimer (2 chains)"),(("all","N-mer"),"N-mer (≥3)")],
     "Marginals")
axc.legend(handles=[Patch(color=TIER[0],label="Acceptable"),Patch(color=TIER[1],label="Medium"),
                    Patch(color=TIER[2],label="High"),Patch(facecolor="white",edgecolor=AF3,lw=1.5,label="AF3 outline"),
                    Patch(facecolor="white",edgecolor=R14,lw=1.5,label="r14 outline")],
           loc="lower right",fontsize=7.8,framealpha=0.96,handlelength=1.3,labelspacing=0.35)

fig.suptitle(f"protein–protein interfaces by multimer type — AF3 vs DF3 r14 (step {STEP//1000}k), FoldBench-small",fontsize=13,color=INK,y=1.02)
fig.text(0.5,-0.04,"homo/hetero from entity grouping (shared sequence) in bench input JSONs; size = total polymer chains in assembly.  "
         f"MMseqs2/ColabFold MSA, {_NS_R14} predictions/target for r14 vs {_NS_AF3} for af3_ref, OST --lddt-no-stereochecks.  rank-1 = top ranking_score; oracle = best sample."
         + ("" if _NS_R14==_NS_AF3 else "  SAMPLE BUDGET NOT MATCHED - oracle is monotone in it."),
         ha="center",fontsize=7.8,color=INK2)
fig.savefig(f"{OUT}/multimer_breakdown.png",facecolor=fig.get_facecolor(),bbox_inches="tight")
print("wrote multimer_breakdown.png")

with open(f"{OUT}/multimer_breakdown_data.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["bucket","n","af3_rank1_succ","af3_oracle_succ","r14_rank1_succ","r14_oracle_succ"])
    for key,lab in [(("homo","dimer"),"homo-dimer"),(("hetero","dimer"),"hetero-dimer"),
                    (("homo","N-mer"),"homo-N-mer"),(("hetero","N-mer"),"hetero-N-mer"),
                    (("homo","all"),"homo-all"),(("hetero","all"),"hetero-all"),
                    (("all","dimer"),"all-dimer"),(("all","N-mer"),"all-N-mer")]:
        sc=lambda L:100*sum(1 for d in L if d>=0.23)/len(L) if L else 0
        w.writerow([lab,len(R[key]["r1"]),round(sc(A[key]["r1"]),0),round(sc(A[key]["orc"]),0),
                    round(sc(R[key]["r1"]),0),round(sc(R[key]["orc"]),0)])
print("wrote multimer_breakdown_data.csv")
