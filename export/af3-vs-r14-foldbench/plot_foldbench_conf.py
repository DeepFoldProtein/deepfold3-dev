"""FoldBench-small, confidence-having models only (AF3, Boltz-1/2, Protenix v1/v2, DF3r14).
rank-1 (top ranking_score) stacked CAPRI tiers + oracle (best sample) caret, per-target on the
designated interface. ppi + agab. OF3p2/EF2 excluded (no ranking_score)."""
import csv, json, glob, os, re, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

EXT="/tmp/claude-1000/-home-jovyan-work-deepfold3-dev/c22730e6-ba92-4965-a51b-c5b63c9bcce4/scratchpad/foldbench_ext"
R14="/home/jovyan/work/deepfold3-dev/bench/runs/df3v4r14/eval"
TGT="/home/jovyan/work/deepfold3-dev/bench/data/targets"
OUT="/home/jovyan/work/deepfold3-dev/export/af3-vs-r14-foldbench"
INK,INK2,GRID="#0b0b0b","#52514e","#d9d8d4"
TIER=["#bcd4ec","#5a9bd4","#1f5c9e"]
R14COL="#eb6834"; R14TIER=["#f6c3a6","#f0946a","#c1490f"]
MODELS=[("af3","AF3"),("b2","BT2"),("b1","BT1"),("ptx_v2","PXv2"),("ptx","PXv1")]
CATS=[("ppi","interface_protein_protein","protein–protein"),
      ("agab","interface_antibody_antigen","antibody–antigen")]

ap=argparse.ArgumentParser(); ap.add_argument("--step",type=int,default=None); A=ap.parse_args()
def latest_step(cat_key):
    st=[int(re.search(r"step_(\d+)",p).group(1)) for p in glob.glob(f"{R14}/{cat_key}/step_*/raw/{cat_key}_ost.csv")]
    return max(st) if st else None
steps_used=set()

def designated(cat_csv):
    return {r["pdb_id"]:frozenset([r["interface_chain_id_1"],r["interface_chain_id_2"]])
            for r in csv.DictReader(open(f"{TGT}/{cat_csv}.csv"))}

def ext_sel(model_stem, cat_key, desig):
    """per-target (rank1 dockq, oracle dockq) on designated interface."""
    d=f"{EXT}/{model_stem}_{cat_key}_full"
    if not os.path.isdir(d): d=f"{EXT}/{model_stem}_{cat_key}"
    # ranking_score per (pdb,seed,sample)
    rank={}
    prf=f"{d}/prediction_reference.csv"
    for r in csv.DictReader(open(prf)):
        try: rank[(r["pdb_id"],r["seed"],r["sample"])]=float(r["ranking_score"])
        except: pass
    by=defaultdict(list)  # pdb -> [(rankscore, dockq)]
    for f in glob.glob(f"{d}/detail/*.json"):
        base=os.path.basename(f).replace("_full.json","")
        parts=base.split("_")            # {pdb-assembly1}_{seed}_{sample}
        pdb="_".join(parts[:-2]); seed,samp=parts[-2],parts[-1]
        if pdb not in desig: continue
        try: js=json.load(open(f))
        except: continue
        refs=js.get("dockq_reference_interfaces") or []; dq=js.get("dockq") or []
        val=next((dq[i] for i,ri in enumerate(refs) if frozenset(ri)==desig[pdb] and i<len(dq)),None)
        if val is None: continue
        rs=rank.get((pdb,seed,samp))
        by[pdb].append((rs,float(val)))
    out={}
    for pdb,v in by.items():
        if not v: continue
        vr=[x for x in v if x[0] is not None]
        r1=max(vr,key=lambda x:x[0])[1] if vr else np.mean([x[1] for x in v])
        orc=max(x[1] for x in v)
        out[pdb]=(r1,orc)
    return out

def r14_sel(cat_key, desig, step):
    raw=list(csv.DictReader(open(f"{R14}/{cat_key}/step_{step}/raw/{cat_key}_ost.csv")))
    by=defaultdict(list)
    for r in raw:
        pdb=r["pdb_id"]
        if pdb not in desig or frozenset([r["ic1"],r["ic2"]])!=desig[pdb]: continue
        rs=r.get("ranking_score"); rs=float(rs) if rs not in (None,"","None") else None
        try: by[pdb].append((rs,float(r["dockq"])))
        except: pass
    out={}
    for pdb,v in by.items():
        if not v: continue
        vr=[x for x in v if x[0] is not None]
        r1=max(vr,key=lambda x:x[0])[1] if vr else np.mean([x[1] for x in v])
        out[pdb]=(r1,max(x[1] for x in v))
    return out

def tiers(vals):
    n=len(vals)
    if not n: return 0,0,0,0
    a=sum(1 for d in vals if 0.23<=d<0.49)/n*100; m=sum(1 for d in vals if 0.49<=d<0.80)/n*100
    h=sum(1 for d in vals if d>=0.80)/n*100; return a,m,h,a+m+h

results={}; common={}; steps={}
for short,cat_key,cat_name in CATS:
    desig=designated(cat_key)
    step=A.step or latest_step(cat_key); steps_used.add(step); steps[short]=step
    pm={"DF3r14":(r14_sel(cat_key,desig,step),True)}
    for stem,lbl in MODELS: pm[lbl]=(ext_sel(stem,short,desig),False)
    comm=set.intersection(*[set(d.keys()) for d,_ in pm.values()]); common[short]=len(comm)
    rows=[]
    for lbl,(d,ours) in pm.items():
        r1=[d[p][0] for p in comm]; orc=[d[p][1] for p in comm]
        rows.append((lbl,tiers(r1),tiers(orc)[3],ours))
    rows.sort(key=lambda r:-r[1][3])
    results[short]=rows
    print(f"[{cat_name}] common={len(comm)}")
    for lbl,t,osucc,ours in rows: print(f"    {lbl:8s} rank1={t[3]:5.1f}  oracle={osucc:5.1f}  (A{t[0]:.0f}/M{t[1]:.0f}/H{t[2]:.0f})")

fig,axes=plt.subplots(1,2,figsize=(14,5.0),dpi=150,gridspec_kw={"wspace":0.4})
fig.patch.set_facecolor("#fcfcfb")
for ax,(short,cat_key,cat_name) in zip(axes,CATS):
    rows=results[short]; y=np.arange(len(rows))[::-1]
    ax.set_facecolor("#fcfcfb")
    for yy,(lbl,t,osucc,ours) in zip(y,rows):
        cols=R14TIER if ours else TIER; left=0
        for frac,col in zip(t[:3],cols):
            ax.barh(yy,frac,0.6,left=left,color=col,edgecolor="white",lw=0.6,zorder=3); left+=frac
        ax.plot([osucc,osucc],[yy-0.3,yy+0.3],color=(R14COL if ours else INK),lw=1.8,ls=(0,(2,1)),zorder=5)
        ax.text(min(max(left,osucc)+1.3,88),yy,f"{t[3]:.0f}%→{osucc:.0f}%",va="center",ha="left",fontsize=8.6,
                color=(R14COL if ours else INK2),fontweight=("bold" if ours else "normal"))
    ax.set_yticks(y); ax.set_yticklabels([("★ "+r[0]) if r[3] else r[0] for r in rows],fontsize=10.5,color=INK)
    for tk,r in zip(ax.get_yticklabels(),rows):
        if r[3]: tk.set_color(R14COL); tk.set_fontweight("bold")
    ax.set_xlim(0,100); ax.set_xlabel("DockQ success rate %  (rank-1 tiers → oracle ┆)",fontsize=9.5,color=INK)
    ax.set_title(f"{cat_name}   (n={common[short]})",fontsize=12,color=INK,pad=8)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x",color=GRID,lw=0.5,alpha=0.5,zorder=0)
axes[1].legend(handles=[Patch(color=TIER[0],label="Acceptable"),Patch(color=TIER[1],label="Medium"),
                        Patch(color=TIER[2],label="High")],loc="center right",bbox_to_anchor=(1.0,0.5),
               fontsize=8.5,framealpha=0.96,title="CAPRI tier",title_fontsize=8.5)
fig.suptitle("FoldBench-small — confidence-having models — rank-1 (deployed) → oracle (ceiling)",fontsize=13,color=INK,y=1.02)
_ns={}
for _short,_ck,_cn in CATS:
    _per={}
    for _r in csv.DictReader(open(f"{R14}/{_ck}/step_{steps[_short]}/raw/{_ck}_ost.csv")):
        _per.setdefault(_r["pdb_id"], set()).add((_r["seed"], _r["sample"]))
    _ns[_short]=max((len(v) for v in _per.values()), default=0)
_budget=("Sample budget matched: 20 predictions/target (4 seeds x 5) for all models."
         if set(_ns.values())=={20} else
         "SAMPLE BUDGET NOT MATCHED - external fold 20/target; DF3r14 folds "
         + ", ".join(f"{n} {_ns[s]}" for s,_,n in CATS) + " (oracle is monotone in it).")
_st=sorted(steps_used); _sttxt=f"step {_st[0]//1000}k" if len(_st)==1 else f"steps {min(_st)//1000}–{max(_st)//1000}k"
fig.text(0.5,-0.055,"Per-target DockQ on FoldBench-designated interface. rank-1 = sample with top ranking_score; oracle ┆ = best sample "
         f"{_budget}  ★ = DF3r14 ({_sttxt}).  AF3/BT=Boltz/PX=Protenix.",
         ha="center",fontsize=7.7,color=INK2)
fig.savefig(f"{OUT}/foldbench_conf_models.png",facecolor=fig.get_facecolor(),bbox_inches="tight")
print("wrote foldbench_conf_models.png")
with open(f"{OUT}/foldbench_conf_data.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["category","r14_step","model","n","rank1_A%","rank1_M%","rank1_H%","rank1_succ%","oracle_succ%"])
    for short,cat_key,cat_name in CATS:
        for lbl,t,osucc,ours in results[short]:
            w.writerow([cat_name,steps[short],lbl,common[short],round(t[0],1),round(t[1],1),round(t[2],1),round(t[3],1),round(osucc,1)])
print("wrote foldbench_conf_data.csv")
