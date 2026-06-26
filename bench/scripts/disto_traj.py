import json, glob, re, os, sys
import numpy as np, gemmi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def coords(cif):
    # distogram rep atom = CB (Gly→CA), matching featurizer.py token_to_rep_atom + loss.py gt_rep.
    # (NOT CA — distogram_loss is trained on CB-CB distances.)
    st=gemmi.read_structure(cif); st.setup_entities(); m=st[0]; d={}
    for ch in m:
        for res in ch:
            at={a.name:a.pos for a in res}
            p = at.get("CA") if res.name=="GLY" else (at.get("CB") or at.get("CA"))
            if p is not None: d[(ch.name,res.seqid.num)]=np.array([p.x,p.y,p.z])
    return d

GTC={}  # cache GT contact mask per (target, tuple of tokens)
def gt_iu(target, tc, tr):
    gt=coords(f"tmp/bench/{target}/_ref.cif"); N=len(tr)
    gtc=np.full((N,3),np.nan)
    for i,(c,r) in enumerate(zip(tc,tr)):
        if (c,r) in gt: gtc[i]=gt[(c,r)]
    v=~np.isnan(gtc).any(1)
    gtd=np.sqrt(((gtc[:,None]-gtc[None])**2).sum(-1))
    seq=np.abs(np.subtract.outer(np.array(tr),np.array(tr)))
    sc=np.array(tc)[:,None]==np.array(tc)[None]; vm=v[:,None]&v[None]
    return np.triu((gtd<8.0)&(seq>=6)&sc&vm)

def point(stepdir, target):
    samples=sorted(glob.glob(f"{stepdir}/seed-*_sample-*"))
    if not samples: return None
    f=f"{samples[0]}/confidences.json"
    if not os.path.exists(f): return None
    conf=json.load(open(f))
    cp=np.array(conf["contact_probs"]); dg=np.array(conf["distogram"])
    iu=gt_iu(target, list(conf["token_chain_ids"]), list(conf["token_res_ids"]))
    if iu.sum()==0: return None
    return np.median(dg[iu]), (cp[iu]>0.5).mean()

def traj(arm, target):
    out=[]
    for d in glob.glob(f"tmp/bench/{target}/{arm}/step_*_n30"):
        m=re.search(r"step_(\d+)_n30", d)
        if not m: continue
        p=point(d, target)
        if p: out.append((int(m.group(1)), p[0], p[1]))
    return sorted(out)

# usage: python bench/scripts/disto_traj.py [ARM_LEFT] [ARM_RIGHT]
#   default compares base df3v1 vs crop640; for option-(A) watch pass the continued arm.
ARM_LEFT  = sys.argv[1] if len(sys.argv) > 1 else "df3-25e5-c10-g35"
ARM_RIGHT = sys.argv[2] if len(sys.argv) > 2 else "df3-stage1-crop640-average"
targets=["9EDF","9OT9","29MW"]; col={"9EDF":"tab:green","9OT9":"tab:orange","29MW":"tab:red"}
fig,ax=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[3,2]})
print("=== distoMed (trunk predicted dist on GT β-contacts; <8Å=knows the sheet) ===\n")
for t in targets:
    b=traj(ARM_LEFT, t); c=traj(ARM_RIGHT, t)
    bs=[x[0] for x in b]; bd=[x[1] for x in b]
    cs=[x[0] for x in c]; cd=[x[1] for x in c]
    ax[0].plot(bs,bd,'-o',ms=3,color=col[t],label=t)
    ax[1].plot(cs,cd,'-o',ms=3,color=col[t],label=t)
    print(f"{t}  base 80k→110k distoMed: {bd[0]:.1f}→{bd[-1]:.1f}Å (min {min(bd):.1f}, max {max(bd):.1f})")
    print(f"{t}  crop640 0→2750 distoMed: {cd[0]:.1f}→{cd[-1]:.1f}Å")
    # print crop640 full
    print("     crop640: "+" ".join(f"{s}:{d:.1f}" for s,d in zip(cs,cd)))
    print()
for a,title in zip(ax,["base df3v1 (smooth_lddt ON)  80k–110k","crop640 (smooth_lddt OFF)  0–2.75k"]):
    a.axhline(8.0,ls='--',c='gray',lw=1,label='8Å (contact)')
    a.set_xlabel("training step"); a.set_ylabel("trunk distoMed on GT β-contacts (Å)")
    a.set_title(title); a.legend(fontsize=8); a.grid(alpha=0.3); a.set_ylim(6,19)
plt.tight_layout()
out="tmp/bench/ss_analysis_full/distoMed_trajectory.png"
plt.savefig(out,dpi=130); print("wrote",out)
