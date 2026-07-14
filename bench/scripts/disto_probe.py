import json, sys, glob, os
import numpy as np
import gemmi

def coords(cif):
    # distogram rep atom = CB (Gly→CA), matching featurizer.py token_to_rep_atom + loss.py gt_rep.
    # (NOT CA — distogram_loss / contact_probs are CB-CB.)
    st = gemmi.read_structure(cif)
    st.setup_entities()
    m = st[0]
    d = {}
    for ch in m:
        for res in ch:
            at = {a.name: a.pos for a in res}
            p = at.get("CA") if res.name == "GLY" else (at.get("CB") or at.get("CA"))
            if p is not None:
                d[(ch.name, res.seqid.num)] = np.array([p.x, p.y, p.z])
    return d

def analyze(target, arm, step, contact_thr=0.5, seq_gap=6, dist_cut=8.0):
    base = f"tmp/bench/{target}/{arm}/step_{step}_n30"
    samples = sorted(glob.glob(f"{base}/seed-*_sample-*"))
    if not samples: return None
    # trunk belief: distogram identical across samples (trunk deterministic) -> take sample-0
    cpath = f"{samples[0]}/confidences.json"
    conf = json.load(open(cpath))
    # contact_probs / expected dist come from the per-seed distogram codec npz
    # (current layout), with a fallback to the inline fields for legacy JSON.
    from deepfold3.io.confidences import load_logit_maps_from_json
    maps = load_logit_maps_from_json(cpath)
    cp = np.asarray(maps["contact_probs"] if "contact_probs" in maps else conf["contact_probs"])  # P(d<8)
    dg = np.asarray(maps["distogram"] if "distogram" in maps else conf["distogram"])              # expected dist
    tchain = list(conf["token_chain_ids"]); tres = list(conf["token_res_ids"])
    N = len(tres)
    gt = coords(f"tmp/bench/{target}/_ref.cif")
    # GT Cα per token (mask missing)
    gtc = np.full((N,3), np.nan)
    for i,(c,r) in enumerate(zip(tchain,tres)):
        if (c,r) in gt: gtc[i]=gt[(c,r)]
    valid = ~np.isnan(gtc).any(1)
    # GT distance map + long-range contact set
    diff = gtc[:,None,:]-gtc[None,:,:]
    gtd = np.sqrt((diff**2).sum(-1))
    seq = np.abs(np.subtract.outer(np.array(tres), np.array(tres)))
    samechain = np.array(tchain)[:,None]==np.array(tchain)[None,:]
    vmask = valid[:,None]&valid[None,:]
    gt_contact = (gtd<dist_cut)&(seq>=seq_gap)&samechain&vmask
    iu = np.triu(gt_contact)
    n_gt = iu.sum()
    if n_gt==0: return dict(target=target,n_gt=0)
    # trunk recall: contact_probs high on GT long-range contacts
    trunk_rec = (cp[iu]>contact_thr).mean()
    trunk_mean_p = cp[iu].mean()
    trunk_dg_med = np.median(dg[iu])  # trunk expected dist on true contacts (should be small)
    # structure realization recall: averaged over all samples
    srecs=[]
    for s in samples:
        pm = coords(f"{s}/model.cif")
        pc = np.full((N,3),np.nan)
        for i,(c,r) in enumerate(zip(tchain,tres)):
            if (c,r) in pm: pc[i]=pm[(c,r)]
        pdiff=pc[:,None,:]-pc[None,:,:]; pd=np.sqrt((pdiff**2).sum(-1))
        srecs.append((pd[iu]<dist_cut).mean())
    struct_rec = float(np.mean(srecs))
    return dict(target=target, n_gt=int(n_gt), trunk_rec=float(trunk_rec),
                trunk_mean_p=float(trunk_mean_p), trunk_dg_med=float(trunk_dg_med),
                struct_rec=struct_rec, n_samples=len(samples))

# usage: python bench/scripts/disto_probe.py [ARM] [STEP1,STEP2,...]
arm = sys.argv[1] if len(sys.argv) > 1 else "df3-stage1-crop640-average"
_steps = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2500, 2750, 750]
print(f"{'target':>6} {'step':>5} | {'n_gt':>5} {'TRUNK recall':>12} {'mean P':>7} {'distoMed':>8} | {'STRUCT recall':>13}")
print("-"*72)
for step in _steps:
    for t in ["29MW","9OT9","9EDF"]:
        r=analyze(t,arm,step)
        if r and r.get("n_gt"):
            print(f"{t:>6} {step:>5} | {r['n_gt']:>5} {r['trunk_rec']*100:>11.1f}% {r['trunk_mean_p']:>7.3f} {r['trunk_dg_med']:>7.2f}Å | {r['struct_rec']*100:>12.1f}%")
    print()
