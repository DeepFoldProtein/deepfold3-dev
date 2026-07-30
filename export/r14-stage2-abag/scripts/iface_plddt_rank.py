"""Can any saved confidence signal rank samples WITHIN a target?

Interface pLDDT = mean predicted lDDT over the residues of (ic1, ic2) that are
in contact in that sample's own model (CB-CB < 10 A). Unlike the global
ranking_score it is interface-local, so if within-target ranking is possible at
all from what the head already emits, it should show here.
"""
import csv, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

CAT = "interface_antibody_antigen"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
CUT, CONTACT = 0.23, 10.0
ROOT = Path("bench/runs/df3v4r14")
RAW = ROOT / "eval" / CAT / f"step_{STEP}" / "raw" / f"{CAT}_ost.csv"
PRED = ROOT / "pred" / CAT / f"step_{STEP}"

def rep_coords(model, chain):
    """(res_id -> CB (CA fallback) coord) for one chain."""
    out = {}
    for res in model[chain]:
        at = res.find_atom("CB", "*") or res.find_atom("CA", "*") or (res[0] if len(res) else None)
        if at is not None:
            out[res.seqid.num] = np.array([at.pos.x, at.pos.y, at.pos.z])
    return out

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2 + 1
            i = j+1
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = (sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den else None

rows = list(csv.DictReader(open(RAW)))
by = defaultdict(list)
for r in rows:
    try: r["dockq"] = float(r["dockq"])
    except (TypeError, ValueError): continue
    by[(r["pdb_id"], r["ic1"], r["ic2"])].append(r)

cache = {}
def sample_data(pdb, seed, sample):
    key = (pdb, seed, sample)
    if key not in cache:
        d = PRED / pdb / f"seed-{seed}_sample-{sample}"
        conf = json.load(open(d / "confidences.json"))
        plddt = {}
        for ch, rid, p in zip(conf["token_chain_ids"], conf["token_res_ids"], conf["token_plddts"]):
            plddt[(ch, rid)] = p
        cache[key] = (plddt, gemmi.read_structure(str(d / "model.cif"))[0])
    return cache[key]

rho_if, rho_glob, picks_if, picks_rs, oracle, means = [], [], [], [], [], []
for (pdb, ic1, ic2), rs in sorted(by.items()):
    vals_if, vals_rs, dqs = [], [], []
    for r in rs:
        try:
            plddt, model = sample_data(pdb, r["seed"], r["sample"])
            a, b = rep_coords(model, ic1), rep_coords(model, ic2)
        except Exception:
            continue
        if not a or not b:
            continue
        ka, va = list(a), np.stack(list(a.values()))
        kb, vb = list(b), np.stack(list(b.values()))
        d = np.linalg.norm(va[:, None, :] - vb[None, :, :], axis=-1)
        ia, ib = np.where(d.min(1) < CONTACT)[0], np.where(d.min(0) < CONTACT)[0]
        sel = [(ic1, ka[i]) for i in ia] + [(ic2, kb[i]) for i in ib]
        vp = [plddt[k] for k in sel if k in plddt]
        if not vp:
            continue
        vals_if.append(sum(vp)/len(vp)); vals_rs.append(float(r["ranking_score"])); dqs.append(r["dockq"])
    if len(dqs) < 5:
        continue
    if len(set(vals_if)) > 1 and len(set(dqs)) > 1:
        rho_if.append(spearman(vals_if, dqs))
        rho_glob.append(spearman(vals_rs, dqs))
    picks_if.append(dqs[max(range(len(dqs)), key=lambda i: vals_if[i])])
    picks_rs.append(dqs[max(range(len(dqs)), key=lambda i: vals_rs[i])])
    oracle.append(max(dqs)); means.append(sum(dqs)/len(dqs))

n = len(oracle)
print(f"{CAT} step {STEP}: {n} interfaces, {len(rho_if)} with variance\n")
print(f"  within-target Spearman vs DockQ:  interface pLDDT {st.mean(rho_if):+.3f}   ranking_score {st.mean(rho_glob):+.3f}")
print(f"  (median {st.median(rho_if):+.3f} / {st.median(rho_glob):+.3f};  frac rho>0: "
      f"{sum(1 for v in rho_if if v>0)/len(rho_if):.2f} / {sum(1 for v in rho_glob if v>0)/len(rho_glob):.2f})\n")
for tag, p in (("random (mean)", means), ("rank-1 ranking_score", picks_rs), ("rank-1 interface pLDDT", picks_if), ("oracle", oracle)):
    print(f"  {tag:24s} success {100*sum(1 for v in p if v>=CUT)/n:6.2f}%   mean DockQ {st.mean(p):.4f}")
