"""What the 0/20 AbAg interfaces actually get wrong.

fnat / fnonnat come straight from OST's own per-interface DockQ decomposition
(matched exactly as eval_ost.parse_interface does), so the epitope question is
answered with OST's chain+residue mapping rather than a hand-rolled one:

  fnat    = fraction of NATIVE contacts the model reproduces (right epitope?)
  fnonnat = fraction of the model's contacts that are NOT native (invented ones)

Cross-sample epitope agreement stays hand-computed (model vs model, one
numbering, so it is safe): mean pairwise Jaccard of the 20 samples' contact sets.
"""
import csv, json, statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

CAT, STEP, CUT, CONTACT = "interface_antibody_antigen", 30000, 0.23, 10.0
ROOT = Path("bench/runs/df3v4r14")
RAW = ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv"
PRED, DETAIL = ROOT/"pred"/CAT/f"step_{STEP}", ROOT/"eval"/CAT/f"step_{STEP}"/"detail"

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def iface_fields(d, ic1, ic2):
    for i, iface in enumerate(d.get("dockq_interfaces") or []):
        if ic1 in iface and ic2 in iface:
            g = lambda k: (d.get(k) or [None]*(i+1))[i] if len(d.get(k) or []) > i else None
            return g("fnat"), g("fnonnat"), g("irmsd"), g("lrmsd")
    return None, None, None, None

def reps(model, chain):
    out = {}
    ch = model.find_chain(chain)
    if ch is None: return out
    for res in ch:
        at = res.find_atom("CB", "*") or res.find_atom("CA", "*") or (res[0] if len(res) else None)
        if at is not None: out[res.seqid.num] = (at.pos.x, at.pos.y, at.pos.z)
    return out

def contact_set(model, c1, c2):
    a, b = reps(model, c1), reps(model, c2)
    if not a or not b: return set()
    ka, va = list(a), np.array(list(a.values()))
    kb, vb = list(b), np.array(list(b.values()))
    d = np.linalg.norm(va[:, None] - vb[None], axis=-1)
    return {(c1, ka[i]) for i in np.where(d.min(1) < CONTACT)[0]} | {(c2, kb[j]) for j in np.where(d.min(0) < CONTACT)[0]}

rows = list(csv.DictReader(open(RAW)))
by = defaultdict(list)
for r in rows:
    if num(r["dockq"]) is not None: by[(r["pdb_id"], r["ic1"], r["ic2"])].append(r)

groups = defaultdict(list)
for key, rs in by.items():
    n = sum(1 for r in rs if num(r["dockq"]) >= CUT)
    groups["never (0/20)" if n == 0 else "partial (1-15/20)" if n < 16 else "solved (>=16/20)"].append((key, rs))

for gname in ("never (0/20)", "partial (1-15/20)", "solved (>=16/20)"):
    items = groups[gname]
    fnat, fnon, irms, lrms, agree, sd_dq, ilddt, lddt = [], [], [], [], [], [], [], []
    for (pdb, ic1, ic2), rs in items:
        f_s, fn_s, ir_s, lr_s = [], [], [], []
        sets = []
        for r in rs:
            try:
                d = json.load(open(DETAIL/f"{pdb}_{r['seed']}_{r['sample']}_structure_ost.json"))
            except Exception:
                continue
            f, fn, ir, lr = iface_fields(d, ic1, ic2)
            for dst, v in ((f_s, f), (fn_s, fn), (ir_s, ir), (lr_s, lr)):
                if v is not None: dst.append(v)
            cm = d.get("chain_mapping") or {}
            try:
                m = gemmi.read_structure(str(PRED/pdb/f"seed-{r['seed']}_sample-{r['sample']}"/"model.cif"))[0]
                s = contact_set(m, cm.get(ic1, ic1), cm.get(ic2, ic2))
                if s: sets.append(s)
            except Exception:
                pass
        if f_s: fnat.append(st.mean(f_s))
        if fn_s: fnon.append(st.mean(fn_s))
        if ir_s: irms.append(st.mean(ir_s))
        if lr_s: lrms.append(st.mean(lr_s))
        if len(sets) > 1:
            js = [len(a & b)/len(a | b) for a, b in combinations(sets, 2) if (a | b)]
            if js: agree.append(st.mean(js))
        dqs = [num(r["dockq"]) for r in rs]
        sd_dq.append(st.pstdev(dqs))
        for dst, k in ((ilddt, "ilddt"), (lddt, "lddt")):
            vs = [num(r[k]) for r in rs if num(r[k]) is not None]
            if vs: dst.append(st.mean(vs))
    print(f"\n### {gname}: {len(items)} interfaces")
    print(f"  fnat (native contacts recovered)  {st.mean(fnat):.3f}  median {st.median(fnat):.3f}   "
          f"fnat<0.05 on {100*sum(1 for v in fnat if v < 0.05)/len(fnat):.0f}% of them")
    print(f"  fnonnat (model contacts not native) {st.mean(fnon):.3f}")
    print(f"  iRMSD {st.mean(irms):6.2f} A   LRMSD {st.mean(lrms):6.2f} A")
    print(f"  whole-model lDDT {st.mean(lddt):.3f}   iLDDT {st.mean(ilddt):.3f}")
    print(f"  cross-sample contact agreement (Jaccard) {st.mean(agree):.3f}  median {st.median(agree):.3f}")
    print(f"  sd(DockQ) over 20 samples {st.mean(sd_dq):.4f}")
