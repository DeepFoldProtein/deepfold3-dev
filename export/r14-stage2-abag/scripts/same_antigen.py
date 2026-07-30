"""Does the model pick the epitope from the ANTIGEN alone, ignoring the antibody?

For antigens that recur across bench entries bound by different antibodies, the
real epitopes differ (that is what different antibodies do). If the predicted
epitopes agree with each other MORE than the real ones do, the model is choosing
a patch on the antigen surface without using CDR complementarity.

Epitope sets live in each antigen's own residue ordinals; the two entries'
antigens are joined by sequence alignment.
"""
import csv, json, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np

CAT, STEP, CONTACT, CUT = "interface_antibody_antigen", 30000, 10.0, 0.23
SEED, NSAMP = "42", 5
ROOT = Path("bench/runs/df3v4r14")
GT = Path("bench/data/ground_truths")
PRED, DETAIL = ROOT/"pred"/CAT/f"step_{STEP}", ROOT/"eval"/CAT/f"step_{STEP}"/"detail"
PAIRS = [("8h3x", "8weo"), ("7ua2", "7uvs"), ("7yv1", "8be3"), ("7yds", "8aom"),
         ("8aok", "8aom"), ("7yds", "8aok"), ("7yxu", "8gye")]

tgt = {}
for r in csv.DictReader(open(f"bench/data/targets/{CAT}.csv")):
    tgt.setdefault(r["pdb_id"], []).append((r["interface_chain_id_1"], r["interface_chain_id_2"],
                                            r["antibody_chain"], r["antigen_chain"]))
dq = defaultdict(list)
for r in csv.DictReader(open(ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv")):
    try: dq[(r["pdb_id"], r["ic1"], r["ic2"])].append(float(r["dockq"]))
    except (TypeError, ValueError): pass

def gt_res(pdb, label_chain):
    st_ = gemmi.read_structure(str(GT/f"{pdb}.cif"))
    return [x for ch in st_[0] for x in ch if x.subchain == label_chain]

def pts(residues):
    out = []
    for res in residues:
        at = res.find_atom("CB", "*") or res.find_atom("CA", "*") or (res[0] if len(res) else None)
        out.append([at.pos.x, at.pos.y, at.pos.z] if at else [1e6]*3)
    return np.array(out)

def contact_idx(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return set(np.where(d.min(1) < CONTACT)[0].tolist()), set(np.where(d.min(0) < CONTACT)[0].tolist())

def aln_map(a, b):
    """ordinal in a -> ordinal in b (both non-gap)."""
    r = gemmi.align_string_sequences(list(a), list(b), [])
    m, i, j = {}, 0, 0
    for x, y in zip(r.add_gaps(a, 1), r.add_gaps(b, 2)):
        if x != "-" and y != "-": m[i] = j
        if x != "-": i += 1
        if y != "-": j += 1
    return m

def model_to_ref(aln_entries, chain):
    for e in aln_entries or []:
        L = e.strip().split("\n")
        if len(L) >= 4 and L[0].split(":")[-1] == chain:
            ref, mdl, m, ri, mi = L[1], L[3], {}, 0, 0
            for a, b in zip(ref, mdl):
                if a != "-" and b != "-": m[mi] = ri
                if a != "-": ri += 1
                if b != "-": mi += 1
            return m
    return {}

def profile(pdb_short):
    pdb = f"{pdb_short}-assembly1"
    out = []
    for ic1, ic2, ab, ag in tgt[pdb]:
        key = (pdb, ic1, ic2)
        if key not in dq: continue
        agr, abr = gt_res(pdb, ag), gt_res(pdb, ab)
        if not agr or not abr: continue
        _, gt_epi = contact_idx(pts(abr), pts(agr))
        seq = gemmi.one_letter_code([r.name for r in agr])
        counter = Counter()
        used = 0
        for s in range(NSAMP):
            try:
                d = json.load(open(DETAIL/f"{pdb}_{SEED}_{s}_structure_ost.json"))
                cm = d.get("chain_mapping") or {}
                m = gemmi.read_structure(str(PRED/pdb/f"seed-{SEED}_sample-{s}"/"model.cif"))[0]
                mab, mag = m.find_chain(cm.get(ab, ab)), m.find_chain(cm.get(ag, ag))
                if mab is None or mag is None: continue
                _, p_epi = contact_idx(pts(list(mab)), pts(list(mag)))
                back = model_to_ref(d.get("aln"), ag)
            except Exception:
                continue
            counter.update({back[i] for i in p_epi if i in back})
            used += 1
        if not used: continue
        pred = {r for r, c in counter.items() if c >= used/2}
        out.append((f"{pdb_short} {ab}-{ag}", seq, gt_epi, pred, st.mean(dq[key])))
    return out

def jac(a, b): return len(a & b)/len(a | b) if (a | b) else float("nan")

print(f"{'pair':34s} {'GT-vs-GT':>9s} {'pred-vs-pred':>13s} {'self recall':>12s} {'cross recall':>13s}")
rows = []
for p1, p2 in PAIRS:
    for n1, s1, g1, r1, d1 in profile(p1):
        for n2, s2, g2, r2, d2 in profile(p2):
            m12 = aln_map(s1, s2)
            G1 = {m12[i] for i in g1 if i in m12}; R1 = {m12[i] for i in r1 if i in m12}
            gg, pp = jac(G1, g2), jac(R1, r2)
            self_rec = st.mean([len(r1 & g1)/len(g1) if g1 else 0, len(r2 & g2)/len(g2) if g2 else 0])
            cross_rec = st.mean([len(R1 & g2)/len(g2) if g2 else 0, len({m12[i] for i in r1 if i in m12} & g2)/len(g2) if g2 else 0])
            print(f"{n1+' | '+n2:34s} {gg:9.3f} {pp:13.3f} {self_rec:12.3f} {cross_rec:13.3f}   DockQ {d1:.2f}/{d2:.2f}")
            rows.append((gg, pp))
print(f"\nmean over {len(rows)} interface pairs:  GT-vs-GT {st.mean([r[0] for r in rows]):.3f}   "
      f"pred-vs-pred {st.mean([r[1] for r in rows]):.3f}")
