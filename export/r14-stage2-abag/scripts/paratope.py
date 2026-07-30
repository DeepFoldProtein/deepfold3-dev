"""Paratope vs epitope: which half of the antibody-antigen pairing is wrong?

The bench's target list labels antibody_chain / antigen_chain, and OST's detail
JSON carries a per-chain reference<->model sequence alignment, so GT and model
residues can be joined by ORDINAL position (the i-th residue of that chain in
structure order) instead of by residue number, which the two do not share.

  paratope recall = predicted antibody-side contacts that are real ones
  epitope  recall = predicted antigen-side contacts that are real ones

High paratope + zero epitope = right CDRs, wrong place on the antigen.
Low paratope = the antibody is not even binding through its CDRs.
"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

CAT, STEP, CUT, CONTACT = "interface_antibody_antigen", 30000, 0.23, 10.0
SEEDS = ("42",)  # 5 samples/target is enough for group-level tendencies
ROOT = Path("bench/runs/df3v4r14")
GT = Path("bench/data/ground_truths")
RAW = ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv"
PRED, DETAIL = ROOT/"pred"/CAT/f"step_{STEP}", ROOT/"eval"/CAT/f"step_{STEP}"/"detail"

labels = {}
for r in csv.DictReader(open(f"bench/data/targets/{CAT}.csv")):
    labels[(r["pdb_id"], r["interface_chain_id_1"], r["interface_chain_id_2"])] = (r["antibody_chain"], r["antigen_chain"])

def parse_aln(entries):
    """reference chain -> (aligned_ref_seq, aligned_model_seq)."""
    out = {}
    for e in entries or []:
        lines = e.strip().split("\n")
        if len(lines) < 4: continue
        out[lines[0].split(":")[-1]] = (lines[1], lines[3])
    return out

def ref2model_ordinal(aln_pair):
    """i-th GT residue -> j-th model residue (both structure order)."""
    ref, mdl = aln_pair
    m, ri, mi = {}, 0, 0
    for a, b in zip(ref, mdl):
        if a != "-" and b != "-": m[ri] = mi
        if a != "-": ri += 1
        if b != "-": mi += 1
    return m

def coords(model, chain, by_label=False):
    """CB (CA fallback) per residue, in structure order.

    Ground truths are keyed by label_asym_id (what OST reports as the reference
    chain name), which is NOT the auth chain name gemmi puts on Chain.name — for
    7txw the interface (A, C) lives on auth chains L and A. Predictions use one
    naming for both, so only the GT side needs the label lookup.
    """
    residues = []
    if by_label:
        residues = [r for ch in model for r in ch if r.subchain == chain]
    else:
        ch = model.find_chain(chain)
        residues = list(ch) if ch is not None else []
    if not residues: return None
    pts = []
    for res in residues:
        at = res.find_atom("CB", "*") or res.find_atom("CA", "*") or (res[0] if len(res) else None)
        pts.append([at.pos.x, at.pos.y, at.pos.z] if at else [1e6, 1e6, 1e6])
    return np.array(pts)

def contacts(model, c1, c2, by_label=False):
    a, b = coords(model, c1, by_label), coords(model, c2, by_label)
    if a is None or b is None or not len(a) or not len(b): return None, None
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return set(np.where(d.min(1) < CONTACT)[0].tolist()), set(np.where(d.min(0) < CONTACT)[0].tolist())

rows = list(csv.DictReader(open(RAW)))
by = defaultdict(list)
for r in rows:
    try: r["dockq"] = float(r["dockq"])
    except (TypeError, ValueError): continue
    by[(r["pdb_id"], r["ic1"], r["ic2"])].append(r)

groups = defaultdict(list)
for key, rs in by.items():
    n = sum(1 for r in rs if r["dockq"] >= CUT)
    groups["never (0/20)" if n == 0 else "partial (1-15/20)" if n < 16 else "solved (>=16/20)"].append((key, rs))

gt_cache = {}
for gname in ("never (0/20)", "partial (1-15/20)", "solved (>=16/20)"):
    par, epi, par_size, skipped = [], [], [], 0
    for (pdb, ic1, ic2), rs in groups[gname]:
        lab = labels.get((pdb, ic1, ic2))
        if lab is None: skipped += 1; continue
        ab, ag = lab
        if pdb not in gt_cache:
            try: gt_cache[pdb] = gemmi.read_structure(str(GT/f"{pdb}.cif"))[0]
            except Exception: gt_cache[pdb] = None
        gm = gt_cache[pdb]
        if gm is None: skipped += 1; continue
        gt_par, gt_epi = contacts(gm, ab, ag, by_label=True)
        if not gt_par or not gt_epi: skipped += 1; continue
        pr, ep = [], []
        for r in rs:
            if r["seed"] not in SEEDS: continue
            try:
                d = json.load(open(DETAIL/f"{pdb}_{r['seed']}_{r['sample']}_structure_ost.json"))
                cm = d.get("chain_mapping") or {}
                aln = parse_aln(d.get("aln"))
                mm = gemmi.read_structure(str(PRED/pdb/f"seed-{r['seed']}_sample-{r['sample']}"/"model.cif"))[0]
                p_par, p_epi = contacts(mm, cm.get(ab, ab), cm.get(ag, ag))
                if p_par is None: continue
                map_ab, map_ag = ref2model_ordinal(aln[ab]), ref2model_ordinal(aln[ag])
            except Exception:
                continue
            gp = {map_ab[i] for i in gt_par if i in map_ab}
            ge = {map_ag[i] for i in gt_epi if i in map_ag}
            if gp: pr.append(len(p_par & gp)/len(gp))
            if ge: ep.append(len(p_epi & ge)/len(ge))
        if pr: par.append(st.mean(pr)); par_size.append(len(gt_par))
        if ep: epi.append(st.mean(ep))
    print(f"\n### {gname}: {len(groups[gname])} interfaces ({len(par)} profiled, {skipped} skipped)")
    print(f"  paratope recall (antibody side): {st.mean(par):.3f}  median {st.median(par):.3f}")
    print(f"  epitope  recall (antigen side):  {st.mean(epi):.3f}  median {st.median(epi):.3f}")
    print(f"  GT paratope size: {st.mean(par_size):.1f} residues")
