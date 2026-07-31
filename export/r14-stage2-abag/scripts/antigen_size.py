"""Is antigen size the driver, or a proxy for complex size?"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

CAT, STEP, CUT = "interface_antibody_antigen", 30000, 0.23
ROOT = Path("bench/runs/df3v4r14"); INP = Path("bench/inputs")/CAT

lab = {}
for r in csv.DictReader(open(f"bench/data/targets/{CAT}.csv")):
    lab[(r["pdb_id"], r["interface_chain_id_1"], r["interface_chain_id_2"])] = (r["antibody_chain"], r["antigen_chain"])
dq = defaultdict(list)
for r in csv.DictReader(open(ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv")):
    try: dq[(r["pdb_id"], r["ic1"], r["ic2"])].append(float(r["dockq"]))
    except (TypeError, ValueError): pass

def inputs_for(pdb):
    out, total = {}, 0
    try: d = json.load(open(INP/f"{pdb}.json"))
    except Exception: return out, 0
    for e in d["sequences"]:
        p = e.get("protein") or e.get("rna") or e.get("dna")
        if not p: continue
        ids = p["id"] if isinstance(p["id"], list) else [p["id"]]
        for cid in ids:
            out[cid] = len(p["sequence"]); total += len(p["sequence"])
    return out, total

cache, rows = {}, []
for key, dqs in dq.items():
    if not dqs: continue  # every sample's OST run failed for this interface
    pdb, ic1, ic2 = key
    ab, ag = lab[key]
    if pdb not in cache: cache[pdb] = inputs_for(pdb)
    lens, total = cache[pdb]
    if ag not in lens: continue
    rows.append({"pdb": pdb, "ag_len": lens[ag], "total": total,
                 "mean_dq": st.mean(dqs), "oracle_dq": max(dqs),
                 "never": sum(1 for v in dqs if v >= CUT) == 0})

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2+1
            i = j+1
        return r
    rx, ry = rk(x), rk(y); n=len(x); mx,my=sum(rx)/n,sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den else 0.0

print(f"n={len(rows)} interfaces")
print(f"Spearman(antigen length, mean DockQ)  = {spearman([r['ag_len'] for r in rows], [r['mean_dq'] for r in rows]):+.3f}")
print(f"Spearman(total tokens,   mean DockQ)  = {spearman([r['total'] for r in rows], [r['mean_dq'] for r in rows]):+.3f}")
print(f"Spearman(antigen length, total tokens)= {spearman([r['ag_len'] for r in rows], [r['total'] for r in rows]):+.3f}")

def table(field, bins, title):
    print(f"\n  {title}")
    print(f"    {'bin':>14s} {'n':>4s} {'never%':>7s} {'mean DockQ':>11s} {'oracle DockQ':>13s}")
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r[field] < hi]
        if not sub: continue
        name = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
        print(f"    {name:>14s} {len(sub):4d} "
              f"{100*sum(r['never'] for r in sub)/len(sub):7.1f} "
              f"{st.median([r['mean_dq'] for r in sub]):11.3f} {st.median([r['oracle_dq'] for r in sub]):13.3f}")

table("ag_len", [(0,100),(100,150),(150,250),(250,10**6)], "by ANTIGEN length")
table("total", [(0,500),(500,700),(700,900),(900,10**6)], "by TOTAL complex length (tokens)")

print("\n  stratified: does antigen length still separate within a complex-size band?")
for lo, hi, nm in ((0,700,"total < 700"), (700,10**6,"total >= 700")):
    sub = [r for r in rows if lo <= r["total"] < hi]
    if len(sub) < 10: continue
    med = st.median([r["ag_len"] for r in sub])
    small = [r for r in sub if r["ag_len"] <= med]; big = [r for r in sub if r["ag_len"] > med]
    print(f"    {nm:14s} n={len(sub):3d}  ag_len<={med:.0f}: never {100*sum(r['never'] for r in small)/len(small):5.1f}% (n={len(small)})"
          f"   ag_len>{med:.0f}: never {100*sum(r['never'] for r in big)/len(big):5.1f}% (n={len(big)})")
print("\n  stratified: does complex size still separate within an antigen-size band?")
for lo, hi, nm in ((0,150,"antigen < 150"), (150,10**6,"antigen >= 150")):
    sub = [r for r in rows if lo <= r["ag_len"] < hi]
    if len(sub) < 10: continue
    med = st.median([r["total"] for r in sub])
    small = [r for r in sub if r["total"] <= med]; big = [r for r in sub if r["total"] > med]
    print(f"    {nm:14s} n={len(sub):3d}  total<={med:.0f}: never {100*sum(r['never'] for r in small)/len(small):5.1f}% (n={len(small)})"
          f"   total>{med:.0f}: never {100*sum(r['never'] for r in big)/len(big):5.1f}% (n={len(big)})")
