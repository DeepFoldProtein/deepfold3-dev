"""What separates the protein-protein interfaces the model never gets right.

Same treatment as the antibody-antigen side, plus the two features that only
exist here: whether the interface is homomeric (its pairing is trivial) and how
deep the PAIRED MSA is (the co-evolution signal antibody-antigen cannot have).
"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

CAT, STEP, CUT = "interface_protein_protein", 30000, 0.23
ROOT = Path("bench/runs/df3v4r14"); INP = Path("bench/inputs")/CAT
PAIRED = Path("bench/data/msa/paired")
DETAIL = ROOT/"eval"/CAT/f"step_{STEP}"/"detail"

dq = defaultdict(list)
for r in csv.DictReader(open(ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv")):
    try: dq[(r["pdb_id"], r["ic1"], r["ic2"])].append(float(r["dockq"]))
    except (TypeError, ValueError): pass

def depth(p):
    try:
        n = 0
        with open(p) as f:
            for line in f:
                if line.startswith(">"): n += 1
        return n
    except Exception:
        return None

def inputs_for(pdb):
    out, total = {}, 0
    try: d = json.load(open(INP/f"{pdb}.json"))
    except Exception: return out, 0
    for e in d["sequences"]:
        p = e.get("protein") or e.get("rna") or e.get("dna")
        if not p: continue
        ids = p["id"] if isinstance(p["id"], list) else [p["id"]]
        tp = p.get("templatesPath")
        ntpl = None
        if tp:
            try: ntpl = len(json.load(open(tp)))
            except Exception: ntpl = None
        for cid in ids:
            out[cid] = {"len": len(p["sequence"]), "seq": p["sequence"], "tpl": ntpl,
                        "msa": depth(p.get("unpairedMsaPath")),
                        "paired": depth(PAIRED/pdb/f"{pdb}__{cid}.paired.a3m")}
            total += len(p["sequence"])
    return out, total

cache, rows = {}, []
for (pdb, ic1, ic2), dqs in dq.items():
    if not dqs: continue
    if pdb not in cache: cache[pdb] = inputs_for(pdb)
    lens, total = cache[pdb]
    # OST names a duplicate chain "A-2"; the input JSON has only the base id.
    c1, c2 = ic1.split("-")[0], ic2.split("-")[0]
    a, b = lens.get(c1), lens.get(c2)
    if not a or not b: continue
    nnat = None
    try:
        d = json.load(open(DETAIL/f"{pdb}_42_0_structure_ost.json"))
        for i, iface in enumerate(d.get("dockq_interfaces") or []):
            if ic1 in iface and ic2 in iface:
                nn = d.get("nnat"); nnat = nn[i] if isinstance(nn, list) and len(nn) > i else nn
                break
    except Exception:
        pass
    n_pass = sum(1 for v in dqs if v >= CUT)
    rows.append({
        "pdb": pdb, "group": "never" if n_pass == 0 else "partial" if n_pass < 16 else "solved",
        "homo": a["seq"] == b["seq"], "nnat": nnat, "total": total,
        "min_len": min(a["len"], b["len"]), "max_len": max(a["len"], b["len"]),
        "min_msa": min(x for x in (a["msa"], b["msa"]) if x is not None) if a["msa"] and b["msa"] else None,
        "min_paired": min(x for x in (a["paired"], b["paired"]) if x is not None) if a["paired"] and b["paired"] else None,
        "min_tpl": min(x for x in (a["tpl"], b["tpl"]) if x is not None) if a["tpl"] is not None and b["tpl"] is not None else None,
        "mean_dq": st.mean(dqs), "never": n_pass == 0,
    })

G = ("never", "partial", "solved")
print(f"n={len(rows)}  " + ", ".join(f"{g}={sum(1 for r in rows if r['group']==g)}" for g in G))
print("\n  medians by group")
for f_, fmt in (("nnat","{:.0f}"),("min_len","{:.0f}"),("max_len","{:.0f}"),("total","{:.0f}"),
                ("min_msa","{:.0f}"),("min_paired","{:.0f}"),("min_tpl","{:.0f}")):
    line = f"    {f_:11s}"
    for g in G:
        v = [r[f_] for r in rows if r["group"] == g and r[f_] is not None]
        line += f"  {g} n={len(v):3d} " + (fmt.format(st.median(v)) if v else " - ").rjust(7)
    print(line)

print("\n  homomeric vs heteromeric")
for h, nm in ((True, "homomer"), (False, "heteromer")):
    sub = [r for r in rows if r["homo"] == h]
    if not sub: continue
    print(f"    {nm:10s} n={len(sub):3d}  never {100*sum(r['never'] for r in sub)/len(sub):5.1f}%   "
          f"median mean-DockQ {st.median([r['mean_dq'] for r in sub]):.3f}   median paired MSA "
          f"{st.median([r['min_paired'] for r in sub if r['min_paired'] is not None]):.0f}")

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
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den else 0.0

print("\n  Spearman vs mean DockQ")
for f_ in ("total", "max_len", "min_len", "nnat", "min_msa", "min_paired"):
    v = [(r[f_], r["mean_dq"]) for r in rows if r[f_] is not None]
    print(f"    {f_:11s} {spearman([a for a,_ in v],[b for _,b in v]):+.3f}   (n={len(v)})")

def table(field, bins, title):
    print(f"\n  {title}")
    for lo, hi in bins:
        sub = [r for r in rows if r[field] is not None and lo <= r[field] < hi]
        if not sub: continue
        nm = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"    {nm:>12s} n={len(sub):3d}  never {100*sum(r['never'] for r in sub)/len(sub):5.1f}%  "
              f"median mean-DockQ {st.median([r['mean_dq'] for r in sub]):.3f}")

table("total", [(0,400),(400,700),(700,1000),(1000,10**9)], "by TOTAL complex length")
table("min_paired", [(0,10),(10,100),(100,1000),(1000,10**9)], "by PAIRED MSA depth (shallower chain)")
table("nnat", [(0,20),(20,40),(40,80),(80,10**9)], "by native contact count")
