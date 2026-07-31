"""What separates the antibody-antigen interfaces the model never gets right."""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

CAT, STEP, CUT = "interface_antibody_antigen", 30000, 0.23
ROOT = Path("bench/runs/df3v4r14")
DETAIL = ROOT/"eval"/CAT/f"step_{STEP}"/"detail"
INP = Path("bench/inputs")/CAT

lab = {}
for r in csv.DictReader(open(f"bench/data/targets/{CAT}.csv")):
    lab[(r["pdb_id"], r["interface_chain_id_1"], r["interface_chain_id_2"])] = (r["antibody_chain"], r["antigen_chain"])

dq = defaultdict(list)
for r in csv.DictReader(open(ROOT/"eval"/CAT/f"step_{STEP}"/"raw"/f"{CAT}_ost.csv")):
    try: dq[(r["pdb_id"], r["ic1"], r["ic2"])].append(float(r["dockq"]))
    except (TypeError, ValueError): pass

def a3m_depth(path):
    try:
        n = 0
        with open(path) as f:
            for line in f:
                if line.startswith(">"): n += 1
        return n
    except Exception:
        return None

def inputs_for(pdb):
    """chain id -> (seq_len, unpaired MSA depth, n templates)."""
    try: d = json.load(open(INP/f"{pdb}.json"))
    except Exception: return {}
    out = {}
    for e in d["sequences"]:
        p = e.get("protein")
        if not p: continue
        tp = p.get("templatesPath")
        ntpl = None
        if tp:
            try: ntpl = len(json.load(open(tp)))
            except Exception: ntpl = None
        ids = p["id"] if isinstance(p["id"], list) else [p["id"]]
        for cid in ids:  # AF3 dialect allows one entry to define several copies
            out[cid] = (len(p["sequence"]), a3m_depth(p.get("unpairedMsaPath")), ntpl)
    return out

# model chain ids in the input JSON are the same label ids OST reports
rows = []
inp_cache = {}
for key, dqs in dq.items():
    pdb, ic1, ic2 = key
    ab, ag = lab[key]
    n_pass = sum(1 for v in dqs if v >= CUT)
    grp = "never" if n_pass == 0 else "partial" if n_pass < 16 else "solved"
    if pdb not in inp_cache: inp_cache[pdb] = inputs_for(pdb)
    inf = inp_cache[pdb]
    nnat = None
    f = DETAIL/f"{pdb}_42_0_structure_ost.json"
    try:
        d = json.load(open(f))
        for i, iface in enumerate(d.get("dockq_interfaces") or []):
            if ic1 in iface and ic2 in iface:
                nn = d.get("nnat")
                nnat = nn[i] if isinstance(nn, list) and len(nn) > i else nn
                break
    except Exception:
        pass
    ab_i, ag_i = inf.get(ab, (None,)*3), inf.get(ag, (None,)*3)
    rows.append({"pdb": pdb, "ab": ab, "ag": ag, "group": grp, "nnat": nnat,
                 "ab_len": ab_i[0], "ab_msa": ab_i[1], "ab_tpl": ab_i[2],
                 "ag_len": ag_i[0], "ag_msa": ag_i[1], "ag_tpl": ag_i[2],
                 "n_iface_in_pdb": 0})

per_pdb = defaultdict(list)
for r in rows: per_pdb[r["pdb"]].append(r)
for pdb, rs in per_pdb.items():
    for r in rs: r["n_iface_in_pdb"] = len(rs)
    if len(rs) == 2:  # Fab: heavy = longer antibody chain
        a, b = rs
        if a["ab_len"] and b["ab_len"]:
            hi = a if a["ab_len"] >= b["ab_len"] else b
            lo = b if hi is a else a
            hi["hl"], lo["hl"] = "H", "L"
for r in rows: r.setdefault("hl", "single" if r["n_iface_in_pdb"] == 1 else "?")

def summarize(field, fmt="{:.1f}"):
    line = f"  {field:14s}"
    for g in ("never", "partial", "solved"):
        v = [r[field] for r in rows if r["group"] == g and r[field] is not None]
        line += f"  {g} n={len(v):3d} med " + (fmt.format(st.median(v)) if v else "  -  ")
    return line

print(f"{len(rows)} interfaces: " + ", ".join(f"{g}={sum(1 for r in rows if r['group']==g)}" for g in ("never","partial","solved")))
for f_, fmt in (("nnat", "{:.0f}"), ("ab_len", "{:.0f}"), ("ag_len", "{:.0f}"),
                ("ab_msa", "{:.0f}"), ("ag_msa", "{:.0f}"), ("ab_tpl", "{:.0f}"), ("ag_tpl", "{:.0f}")):
    print(summarize(f_, fmt))

print("\n  by antibody chain type (share of that class that is 'never'):")
for hl in ("H", "L", "single"):
    sub = [r for r in rows if r["hl"] == hl]
    if not sub: continue
    nev = sum(1 for r in sub if r["group"] == "never")
    print(f"    {hl:6s} n={len(sub):3d}  never {100*nev/len(sub):5.1f}%   median nnat "
          f"{st.median([r['nnat'] for r in sub if r['nnat'] is not None]):.0f}")

print("\n  is failure per-complex? (2-interface entries)")
both = [rs for rs in per_pdb.values() if len(rs) == 2]
agree = sum(1 for rs in both if (rs[0]["group"] == "never") == (rs[1]["group"] == "never"))
print(f"    {agree}/{len(both)} entries have both interfaces on the same side of 'never'")

print("\n  nnat quartiles by group:")
for g in ("never", "partial", "solved"):
    v = sorted(r["nnat"] for r in rows if r["group"] == g and r["nnat"] is not None)
    if v:
        q = lambda p: v[int(p*(len(v)-1))]
        print(f"    {g:8s} n={len(v):3d}  Q1 {q(.25):3.0f}  med {q(.5):3.0f}  Q3 {q(.75):3.0f}  min {v[0]:.0f}")
