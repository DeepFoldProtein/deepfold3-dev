"""Does per-interface chain-pair ipTM rank samples better than the global score?"""
import csv, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

CUT = 0.23
ROOT = Path("bench/runs/df3v4r14")

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2+1
            i = j+1
        return r
    rx, ry = rk(x), rk(y); n=len(x); mx,my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den = (sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den else None

def run(cat, step):
    raw = ROOT/"eval"/cat/f"step_{step}"/"raw"/f"{cat}_ost.csv"
    pred = ROOT/"pred"/cat/f"step_{step}"
    by = defaultdict(list)
    for r in csv.DictReader(open(raw)):
        try: r["dockq"] = float(r["dockq"])
        except (TypeError, ValueError): continue
        by[(r["pdb_id"], r["ic1"], r["ic2"])].append(r)

    detail = ROOT/"eval"/cat/f"step_{step}"/"detail"
    # OST reports interfaces by REFERENCE chain name; chain_pair_iptm is keyed by
    # MODEL chain name. ``chain_mapping`` (reference -> model) is what joins them,
    # and it is per (pdb, seed, sample) because OST re-maps for every model.
    def ref2model(pdb, seed, sample):
        f = detail/f"{pdb}_{seed}_{sample}_structure_ost.json"
        try: return json.load(open(f)).get("chain_mapping") or {}
        except Exception: return {}

    cache = {}
    def pair_iptm(pdb, seed, sample):
        key = (pdb, seed, sample)
        if key not in cache:
            p = pred/pdb/f"seed-{seed}_sample-{sample}"/"confidences.json"
            try: cache[key] = json.load(open(p))["summary"].get("chain_pair_iptm") or {}
            except Exception: cache[key] = {}
        return cache[key]

    rho_p, rho_g, pick_p, pick_g, orac, mean = [], [], [], [], [], []
    missing = 0
    for (pdb, ic1, ic2), rs in sorted(by.items()):
        vp, vg, dq = [], [], []
        for r in rs:
            m = pair_iptm(pdb, r["seed"], r["sample"])
            cm = ref2model(pdb, r["seed"], r["sample"])
            m1, m2 = cm.get(ic1, ic1), cm.get(ic2, ic2)
            v = m.get(f"{m1}-{m2}", m.get(f"{m2}-{m1}"))
            if v is None: continue
            vp.append(v); vg.append(float(r["ranking_score"])); dq.append(r["dockq"])
        if len(dq) < 5:
            missing += 1
            continue
        if len(set(vp)) > 1 and len(set(dq)) > 1:
            a, b = spearman(vp, dq), spearman(vg, dq)
            if a is not None and b is not None:
                rho_p.append(a); rho_g.append(b)
        pick_p.append(dq[max(range(len(dq)), key=lambda i: vp[i])])
        pick_g.append(dq[max(range(len(dq)), key=lambda i: vg[i])])
        orac.append(max(dq)); mean.append(sum(dq)/len(dq))

    n = len(orac)
    print(f"\n=== {cat} step {step}: {n} interfaces scored ({missing} skipped: no matching chain-pair key)")
    print(f"  within-target rho vs DockQ:  chain-pair ipTM {st.mean(rho_p):+.3f}   global ranking_score {st.mean(rho_g):+.3f}")
    for tag, p in (("random (mean)", mean), ("rank-1 global", pick_g), ("rank-1 chain-pair ipTM", pick_p), ("oracle", orac)):
        print(f"  {tag:24s} success {100*sum(1 for v in p if v>=CUT)/n:6.2f}%   mean DockQ {st.mean(p):.4f}")
    gain = (100*sum(1 for v in pick_p if v>=CUT)/n) - (100*sum(1 for v in pick_g if v>=CUT)/n)
    head = (100*sum(1 for v in orac if v>=CUT)/n) - (100*sum(1 for v in mean if v>=CUT)/n)
    print(f"  -> chain-pair vs global: {gain:+.2f} pt   (headroom oracle-random = {head:.2f} pt)")

for step in (28000, 29000, 30000):
    run("interface_antibody_antigen", step)
run("interface_protein_protein", 30000)
