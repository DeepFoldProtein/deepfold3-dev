"""One-time: parse the external FoldBench results (fixed reference models) into a small
per-target cache, so the per-checkpoint updater never re-parses thousands of JSONs.

Cache = per-target MEAN DockQ on the FoldBench-designated interface, per model, per category.
Re-run only if foldbench_results.tar.gz changes. Extracts the tarball to a temp dir if the
scratchpad extraction is absent (so it works in a fresh session).

Output: export/af3-vs-r14-foldbench/foldbench_ext_cache.json
"""
import csv, json, glob, os, tarfile, tempfile
from collections import defaultdict
import numpy as np

REPO="/home/jovyan/work/deepfold3-dev"
TGT=f"{REPO}/bench/data/targets"
OUT=f"{REPO}/export/af3-vs-r14-foldbench"
TARBALL=f"{REPO}/foldbench_results.tar.gz"
SCRATCH="/tmp/claude-1000/-home-jovyan-work-deepfold3-dev/c22730e6-ba92-4965-a51b-c5b63c9bcce4/scratchpad/foldbench_ext"

# stem -> display label
MODELS={"af3":"AF3","b2":"BT2","b1":"BT1","ptx_v2":"PXv2","ptx":"PXv1","of3":"OF3p2","ef2":"EF2"}
CATS=[("ppi","interface_protein_protein"),("agab","interface_antibody_antigen")]

def find_ext_root():
    if os.path.isdir(SCRATCH) and glob.glob(f"{SCRATCH}/*_ppi_full"):
        return SCRATCH, None
    tmp=tempfile.mkdtemp(prefix="fb_ext_")
    print(f"[precompute] extracting {TARBALL} -> {tmp} (scratch absent)")
    with tarfile.open(TARBALL) as t: t.extractall(tmp)
    return tmp, tmp

def designated(cat_full):
    return {r["pdb_id"]:frozenset([r["interface_chain_id_1"],r["interface_chain_id_2"]])
            for r in csv.DictReader(open(f"{TGT}/{cat_full}.csv"))}

def per_target(root, stem, cat_short, desig):
    d=f"{root}/{stem}_{cat_short}_full"
    if not os.path.isdir(d): d=f"{root}/{stem}_{cat_short}"
    by=defaultdict(list)
    for f in glob.glob(f"{d}/detail/*.json"):
        pdb=os.path.basename(f).split("-assembly1")[0]+"-assembly1"
        if pdb not in desig: continue
        try: js=json.load(open(f))
        except Exception: continue
        refs=js.get("dockq_reference_interfaces") or []; dq=js.get("dockq") or []
        val=next((dq[i] for i,ri in enumerate(refs) if frozenset(ri)==desig[pdb] and i<len(dq)),None)
        if val is not None: by[pdb].append(float(val))
    # per target: [mean over samples, oracle (best sample)]
    return {p:[round(float(np.mean(v)),4),round(float(np.max(v)),4)] for p,v in by.items() if v}

root, cleanup = find_ext_root()
cache={"_meta":{"models":MODELS,"metric":"per-target [mean, oracle(max)] DockQ on designated interface",
                "value_format":"[mean, oracle]","source":os.path.basename(TARBALL)}}
for short, full in CATS:
    desig=designated(full)
    cache[short]={"_designated_n":len(desig)}
    for stem,lbl in MODELS.items():
        pt=per_target(root,stem,short,desig)
        cache[short][stem]=pt
        print(f"[{short}] {lbl}: {len(pt)} targets")
json.dump(cache, open(f"{OUT}/foldbench_ext_cache.json","w"))
print(f"wrote {OUT}/foldbench_ext_cache.json")
if cleanup:
    import shutil; shutil.rmtree(cleanup, ignore_errors=True)
