"""Classify FoldBench monomer-protein targets by secondary-structure content.

Runs DSSP on each ground-truth CIF, collapses the 8-state labels to 3-state
(H = helix {H,G,I}, E = strand {E,B}, C = coil/other), and assigns a fold
class from the helix/strand fractions so prediction accuracy can be broken
down by class (the all-alpha vs all-beta split is where the FAPE/ss-learning
work showed strand recall lagging helix recall).

Fold-class rule (SCOP/CATH-style, on 3-state fractions):
    alpha       pct_H >= 0.15 and pct_E <  0.05
    beta        pct_E >= 0.15 and pct_H <  0.05
    alpha_beta  pct_H >= 0.05 and pct_E >= 0.05
    coil        otherwise (little regular SS)

Reuses ``run_dssp`` from scripts/dssp_f1.py (mkdssp + gemmi cif->pdb stub).

Output: bench/inputs/{category}_dssp.csv
    pdb_id, n_res, pct_H, pct_E, pct_C, fold_class

Usage::

    python bench/scripts/dssp_classify.py --category monomer_protein
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_ROOT = BENCH / "data" / "ground_truths"
sys.path.insert(0, str(REPO / "scripts"))  # for dssp_f1.run_dssp + its check_chirality import

from dssp_f1 import run_dssp  # noqa: E402

HELIX = {"H", "G", "I"}
STRAND = {"E", "B"}


def classify(cif: Path) -> tuple[int, float, float, float, str]:
    res = run_dssp(cif, cache=False)
    n = len(res)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, "unknown"
    h = sum(1 for _aa, ss in res if ss in HELIX)
    e = sum(1 for _aa, ss in res if ss in STRAND)
    pct_h, pct_e = h / n, e / n
    pct_c = 1.0 - pct_h - pct_e
    if pct_h >= 0.15 and pct_e < 0.05:
        cls = "alpha"
    elif pct_e >= 0.15 and pct_h < 0.05:
        cls = "beta"
    elif pct_h >= 0.05 and pct_e >= 0.05:
        cls = "alpha_beta"
    else:
        cls = "coil"
    return n, round(pct_h, 4), round(pct_e, 4), round(pct_c, 4), cls


def _job(pdb_id: str) -> dict:
    cif = GT_ROOT / f"{pdb_id}.cif"
    try:
        n, ph, pe, pc, cls = classify(cif)
    except Exception as ex:  # noqa: BLE001
        print(f"  FAIL {pdb_id}: {ex}", file=sys.stderr)
        return {"pdb_id": pdb_id, "n_res": 0, "pct_H": 0, "pct_E": 0, "pct_C": 0, "fold_class": "error"}
    return {"pdb_id": pdb_id, "n_res": n, "pct_H": ph, "pct_E": pe, "pct_C": pc, "fold_class": cls}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--small-only", action="store_true", default=True,
                    help="Only targets in the FoldBench-small manifest (default).")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    manifest = BENCH / "inputs" / f"{args.category}_manifest.csv"
    rows = list(csv.DictReader(open(manifest)))
    pdb_ids = [r["pdb_id"] for r in rows if (not args.small_only or int(r["in_small"]))]
    print(f"[dssp] classifying {len(pdb_ids)} {args.category} targets on {args.workers} workers ...")

    out_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, p) for p in pdb_ids]
        for fut in as_completed(futs):
            out_rows.append(fut.result())
    out_rows.sort(key=lambda r: r["pdb_id"])

    out_csv = BENCH / "inputs" / f"{args.category}_dssp.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "n_res", "pct_H", "pct_E", "pct_C", "fold_class"])
        w.writeheader()
        w.writerows(out_rows)

    from collections import Counter
    dist = Counter(r["fold_class"] for r in out_rows)
    print(f"[dssp] -> {out_csv}")
    print(f"[dssp] fold-class distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
