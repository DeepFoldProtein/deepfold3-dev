"""Build FoldBench ``prediction_reference.csv`` from our inference tree.

FoldBench's ``evaluate.py`` consumes, per algorithm, a CSV:

    pdb_id, seed, sample, ranking_score, prediction_path

We map one df3 checkpoint to one FoldBench "algorithm" named ``step_{N}``,
so the eval dir layout is::

    bench/runs/{series}/eval/step_{N}/prediction_reference.csv

walking ``bench/runs/{series}/pred/step_{N}/{pdb}/`` for
``seed-*_sample-*/model.cif`` and pulling each sample's ranking_score from
the sibling ``ranking_scores.csv`` (written by inference-batch).

Usage::

    python bench/scripts/make_pred_ref.py --series df3v1 --step 90000
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
_SEED_SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")


def ranking_map(target_dir: Path) -> dict[tuple[int, int], float]:
    """(seed, sample) -> ranking_score from a target's ranking_scores.csv."""
    out: dict[tuple[int, int], float] = {}
    csv_path = target_dir / "ranking_scores.csv"
    if not csv_path.exists():
        return out
    for r in csv.DictReader(open(csv_path)):
        try:
            out[(int(r["seed"]), int(r["sample"]))] = float(r["ranking_score"])
        except (KeyError, ValueError):
            continue
    return out


def build(series: str, step: int, category: str) -> Path:
    pred_root = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    eval_dir = BENCH / "runs" / series / "eval" / category / f"step_{step}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / "prediction_reference.csv"

    rows: list[dict] = []
    n_targets = 0
    for target_dir in sorted(pred_root.iterdir()):
        if not target_dir.is_dir() or target_dir.name.startswith("_"):
            continue
        pdb_id = target_dir.name
        rmap = ranking_map(target_dir)
        found = False
        for sample_dir in sorted(target_dir.glob("seed-*_sample-*")):
            m = _SEED_SAMPLE.search(sample_dir.name)
            if not m:
                continue
            seed, sample = int(m.group(1)), int(m.group(2))
            cif = sample_dir / "model.cif"
            if not cif.exists():
                continue
            rows.append({
                "pdb_id": pdb_id,
                "seed": seed,
                "sample": sample,
                "ranking_score": rmap.get((seed, sample), 0.0),
                "prediction_path": str(cif.resolve()),
            })
            found = True
        n_targets += int(found)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "seed", "sample", "ranking_score", "prediction_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"[pred_ref] {series} step {step}: {n_targets} targets, {len(rows)} predictions -> {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--category", default="monomer_protein")
    args = ap.parse_args()
    build(args.series, args.step, args.category)


if __name__ == "__main__":
    main()
