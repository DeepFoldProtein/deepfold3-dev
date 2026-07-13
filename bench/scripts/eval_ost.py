"""Evaluate df3 predictions with OpenStructure, FoldBench-faithful.

Calls the ``ost`` binary directly from our venv (the foldbench conda env's
bundled python can't import its own pip deps), parallelising across CPUs. The
``compare-structures`` command string and the JSON field extraction match
FoldBench exactly, so numbers are leaderboard-comparable (modulo the
no-stereochecks lDDT note below).

Two modes, by category:
  - monomer_*            : per-target LDDT / TM / GDT-TS
  - interface_protein_*  : per-interface DockQ / iRMSD / lRMSD (+ global LDDT),
                           success = DockQ >= 0.23 (FoldBench threshold)

df3v1 has NO trained confidence head, so FoldBench's ranking_score "rank"
metric is meaningless; we report confidence-free aggregates instead:
  mean   = mean over samples
  oracle = best sample by the metric itself (best-of-N upper bound)

Outputs under bench/runs/{series}/eval/step_{N}/:
  detail/*.json   raw OST   |   raw/{category}_ost.csv   per-sample(-interface)
  summary.csv     aggregates

Usage::

    python bench/scripts/eval_ost.py --series df3v1 --step 90000 --category monomer_protein
    python bench/scripts/eval_ost.py --series df3v1 --step 90000 --category interface_protein_protein
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import make_pred_ref  # sibling script

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_ROOT = BENCH / "data" / "ground_truths"
TARGETS_ROOT = BENCH / "data" / "targets"
OST = "/home/jovyan/.conda/envs/foldbench/bin/ost"

# Same OST command for both modes (--dockq already covers interfaces).
# --lddt-no-stereochecks: df3v1 is mid-training and emits bad local geometry;
# OST's default lDDT stereocheck DELETES those atoms, collapsing lDDT to ~0
# even when topology is partly right (TM/GDT/DockQ use Ca superposition and
# are unaffected). No-stereocheck lDDT tracks topology across our checkpoints
# (NOT directly comparable to the FoldBench leaderboard's stereocheck-on lDDT).
STRUCT_FLAGS = [
    "--fault-tolerant",
    "--min-pep-length",
    "4",
    "--min-nuc-length",
    "4",
    "--lddt",
    "--lddt-no-stereochecks",
    "--rigid-scores",
    "--tm-score",
    "--dockq",
]

INTERFACE_CATEGORIES = {
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_dna",
    "interface_protein_rna",
    "interface_protein_peptide",
}
DOCKQ_SUCCESS = 0.23


def run_ost(model: Path, ref: Path, out_json: Path) -> None:
    if out_json.exists():
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [OST, "compare-structures", "-m", str(model), "-r", str(ref), "-o", str(out_json), *STRUCT_FLAGS]
    subprocess.run(cmd, capture_output=True, check=False)


def parse_monomer(d: dict) -> dict:
    return {m: d.get(m) for m in ("lddt", "tm_score", "oligo_gdtts", "rmsd")}


def parse_interface(d: dict, ic1: str, ic2: str) -> dict:
    """DockQ/iRMSD/lRMSD for the (ic1, ic2) interface (FoldBench matching),
    plus the global (whole-model) lDDT / TM-score / GDT-TS."""
    out = {
        "dockq": None,
        "irmsd": None,
        "lrmsd": None,
        "lddt": d.get("lddt"),
        "tm_score": d.get("tm_score"),
        "oligo_gdtts": d.get("oligo_gdtts"),
    }
    interfaces = d.get("dockq_interfaces") or []
    for i, iface in enumerate(interfaces):
        if ic1 in iface and ic2 in iface:
            out["dockq"] = (d.get("dockq") or [None] * len(interfaces))[i]
            out["irmsd"] = (d.get("irmsd") or [None] * len(interfaces))[i]
            out["lrmsd"] = (d.get("lrmsd") or [None] * len(interfaces))[i]
            break
    return out


def load_interfaces(category: str) -> dict[str, list[tuple[str, str]]]:
    """pdb_id -> [(interface_chain_id_1, interface_chain_id_2), ...]."""
    out: dict[str, list[tuple[str, str]]] = {}
    for r in csv.DictReader(open(TARGETS_ROOT / f"{category}.csv")):
        out.setdefault(r["pdb_id"], []).append((r["interface_chain_id_1"], r["interface_chain_id_2"]))
    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate(series: str, step: int, category: str, workers: int) -> None:
    is_interface = category in INTERFACE_CATEGORIES
    eval_dir = BENCH / "runs" / series / "eval" / category / f"step_{step}"
    detail = eval_dir / "detail"
    (eval_dir / "raw").mkdir(parents=True, exist_ok=True)

    pred_ref = make_pred_ref.build(series, step, category)
    rows = list(csv.DictReader(open(pred_ref)))
    if not rows:
        print(f"[eval] no predictions for {series} step {step}")
        return

    # 1. run OST per prediction (subprocess => threads fine), cached
    def job(r):
        out = detail / f"{r['pdb_id']}_{r['seed']}_{r['sample']}_structure_ost.json"
        run_ost(Path(r["prediction_path"]), GT_ROOT / f"{r['pdb_id']}.cif", out)
        return r, out

    print(f"[eval] {len(rows)} OST comparisons on {workers} workers ({'interface' if is_interface else 'monomer'}) ...")
    parsed = []
    done = 0
    n_ost_failed = 0  # OST returned status != SUCCESS (e.g. its chain-mapper crashed)
    interfaces = load_interfaces(category) if is_interface else {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for fut in as_completed(futs):
            r, out = fut.result()
            try:
                d = json.load(open(out))
            except (FileNotFoundError, json.JSONDecodeError):
                d = {}
            if d.get("status") != "SUCCESS":
                n_ost_failed += 1
            if is_interface:
                for ic1, ic2 in interfaces.get(r["pdb_id"], []):
                    parsed.append({**r, "ic1": ic1, "ic2": ic2, **parse_interface(d, ic1, ic2)})
            else:
                parsed.append({**r, **parse_monomer(d)})
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}")

    # 2. raw per-sample CSV
    if is_interface:
        fields = [
            "pdb_id",
            "ic1",
            "ic2",
            "seed",
            "sample",
            "dockq",
            "irmsd",
            "lrmsd",
            "lddt",
            "tm_score",
            "oligo_gdtts",
        ]
    else:
        fields = ["pdb_id", "seed", "sample", "lddt", "tm_score", "oligo_gdtts", "rmsd"]
    with open(eval_dir / "raw" / f"{category}_ost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(parsed)

    # 3. aggregate
    if is_interface:
        by: dict[tuple, list] = {}
        for r in parsed:
            by.setdefault((r["pdb_id"], r["ic1"], r["ic2"]), []).append(r)
        mean_dq, oracle_dq, mean_ld, mean_tm, mean_gd = [], [], [], [], []
        succ_oracle = succ_mean = n = 0
        for _k, rs in by.items():
            dqs = [r["dockq"] for r in rs if isinstance(r["dockq"], (int, float))]
            if not dqs:
                continue
            n += 1
            md, od = sum(dqs) / len(dqs), max(dqs)
            mean_dq.append(md)
            oracle_dq.append(od)
            mean_ld.append(mean([r["lddt"] for r in rs]))
            mean_tm.append(mean([r["tm_score"] for r in rs]))
            mean_gd.append(mean([r["oligo_gdtts"] for r in rs]))
            succ_oracle += int(od >= DOCKQ_SUCCESS)
            succ_mean += int(md >= DOCKQ_SUCCESS)
        summary = {
            "series": series,
            "step": step,
            "category": category,
            "n_interfaces": n,
            "n_ost_failed": n_ost_failed,
            "success_rate_oracle": round(100 * succ_oracle / n, 2) if n else None,
            "success_rate_mean": round(100 * succ_mean / n, 2) if n else None,
            "mean_dockq": mean(mean_dq),
            "oracle_dockq": mean(oracle_dq),
            "mean_lddt": mean(mean_ld),
            "mean_tm_score": mean(mean_tm),
            "mean_gdt_ts": mean(mean_gd),
        }
        msg = (
            f"n={n}  succ@0.23 oracle={summary['success_rate_oracle']}% "
            f"mean={summary['success_rate_mean']}%  "
            f"mean_dockq={summary['mean_dockq']} oracle_dockq={summary['oracle_dockq']}  "
            f"ost_failed={n_ost_failed}"
        )
    else:
        by = {}
        for r in parsed:
            by.setdefault(r["pdb_id"], []).append(r)
        agg = {f"{s}_{m}": [] for s in ("mean", "oracle") for m in ("lddt", "tm_score", "gdt_ts")}
        n = 0
        for _p, rs in by.items():
            ld = [r["lddt"] for r in rs if isinstance(r["lddt"], (int, float))]
            if not ld:
                continue
            n += 1
            for m, col in (("lddt", "lddt"), ("tm_score", "tm_score"), ("gdt_ts", "oligo_gdtts")):
                vals = [r[col] for r in rs if isinstance(r[col], (int, float))]
                if vals:
                    agg[f"mean_{m}"].append(sum(vals) / len(vals))
                    agg[f"oracle_{m}"].append(max(vals))
        summary = {
            "series": series,
            "step": step,
            "category": category,
            "n_scored": n,
            "n_ost_failed": n_ost_failed,
            **{k: mean(v) for k, v in agg.items()},
        }
        msg = (
            f"n={n}  mean_lddt={summary['mean_lddt']} oracle_lddt={summary['oracle_lddt']}  "
            f"mean_tm={summary['mean_tm_score']} mean_gdt={summary['mean_gdt_ts']}  "
            f"ost_failed={n_ost_failed}"
        )

    with open(eval_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    print(f"[eval] {series} step {step} {category}: {msg}")
    print(f"[eval] -> {eval_dir / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    evaluate(args.series, args.step, args.category, args.workers)


if __name__ == "__main__":
    main()
