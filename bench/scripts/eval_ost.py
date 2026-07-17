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

We report three aggregates per target/interface:
  mean   = mean over samples (the confidence-free floor)
  oracle = best sample by the metric itself (best-of-N upper bound / ceiling)
  conf   = confidence top-1: the sample with the highest ranking_score
           (0.8*ipTM+0.2*pTM) — the deployed number between mean and oracle.

The conf aggregate is only meaningful once a run trains a confidence head
(df3v2+); for confidence-head-free runs (df3v1) ranking_score is untrained, so
conf sits near mean and the mean/oracle band is what to read. ranking_score,
ptm, iptm, mean_plddt are carried per sample into the raw CSV for selection and
downstream calibration analysis.

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
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import make_pred_ref  # sibling script

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_ROOT = BENCH / "data" / "ground_truths"
TARGETS_ROOT = BENCH / "data" / "targets"

# OST invocation. This cluster has no conda `ost`; OpenStructure ships as an
# apptainer image (ost.sif) whose runscript is `ost`, so `apptainer run <sif>
# compare-structures ...` is the entry point. Both knobs are env-overridable:
#   OST_SIF  — path to the .sif (default: the casp17 foldbench image)
#   OST_CMD  — full command prefix up to and including the OST subcommand
#              (e.g. "ost" for a native binary, or a custom apptainer line).
# The bind mounts must cover wherever models/refs/outputs live (/store, /gpfs).
OST_SIF = os.environ.get("OST_SIF", "/store/deepfold/casp17_sync/benchmark/foldbench/ost.sif")
OST_BIND = os.environ.get("OST_BIND", "/gpfs,/store")


def _ost_cmd() -> list[str]:
    """Command prefix ending in the ``compare-structures`` subcommand."""
    override = os.environ.get("OST_CMD")
    if override:
        return shlex.split(override)
    return ["apptainer", "run", "--bind", OST_BIND, OST_SIF, "compare-structures"]


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
    "--lddt",  # global lDDT over ALL atoms (backbone + sidechain), key "lddt"
    "--bb-lddt",  # global lDDT over BACKBONE only (CA for peptides, C3' for nucleotides), key "bb_lddt".
    # bb_lddt - lddt isolates the non-CA (mostly sidechain) contribution: bb_lddt high & lddt low =>
    # backbone trace placed but sidechains/local all-atom geometry lag (the mid-training signature).
    "--local-lddt",  # per-residue all-atom lDDT, key "local_lddt" (array; drill-down, detail JSON only)
    "--bb-local-lddt",  # per-residue backbone lDDT, key "bb_local_lddt" (array; pairs with local_lddt)
    "--lddt-no-stereochecks",
    "--ilddt",  # interface lDDT: global lDDT over inter-chain contacts only (collapse-sensitive)
    "--qs-score",  # QS-score: standard multimer interface-quality metric (collapse-sensitive)
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


def run_ost(model: Path, ref: Path, out_json: Path, force: bool = False) -> None:
    if out_json.exists() and not force:
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*_ost_cmd(), "-m", str(model), "-r", str(ref), "-o", str(out_json), *STRUCT_FLAGS]
    subprocess.run(cmd, capture_output=True, check=False)


def parse_monomer(d: dict) -> dict:
    return {m: d.get(m) for m in ("lddt", "bb_lddt", "tm_score", "oligo_gdtts", "rmsd")}


def parse_interface(d: dict, ic1: str, ic2: str) -> dict:
    """DockQ/iRMSD/lRMSD for the (ic1, ic2) interface (FoldBench matching),
    plus the global (whole-model) lDDT / TM-score / GDT-TS."""
    out = {
        "dockq": None,
        "irmsd": None,
        "lrmsd": None,
        "lddt": d.get("lddt"),
        "bb_lddt": d.get("bb_lddt"),  # whole-model backbone (CA) lDDT; None for pre-bb-lddt cached JSONs
        "ilddt": d.get("ilddt"),  # whole-model interface (inter-chain) lDDT; None for pre-11k cached JSONs
        "qs_global": d.get("qs_global"),  # QS-score; None for pre-11k cached JSONs
        "qs_best": d.get("qs_best"),
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


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def conf_pick(rs: list[dict], key: str):
    """Confidence top-1 value of ``key``: the metric of the sample with the
    highest ``ranking_score``, skipping samples whose ``key`` is missing (e.g.
    OST failed for that interface). ``None`` if no ranked sample has a value.

    Selection is by the whole-prediction ``ranking_score`` (0.8·ipTM+0.2·pTM),
    so for interfaces every row of a given (seed, sample) shares the score and
    the pick reflects the model's own top-ranked structure — the deployed number
    between the mean floor and the oracle (best-of-N) ceiling.
    """
    best = best_score = None
    for r in rs:
        v = r.get(key)
        if not isinstance(v, (int, float)):
            continue
        s = _fnum(r.get("ranking_score"))
        if s is None:
            continue
        if best_score is None or s > best_score:
            best_score, best = s, v
    return best


def evaluate(series: str, step: int, category: str, workers: int, force: bool = False) -> None:
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
        run_ost(Path(r["prediction_path"]), GT_ROOT / f"{r['pdb_id']}.cif", out, force=force)
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
    # Confidence columns (carried from prediction_reference.csv per sample) so
    # the raw CSV can drive confidence top-1 selection in summarize.py.
    conf_cols = ["ranking_score", "ptm", "iptm", "mean_plddt"]
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
            "bb_lddt",
            "ilddt",
            "qs_global",
            "qs_best",
            "tm_score",
            "oligo_gdtts",
            *conf_cols,
        ]
    else:
        fields = ["pdb_id", "seed", "sample", "lddt", "bb_lddt", "tm_score", "oligo_gdtts", "rmsd", *conf_cols]
    with open(eval_dir / "raw" / f"{category}_ost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(parsed)

    # 3. aggregate
    if is_interface:
        by: dict[tuple, list] = {}
        for r in parsed:
            by.setdefault((r["pdb_id"], r["ic1"], r["ic2"]), []).append(r)
        mean_dq, oracle_dq, conf_dq, mean_ld, mean_tm, mean_gd = [], [], [], [], [], []
        mean_ild, oracle_ild, conf_ild, mean_qs, oracle_qs, conf_qs = [], [], [], [], [], []
        conf_ld, conf_tm, conf_gd = [], [], []
        mean_bld = []  # backbone (CA) lDDT; None for pre-bb-lddt cached JSONs
        succ_oracle = succ_mean = succ_conf = n = 0
        n_conf = 0  # interfaces with a valid ranking_score (confidence top-1 selectable)
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
            blds = [r["bb_lddt"] for r in rs if isinstance(r["bb_lddt"], (int, float))]
            if blds:
                mean_bld.append(sum(blds) / len(blds))
            # interface lDDT / QS-score (present only for step ≥ 11k JSONs; None-safe)
            ilds = [r["ilddt"] for r in rs if isinstance(r["ilddt"], (int, float))]
            qss = [r["qs_global"] for r in rs if isinstance(r["qs_global"], (int, float))]
            if ilds:
                mean_ild.append(sum(ilds) / len(ilds))
                oracle_ild.append(max(ilds))
            if qss:
                mean_qs.append(sum(qss) / len(qss))
                oracle_qs.append(max(qss))
            # Confidence top-1: the ranking_score-selected sample's metric. Only
            # present once the run trains a confidence head (else ranking_score is
            # absent/untrained and cd is None → interface skipped from conf stats).
            cd = conf_pick(rs, "dockq")
            if cd is not None:
                n_conf += 1
                conf_dq.append(cd)
                succ_conf += int(cd >= DOCKQ_SUCCESS)
                for dst, key in ((conf_ld, "lddt"), (conf_tm, "tm_score"), (conf_gd, "oligo_gdtts")):
                    cv = conf_pick(rs, key)
                    if cv is not None:
                        dst.append(cv)
                cil, cqs = conf_pick(rs, "ilddt"), conf_pick(rs, "qs_global")
                if cil is not None:
                    conf_ild.append(cil)
                if cqs is not None:
                    conf_qs.append(cqs)
            succ_oracle += int(od >= DOCKQ_SUCCESS)
            succ_mean += int(md >= DOCKQ_SUCCESS)
        summary = {
            "series": series,
            "step": step,
            "category": category,
            "n_interfaces": n,
            "n_conf": n_conf,
            "n_ost_failed": n_ost_failed,
            "success_rate_oracle": round(100 * succ_oracle / n, 2) if n else None,
            "success_rate_mean": round(100 * succ_mean / n, 2) if n else None,
            "success_rate_conf": round(100 * succ_conf / n_conf, 2) if n_conf else None,
            "mean_dockq": mean(mean_dq),
            "oracle_dockq": mean(oracle_dq),
            "conf_dockq": mean(conf_dq),
            "mean_lddt": mean(mean_ld),
            "conf_lddt": mean(conf_ld),
            "mean_bb_lddt": mean(mean_bld),
            "mean_ilddt": mean(mean_ild),
            "oracle_ilddt": mean(oracle_ild),
            "conf_ilddt": mean(conf_ild),
            "mean_qs_score": mean(mean_qs),
            "oracle_qs_score": mean(oracle_qs),
            "conf_qs_score": mean(conf_qs),
            "mean_tm_score": mean(mean_tm),
            "conf_tm_score": mean(conf_tm),
            "mean_gdt_ts": mean(mean_gd),
            "conf_gdt_ts": mean(conf_gd),
        }
        msg = (
            f"n={n}  succ@0.23 oracle={summary['success_rate_oracle']}% "
            f"mean={summary['success_rate_mean']}% conf={summary['success_rate_conf']}%  "
            f"mean_dockq={summary['mean_dockq']} oracle_dockq={summary['oracle_dockq']} "
            f"conf_dockq={summary['conf_dockq']}  "
            f"ost_failed={n_ost_failed}"
        )
    else:
        by = {}
        for r in parsed:
            by.setdefault(r["pdb_id"], []).append(r)
        agg = {f"{s}_{m}": [] for s in ("mean", "oracle", "conf") for m in ("lddt", "bb_lddt", "tm_score", "gdt_ts")}
        n = 0
        for _p, rs in by.items():
            ld = [r["lddt"] for r in rs if isinstance(r["lddt"], (int, float))]
            if not ld:
                continue
            n += 1
            for m, col in (
                ("lddt", "lddt"),
                ("bb_lddt", "bb_lddt"),
                ("tm_score", "tm_score"),
                ("gdt_ts", "oligo_gdtts"),
            ):
                vals = [r[col] for r in rs if isinstance(r[col], (int, float))]
                if vals:
                    agg[f"mean_{m}"].append(sum(vals) / len(vals))
                    agg[f"oracle_{m}"].append(max(vals))
                cv = conf_pick(rs, col)  # confidence top-1 (ranking_score-selected sample)
                if cv is not None:
                    agg[f"conf_{m}"].append(cv)
        summary = {
            "series": series,
            "step": step,
            "category": category,
            "n_scored": n,
            "n_conf": len(agg["conf_lddt"]),
            "n_ost_failed": n_ost_failed,
            **{k: mean(v) for k, v in agg.items()},
        }
        msg = (
            f"n={n}  mean_lddt={summary['mean_lddt']} oracle_lddt={summary['oracle_lddt']} "
            f"conf_lddt={summary['conf_lddt']}  "
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
    ap.add_argument("--force", action="store_true", help="recompute OST even if detail JSON is cached")
    args = ap.parse_args()
    evaluate(args.series, args.step, args.category, args.workers, force=args.force)


if __name__ == "__main__":
    main()
