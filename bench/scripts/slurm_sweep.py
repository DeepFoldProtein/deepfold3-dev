"""Cluster-wide FoldBench inference sweep via Slurm.

Where ``run_sweep.py`` fans a df3 checkpoint across the *local* GPUs of one
machine, this launcher fans it across **every GPU in the Slurm cluster**. It
shards the FoldBench-small targets (union over the requested categories) and
submits a Slurm **job array** — one array task per shard, each requesting a
single GPU and running exactly one ``deepfold3 inference-batch`` process. Slurm
packs the tasks onto whatever GPUs are free (across all nodes) and queues the
rest, so the sweep automatically uses the whole cluster and drains as GPUs free.

Design choices for this cluster (ada nodes, 64 CPU / node, no Slurm mem
accounting):
  * one GPU per task (``--gres=gpu:<type>:1``); the process sees it as cuda:0.
  * ``--cpus-per-task`` capped so that even a fully packed node (up to 6 GPUs)
    stays at ~half the 64 cores (default 5 → ≤30 threads/node).
  * Triton autotune + JIT caches redirected to a **shared GPFS dir** keyed by
    GPU compute-capability, so a single ``warmup-triton`` on one RTX4090 warms
    every RTX4090 node.
  * fully **resumable**: ``inference-batch --skip-existing`` skips any
    (target, seed, sample) whose CIF exists, so re-submitting fills only gaps.

One array is submitted **per step** (an ``inference-batch`` process loads a
single ``--params`` checkpoint). Steps are submitted latest-first.

Layout written (matches run_sweep.py / the df3v3 history):

    bench/runs/{series}/pred/{category}/step_{N}/{pdb_id}/seed-{s}_sample-{i}/model.cif
    bench/runs/{series}/pred/_slurm/step_{N}/shard_{k}.tsv        # shard inputs
    bench/runs/{series}/pred/_slurm/step_{N}/sweep.sbatch         # generated job
    bench/runs/{series}/logs/sweep_step{N}_shard_{k}.out          # per-task log

Usage::

    # steps 5000 & 6000, all 8 swept categories, 4 seeds, across all RTX4090s
    python bench/scripts/slurm_sweep.py --series df3v3 --steps 5000,6000

    # just monomer_protein on 6000, print the sbatch and don't submit
    python bench/scripts/slurm_sweep.py --series df3v3 --steps 6000 \
        --categories monomer_protein --dry-run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
DF3 = REPO / ".venv" / "bin" / "deepfold3"
# Shared, cross-node cache so warmup on one GPU is reused everywhere.
CACHE_ROOT = BENCH / "cache"

# The 8 categories the df3v3 series sweeps (interface_protein_ligand is excluded:
# no ligand-aware OST scorer). Order = *priority*: each GPU works through the
# categories in this order, so the headline categories (monomer_protein, then
# protein-protein, then antibody-antigen) complete cluster-wide first.
DEFAULT_CATEGORIES = [
    "monomer_protein",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "monomer_rna",
    "monomer_dna",
    "interface_protein_dna",
    "interface_protein_rna",
    "interface_protein_peptide",
]


def resolve_ckpt(series: str, step: int) -> Path | None:
    base = BENCH / "ckpts" / series
    for ext in (".safetensors", ".pt"):
        p = base / f"step_{step}{ext}"
        if p.exists():
            return p
    return None


# States a node can be in and still accept/run work. Anything else (down,
# drain, maint, ...) is excluded from the GPU census so we don't size the sweep
# to hardware that will never schedule.
_USABLE_STATES = {"idle", "mix", "mixed", "alloc", "allocated", "comp", "completing"}


def detect_cluster(partition: str, gpu_type: str) -> dict:
    """Census the partition's GPUs from ``sinfo`` — no hardcoded node list.

    Returns ``{"gpu_type", "total_gpus", "max_gpus_per_node", "cpus_per_node",
    "by_type"}``. ``gpu_type="auto"`` selects the most abundant type. Falls back
    to empty/zeros if sinfo is unavailable (caller then needs explicit flags).
    """
    by_type: dict[str, list[tuple[int, int]]] = {}  # type -> [(gpu_count, node_cpus)]
    try:
        out = subprocess.run(
            ["sinfo", "-h", "-N", "-p", partition, "-o", "%n|%G|%t|%c"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"gpu_type": gpu_type, "total_gpus": 0, "max_gpus_per_node": 0, "cpus_per_node": 0, "by_type": {}}

    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        _node, gres, state, cpus = parts
        state = state.rstrip("*$~#").lower()  # strip status flags (*, ~, #, $)
        if state not in _USABLE_STATES:
            continue
        try:
            ncpu = int(cpus)
        except ValueError:
            ncpu = 0
        for tok in gres.split(","):
            if not tok.startswith("gpu:"):
                continue
            bits = tok.split(":")
            if len(bits) < 3:
                continue
            gtype = bits[1]
            count = int(bits[2].split("(")[0])
            by_type.setdefault(gtype, []).append((count, ncpu))

    if gpu_type == "auto":
        gpu_type = max(by_type, key=lambda t: sum(c for c, _ in by_type[t])) if by_type else gpu_type
    entries = by_type.get(gpu_type, [])
    total = sum(c for c, _ in entries)
    max_per_node = max((c for c, _ in entries), default=0)
    cpus_per_node = min((n for _, n in entries if n), default=0)
    return {
        "gpu_type": gpu_type,
        "total_gpus": total,
        "max_gpus_per_node": max_per_node,
        "cpus_per_node": cpus_per_node,
        "by_type": {t: sum(c for c, _ in v) for t, v in by_type.items()},
    }


def small_targets(category: str) -> list[tuple[str, int]]:
    """FoldBench-small (in_small=1) targets for a category as (pdb_id, tokens)."""
    manifest = BENCH / "inputs" / f"{category}_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest} — run build_inputs.py --category {category} first")
    rows = []
    for r in csv.DictReader(open(manifest)):
        if int(r["in_small"]):
            rows.append((r["pdb_id"], int(r["num_tokens"])))
    return rows


def collect_jobs(series: str, step: int, categories: list[str]) -> list[tuple[str, list[tuple[Path, Path, int]]]]:
    """Per-category ``(json_path, output_dir, num_tokens)`` lists, in the given
    category (priority) order. Missing per-target JSONs are skipped with a
    warning (a target can be over-cap in one category yet listed in another)."""
    by_cat: list[tuple[str, list[tuple[Path, Path, int]]]] = []
    for category in categories:
        in_dir = BENCH / "inputs" / category
        pred_root = BENCH / "runs" / series / "pred" / category / f"step_{step}"
        cjobs: list[tuple[Path, Path, int]] = []
        for pdb_id, tok in small_targets(category):
            jp = in_dir / f"{pdb_id}.json"
            if not jp.exists():
                print(f"[slurm-sweep] WARN {category}/{pdb_id}: no input JSON, skipping", file=sys.stderr)
                continue
            cjobs.append((jp, pred_root / pdb_id, tok))
        if cjobs:
            by_cat.append((category, cjobs))
    return by_cat


def shard_balanced(by_cat: list[tuple[str, list[tuple[Path, Path, int]]]], n: int) -> list[list[tuple[Path, Path]]]:
    """Category-priority + length-balanced sharding by **greedy LPT**.

    Two goals at once:
      * **Priority order** — categories are processed in the order given, so
        every GPU folds all of monomer_protein before protein-protein before
        antibody-antigen, etc. (each shard's list is category-1 targets first,
        then category-2, …). The headline categories thus finish cluster-wide
        first and can be scored while the rest are still folding.
      * **Length balance** — within the running total, each category's targets
        are placed heaviest-first onto the currently-lightest shard (LPT on a
        ``tokens**2`` cost proxy, since the trunk forward is ~quadratic). This
        keeps per-GPU load even so no single shard gates the step.
    """
    shards: list[list[tuple[Path, Path]]] = [[] for _ in range(n)]
    loads = [0] * n
    for _cat, cjobs in by_cat:
        for jp, od, tok in sorted(cjobs, key=lambda t: -t[2]):
            k = min(range(n), key=lambda i: loads[i])
            shards[k].append((jp, od))
            loads[k] += tok * tok
    return shards


def write_sbatch(
    series: str,
    step: int,
    ckpt: Path,
    shard_dir: Path,
    log_dir: Path,
    n_shards: int,
    gpu_type: str,
    cpus: int,
    partition: str,
    num_samples: int,
    num_recycles: int,
    seeds: list[int],
    max_concurrent: int | None,
    save_distogram: bool,
    tag: str = "",
) -> Path:
    gres = f"gpu:{gpu_type}:1" if gpu_type and gpu_type != "any" else "gpu:1"
    array = f"0-{n_shards - 1}" + (f"%{max_concurrent}" if max_concurrent else "")
    seed_flags = " ".join(f"--seed {s}" for s in seeds)
    disto = "--save-distogram" if save_distogram else "--no-save-distogram"
    suffix = f"_{tag}" if tag else ""
    sbatch = shard_dir / "sweep.sbatch"
    sbatch.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=fb_{series}_{step}{suffix}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --array={array}
#SBATCH --output={log_dir}/sweep_step{step}{suffix}_shard_%a.out
#SBATCH --open-mode=append
set -euo pipefail
cd {REPO}

T={cpus}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
# expandable_segments lets the CUDA caching allocator grow/reuse one segment
# instead of fragmenting into many fixed blocks — the standard fix for the
# intermittent fragmentation OOM that varied bucket sizes + Triton autotune
# scratch cause across a long shard.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Shared cross-node caches (keyed by GPU arch): warmup once, reuse everywhere.
export DEEPFOLD3_TRITON_CACHE_DIR={CACHE_ROOT}/deepfold3-triton
export TRITON_CACHE_DIR={CACHE_ROOT}/triton
mkdir -p "$DEEPFOLD3_TRITON_CACHE_DIR" "$TRITON_CACHE_DIR"

SHARD={shard_dir}/shard_${{SLURM_ARRAY_TASK_ID}}.tsv
echo "[$(date +%T)] $(hostname) task $SLURM_ARRAY_TASK_ID  GPU=$CUDA_VISIBLE_DEVICES  shard=$SHARD"
nvidia-smi -L || true

# Retry loop: shared-filesystem hiccups (e.g. "Stale file handle") can kill a
# shard mid-run. --skip-existing makes every attempt resume where the last left
# off, so a few retries drain the shard without redoing finished work.
rc=1
for attempt in 1 2 3 4 5; do
    if {DF3} inference-batch \\
        --inputs-file "$SHARD" \\
        --params {ckpt} \\
        --optim inference \\
        --num-samples {num_samples} \\
        --num-recycles {num_recycles} \\
        {seed_flags} \\
        --skip-existing \\
        {disto}; then
        rc=0; break
    fi
    echo "[$(date +%T)] attempt $attempt failed — retrying in 30s"
    sleep 30
done
exit $rc
"""
    )
    return sbatch


def submit_step(args, step: int, categories: list[str], tag: str = "") -> str | None:
    """Submit one sweep array for ``categories`` at ``step``.

    ``tag`` namespaces the shard dir / job name / logs so several arrays for the
    same step (one per category, in per-category mode) don't collide. Returns
    the Slurm array job id (str) or None (skipped / dry-run / failure)."""
    ckpt = resolve_ckpt(args.series, step)
    if ckpt is None:
        print(
            f"[slurm-sweep] SKIP step {step}: no ckpt at ckpts/{args.series}/step_{step}.{{safetensors,pt}}",
            file=sys.stderr,
        )
        return None

    by_cat = collect_jobs(args.series, step, categories)
    n_targets = sum(len(c) for _c, c in by_cat)
    if not n_targets:
        print(f"[slurm-sweep] step {step} [{tag or 'all'}]: no targets — did you run build_inputs?", file=sys.stderr)
        return None

    n_shards = min(args.shards, n_targets)
    shards = shard_balanced(by_cat, n_shards)

    base = BENCH / "runs" / args.series / "pred" / "_slurm" / f"step_{step}"
    shard_dir = base / tag if tag else base
    log_dir = BENCH / "runs" / args.series / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for k, shard in enumerate(shards):
        with open(shard_dir / f"shard_{k}.tsv", "w") as f:
            for jp, od in shard:
                f.write(f"{jp}\t{od}\n")

    sbatch = write_sbatch(
        args.series,
        step,
        ckpt,
        shard_dir,
        log_dir,
        n_shards,
        args.gpu_type,
        args.cpus_per_task,
        args.partition,
        args.num_samples,
        args.num_recycles,
        [int(s) for s in args.seed.split(",") if s != ""],
        args.max_concurrent,
        not args.no_distogram,
        tag,
    )
    order = " > ".join(c for c, _ in by_cat)
    print(
        f"[slurm-sweep] step {step} [{tag or 'all'}]: {n_targets} targets over {len(by_cat)} cats "
        f"-> {n_shards} shards (~{n_targets // n_shards}/shard)\n              priority: {order}"
    )
    if args.dry_run:
        print(f"[slurm-sweep] --dry-run: not submitting. Inspect {sbatch}")
        return None
    out = subprocess.run(["sbatch", str(sbatch)], capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed (rc={out.returncode})")
    # "Submitted batch job 12345" -> record jobid so downstream eval can depend on it.
    jobid = out.stdout.strip().split()[-1] if out.stdout.strip() else ""
    if jobid.isdigit():
        (shard_dir / "jobid.txt").write_text(jobid + "\n")
        return jobid
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v3")
    ap.add_argument("--steps", required=True, help="Comma list, e.g. 5000,6000. Submitted latest-first.")
    ap.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma list or 'all' (= the 8 swept categories). Default: all 8.",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Array tasks = max concurrent GPUs used per step. Default: auto (total GPUs of the chosen type in the partition, from sinfo).",
    )
    ap.add_argument(
        "--gpu-type",
        default="auto",
        help="Slurm GRES gpu type ('auto' = most abundant in partition, an explicit type e.g. 'RTX4090', or 'any' for untyped).",
    )
    ap.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="CPUs per GPU task. Default: auto (~half a node's cores split across its GPUs, so a full node stays near half-CPU).",
    )
    ap.add_argument("--partition", default="normal")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--seed", default="42,43,44,45", help="Comma list of model seeds (FoldBench uses 4).")
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Cap simultaneously-running array tasks (Slurm '%%N'). Default: unlimited (GPU-bound).",
    )
    ap.add_argument("--no-distogram", action="store_true", help="Disable --save-distogram.")
    ap.add_argument(
        "--per-category",
        dest="per_category",
        action="store_true",
        default=True,
        help="Submit one array per (step, category) instead of one big array per step, "
        "so each category can be scored + plotted as soon as it finishes (default).",
    )
    ap.add_argument(
        "--single-array",
        dest="per_category",
        action="store_false",
        help="One array per step over all categories (evaluate only after the whole sweep).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Write shards + sbatch but do not submit.")
    args = ap.parse_args()

    cats = DEFAULT_CATEGORIES if args.categories == "all" else [c for c in args.categories.split(",") if c]
    steps = sorted({int(s) for s in args.steps.split(",") if s != ""}, reverse=True)

    # Resolve cluster-shaped defaults from sinfo (no hardcoded node list).
    census = detect_cluster(args.partition, args.gpu_type)
    args.gpu_type = census["gpu_type"]
    if args.shards is None:
        args.shards = census["total_gpus"] or 8  # fallback if sinfo gave nothing
    if args.cpus_per_task is None:
        half = (census["cpus_per_node"] or 64) // 2
        per_node = census["max_gpus_per_node"] or 1
        args.cpus_per_task = max(1, half // per_node)
    mode = "per-category" if args.per_category else "single-array"
    print(
        f"[slurm-sweep] cluster census (partition={args.partition}): {census['by_type']}\n"
        f"[slurm-sweep] using gpu_type={args.gpu_type} total_gpus={census['total_gpus']} "
        f"-> shards={args.shards}, cpus/task={args.cpus_per_task}\n"
        f"[slurm-sweep] series={args.series} steps={steps} (latest-first) categories={len(cats)} mode={mode}"
    )

    # (category, step, jobid) records — the eval driver reads this to gate each
    # category's OST-eval on just that category's arrays (incremental results).
    records: list[tuple[str, int, str]] = []
    if args.per_category:
        # Submit in priority order: each category (both steps) before the next,
        # so job-id order makes Slurm drain monomer_protein first, then ppi, …
        # (Slurm backfills later categories onto idle GPUs, so utilization holds).
        for cat in cats:
            for step in steps:
                jid = submit_step(args, step, [cat], tag=cat)
                if jid:
                    records.append((cat, step, jid))
    else:
        for step in steps:
            jid = submit_step(args, step, cats, tag="")
            if jid:
                records.append(("all", step, jid))

    if not args.dry_run:
        jobids = BENCH / "runs" / args.series / "pred" / "_slurm" / "jobids.tsv"
        jobids.parent.mkdir(parents=True, exist_ok=True)
        with open(jobids, "w") as f:
            for cat, step, jid in records:
                f.write(f"{cat}\t{step}\t{jid}\n")
        print(f"[slurm-sweep] {len(records)} arrays submitted -> {jobids}")


if __name__ == "__main__":
    main()
