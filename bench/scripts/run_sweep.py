"""Multi-GPU FoldBench inference dispatcher for a df3 checkpoint series.

Parallelism is at the *target* level: shard the FoldBench-small targets
across the local GPUs (length-balanced round-robin to even out the N^2
forward cost), and run one ``deepfold3 inference-batch`` process per GPU.
Each process loads the 5.8 GB checkpoint once and folds its whole shard, so
weight loads = n_gpus, not n_targets.

Resumable: ``inference-batch --skip-existing`` skips any (target, seed) whose
sample CIFs already exist, so re-running fills only the gaps — including a
sweep interrupted partway through its seed list.

Multi-seed: ``--seed`` takes a comma list (FoldBench folds 4 seeds); each
target is folded once per seed under its own ``seed-{seed}_sample-*`` subdir.

Layout written (one model series, one step):

    bench/runs/{series}/pred/{category}/step_{N}/{pdb_id}/seed-{seed}_sample-{s}/model.cif
                                                         /ranking_scores.csv

Usage::

    # latest checkpoint, all 4 GPUs (default), 5 samples, 4 seeds
    python bench/scripts/run_sweep.py --series df3v1 --steps 90000 \
        --category monomer_protein --seed 42,43,44,45

    # the full 5k-grid trajectory, latest first
    python bench/scripts/run_sweep.py --series df3v1 --steps all --category monomer_protein
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# Per-worker CPU-thread cap. Without this, each inference-batch process spawns
# ~one OMP/MKL thread per core (~64+), so 4 workers => 800+ threads on a
# 64-CPU budget and they hang in futex_wait during model build / first forward
# (see memory: "Concurrent runs need thread caps"). 8 threads x 4 workers = 32.
THREADS_PER_WORKER = 8

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
DF3 = str(REPO / ".venv" / "bin" / "deepfold3")

# 5k grid, latest first (so good results show up early).
ALL_STEPS = list(range(90000, 5000, -5000))  # 90000, 85000, ..., 10000


def resolve_ckpt(series: str, step: int) -> Path | None:
    """Return the checkpoint for (series, step), or None if absent.

    Accepts either an exported ``.safetensors`` (inference weights, already
    EMA-averaged — ``--snapshot`` is a no-op for these) or a raw ``.pt``
    training checkpoint. ``.safetensors`` wins if both exist.
    """
    base = BENCH / "ckpts" / series
    for ext in (".safetensors", ".pt"):
        p = base / f"step_{step}{ext}"
        if p.exists():
            return p
    return None


def small_targets(category: str) -> list[tuple[str, int]]:
    """Return [(pdb_id, num_tokens), ...] for the FoldBench-small subset,
    sorted by token count descending (for length-balanced sharding)."""
    manifest = BENCH / "inputs" / f"{category}_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest} — run build_inputs.py --category {category} first")
    rows = []
    for r in csv.DictReader(open(manifest)):
        if int(r["in_small"]):
            rows.append((r["pdb_id"], int(r["num_tokens"])))
    rows.sort(key=lambda t: -t[1])
    return rows


def shard_round_robin(targets: list[tuple[str, int]], n: int) -> list[list[str]]:
    """Length-sorted round-robin: target i -> shard (i % n). Balances N^2."""
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, (pdb_id, _tok) in enumerate(targets):
        shards[i % n].append(pdb_id)
    return shards


def run_step(
    series: str,
    step: int,
    category: str,
    gpus: list[int],
    num_samples: int,
    num_recycles: int,
    seeds: list[int],
    snapshot: str,
    msa_crop_size: int | None = None,
) -> None:
    params = resolve_ckpt(series, step)
    if params is None:
        print(
            f"[sweep] SKIP step {step}: no checkpoint at {BENCH / 'ckpts' / series}/step_{step}.{{safetensors,pt}}",
            file=sys.stderr,
        )
        return
    in_dir = BENCH / "inputs" / category
    pred_root = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    shard_dir = pred_root / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    targets = small_targets(category)
    shards = shard_round_robin(targets, len(gpus))
    print(
        f"[sweep] {series} step {step} | {category} | {len(targets)} targets "
        f"-> {len(gpus)} GPUs ({[len(s) for s in shards]})"
    )

    procs = []
    t0 = time.perf_counter()
    for gpu, shard in zip(gpus, shards, strict=False):
        if not shard:
            continue
        shard_file = shard_dir / f"gpu{gpu}.txt"
        with open(shard_file, "w") as f:
            for pdb_id in shard:
                jp = in_dir / f"{pdb_id}.json"
                od = pred_root / pdb_id
                f.write(f"{jp}\t{od}\n")
        log = pred_root / f"gpu{gpu}.log"
        t = str(THREADS_PER_WORKER)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": t,
            "MKL_NUM_THREADS": t,
            "OPENBLAS_NUM_THREADS": t,
            "NUMEXPR_NUM_THREADS": t,
        }
        cmd = [
            DF3,
            "inference-batch",
            "--inputs-file",
            str(shard_file),
            "--params",
            str(params),
            "--snapshot",
            snapshot,
            "--optim",
            "inference",
            "--num-samples",
            str(num_samples),
            "--num-recycles",
            str(num_recycles),
            *(["--msa-crop-size", str(msa_crop_size)] if msa_crop_size else []),
            "--skip-existing",
            "--save-distogram",  # per-seed trunk distogram for interface/contact post-analysis
        ]
        for sd in seeds:
            cmd += ["--seed", str(sd)]
        print(f"  GPU {gpu}: {len(shard)} targets -> {log.name}")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        time.sleep(3)  # stagger model-build / first-compile across workers

    rc = 0
    for gpu, p in procs:
        r = p.wait()
        rc |= r
        print(f"  GPU {gpu}: exit {r}")
    dt = time.perf_counter() - t0
    print(f"[sweep] step {step} done in {dt / 60:.1f} min (rc={rc})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--steps", default="90000", help="'all' (5k grid, latest first), or comma list e.g. 90000,85000")
    ap.add_argument("--gpus", default="0,1,2,3", help="comma list of GPU ids")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument(
        "--msa-crop-size",
        type=int,
        default=None,
        help=(
            "Override the checkpoint's per-recycle MSA crop. Training uses a "
            "deliberately small crop (r14: 1024) and randomises the effective row "
            "count, so bench should serve the full pool: pass 16384. Omitted = "
            "whatever the checkpoint was trained with."
        ),
    )
    ap.add_argument(
        "--seed",
        default="42",
        help="Comma list of model seeds, e.g. 42,43,44,45. Each target is folded "
        "once per seed (FoldBench uses 4). Default: 42.",
    )
    ap.add_argument("--snapshot", default="average", choices=["auto", "average", "shadow", "fast"])
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    seeds = [int(s) for s in str(args.seed).split(",") if s != ""]
    if args.steps == "all":
        steps = ALL_STEPS
    else:
        steps = [int(s) for s in args.steps.split(",") if s != ""]

    print(
        f"[sweep] series={args.series} steps={steps} gpus={gpus} "
        f"samples={args.num_samples} recycles={args.num_recycles} "
        f"msa_crop={args.msa_crop_size or 'ckpt'} "
        f"seeds={seeds} snapshot={args.snapshot}"
    )
    for step in steps:
        run_step(
            args.series, step, args.category, gpus, args.num_samples, args.num_recycles,
            seeds, args.snapshot, args.msa_crop_size,
        )


if __name__ == "__main__":
    main()
