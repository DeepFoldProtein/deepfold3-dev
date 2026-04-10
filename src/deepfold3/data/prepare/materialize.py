"""Parallel A3M → NPZ materialization helpers.

A3M parsing is the bottleneck of every import command (~1 file/sec/worker
on multi-MB MMseqs2 outputs), so the work is fanned out to a process pool.
The same materialize task type is shared by ``import-msa``,
``import-paired-msa``, and ``msa-batch`` so they all behave identically
(resume-safe, optional A3M sibling, hardlink fast path).
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

from deepfold3.data.parse.a3m import convert_a3m_file_to_npz

# Pre-create one shard per hex byte. Hex msa_ids cover 256 buckets;
# PDB-id-prefixed paired ids cover ~324 buckets but reuse this set as a
# starting point. Worker mkdir(exist_ok=True) handles any extras.
_HEX_SHARDS = tuple(f"{i:02x}" for i in range(256))


class MaterializeTask(NamedTuple):
    """One unit of work for the materialize pool."""

    msa_id: str
    src: str
    dest_npz: str


def precreate_shards(root: Path) -> None:
    """Create the 256 hex shard subdirectories under ``root`` upfront.

    Saves N×256 ``mkdir(exist_ok=True)`` syscalls when materializing
    hundreds of thousands of NPZs into a sharded layout. Idempotent.
    """
    root.mkdir(parents=True, exist_ok=True)
    for s in _HEX_SHARDS:
        (root / s).mkdir(exist_ok=True)


def _materialize_one(args: tuple[MaterializeTask, bool]) -> tuple[str, bool]:
    """Worker: parse one A3M into ``dest_npz``, optionally hardlinking the source A3M.

    The planner is responsible for filtering out ``dest_npz`` paths that already
    exist, so the worker doesn't re-check. Module-level for picklability.
    """
    task, keep_a3m = args
    dest_npz = Path(task.dest_npz)
    src = Path(task.src)
    if not convert_a3m_file_to_npz(src, dest_npz):
        return task.msa_id, False
    if keep_a3m:
        dest_a3m = dest_npz.with_suffix(".a3m")
        if not dest_a3m.exists():
            try:
                os.link(src, dest_a3m)
            except OSError:
                shutil.copy2(src, dest_a3m)
    return task.msa_id, True


def run_materialize(
    work: list[MaterializeTask],
    workers: int,
    log: logging.Logger,
    label: str = "materialized",
    keep_a3m: bool = False,
) -> tuple[int, int]:
    """Materialize a list of tasks (serial or parallel). Returns ``(ok, failed)``."""
    if not work:
        return 0, 0
    items = [(t, keep_a3m) for t in work]
    ok_count = 0
    failed = 0
    if workers <= 1:
        for i, item in enumerate(items):
            _, ok = _materialize_one(item)
            if ok:
                ok_count += 1
            else:
                failed += 1
            if (i + 1) % 1000 == 0:
                log.info("  %s %d/%d...", label, i + 1, len(items))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            for _, ok in ex.map(_materialize_one, items, chunksize=64):
                done += 1
                if ok:
                    ok_count += 1
                else:
                    failed += 1
                if done % 1000 == 0:
                    log.info("  %s %d/%d...", label, done, len(items))
    return ok_count, failed
