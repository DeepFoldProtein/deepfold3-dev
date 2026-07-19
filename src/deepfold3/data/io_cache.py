"""Page-cache policy for NPZ reads on the training data path.

Training draws records at random from datasets of 10^6–10^7 structures, so
a given NPZ is essentially never read twice within a job. The kernel still
caches every one of those reads, and under a container memory cgroup that
cache grows until allocations start hitting **direct reclaim** — a
synchronous stall in the allocating process. Under DDP one stalled rank
holds the whole world at the next collective. Bytes read per step are
constant, which is why the resulting stalls recur on a step-tied period
(see ``export/burst-periodic-stall/``).

:func:`read_npz` is the read used by every dataset-path loader. By default
it is plain :func:`numpy.load`; a process that calls
:func:`set_drop_page_cache` — in practice only the trainer — switches it to
read the file whole and drop it from the page cache with
``POSIX_FADV_DONTNEED``. Offline CLIs (``prepare``, ``import-of3``,
``refilter``, …) keep normal caching, where re-reads and readahead are real
wins and no cgroup pressure exists to relieve.

Callers with genuine reuse must not use this module: the per-chain template
cache in :mod:`deepfold3.data.parse.pdb_chain_cache` runs at a ~99.8% hit
rate, and dropping its pages would turn every hit into a re-read.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ``posix_fadvise`` is Linux-only; elsewhere the policy silently stays off.
_HAS_FADVISE = hasattr(os, "posix_fadvise")

_drop_page_cache = False


def set_drop_page_cache(enabled: bool) -> None:
    """Enable/disable page-cache dropping for :func:`read_npz` process-wide.

    Set once during process setup. Dataloader workers inherit it through
    the fork, so no per-worker propagation is needed.
    """
    global _drop_page_cache
    if enabled and not _HAS_FADVISE:
        logger.warning("posix_fadvise unavailable; NPZ reads keep normal page caching")
        return
    _drop_page_cache = enabled


def drop_from_page_cache(fd: int) -> None:
    """Best-effort ``POSIX_FADV_DONTNEED`` over an entire open file.

    Only drops *clean* pages that no one else holds, so this is safe to
    call on shared files — a concurrent reader keeps its own pages.
    """
    if not _HAS_FADVISE:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:  # unsupported filesystem — the caller's I/O still succeeded
        pass


def read_npz(path: str | Path, *, allow_pickle: bool = False) -> Any:
    """:func:`numpy.load`, honoring the process page-cache policy.

    With the policy off this *is* ``np.load(path)``. With it on, the file is
    read whole, evicted from the page cache, and handed to numpy as an
    in-memory buffer — the members are DEFLATE-compressed either way, so
    this costs one compressed-file-sized buffer and no extra decode.

    Args:
        path: NPZ (or NPY) file to read.
        allow_pickle: Forwarded to :func:`numpy.load`. Required by the
            OpenFold-3 alignment / template files, which store object arrays.

    Returns:
        Whatever :func:`numpy.load` returns for this file.
    """
    if not _drop_page_cache:
        return np.load(path, allow_pickle=allow_pickle)
    with open(path, "rb") as f:
        buf = f.read()
        drop_from_page_cache(f.fileno())
    return np.load(io.BytesIO(buf), allow_pickle=allow_pickle)
