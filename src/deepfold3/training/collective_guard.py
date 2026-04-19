"""Named barrier fences around collective regions.

The guard's purpose is **attribution**, not stricter timeout enforcement.
NCCL already kills the process after ``init_process_group``'s configured
``timeout=`` — the guard renames the failure so the error points at the
specific region that desynced (e.g. ``step_or_skip`` vs. eval) instead
of an anonymous all_gather / all_reduce deep inside another function.

Defends against the class of bug where a code-path difference on one
rank (``try/except`` around a DDP op, an extra ``continue``) leaves
other ranks blocked on the next collective until the watchdog fires.
See ``memory/project_try_except_continue_deadlock.md`` and
``memory/project_bp2_seed42_deadlock.md``.

Cost: two 1-element all-reduces per guarded region. On 8×NVLink that's
<1ms; per sync step at DF3's 1-2s cadence it is well below the noise
floor.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed as dist

from deepfold3.training.distributed import DistInfo

logger = logging.getLogger(__name__)


def _barrier(name: str, side: str, dist_info: DistInfo) -> None:
    """Run ``dist.barrier`` and re-raise any failure with the guard name."""
    try:
        dist.barrier(device_ids=[dist_info.local_rank])
    except RuntimeError as e:
        raise RuntimeError(f"collective_guard({name!r}) {side} barrier: {e}") from e


@contextmanager
def collective_guard(name: str, dist_info: DistInfo) -> Iterator[None]:
    """Fence a region that every rank must enter and leave together.

    Single-process / non-distributed mode is a no-op. In distributed
    mode, issues one barrier on entry and one on exit; body exceptions
    propagate unchanged so the exit barrier is skipped on the failing
    rank (which is fine — the rank is terminating and other ranks will
    hit the NCCL watchdog on their next collective).

    The guard does NOT enforce a stricter deadline than the watchdog.
    It only attaches a human-readable name to the failure so hangs are
    diagnosable from a single stack trace instead of requiring NCCL log
    forensics.
    """
    if not dist_info.is_distributed:
        yield
        return
    _barrier(name, "entry", dist_info)
    yield
    _barrier(name, "exit", dist_info)
