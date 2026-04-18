"""Process-group bring-up / teardown and a small ``DistInfo`` value object.

Design choice: we do NOT adopt accelerate's ``PartialState`` / singleton
pattern. Explicit ``DistInfo`` passing surfaces collective-ordering
bugs at construction time instead of hiding them behind global state —
worth the plumbing cost given prior debugging history (see
``memory/project_try_except_continue_deadlock.md``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistInfo:
    """Rank-local distributed state captured once at startup."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_process_group() -> DistInfo:
    """Initialize NCCL and return the captured :class:`DistInfo`.

    Reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` from the environment
    (populated by ``torchrun``). In single-process mode, returns a
    trivial :class:`DistInfo` without initializing a process group.

    ``torch.cuda.set_device(local_rank)`` MUST run before any model
    construction — it pins the Triton autotune cache fingerprint to the
    correct GPU (see ``memory/reference_triton_cache.md``).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # ``set_device`` must run BEFORE ``init_process_group`` — the latter
        # captures the current device for its internal barriers, and
        # passing ``device_id`` avoids the
        # ``barrier(): using the device under current context`` warning.
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def destroy_process_group_safely() -> None:
    """Collective teardown of the NCCL process group.

    Per the PyTorch reference: every trainer process must call
    ``destroy_process_group`` exactly once. The call is a collective —
    all ranks must reach it, so it must NOT live in a ``try/finally``
    that could fire at different times on different ranks and cause a
    deadlock. The ``barrier`` ensures all ranks finished before
    teardown.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
