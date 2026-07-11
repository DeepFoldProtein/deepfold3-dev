"""Gradient-accumulation + chunked-diffusion DDP-sync gating.

A single :class:`AccumulationLoop` owns the two ``no_sync()`` decisions
that used to be scattered across ``trainer.py``:

1. Chunked-diffusion: ``forward_training`` back-props per chunk
   internally; those per-chunk backwards must run under
   ``model.no_sync()`` so DDP doesn't fire an all_reduce per chunk. The
   final aux backward runs OUTSIDE ``no_sync`` and fires the single
   DDP all_reduce for that microstep.

2. Gradient accumulation: non-final microsteps run under ``no_sync()``
   so DDP defers the gradient all_reduce until the last microstep.

Keeping both behind one class avoids the bug class where one path
forgets to set up the context. Every rank MUST call
``step_context`` / ``chunk_context`` on every call — dropping a call
on any rank desyncs collective participation and hangs NCCL.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch.nn as nn

from deepfold3.training.distributed import DistInfo


class AccumulationLoop:
    """Own the ``no_sync`` gating for both gradient accumulation and chunked diffusion."""

    def __init__(
        self, ddp_model: nn.Module, dist: DistInfo, num_steps: int, *, per_rank_clip: bool = False
    ) -> None:
        self.ddp_model = ddp_model
        self.dist = dist
        self.num_steps = num_steps
        # Per-rank grad clip defers ALL gradient all-reduce to a manual
        # reduce after clipping, so even the sync-step backward runs under
        # no_sync (there is no DDP all-reduce to leave un-suppressed).
        self.per_rank_clip = per_rank_clip

    def is_sync_step(self, global_step_before_increment: int) -> bool:
        """Return True when the microstep at ``global_step + 1`` is a sync boundary."""
        return (global_step_before_increment + 1) % self.num_steps == 0

    def step_context(self, *, is_sync_step: bool) -> AbstractContextManager:
        """Context manager for the outer (aux) backward inside ``_optimizer_step``.

        Suppress DDP all_reduce on non-final accumulation microsteps
        when running distributed. In per-rank-clip mode the sync-step
        backward is ALSO suppressed — the all-reduce is done manually after
        the per-rank clip in ``WrappedOptimizer._unscale_clip_and_measure``.
        """
        defer_all_reduce = self.per_rank_clip or not is_sync_step
        if defer_all_reduce and self.dist.is_distributed:
            return self.ddp_model.no_sync()
        return nullcontext()

    def chunk_context(self, *, chunked_diffusion: bool) -> AbstractContextManager:
        """Context manager wrapping ``forward_training`` under chunked diffusion.

        In chunked mode the per-chunk backwards run under ``no_sync`` so
        DDP doesn't fire per chunk; the single aux backward in
        ``_optimizer_step`` triggers the per-microstep all_reduce.
        """
        if chunked_diffusion and self.dist.is_distributed:
            return self.ddp_model.no_sync()
        return nullcontext()
