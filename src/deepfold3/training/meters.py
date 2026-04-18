"""Distributed average meters with deferred all-reduce at log time."""

from __future__ import annotations

import torch
import torch.distributed as dist


class DistributedAverageMeters:
    """Group of named metrics with deferred DDP all-reduce.

    Per-step ``update`` is collective-free — values accumulate locally as GPU
    tensors. ``sync`` packs all metric sums + a shared count into a single
    f64 tensor and does ONE ``all_reduce`` to compute the global weighted
    average across ranks. The "all-reduce only at log time" pattern avoids
    the per-step sync barrier that would otherwise serialize fast and slow
    ranks at every step.

    Important: ``sync`` MUST be called by every rank in lockstep (it issues a
    collective). Place the call right after the optimizer step, where DDP's
    own gradient all-reduce has already synchronized the ranks for free —
    this makes the meter's all-reduce essentially zero-cost.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}  # f64 device tensors
        self._count: int = 0

    def update(self, values: dict[str, torch.Tensor], n: int = 1) -> None:
        """Accumulate a step's metrics. Tensors are detached and stored on device."""
        for k, v in values.items():
            # Squeeze to scalar — with batch_per_rank > 1 the per-batch
            # loss loop may produce 0-d or 1-element tensors.
            v_f64 = v.detach().to(torch.float64).squeeze() * n
            if k in self._sums:
                self._sums[k] += v_f64
            else:
                self._sums[k] = v_f64.clone()
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    def sync(self) -> tuple[dict[str, float], dict[str, float]]:
        """All-reduce across ranks and return ``(averages, sums)``.

        Both dicts share the same keys. ``averages`` divides by global count
        (for losses). ``sums`` gives the raw global totals (for counters like
        token counts that should not be averaged).

        Per-rank key sets can diverge during a window: ``training_loss``
        returns early for distillation samples (no distogram/confidence
        keys), so ranks that drew mostly distillation samples end up with
        fewer keys than others. Packing different-sized tensors into
        ``all_reduce`` causes NCCL to hang / time out. We fix this by
        ``all_gather_object`` of the local key list and pack the union,
        filling missing entries with zero — the pack shape is then
        invariant across ranks.
        """
        multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if multi:
            local_keys = sorted(self._sums.keys())
            gathered: list[list[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_keys)
            keys = sorted({k for ks in gathered if ks for k in ks})
        else:
            keys = sorted(self._sums.keys())

        if not keys:
            return {}, {}

        # Pack [sum_k0, sum_k1, ..., sum_kN-1, count] into one f64 tensor.
        # Keys missing locally are packed as 0 without mutating ``self._sums``
        # so the window-local key set stays bounded to what this rank saw.
        packed = torch.zeros(len(keys) + 1, dtype=torch.float64, device=self.device)
        for i, k in enumerate(keys):
            if k in self._sums:
                packed[i] = self._sums[k]
        packed[-1] = float(self._count)
        if multi:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_count = packed[-1].item()
        if global_count == 0:
            return dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0.0)
        # Single GPU→CPU sync at the very end.
        raw = packed[:-1].tolist()
        sums = dict(zip(keys, raw, strict=True))
        averaged = {k: v / global_count for k, v in sums.items()}
        return averaged, sums

    def reset(self) -> None:
        for t in self._sums.values():
            t.zero_()
        self._count = 0
