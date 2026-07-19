"""Distributed average meters with deferred all-reduce at log time."""

from __future__ import annotations

import torch
import torch.distributed as dist


class DistributedAverageMeters:
    """Group of named metrics with deferred DDP all-reduce.

    Per-step ``update`` is collective-free — values accumulate locally as GPU
    tensors. ``sync`` packs all metric (sum, count) pairs into a single
    f64 tensor and does ONE ``all_reduce`` to compute the global weighted
    average across ranks. The "all-reduce only at log time" pattern avoids
    the per-step sync barrier that would otherwise serialize fast and slow
    ranks at every step.

    Important: ``sync`` MUST be called by every rank in lockstep (it issues a
    collective). Place the call right after the optimizer step, where DDP's
    own gradient all-reduce has already synchronized the ranks for free —
    this makes the meter's all-reduce essentially zero-cost.

    Per-key counts: ``training_loss`` returns early for distillation samples
    (no distogram/confidence keys), so per-rank key sets diverge during a
    window. Each key tracks its own count of finite contributions so the
    averaged value is ``global_sum / global_count_for_that_key``, not
    ``global_sum / total_microsteps``. Non-finite values (NaN/Inf from
    skipped windows) are dropped at update time so the displayed averages
    aren't poisoned by a single bad batch.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}  # f64 device tensors
        self._counts: dict[str, int] = {}  # finite-contribution count, per key

    def update(self, values: dict[str, torch.Tensor], n: int | dict[str, int] = 1) -> None:
        """Accumulate a step's metrics. Non-finite values are dropped.

        Detached, cast to f64, squeezed to scalar (the per-batch loss loop
        with ``batch_per_rank > 1`` produces 0-d or 1-element tensors).

        Args:
            values: Metric name → scalar value for this step.
            n: Weight for this sample. Pass a dict to weight per key, which is
                what the training loop does: not every batch element emits
                every key (a distillation sample skips the confidence terms),
                so a rank that contributed one sample must not be averaged
                against one that contributed two as if they were equal.
                Missing keys default to 1.
        """
        for k, v in values.items():
            n_k = n.get(k, 1) if isinstance(n, dict) else n
            if n_k <= 0:
                continue
            v_f64 = v.detach().to(torch.float64).squeeze()
            if not bool(torch.isfinite(v_f64).all()):
                # NaN/Inf from a skipped window — drop it. Without this,
                # one bad microstep poisons every subsequent ``sync`` until
                # ``reset`` runs at the next log boundary.
                continue
            v_f64_n = v_f64 * n_k
            if k in self._sums:
                self._sums[k] += v_f64_n
                self._counts[k] += n_k
            else:
                self._sums[k] = v_f64_n.clone()
                self._counts[k] = n_k

    @property
    def count(self) -> int:
        """Max contribution count across keys; 0 if nothing accumulated."""
        return max(self._counts.values(), default=0)

    def sync(self) -> tuple[dict[str, float], dict[str, float]]:
        """All-reduce across ranks and return ``(averages, sums)``.

        Both dicts share the same keys. ``averages`` divides by per-key
        global count (correct under rank-divergent key sets — distillation
        ranks miss the distogram key, but only ranks that contributed get
        counted). ``sums`` gives the raw global totals (for counters like
        token counts that should not be averaged).

        Per-rank key sets can diverge during a window: ``training_loss``
        returns early for distillation samples (no distogram/confidence
        keys). We ``all_gather_object`` the local key list, pack the union,
        and fill missing entries with (0, 0) — the pack shape is then
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

        # Interleaved pack: [sum_k0, count_k0, sum_k1, count_k1, ...].
        # One all_reduce yields both the global sum AND the global count
        # for every key, so a key contributed by N of W ranks averages over
        # N (the correct denominator), not over the full microstep total.
        packed = torch.zeros(2 * len(keys), dtype=torch.float64, device=self.device)
        for i, k in enumerate(keys):
            if k in self._sums:
                packed[2 * i] = self._sums[k]
                packed[2 * i + 1] = float(self._counts[k])
        if multi:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        # Single GPU→CPU sync at the very end.
        raw = packed.tolist()
        sums: dict[str, float] = {}
        averaged: dict[str, float] = {}
        for i, k in enumerate(keys):
            global_sum = raw[2 * i]
            global_count = raw[2 * i + 1]
            sums[k] = global_sum
            averaged[k] = global_sum / global_count if global_count > 0 else 0.0
        return averaged, sums

    def reset(self) -> None:
        for t in self._sums.values():
            t.zero_()
        for k in self._counts:
            self._counts[k] = 0
