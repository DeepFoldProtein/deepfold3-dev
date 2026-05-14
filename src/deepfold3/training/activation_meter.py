"""Per-layer activation RMS meter for the PairFormer trunk.

Always-on sentinel that captures ``mean(x²)`` of the single and pair
representations at the PairFormer input plus each of the 48 layer outputs.
Costs:

* One ``mean(x²)`` per measurement point per forward (98 scalar reductions
  for the default 48-layer trunk).
* Local-rank accumulation in pre-allocated GPU tensors (no per-step ``.item()``,
  no per-step ``all_reduce``).
* One packed ``all_reduce`` at ``flush()`` boundary (typically every
  ``log_interval`` steps), interleaved with the existing
  :class:`DistributedAverageMeters` sync window.

The meter is attached to an :class:`Evoformer` instance via
``Evoformer._act_meter`` and consumed by ``Evoformer.forward``. Trainer
toggles ``enabled`` around training forwards so eval forwards do not
contaminate the train averages.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ActivationRMSMeter:
    """Accumulates single/pair RMS at PF input + each layer output.

    Layout: ``self._sum`` is a ``[2, num_points]`` f64 device tensor where
    row 0 is single, row 1 is pair; ``num_points = num_layers + 1`` (index 0
    is the pre-stack input, indices 1..L are the output of layer ``i-1``).
    """

    def __init__(self, num_layers: int, device: torch.device) -> None:
        self.num_layers = int(num_layers)
        self.num_points = self.num_layers + 1
        self.device = device
        self._sum = torch.zeros((2, self.num_points), dtype=torch.float64, device=device)
        self._count = torch.zeros((self.num_points,), dtype=torch.float64, device=device)
        self.enabled = False

    @torch.no_grad()
    def record(self, point_idx: int, single: torch.Tensor, pair: torch.Tensor) -> None:
        """Add one forward's mean(x²) for both tracks at ``point_idx``.

        Cheap: two reductions, two scalar add-in-place. No host sync.
        Called from inside ``Evoformer.forward``; recompute under gradient
        checkpointing harmlessly double-counts (sum and count both ×2 → average
        unchanged).
        """
        if not self.enabled:
            return
        # ``mean()`` over all elements (B, N[, N], C). Cast to f64 for stable
        # accumulation across many microsteps; the .pow(2).mean() is the cheap
        # part — the cast adds ~zero overhead since the result is scalar.
        self._sum[0, point_idx] += single.detach().float().pow(2).mean().to(torch.float64)
        self._sum[1, point_idx] += pair.detach().float().pow(2).mean().to(torch.float64)
        self._count[point_idx] += 1.0

    def reset(self) -> None:
        self._sum.zero_()
        self._count.zero_()

    @torch.no_grad()
    def flush_to_dict(self, prefix: str = "act_rms") -> dict[str, float]:
        """All-reduce and return a flat dict of per-layer RMS values.

        Keys: ``{prefix}/pf_in_single``, ``{prefix}/pf_in_pair``,
        ``{prefix}/pf_<ii>_single``, ``{prefix}/pf_<ii>_pair`` for
        ``ii = 00..(L-1)``.

        Must be called by every rank in lockstep (collective). Returns
        empty dict if nothing accumulated. Caller is responsible for
        ``reset()`` afterwards.
        """
        # Pack [sum_single (P,), sum_pair (P,), count (P,)] for one all_reduce.
        P = self.num_points
        packed = torch.empty(3 * P, dtype=torch.float64, device=self.device)
        packed[:P] = self._sum[0]
        packed[P : 2 * P] = self._sum[1]
        packed[2 * P :] = self._count
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        raw = packed.tolist()  # single host sync

        out: dict[str, float] = {}
        for i in range(P):
            n = raw[2 * P + i]
            if n <= 0:
                continue
            mean_sq_s = raw[i] / n
            mean_sq_p = raw[P + i] / n
            # RMS = sqrt(mean(x²)).
            rms_s = mean_sq_s**0.5 if mean_sq_s > 0 else 0.0
            rms_p = mean_sq_p**0.5 if mean_sq_p > 0 else 0.0
            label = "in" if i == 0 else f"{i - 1:02d}"
            out[f"{prefix}/pf_{label}_single"] = rms_s
            out[f"{prefix}/pf_{label}_pair"] = rms_p
        return out


def attach_activation_meter(model: nn.Module, device: torch.device) -> ActivationRMSMeter:
    """Attach an :class:`ActivationRMSMeter` to ``model.evoformer``.

    Sets ``evoformer._act_meter`` so ``Evoformer.forward`` records into it
    on each PairFormer-layer boundary. Returns the meter for the trainer
    to flush at log boundaries.
    """
    evo = _unwrap(model).evoformer
    num_layers = len(evo.trunk_pairformer)
    meter = ActivationRMSMeter(num_layers, device)
    evo._act_meter = meter
    return meter
