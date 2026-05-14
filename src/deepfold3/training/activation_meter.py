"""Per-layer activation RMS meter for the PairFormer trunk.

Always-on sentinel that captures ``mean(x²)`` of the single and pair
representations at the PairFormer input plus each of the 48 layer outputs,
and (per layer) the magnitude of the **branch contribution** ``Δ = h_{l+1} - h_l``.
The latter gives ``ρ_l = RMS(Δh_l) / RMS(h_{l+1})``, the standard
residual-injection ratio (cf. Karras et al. 2024). Residual-stack stability
requires ``ρ_l ≲ c/√L``; persistently larger values for many layers signal
runaway energy injection.

Costs:

* Two ``mean(x²)`` per measurement point per forward, plus two
  ``mean(x²)`` on ``Δsingle``/``Δpair`` per PF block. ``num_layers + 1``
  stream points + ``num_layers`` branch points = ``2L + 1`` total points
  × 2 tracks ≈ 4L+2 scalar reductions per microstep.
* Local-rank accumulation in pre-allocated GPU tensors (no per-step
  ``.item()``, no per-step ``all_reduce``).
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
    """Accumulates stream + branch RMS for each PF block.

    Stream points: ``num_layers + 1`` (PF input plus output of each layer).
    Branch points: ``num_layers`` (Δ = post-block − pre-block, per layer).

    Two GPU accumulators of mean-square:

    * ``_stream_sum[track, point]`` — Σ RMS²(h) at ``point`` for ``track``
      ∈ {single, pair}; ``point ∈ [0, num_layers]`` with 0 = PF input,
      ``i + 1`` = output of layer ``i``.
    * ``_branch_sum[track, layer]`` — Σ RMS²(Δh) for layer ``∈ [0, num_layers)``.

    Counts use the same per-point granularity so distillation samples or
    NaN-skipped microsteps don't bias the window average.
    """

    def __init__(self, num_layers: int, device: torch.device) -> None:
        self.num_layers = int(num_layers)
        self.num_points = self.num_layers + 1
        self.device = device
        self._stream_sum = torch.zeros((2, self.num_points), dtype=torch.float64, device=device)
        self._stream_count = torch.zeros((self.num_points,), dtype=torch.float64, device=device)
        self._branch_sum = torch.zeros((2, self.num_layers), dtype=torch.float64, device=device)
        self._branch_count = torch.zeros((self.num_layers,), dtype=torch.float64, device=device)
        self.enabled = False

    @torch.no_grad()
    def record(self, point_idx: int, single: torch.Tensor, pair: torch.Tensor) -> None:
        """Add one forward's mean(x²) for both tracks at the stream point ``point_idx``.

        Cheap: two reductions, two scalar add-in-place. No host sync.
        Called from inside ``Evoformer.forward``; recompute under gradient
        checkpointing harmlessly double-counts (sum and count both ×2 →
        window average unchanged).
        """
        if not self.enabled:
            return
        # ``mean()`` over all elements (B, N[, N], C). Cast to f64 for stable
        # accumulation across many microsteps; the .pow(2).mean() is the cheap
        # part — the cast adds ~zero overhead since the result is scalar.
        self._stream_sum[0, point_idx] += single.detach().float().pow(2).mean().to(torch.float64)
        self._stream_sum[1, point_idx] += pair.detach().float().pow(2).mean().to(torch.float64)
        self._stream_count[point_idx] += 1.0

    @torch.no_grad()
    def record_branch(
        self,
        layer_idx: int,
        single_delta: torch.Tensor,
        pair_delta: torch.Tensor,
    ) -> None:
        """Add one forward's mean(Δ²) for the branch contribution of layer ``layer_idx``.

        ``single_delta`` / ``pair_delta`` are the post-block minus pre-block
        residual differences — the total branch injection for that layer.
        Allocates a temporary subtracted tensor at the call site (one extra
        ``single`` + ``pair`` allocation, freed after the reduction).
        """
        if not self.enabled:
            return
        self._branch_sum[0, layer_idx] += single_delta.detach().float().pow(2).mean().to(torch.float64)
        self._branch_sum[1, layer_idx] += pair_delta.detach().float().pow(2).mean().to(torch.float64)
        self._branch_count[layer_idx] += 1.0

    def reset(self) -> None:
        self._stream_sum.zero_()
        self._stream_count.zero_()
        self._branch_sum.zero_()
        self._branch_count.zero_()

    @torch.no_grad()
    def flush_to_dict(self, prefix: str = "act_rms") -> dict[str, float]:
        """All-reduce and return a flat dict of per-layer RMS / branch / ρ values.

        Stream keys: ``{prefix}/pf_in_{single,pair}``,
        ``{prefix}/pf_<ii>_{single,pair}`` for ``ii = 00..(L-1)``.
        Branch keys: ``{prefix}/pf_<ii>_{single,pair}_branch``.
        Ratio keys: ``{prefix}/pf_<ii>_{single,pair}_rho``
        (RMS(branch) / RMS(post-block stream)).

        Must be called by every rank in lockstep (collective). Returns
        empty dict if nothing accumulated. Caller is responsible for
        ``reset()`` afterwards.
        """
        P, L = self.num_points, self.num_layers
        # Pack [stream_sum_single (P,), stream_sum_pair (P,), stream_count (P,),
        #       branch_sum_single (L,), branch_sum_pair (L,), branch_count (L,)]
        # for one all_reduce.
        packed = torch.empty(3 * P + 3 * L, dtype=torch.float64, device=self.device)
        packed[:P] = self._stream_sum[0]
        packed[P : 2 * P] = self._stream_sum[1]
        packed[2 * P : 3 * P] = self._stream_count
        off = 3 * P
        packed[off : off + L] = self._branch_sum[0]
        packed[off + L : off + 2 * L] = self._branch_sum[1]
        packed[off + 2 * L : off + 3 * L] = self._branch_count
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        raw = packed.tolist()  # single host sync

        out: dict[str, float] = {}
        # Stream RMS at each point.
        post_rms_single: list[float | None] = [None] * P
        post_rms_pair: list[float | None] = [None] * P
        for i in range(P):
            n = raw[2 * P + i]
            if n <= 0:
                continue
            mean_sq_s = raw[i] / n
            mean_sq_p = raw[P + i] / n
            rms_s = mean_sq_s**0.5 if mean_sq_s > 0 else 0.0
            rms_p = mean_sq_p**0.5 if mean_sq_p > 0 else 0.0
            post_rms_single[i] = rms_s
            post_rms_pair[i] = rms_p
            label = "in" if i == 0 else f"{i - 1:02d}"
            out[f"{prefix}/pf_{label}_single"] = rms_s
            out[f"{prefix}/pf_{label}_pair"] = rms_p

        # Branch RMS + ρ for each layer.
        # Layer i's branch contribution feeds into stream point (i + 1) — the
        # post-block residual — so ρ = RMS(branch_i) / RMS(stream_{i+1}).
        for li in range(L):
            n = raw[off + 2 * L + li]
            if n <= 0:
                continue
            mean_sq_bs = raw[off + li] / n
            mean_sq_bp = raw[off + L + li] / n
            rms_bs = mean_sq_bs**0.5 if mean_sq_bs > 0 else 0.0
            rms_bp = mean_sq_bp**0.5 if mean_sq_bp > 0 else 0.0
            label = f"{li:02d}"
            out[f"{prefix}/pf_{label}_single_branch"] = rms_bs
            out[f"{prefix}/pf_{label}_pair_branch"] = rms_bp
            post_s = post_rms_single[li + 1]
            post_p = post_rms_pair[li + 1]
            if post_s and post_s > 0:
                out[f"{prefix}/pf_{label}_single_rho"] = rms_bs / post_s
            if post_p and post_p > 0:
                out[f"{prefix}/pf_{label}_pair_rho"] = rms_bp / post_p
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
