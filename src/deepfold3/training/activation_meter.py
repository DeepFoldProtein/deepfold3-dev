"""Per-layer activation RMS meter for the PairFormer trunk.

Always-on sentinel that captures ``mean(x²)`` of the single and pair
representations at the PairFormer input plus each of the 48 layer outputs,
the per-layer block-level branch contribution ``Δ = h_{l+1} - h_l``, and
each of the seven sub-branches inside every PF block:

* ``b_tri_out``, ``b_tri_in`` — triangle-multiplication branches (pair)
* ``b_pair_attn_row``, ``b_pair_attn_col`` — pair-stream attention
* ``b_pair_ffn`` — pair_transition (FFN)
* ``b_single_attn`` — single-stream attention
* ``b_single_ffn`` — single_transition (FFN)

Per branch we log ``RMS(branch)`` and ``ρ = RMS(branch) / RMS(post_block_stream)``
so attention-vs-FFN-vs-triangle injection imbalance is directly observable.
Stack stability requires ``ρ ≲ c/√L`` (Karras et al. 2024). The aggregate
block-level Δ is kept too because cross-branch interference means the
per-branch RMS values are not independently additive.

Costs:

* Two ``mean(x²)`` per stream point + two per block-level Δ + seven per
  block sub-branches ≈ 11·L + 2 scalar reductions per microstep (~530
  reductions for L=48).
* Local-rank accumulation in pre-allocated GPU tensors (no per-step
  ``.item()``, no per-step ``all_reduce``).
* One packed ``all_reduce`` at ``flush()`` boundary (typically every
  ``log_interval`` steps), interleaved with the existing
  :class:`DistributedAverageMeters` sync window.

The meter is attached via ``Evoformer._act_meter`` (used by
``Evoformer.forward`` for stream + block-level Δ) and ``PairFormerBlock._act_meter``
+ ``_act_layer_idx`` (used inside the block for per-branch). Trainer
toggles ``enabled`` around training forwards so eval forwards do not
contaminate train averages.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

# Single-track sub-branches captured inside the PF block. Index = bucket position
# in the per-branch accumulator. ``track`` is the residual stream the branch
# feeds, used for ρ denominator selection.
_BRANCHES: tuple[tuple[str, str], ...] = (
    ("tri_out", "pair"),
    ("tri_in", "pair"),
    ("pair_attn_row", "pair"),
    ("pair_attn_col", "pair"),
    ("pair_ffn", "pair"),
    ("single_attn", "single"),
    ("single_ffn", "single"),
)
_NUM_BRANCHES = len(_BRANCHES)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ActivationRMSMeter:
    """Accumulates stream + block-Δ + per-branch RMS for each PF block.

    Layout:

    * ``_stream_sum[track, point]`` — Σ mean(h²) at stream point
      (``num_layers + 1`` points; 0 = PF input, ``i+1`` = layer i output).
    * ``_blockdelta_sum[track, layer]`` — Σ mean(Δh²) for the post-block
      minus pre-block residual difference.
    * ``_branch_sum[bucket, layer]`` — Σ mean(branch²) for each of the
      seven sub-branches (see ``_BRANCHES``).
    """

    def __init__(self, num_layers: int, device: torch.device) -> None:
        self.num_layers = int(num_layers)
        self.num_points = self.num_layers + 1
        self.num_branches = _NUM_BRANCHES
        self.device = device
        self._stream_sum = torch.zeros((2, self.num_points), dtype=torch.float64, device=device)
        self._stream_count = torch.zeros((self.num_points,), dtype=torch.float64, device=device)
        self._blockdelta_sum = torch.zeros((2, self.num_layers), dtype=torch.float64, device=device)
        self._blockdelta_count = torch.zeros((self.num_layers,), dtype=torch.float64, device=device)
        self._branch_sum = torch.zeros((self.num_branches, self.num_layers), dtype=torch.float64, device=device)
        self._branch_count = torch.zeros((self.num_layers,), dtype=torch.float64, device=device)
        self.enabled = False

    @torch.no_grad()
    def record(self, point_idx: int, single: torch.Tensor, pair: torch.Tensor) -> None:
        """Add one forward's mean(x²) for both tracks at the stream point ``point_idx``."""
        if not self.enabled:
            return
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
        """Add one forward's mean(Δ²) for the block-level branch contribution.

        ``Δ`` is post-block minus pre-block — the total residual injection
        from the layer. Independent of the per-branch recording, since
        cross-branch sequential dependence makes ``|Δ|² ≠ Σ|branch_i|²``.
        """
        if not self.enabled:
            return
        self._blockdelta_sum[0, layer_idx] += single_delta.detach().float().pow(2).mean().to(torch.float64)
        self._blockdelta_sum[1, layer_idx] += pair_delta.detach().float().pow(2).mean().to(torch.float64)
        self._blockdelta_count[layer_idx] += 1.0

    @torch.no_grad()
    def record_all_branches(
        self,
        layer_idx: int,
        *,
        b_tri_out: torch.Tensor,
        b_tri_in: torch.Tensor,
        b_pair_attn_row: torch.Tensor,
        b_pair_attn_col: torch.Tensor,
        b_pair_ffn: torch.Tensor,
        b_single_attn: torch.Tensor,
        b_single_ffn: torch.Tensor,
    ) -> None:
        """Add one forward's mean(branch²) for each PF sub-branch.

        Branch tensors are captured BEFORE the dropout-add in the PF block
        forward — i.e., the "intended" residual injection (interpretation
        stable across dropout/no-dropout regimes).
        """
        if not self.enabled:
            return
        branches = (
            b_tri_out,
            b_tri_in,
            b_pair_attn_row,
            b_pair_attn_col,
            b_pair_ffn,
            b_single_attn,
            b_single_ffn,
        )
        for bi, b in enumerate(branches):
            self._branch_sum[bi, layer_idx] += b.detach().float().pow(2).mean().to(torch.float64)
        self._branch_count[layer_idx] += 1.0

    def reset(self) -> None:
        self._stream_sum.zero_()
        self._stream_count.zero_()
        self._blockdelta_sum.zero_()
        self._blockdelta_count.zero_()
        self._branch_sum.zero_()
        self._branch_count.zero_()

    @torch.no_grad()
    def flush_to_dict(self, prefix: str = "act_rms") -> dict[str, float]:
        """All-reduce and return a flat dict of per-layer RMS / branch / ρ values.

        Stream:     ``{prefix}/pf_in_{single,pair}``,
                    ``{prefix}/pf_<ii>_{single,pair}`` (ii = 00..L-1).
        Block Δ:    ``{prefix}/pf_<ii>_{single,pair}_branch``,
                    ``{prefix}/pf_<ii>_{single,pair}_rho``.
        Per branch: ``{prefix}/pf_<ii>_b_<branch>`` (RMS),
                    ``{prefix}/pf_<ii>_b_<branch>_rho`` (= RMS / post-stream RMS
                    of the track the branch feeds).

        Must be called by every rank in lockstep (collective). Returns
        empty dict if nothing accumulated. Caller is responsible for
        ``reset()`` afterwards.
        """
        P, L, B = self.num_points, self.num_layers, self.num_branches
        # Pack everything into one flat buffer for a single all_reduce.
        # Layout:
        #   [stream_sum_s (P), stream_sum_p (P), stream_count (P),
        #    blockdelta_sum_s (L), blockdelta_sum_p (L), blockdelta_count (L),
        #    branch_sum (B × L flat), branch_count (L)]
        size = 3 * P + 3 * L + B * L + L
        packed = torch.empty(size, dtype=torch.float64, device=self.device)
        o = 0
        packed[o : o + P] = self._stream_sum[0]
        o += P
        packed[o : o + P] = self._stream_sum[1]
        o += P
        packed[o : o + P] = self._stream_count
        o += P
        packed[o : o + L] = self._blockdelta_sum[0]
        o += L
        packed[o : o + L] = self._blockdelta_sum[1]
        o += L
        packed[o : o + L] = self._blockdelta_count
        o += L
        packed[o : o + B * L] = self._branch_sum.reshape(-1)
        o += B * L
        packed[o : o + L] = self._branch_count
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        raw = packed.tolist()  # single host sync

        out: dict[str, float] = {}
        # ----- Stream RMS -----
        post_rms_s: list[float | None] = [None] * P
        post_rms_p: list[float | None] = [None] * P
        for i in range(P):
            n = raw[2 * P + i]
            if n <= 0:
                continue
            rms_s = (raw[i] / n) ** 0.5 if raw[i] > 0 else 0.0
            rms_p = (raw[P + i] / n) ** 0.5 if raw[P + i] > 0 else 0.0
            post_rms_s[i] = rms_s
            post_rms_p[i] = rms_p
            label = "in" if i == 0 else f"{i - 1:02d}"
            out[f"{prefix}/pf_{label}_single"] = rms_s
            out[f"{prefix}/pf_{label}_pair"] = rms_p

        # ----- Block-level Δ + aggregate ρ -----
        off_bd = 3 * P
        for li in range(L):
            n = raw[off_bd + 2 * L + li]
            if n <= 0:
                continue
            rms_bs = (raw[off_bd + li] / n) ** 0.5 if raw[off_bd + li] > 0 else 0.0
            rms_bp = (raw[off_bd + L + li] / n) ** 0.5 if raw[off_bd + L + li] > 0 else 0.0
            label = f"{li:02d}"
            out[f"{prefix}/pf_{label}_single_branch"] = rms_bs
            out[f"{prefix}/pf_{label}_pair_branch"] = rms_bp
            post_s = post_rms_s[li + 1]
            post_p = post_rms_p[li + 1]
            if post_s and post_s > 0:
                out[f"{prefix}/pf_{label}_single_rho"] = rms_bs / post_s
            if post_p and post_p > 0:
                out[f"{prefix}/pf_{label}_pair_rho"] = rms_bp / post_p

        # ----- Per-branch RMS + ρ -----
        off_br = 3 * P + 3 * L
        for li in range(L):
            n = raw[off_br + B * L + li]
            if n <= 0:
                continue
            label = f"{li:02d}"
            for bi, (name, track) in enumerate(_BRANCHES):
                s_branch = raw[off_br + bi * L + li]
                if s_branch <= 0:
                    continue
                rms_b = (s_branch / n) ** 0.5
                out[f"{prefix}/pf_{label}_b_{name}"] = rms_b
                post = post_rms_s[li + 1] if track == "single" else post_rms_p[li + 1]
                if post and post > 0:
                    out[f"{prefix}/pf_{label}_b_{name}_rho"] = rms_b / post
        return out


def attach_activation_meter(model: nn.Module, device: torch.device) -> ActivationRMSMeter:
    """Attach an :class:`ActivationRMSMeter` to ``model.evoformer`` + every PF block.

    Sets ``evoformer._act_meter`` (so ``Evoformer.forward`` records stream
    points and block-level Δ) and ``block._act_meter`` / ``block._act_layer_idx``
    on every PairFormer block (so the block's forward can record per-branch
    contributions). Returns the meter for the trainer to flush at log
    boundaries.
    """
    evo = _unwrap(model).evoformer
    num_layers = len(evo.trunk_pairformer)
    meter = ActivationRMSMeter(num_layers, device)
    evo._act_meter = meter
    for i, blk in enumerate(evo.trunk_pairformer):
        blk._act_meter = meter
        blk._act_layer_idx = i
    return meter
