"""Single-track stability probe for the PairFormer trunk.

Captures per-block diagnostics during selected forward passes — the alignment
between the residual single representation ``s`` and the next ``single_transition``
update ``u``, plus the magnitudes of both. Writes one JSONL line per probed
step to a file (no WandB dependency); meant to be opened concurrently by
multiple runs (each writes to its own file).

Usage:
    probe = SingleTrackProbe(model, log_path)
    probe.begin_step(target_effective_step)
    # ... forward pass populates probe via PairFormerBlock hooks ...
    probe.end_step(extra={"loss": ..., "step": ...})
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class SingleTrackProbe:
    """Captures single-track residual / update statistics on demand.

    Attaches a ``_probe`` reference + ``_probe_layer_idx`` index to every
    ``PairFormerBlock`` in ``model.evoformer.trunk_pairformer``. The block's
    forward, when ``probe.enabled`` is True, calls ``record_block`` with the
    pre-update ``single`` (= ``s``), the raw ``single_transition(single)``
    (= ``u``), and the residual scale ``γ`` ( = ``s_single_ffn``).

    Each call to ``begin_step`` clears the per-block buffer; ``end_step``
    aggregates and appends one JSONL record to ``log_path``. Probing is
    safe under ``torch.autocast`` since stats are computed in ``no_grad``
    on detached fp32 copies.
    """

    def __init__(self, model: nn.Module, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = False
        self._records: list[dict] = []
        self._target_step: int | None = None

        evo = _unwrap(model).evoformer
        self._blocks: list[nn.Module] = list(evo.trunk_pairformer)
        for i, blk in enumerate(self._blocks):
            blk._probe = self
            blk._probe_layer_idx = i
        self._num_layers = len(self._blocks)

    def begin_step(self, target_step: int) -> None:
        self.enabled = True
        self._records = []
        self._target_step = int(target_step)

    @torch.no_grad()
    def record_block(
        self,
        layer_idx: int,
        single_in: torch.Tensor,
        u: torch.Tensor,
        gamma: float,
    ) -> None:
        s = single_in.detach().float()
        uf = u.detach().float()
        gu = gamma * uf
        rms_s = s.pow(2).mean().sqrt().item()
        rms_u = uf.pow(2).mean().sqrt().item()
        rms_gu = gu.pow(2).mean().sqrt().item()
        # Per-token cosine over the channel axis, then mean across tokens / batch.
        ns = s.norm(dim=-1).clamp_min(1e-8)
        nu = uf.norm(dim=-1).clamp_min(1e-8)
        dot = (s * uf).sum(-1)
        cos_su = (dot / (ns * nu)).mean().item()
        # Per-token alpha = (s · u) / ‖s‖². Sign tells us whether u is reinforcing
        # or opposing the residual stream; magnitude is the radial-gain coefficient.
        alpha = (dot / s.pow(2).sum(-1).clamp_min(1e-8)).mean().item()
        self._records.append(
            {
                "layer": int(layer_idx),
                "rms_s": rms_s,
                "rms_u": rms_u,
                "rms_gu": rms_gu,
                "cos_su": cos_su,
                "alpha_su": alpha,
                "gamma": float(gamma),
            }
        )

    @torch.no_grad()
    def _weight_norms(self) -> dict[str, float]:
        """Frobenius norm of single_transition.linear2 and single_attn.out_proj
        per block, summarised as mean / min / max across layers."""
        w2_norms = []
        wo_norms = []
        for blk in self._blocks:
            w2 = blk.single_transition.linear2.weight.detach().float()
            wo = blk.single_attn.out_proj.weight.detach().float()
            w2_norms.append(w2.norm().item())
            wo_norms.append(wo.norm().item())
        import statistics as st

        return {
            "w_single_trans_l2_mean": st.fmean(w2_norms),
            "w_single_trans_l2_min": min(w2_norms),
            "w_single_trans_l2_max": max(w2_norms),
            "w_single_trans_l2_l0": w2_norms[0],
            "w_single_trans_l2_l_last": w2_norms[-1],
            "w_single_attn_out_mean": st.fmean(wo_norms),
            "w_single_attn_out_min": min(wo_norms),
            "w_single_attn_out_max": max(wo_norms),
            "w_single_attn_out_l0": wo_norms[0],
            "w_single_attn_out_l_last": wo_norms[-1],
        }

    def end_step(self, extra: dict | None = None) -> None:
        """Aggregate per-block records, append one JSONL row, and disable."""
        if not self.enabled:
            return
        self.enabled = False
        if not self._records:
            self._target_step = None
            return

        per_layer = {r["layer"]: r for r in self._records}
        L = self._num_layers
        keys = ["rms_s", "rms_u", "rms_gu", "cos_su", "alpha_su"]
        out: dict = {"step": self._target_step}
        if extra:
            out.update(extra)
        for k in keys:
            vals = [per_layer[i][k] for i in range(L) if i in per_layer]
            if not vals:
                continue
            out[f"{k}_mean"] = sum(vals) / len(vals)
            out[f"{k}_min"] = min(vals)
            out[f"{k}_max"] = max(vals)
            out[f"{k}_l0"] = per_layer.get(0, {}).get(k)
            out[f"{k}_l_mid"] = per_layer.get(L // 2, {}).get(k)
            out[f"{k}_l_last"] = per_layer.get(L - 1, {}).get(k)
        # Per-step aligned-gain budget. Under aligned gain (cos(s,u) > 0),
        # one forward grows ‖s‖ by Π_k (1 + γ_k·α_k) ≈ exp(Σ_k γ_k·α_k).
        # ``sum_gamma_alpha`` is the direct depth-summed gain coefficient
        # for this step; ``max_gamma_alpha`` flags any single block doing
        # the bulk of the radial work.
        ga = [per_layer[i]["gamma"] * per_layer[i]["alpha_su"] for i in range(L) if i in per_layer]
        if ga:
            out["sum_gamma_alpha"] = float(sum(ga))
            out["max_gamma_alpha"] = float(max(ga, key=abs))
        out.update(self._weight_norms())
        with self.log_path.open("a") as f:
            f.write(json.dumps(out) + "\n")
        self._records = []
        self._target_step = None

    def discard_step(self) -> None:
        """Drop the in-flight buffer (e.g. on NaN-skip, OOM)."""
        self.enabled = False
        self._records = []
        self._target_step = None
