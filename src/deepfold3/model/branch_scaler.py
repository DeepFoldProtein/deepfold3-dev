"""Residual-branch scaler — extensibility hook for PairFormer residual branches.

A *branch scaler* is the function ``F_l(x) → s(F_l(x))`` applied to every
residual-stream branch output **before** the per-branch ``γ`` multiplier.
The default (``"identity"``) leaves the branch raw — pre-norm AF3 behaviour.
``"rmsnorm"`` forces every branch to unit-RMS so the residual injection
magnitude is decoupled from the branch's internal scale, the (non-learnable)
counterpart to Karras et al. 2024's magnitude-conserving residual blocks.

Currently exposed kinds:

* ``identity`` — no-op (AF3 baseline)
* ``rmsnorm`` — non-learnable RMSNorm; forces ``|branch|_rms = 1`` per token

Adding a new scaler is a single subclass + one branch in
:func:`make_branch_scaler`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from deepfold3.model.primitives import rms_normalize

BranchScalerKind = Literal["identity", "rmsnorm"]


class IdentityScaler(nn.Module):
    """``s(x) = x`` — leaves the branch output untouched (AF3 default)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RMSNormScaler(nn.Module):
    """``s(x) = x / RMS(x)`` along the last dim — non-learnable unit-RMS clamp.

    Matches the old ``residual_branch_rmsnorm=True`` behaviour: each branch
    output becomes unit-RMS per token regardless of its internal scale, so
    the per-branch γ is the *sole* knob controlling residual injection
    magnitude.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize(x)


def make_branch_scaler(kind: BranchScalerKind) -> nn.Module:
    """Construct a branch-scaler module by name.

    Channel-dim agnostic: the current scalers (identity, rmsnorm) are
    parameter-free, so the same module instance can be safely reused
    across branches with different channel counts. Future learnable
    scalers (LayerNorm, learned-γ RMSNorm) will require per-branch
    instances — at that point ``make_branch_scaler`` should also take
    a ``channels`` argument.
    """
    if kind == "identity":
        return IdentityScaler()
    if kind == "rmsnorm":
        return RMSNormScaler()
    raise ValueError(f"Unknown branch_scaler kind: {kind!r}")
