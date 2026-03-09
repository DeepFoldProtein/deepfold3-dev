"""Exponential Moving Average for model parameters.

Maintains shadow copies of model parameters, updated each optimizer step.
Swap in EMA weights for evaluation, then restore originals.

Fused kernel parameters (names containing ``_fused_gate.``,
``_fused_ada_ln_linear.``, ``_fused_swiglu.``, ``_ln_linear.``, ``_swiglu.``)
are excluded — they are derived from base weights and re-synced after
``apply()``/``restore()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.config import OptimConfig


def _is_fused(name: str) -> bool:
    return any(m in name for m in OptimConfig.FUSED_STATE_DICT_MARKERS)


def _sync_fused(model: nn.Module) -> None:
    """Re-sync any fused kernel parameters from their base weights."""
    if hasattr(model, "_sync_fused_kernels"):
        model._sync_fused_kernels()


class EMA:
    """Exponential moving average of model parameters.

    Attributes:
        decay: Target EMA decay factor.
        num_updates: Number of update calls so far.
        shadow: Shadow copies of trainable parameters (excludes fused kernels).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initializes EMA from the current model parameters.

        Args:
            model: The model whose parameters to track.
            decay: EMA decay factor (e.g. 0.999).
        """
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad and not _is_fused(name):
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters.

        Uses warm-start decay: ``min(decay, (1 + n) / (10 + n))``.

        Args:
            model: Model with updated parameters.
        """
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].lerp_(param.data, 1.0 - decay)

    def apply(self, model: nn.Module) -> None:
        """Swap model parameters with EMA shadow parameters in-place.

        Call ``restore()`` after evaluation to put original weights back.
        Uses pointer swapping (zero-copy, zero-allocation).
        After swapping, fused kernels are re-synced from the new base weights.

        Args:
            model: Model whose parameters will be replaced.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name], param.data = param.data, self.shadow[name]
        _sync_fused(model)

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after ``apply()``.

        Swaps the shadow and model weights back, then re-syncs fused kernels.

        Args:
            model: Model whose parameters will be restored.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name], param.data = param.data, self.shadow[name]
        _sync_fused(model)

    def state_dict(self) -> dict:
        """Returns serializable EMA state."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Loads EMA state from a dict.

        Args:
            state: Dict with keys ``decay``, ``num_updates``, ``shadow``.
        """
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
