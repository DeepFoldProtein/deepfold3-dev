"""Exponential Moving Average for model parameters.

Constant-decay EMA over `requires_grad` parameters (matches OF3
``ExponentialMovingAverage`` semantics — no warm-start). Buffers are not
tracked: every DF3 buffer is a constant initialised once (Fourier
weights, ref-atom tables) or a derived shadow re-synced from params
(fused/tiled kernels), so EMAing them is a no-op or actively wrong.

Legacy fused-kernel parameter names are filtered out of the shadow dict
so old EMA state_dicts still load cleanly and so the EMA update doesn't
double-decay against the base params they shadow.

SF + EMA(x) compose pattern: when tracking Schedule-Free's averaged
iterate ``x`` rather than the live ``y``, the trainer wraps
``ema.update(model)`` in ``optimizer_eval_mode(optimizer)`` so
``param.data`` holds ``x`` for the duration of the foreach_lerp. See
``trainer.py`` for the wiring. ``reseed_from(model)`` re-clones the
shadow from the live params after a resume that has no saved EMA state
(e.g. when an EMA-on-x arm starts from a vanilla-SF checkpoint).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.config import OptimConfig


def _is_fused(name: str) -> bool:
    return any(m in name for m in OptimConfig.FUSED_STATE_DICT_MARKERS)


class EMA:
    """Exponential moving average of model parameters.

    Attributes:
        decay: Constant EMA decay factor.
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

        # Parallel-indexed lists consumed by ``torch._foreach_lerp_`` in
        # ``update``. Invalidated whenever ``shadow``/``param.data`` references
        # change (apply/restore swap, load_state_dict).
        self._cached_model_id: int | None = None
        self._shadow_list: list[torch.Tensor] = []
        self._param_list: list[torch.Tensor] = []

    def _invalidate_cache(self) -> None:
        self._cached_model_id = None

    def _rebuild_foreach_lists(self, model: nn.Module) -> None:
        self._shadow_list = []
        self._param_list = []
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._shadow_list.append(self.shadow[name])
                self._param_list.append(param.data)
        self._cached_model_id = id(model)

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters.

        Constant decay (no warm-start), matches OF3 convention.

        Args:
            model: Model with updated parameters.
        """
        self.num_updates += 1
        if self._cached_model_id != id(model):
            self._rebuild_foreach_lists(model)

        with torch.no_grad():
            torch._foreach_lerp_(self._shadow_list, self._param_list, 1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Swap model parameters with EMA shadow parameters in-place.

        Call ``restore()`` after evaluation to put original weights back.
        Uses pointer swapping (zero-copy, zero-allocation).

        Args:
            model: Model whose parameters will be replaced.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name], param.data = param.data, self.shadow[name]
        self._invalidate_cache()

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after ``apply()``.

        Args:
            model: Model whose parameters will be restored.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name], param.data = param.data, self.shadow[name]
        self._invalidate_cache()

    @torch.no_grad()
    def reseed_from(self, model: nn.Module) -> None:
        """Re-clone shadow buffers from the (possibly just-loaded) ``model.parameters``.

        Used on resume when the checkpoint has model + optimizer state but
        no saved EMA shadow (e.g. when an EMA-on-x arm starts from a
        vanilla-SF checkpoint). Without this, the shadow remains at the
        random init from ``__init__`` while the model holds the resumed
        weights — the EMA becomes meaningless until the resumed-weight
        signal decays through.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = param.data.clone()
        self._invalidate_cache()

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
        self._invalidate_cache()
