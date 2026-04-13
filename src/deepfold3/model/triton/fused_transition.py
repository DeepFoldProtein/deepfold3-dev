"""Fused TransitionBlock: LayerNorm + Linear + SwiGLU + Linear in one module.

Composes FusedLayerNormLinear + FusedSwiGLU + LinearNoBias into a single
module that is a drop-in replacement for deepfold3.model.primitives.TransitionBlock.

Weight names match TransitionBlock for load_state_dict compatibility:
    norm.weight, norm.bias  -> FusedLayerNormLinear.weight, .bias
    linear1.weight          -> FusedLayerNormLinear.linear_weight (transposed)
    linear2.weight          -> self.linear2.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.primitives import LinearNoBias

try:
    from deepfold3.model.triton.layernorm_linear import HAS_TRITON as HAS_LNL
    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear
except ImportError:
    HAS_LNL = False

try:
    from deepfold3.model.triton.swiglu import HAS_TRITON as HAS_SWIGLU
    from deepfold3.model.triton.swiglu import FusedSwiGLU
except ImportError:
    HAS_SWIGLU = False

HAS_TRITON = HAS_LNL and HAS_SWIGLU


class FusedTransitionBlock(nn.Module):
    """Drop-in replacement for TransitionBlock using fused Triton kernels.

    Falls back to unfused ops when Triton is unavailable or input is on CPU.

    Args:
        channels: Input/output dimension.
        num_intermediate_factor: Expansion factor for intermediate dimension.
    """

    def __init__(self, channels: int, num_intermediate_factor: int = 4):
        super().__init__()
        self.channels = channels
        num_intermediate = channels * num_intermediate_factor

        if HAS_TRITON:
            # Fused LN + Linear (projects to 2 * intermediate for SwiGLU split)
            self._ln_linear = FusedLayerNormLinear(channels, num_intermediate * 2, has_linear_bias=False)
            self._swiglu = FusedSwiGLU()
        else:
            self._ln_linear = None
            self._swiglu = None

        # These are always created for weight compatibility
        # AF3: transition1='relu', transition2=final_init ('zeros')
        self.norm = nn.LayerNorm(channels)
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")

        if HAS_TRITON:
            self._sync_to_fused()

    def _sync_to_fused(self):
        """Copy weights from standard modules to fused kernel parameters."""
        if self._ln_linear is not None:
            with torch.no_grad():
                self._ln_linear.weight.data.copy_(self.norm.weight.data)
                self._ln_linear.bias.data.copy_(self.norm.bias.data)
                self._ln_linear.linear_weight.data.copy_(self.linear1.weight.data.T)

    def _sync_from_fused(self):
        """Copy weights from fused kernel parameters back to standard modules."""
        if self._ln_linear is not None:
            with torch.no_grad():
                self.norm.weight.data.copy_(self._ln_linear.weight.data)
                self.norm.bias.data.copy_(self._ln_linear.bias.data)
                self.linear1.weight.data.copy_(self._ln_linear.linear_weight.data.T)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict then sync to fused parameters.

        Fused parameters (_ln_linear.*, _swiglu.*) are populated via
        _sync_to_fused() from the standard norm/linear1 weights, so they
        are never expected in the incoming state_dict.
        """
        # The fused params are derived — never require them from the state dict
        if HAS_TRITON and strict:
            fused_prefixes = ("_ln_linear.", "_swiglu.")
            own_keys = {k for k in self.state_dict() if any(k.startswith(p) for p in fused_prefixes)}
            missing_fused = own_keys - set(state_dict)
            if missing_fused:
                strict = False
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if HAS_TRITON:
            self._sync_to_fused()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda and self._ln_linear is not None:
            # Fused path: LN+Linear -> SwiGLU -> Linear
            h = self._ln_linear(x)
            h = self._swiglu(h)
            return self.linear2(h)
        # Fallback path: standard ops
        from deepfold3.model.primitives import SwiGLU

        h = self.norm(x)
        h = SwiGLU()(self.linear1(h))
        return self.linear2(h)
