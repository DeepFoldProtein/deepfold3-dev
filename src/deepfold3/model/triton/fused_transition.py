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

    The `norm, linear1, linear2` modules defined here are the single source
    of truth for weights. The fused LN+Linear path reads them at call time
    (linear1.weight is transposed on the fly; autograd tracks the transpose),
    so there are no duplicate parameters and no sync.

    Args:
        channels: Input/output dimension.
        num_intermediate_factor: Expansion factor for intermediate dimension.
    """

    def __init__(self, channels: int, num_intermediate_factor: int = 4):
        super().__init__()
        self.channels = channels
        num_intermediate = channels * num_intermediate_factor

        # Stateless leaves — hold no params.
        if HAS_TRITON:
            self._ln_linear = FusedLayerNormLinear(channels, num_intermediate * 2, has_linear_bias=False)
            self._swiglu = FusedSwiGLU()
        else:
            self._ln_linear = None
            self._swiglu = None

        # AF3: transition1='relu', transition2=final_init ('zeros').
        self.norm = nn.LayerNorm(channels)
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda and self._ln_linear is not None:
            # Fused path: LN+Linear -> SwiGLU -> Linear
            h = self._ln_linear(
                x,
                self.norm.weight,
                self.norm.bias,
                self.linear1.weight.T,
            )
            h = self._swiglu(h)
            return self.linear2(h)
        # Fallback path: standard ops
        from deepfold3.model.primitives import SwiGLU

        h = self.norm(x)
        h = SwiGLU()(self.linear1(h))
        return self.linear2(h)
