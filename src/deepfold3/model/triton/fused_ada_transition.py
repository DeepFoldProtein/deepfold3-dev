"""Fused AdaTransitionBlock: AdaLN + Linear + SwiGLU + Linear in one module.

Composes FusedAdaLayerNormLinear + FusedSwiGLU + LinearNoBias into a single
module that is a drop-in replacement for deepfold3.model.primitives.AdaTransitionBlock.

The conditioning pre-computation (LN(cond) * w -> sigmoid(cond_scale(c)), etc.)
remains in PyTorch since it's lightweight. The heavy path (LN + scale/bias + GEMM
+ SwiGLU) is fused via Triton.

Weight names match AdaTransitionBlock for load_state_dict compatibility:
    norm (unused — no affine LN)
    cond_norm_weight, cond_scale, cond_bias    -> conditioning (PyTorch)
    linear1.weight                              -> FusedAdaLayerNormLinear.linear_weight (transposed)
    linear2.weight                              -> self.linear2.weight
    zero_cond.weight, zero_cond.bias           -> gating (PyTorch)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.primitives import LinearNoBias, SwiGLU

try:
    from deepfold3.model.triton.ada_layernorm_linear import HAS_TRITON as HAS_ADA_LNL
    from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear
except ImportError:
    HAS_ADA_LNL = False

try:
    from deepfold3.model.triton.swiglu import HAS_TRITON as HAS_SWIGLU
    from deepfold3.model.triton.swiglu import FusedSwiGLU
except ImportError:
    HAS_SWIGLU = False

HAS_TRITON = HAS_ADA_LNL and HAS_SWIGLU


class FusedAdaTransitionBlock(nn.Module):
    """Drop-in replacement for AdaTransitionBlock using fused Triton kernels.

    Falls back to unfused ops when Triton is unavailable or input is on CPU.

    This module owns the standard weights (linear1, linear2, cond_*, etc.);
    the fused AdaLN+Linear path reads linear1.weight at call time (transposed
    on the fly; autograd tracks the transpose). No duplicate parameters, no
    sync.

    Args:
        channels: Input/output dimension.
        cond_dim: Conditioning dimension.
        num_intermediate_factor: Expansion factor for intermediate dimension.
    """

    def __init__(self, channels: int, cond_dim: int, num_intermediate_factor: int = 2):
        super().__init__()
        self.channels = channels
        num_intermediate = channels * num_intermediate_factor

        # Stateless leaves — hold no params.
        if HAS_TRITON:
            self._ada_ln_linear = FusedAdaLayerNormLinear(channels, num_intermediate * 2)
            self._swiglu = FusedSwiGLU()
        else:
            self._ada_ln_linear = None
            self._swiglu = None

        # AF3: single_cond_scale/bias use initializer='zeros', linear1='relu', linear2='zeros'.
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.cond_norm_weight = nn.Parameter(torch.ones(cond_dim))
        self.cond_scale = nn.Linear(cond_dim, channels)
        from deepfold3.model.primitives import af3_linear_init_

        af3_linear_init_(self.cond_scale.weight, "zeros")
        nn.init.zeros_(self.cond_scale.bias)
        self.cond_bias = LinearNoBias(cond_dim, channels, initializer="zeros")
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")
        self.zero_cond = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.zero_cond.weight)
        nn.init.constant_(self.zero_cond.bias, -2.0)
        self._act = SwiGLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Conditioning pre-computation (lightweight, stays in PyTorch)
        c = F.layer_norm(cond, [cond.shape[-1]]) * self.cond_norm_weight
        scale = torch.sigmoid(self.cond_scale(c))
        bias = self.cond_bias(c)

        if HAS_TRITON and x.is_cuda and self._ada_ln_linear is not None:
            # Fused path: AdaLN+Linear -> SwiGLU -> Linear
            h = self._ada_ln_linear(x, scale, bias, self.linear1.weight.T)
            h = self._swiglu(h)
            out = self.linear2(h)
        else:
            # Fallback path: standard ops
            xn = F.layer_norm(x, [x.shape[-1]])
            xn = scale * xn + bias
            out = self.linear2(self._act(self.linear1(xn)))

        # AdaLN-zero gate
        gate = torch.sigmoid(self.zero_cond(cond))
        return gate * out
