"""Primitive building blocks for DeepFold3.

Attribute names are chosen to exactly match the Haiku → PyTorch converter's
expected mapping so that load_state_dict() works without manual renaming.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
# Used to adjust stddev for truncation at ±2 sigma.
TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978


def af3_linear_init_(weight: torch.Tensor, initializer: str = "linear") -> None:
    """Initialize a weight tensor using AF3's variance-scaled truncated normal.

    Args:
        weight: Parameter tensor of shape ``[out_features, in_features]``.
        initializer: One of ``'linear'``, ``'relu'``, or ``'zeros'``.
    """
    if initializer == "zeros":
        nn.init.zeros_(weight)
        return
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    scale = 1.0 / fan_in
    if initializer == "relu":
        scale *= 2.0
    stddev = math.sqrt(scale) / TRUNCATED_NORMAL_STDDEV_FACTOR
    nn.init.trunc_normal_(weight, mean=0.0, std=stddev, a=-2.0, b=2.0)


class LinearNoBias(nn.Linear):
    """nn.Linear with bias=False, matching AF3's `hm.Linear(use_bias=False)`.

    Supports AF3-compatible initialization via the ``initializer`` parameter:
        ``'linear'``  — truncated normal, stddev = sqrt(1/fan_in) / 0.8796 (default)
        ``'relu'``    — truncated normal, stddev = sqrt(2/fan_in) / 0.8796
        ``'zeros'``   — all zeros
    """

    def __init__(self, in_features: int, out_features: int, initializer: str = "linear"):
        super().__init__(in_features, out_features, bias=False)
        af3_linear_init_(self.weight, initializer)


class LayerNorm(nn.LayerNorm):
    """nn.LayerNorm wrapper.

    AF3 always uses create_scale=True, create_offset=True (the default), so
    this is simply a named alias that makes the intent explicit.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__(normalized_shape, elementwise_affine=True)


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on a single embedding.

    Used in the diffusion transformer (AdaLN-zero).

    The base LN has no learned scale/offset; instead, these are predicted from
    a conditioning vector via two linear projections.

    Haiku names inside adaptive_layernorm():
        {name}layer_norm             → norm        (no scale/offset)
        {name}single_cond_layer_norm → cond_norm   (scale only)
        {name}single_cond_scale      → cond_scale  (Linear w/ bias)
        {name}single_cond_bias       → cond_bias   (Linear w/ bias)
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_norm = nn.LayerNorm(cond_dim, elementwise_affine=False, bias=False)
        # weight-only LN (create_offset=False)
        self.cond_norm_weight = nn.Parameter(torch.ones(cond_dim))
        self.cond_scale = nn.Linear(cond_dim, dim)
        self.cond_bias = LinearNoBias(cond_dim, dim, initializer="zeros")
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        af3_linear_init_(self.cond_scale.weight, "zeros")
        nn.init.zeros_(self.cond_scale.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        c = F.layer_norm(cond, [cond.shape[-1]]) * self.cond_norm_weight
        scale = torch.sigmoid(self.cond_scale(c))
        bias = self.cond_bias(c)
        return scale * x + bias


class SwiGLU(nn.Module):
    """SwiGLU activation: swish(a) * b, where [a, b] = split(linear(x), 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class TransitionBlock(nn.Module):
    """Feed-forward transition block used in Evoformer / PairFormer.

    AF3 Haiku module names:
        input_layer_norm → norm
        transition1      → linear1  (projects to 2 × intermediate)
        transition2      → linear2  (projects back to c)
    """

    def __init__(self, channels: int, num_intermediate_factor: int = 4, optim=None):
        super().__init__()
        num_intermediate = channels * num_intermediate_factor
        self.norm = LayerNorm(channels)
        # AF3 uses SwiGLU: transition1 projects to 2 × intermediate
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")
        self._act = SwiGLU()
        if optim is not None and optim.use_fused_swiglu:
            try:
                from deepfold3.model.triton.swiglu import FusedSwiGLU

                self._act = FusedSwiGLU()
            except ImportError:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self._act(self.linear1(x))
        return self.linear2(x)


class AdaTransitionBlock(nn.Module):
    """Transition block conditioned via AdaLN-zero (used in diffusion transformer).

    Haiku names (with prefix ``name``):
        {name}ffw_layer_norm                  → norm         (no scale/offset)
        {name}ffw_single_cond_layer_norm      → cond_norm    (scale only)
        {name}ffw_single_cond_scale           → cond_scale
        {name}ffw_single_cond_bias            → cond_bias
        {name}ffw_transition1                 → linear1
        {name}ffw_transition2                 → linear2
        {name}ffw_adaptive_zero_cond          → zero_cond    (gate, init bias=-2)
    """

    def __init__(self, channels: int, cond_dim: int, num_intermediate_factor: int = 2, optim=None):
        super().__init__()
        num_intermediate = channels * num_intermediate_factor
        # AdaLN parts (no offset on base norm, no offset on cond_norm)
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.cond_norm_weight = nn.Parameter(torch.ones(cond_dim))
        self.cond_scale = nn.Linear(cond_dim, channels)
        self.cond_bias = LinearNoBias(cond_dim, channels, initializer="zeros")
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        af3_linear_init_(self.cond_scale.weight, "zeros")
        nn.init.zeros_(self.cond_scale.bias)
        # FFW
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")
        # AdaLN-zero gate (init bias -2 ≈ sigmoid(-2) ~ 0.1)
        self.zero_cond = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.zero_cond.weight)
        nn.init.constant_(self.zero_cond.bias, -2.0)
        self._act = SwiGLU()
        if optim is not None and optim.use_fused_swiglu:
            try:
                from deepfold3.model.triton.swiglu import FusedSwiGLU

                self._act = FusedSwiGLU()
            except ImportError:
                pass

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # AdaLN
        xn = F.layer_norm(x, [x.shape[-1]])
        c = F.layer_norm(cond, [cond.shape[-1]]) * self.cond_norm_weight
        scale = torch.sigmoid(self.cond_scale(c))
        bias = self.cond_bias(c)
        xn = scale * xn + bias
        # FFW with SwiGLU
        xn = self._act(self.linear1(xn))
        out = self.linear2(xn)
        # AdaLN-zero gate
        gate = torch.sigmoid(self.zero_cond(cond))
        return gate * out
