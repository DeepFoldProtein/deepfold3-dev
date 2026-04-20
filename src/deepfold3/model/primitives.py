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


def sum_fusion_init_(
    weight: torch.Tensor,
    target_output_std: float,
    input_kind: str = "one_hot",
) -> None:
    """Init one stream of a K-way sum-fusion so its output stddev matches peers.

    Default AF3 init uses ``stddev(W) = sqrt(c/fan_in)``. For a single Linear
    that's correct, but when K streams are summed and their inputs differ in
    scale/sparsity (one-hots vs dense vs {0,1} masks), peer streams end up
    with wildly different output magnitudes at init, causing one stream to
    dominate the sum. This helper picks ``stddev(W)`` so that each stream's
    output has stddev ≈ ``target_output_std`` regardless of fan-in:

      * ``input_kind='one_hot'`` — input has exactly one active entry equal to
        1 (one-hot, softmax-concentrated, or {0,1} mask). Output per neuron is
        a single weight, so ``stddev(W) = target_output_std``.
      * ``input_kind='dense'`` — input entries have stddev ≈ 1 (post-LN, or
        continuous features with per-entry variance ≈ 1). Output stddev is
        ``sqrt(fan_in) · stddev(W)``, so ``stddev(W) = target_output_std / sqrt(fan_in)``.

    For a K-way sum with independent streams, the natural choice is
    ``target_output_std = 1 / sqrt(K)`` (unit-variance sum).

    Args:
        weight: Linear weight of shape ``[out_features, in_features]``.
        target_output_std: Desired per-stream output stddev.
        input_kind: ``'one_hot'`` or ``'dense'``.
    """
    if input_kind == "one_hot":
        stddev = target_output_std / TRUNCATED_NORMAL_STDDEV_FACTOR
    elif input_kind == "dense":
        fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
        stddev = (target_output_std / math.sqrt(fan_in)) / TRUNCATED_NORMAL_STDDEV_FACTOR
    else:
        raise ValueError(f"input_kind must be 'one_hot' or 'dense', got {input_kind!r}")
    nn.init.trunc_normal_(weight, mean=0.0, std=stddev, a=-2.0, b=2.0)


def chunked_linear_init_(weight: torch.Tensor, feat_dims: list[int], initializer: str = "linear") -> None:
    """Init a fused ``cat(*feats) @ W`` weight as if it were separate per-feature linears.

    IntFold §5.4.2: ``cat([a, b]) @ W`` uses ``fan_in = a_dim + b_dim`` for
    Lecun/Kaiming scaling — stddev too small, outputs under-scaled. The
    forward/backward of concat-then-linear is mathematically equivalent to
    separate per-feature linears + sum (``a @ W[:a_dim] + b @ W[a_dim:]``),
    so we keep the fused code and only fix init: each ``W[...a_dim_i]`` chunk
    gets its own ``fan_in = a_dim_i`` for correct Lecun Normal scaling.

    Args:
        weight: Parameter tensor of shape ``[out_features, sum(feat_dims)]``.
        feat_dims: Per-feature input dimensions, in concat order. Must sum to
            ``weight.shape[-1]``.
        initializer: ``'linear'`` or ``'relu'`` (``'zeros'`` is rejected — chunking
            zeros is pointless).
    """
    if initializer == "zeros":
        raise ValueError("chunked_linear_init_ does not support 'zeros'")
    total_in = sum(feat_dims)
    if weight.shape[-1] != total_in:
        raise ValueError(f"weight.shape[-1]={weight.shape[-1]} != sum(feat_dims)={total_in}")
    offset = 0
    for d in feat_dims:
        af3_linear_init_(weight[:, offset : offset + d], initializer)
        offset += d


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
