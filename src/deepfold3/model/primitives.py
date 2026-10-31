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


def block_residual_scale(layer_idx: int, num_layer: int, per_layer: bool, uniform: bool) -> float:
    """Resolve the Residual Output Scaling factor for layer ``layer_idx`` of a
    depth-``num_layer`` residual stack.

    Each F sub-module of the block has its output multiplied by this scalar
    before being added to the residual stream:

        x ← x + s · F(x)

    Two mutually exclusive policies (see :class:`EvoformerConfig`):

      * ``per_layer`` — per-layer ``1/sqrt(layer_idx + 1)``  (Sun-et-al.-flavoured;
        not literal LayerNorm Scaling because we scale F's output, not LN's).
      * ``uniform``  — depth-flat ``1/sqrt(num_layer)`` for every layer.

    Returns ``1.0`` (no rescale) when neither is set. Raises ``ValueError`` if
    both are set on the same stack.
    """
    if per_layer and uniform:
        raise ValueError("residual_output_scaling and uniform_residual_scale are mutually exclusive on the same stack")
    if per_layer:
        return 1.0 / math.sqrt(float(layer_idx + 1))
    if uniform:
        return 1.0 / math.sqrt(float(num_layer))
    return 1.0


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


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Non-learnable RMSNorm over the last channel axis.

    Thin wrapper over :func:`torch.nn.functional.rms_norm` (no learnable
    scale or shift). Used at PairFormer residual-branch outputs to force
    each branch contribution to unit per-token RMS before the per-block
    ``γ`` multiplier (used by :class:`deepfold3.model.branch_scaler.RMSNormScaler`).
    """
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


def layer_norm(
    x: torch.Tensor,
    normalized_shape: int | tuple[int, ...] | list[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm in fp32 with bf16 passthrough.

    Upcasts the input (and any affine weight/bias) to fp32, runs the
    normalisation, then casts back to the input dtype. The surrounding
    compute stays in bf16; only the LN-internal mean/variance/rsqrt and
    affine multiply are fp32.

    No ``autocast`` context manager — PyTorch's autocast policy keeps
    ``F.layer_norm`` in fp32 already, and a context-manager scope inside a
    callable that may be re-invoked under
    ``torch.utils.checkpoint(use_reentrant=False)`` causes saved-tensor
    hook divergence (see ``tiled_linear`` module docstring).
    """
    orig_dtype = x.dtype
    x32 = x.float()
    w32 = weight.float() if weight is not None else None
    b32 = bias.float() if bias is not None else None
    out = F.layer_norm(x32, normalized_shape, w32, b32, eps)
    return out.to(orig_dtype)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that always runs forward in fp32.

    Subclass of ``nn.LayerNorm`` — accepts ``elementwise_affine`` and
    ``bias`` like the base class. AF3 trunk LNs use the default
    ``elementwise_affine=True``; AdaLN-zero patterns pass
    ``elementwise_affine=False`` for a parameter-free LN.

    ``output_scale`` is a constant multiplier applied to the LN output.
    Stored on the module so a constant scalar can be folded into the LN
    forward path without a separate multiply. Currently unused by any
    config — kept as a primitive utility (e.g. for future absorption of a
    constant gain into LN). Default ``1.0`` keeps the unscaled fast path
    intact (the multiply is skipped).
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        output_scale: float = 1.0,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )
        self.output_scale = float(output_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.output_scale != 1.0:
            out = out * self.output_scale
        return out


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
        self.norm = LayerNorm(dim, elementwise_affine=False)
        self.cond_norm = LayerNorm(cond_dim, elementwise_affine=False, bias=False)
        # weight-only LN (create_offset=False)
        self.cond_norm_weight = nn.Parameter(torch.ones(cond_dim))
        self.cond_scale = nn.Linear(cond_dim, dim)
        self.cond_bias = LinearNoBias(cond_dim, dim, initializer="zeros")
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        af3_linear_init_(self.cond_scale.weight, "zeros")
        nn.init.zeros_(self.cond_scale.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        c = layer_norm(cond, [cond.shape[-1]]) * self.cond_norm_weight
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

    def __init__(
        self,
        channels: int,
        num_intermediate_factor: int = 4,
        optim=None,
        linear_out_init_gain: float | None = None,
    ):
        super().__init__()
        num_intermediate = channels * num_intermediate_factor
        self.norm = LayerNorm(channels)
        # SwiGLU transition. Internal attribute names ``linear1`` /
        # ``linear2`` are preserved (state-dict compat with existing
        # checkpoints / AF3 converter); semantically these are the
        # **linear_gate** (projects channels → 2× intermediate, fed to
        # SwiGLU) and **linear_out** (projects intermediate → channels)
        # respectively.
        self.linear1 = LinearNoBias(channels, num_intermediate * 2, initializer="relu")  # linear_gate
        self.linear2 = LinearNoBias(num_intermediate, channels, initializer="zeros")  # linear_out
        # ``linear_out_init_gain``: replace the AF3 zeros init on
        # ``linear_out`` (= ``linear2``) with a scaled row-orthogonal init
        # ``W = gain · Q`` (`Q` row-orthonormal). Breaks the zero-wake-up
        # bias that otherwise pushes ``linear_out`` along the loss gradient
        # direction during the first updates — random orthogonal rows give
        # backward signal in all directions immediately, so the residual
        # update ``u`` does not get aligned with the residual stream ``s``
        # by construction. ``gain = 0.01`` keeps the residual update tiny
        # at init (output stddev ≈ gain). ``None`` keeps the AF3 zeros init.
        if linear_out_init_gain is not None:
            nn.init.orthogonal_(self.linear2.weight, gain=float(linear_out_init_gain))
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
        self.norm = LayerNorm(channels, elementwise_affine=False)
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
        xn = layer_norm(x, [x.shape[-1]])
        c = layer_norm(cond, [cond.shape[-1]]) * self.cond_norm_weight
        scale = torch.sigmoid(self.cond_scale(c))
        bias = self.cond_bias(c)
        xn = scale * xn + bias
        # FFW with SwiGLU
        xn = self._act(self.linear1(xn))
        out = self.linear2(xn)
        # AdaLN-zero gate
        gate = torch.sigmoid(self.zero_cond(cond))
        return gate * out


def map_row_chunks(fn, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """``fn(x)`` over ``chunk`` rows of dim ``-3`` at a time, into one output.

    ``fn`` must treat rows independently and keep the shape. Inference only:
    with grad enabled, or ``chunk`` of 0 or at least the row count, this is
    plain ``fn(x)``.

    Args:
        fn: Row-independent function of ``[..., S, N, C]``.
        x: Input ``[..., S, N, C]``.
        chunk: Rows per call.

    Returns:
        The same as ``fn(x)``.
    """
    S = x.shape[-3]
    if chunk <= 0 or S <= chunk or torch.is_grad_enabled():
        return fn(x)
    first = fn(x[..., :chunk, :, :])
    out = first.new_empty(*x.shape[:-3], S, *first.shape[-2:])
    out[..., :chunk, :, :] = first
    del first
    for s0 in range(chunk, S, chunk):
        out[..., s0 : s0 + chunk, :, :] = fn(x[..., s0 : s0 + chunk, :, :])
    return out
