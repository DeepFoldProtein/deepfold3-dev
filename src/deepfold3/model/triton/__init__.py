"""Triton kernel replacements for DeepFold3 model operations.

All imports are guarded — modules gracefully degrade to None when
triton is not installed, allowing the rest of the package to work.
"""

from __future__ import annotations

import enum

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ── TF32 Precision Control (adapted from NVIDIA cuEquivariance) ──────────


class Precision(enum.IntEnum):
    """GEMM precision modes for Triton kernels.

    DEFAULT: Native precision (hardware decides TF32 behavior).
    TF32:    Explicit TF32 rounding via bit masking (19-bit mantissa).
    TF32x3:  Triton's built-in tf32x3 accumulation (higher precision than TF32).
    IEEE:    Full IEEE754 float32 (slowest, most precise).
    """

    DEFAULT = 0
    TF32 = 1
    TF32x3 = 2
    IEEE = 3


if HAS_TRITON:

    @triton.jit
    def cvt_tf32_rn(x):
        """Convert float32 to TF32 with round-to-nearest-even.

        Masks the lower 13 mantissa bits to produce TF32 (19-bit mantissa).
        This gives deterministic TF32 rounding rather than relying on
        hardware defaults which may vary across GPU generations.
        """
        return (x.to(tl.float32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)

    @triton.jit
    def precision_dot(a, b, acc, PRECISION: tl.constexpr):
        """Dispatch tl.dot with the requested precision mode."""
        if PRECISION == 0:  # DEFAULT
            return tl.dot(a, b, acc)
        elif PRECISION == 1:  # TF32 (explicit rounding)
            a = cvt_tf32_rn(a)
            b = cvt_tf32_rn(b)
            return tl.dot(a, b, acc, input_precision="tf32")
        elif PRECISION == 2:  # TF32x3
            return tl.dot(a, b, acc, input_precision="tf32x3")
        elif PRECISION == 3:  # IEEE
            return tl.dot(a, b, acc, input_precision="ieee")
        else:
            tl.static_assert(False, "PRECISION must be 0 (default), 1 (tf32), 2 (tf32x3) or 3 (ieee)")
            return acc  # unreachable


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate BLOCK_SIZE (next power of 2) and num_warps for a given dimension."""
    block_size = max(triton.next_power_of_2(n), 16)
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def _try_import(attr: str) -> type | None:
    """Lazily import a triton kernel class, returning None if unavailable."""
    _map = {
        "FusedSwiGLU": ("deepfold3.model.triton.swiglu", "FusedSwiGLU"),
        "FusedLayerNormLinear": ("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"),
        "FusedTransitionBlock": ("deepfold3.model.triton.fused_transition", "FusedTransitionBlock"),
        "fused_evo_attention": ("deepfold3.model.triton.evo_attention", "fused_evo_attention"),
        "FusedAdaLayerNormLinear": ("deepfold3.model.triton.ada_layernorm_linear", "FusedAdaLayerNormLinear"),
        "FusedAdaTransitionBlock": ("deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock"),
        "FusedTriangleMulGate": ("deepfold3.model.triton.triangle_mul_gate", "FusedTriangleMulGate"),
        "FusedLayerNormTranspose": ("deepfold3.model.triton.layernorm_transpose", "FusedLayerNormTranspose"),
        "FusedGatedDualGEMM": ("deepfold3.model.triton.fused_gated_dual_gemm", "FusedGatedDualGEMM"),
        "FusedPairBias": ("deepfold3.model.triton.fused_pair_bias", "FusedPairBias"),
    }
    if attr not in _map:
        return None
    mod_path, name = _map[attr]
    try:
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
