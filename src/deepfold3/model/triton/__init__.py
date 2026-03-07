"""Triton kernel replacements for DeepFold3 model operations.

All imports are guarded — modules gracefully degrade to None when
triton is not installed, allowing the rest of the package to work.
"""

from __future__ import annotations

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate BLOCK_SIZE (next power of 2) and num_warps for a given dimension."""
    block_size = max(triton.next_power_of_2(n), 16)
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def _try_import(attr: str):
    """Lazily import a triton kernel class, returning None if unavailable."""
    _map = {
        "FusedSwiGLU": ("deepfold3.model.triton.swiglu", "FusedSwiGLU"),
        "FusedLayerNormLinear": ("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"),
        "FusedTransitionBlock": ("deepfold3.model.triton.fused_transition", "FusedTransitionBlock"),
        "fused_evo_attention": ("deepfold3.model.triton.evo_attention", "fused_evo_attention"),
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
