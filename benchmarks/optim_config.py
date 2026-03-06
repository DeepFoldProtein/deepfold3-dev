"""Optimization configuration for benchmarks.

Re-exports the model's OptimConfig and provides benchmark-specific helpers
for checking backend availability and constructing optimized modules.

Usage:
    from benchmarks.optim_config import OptimConfig, get_optimized_module

    cfg = OptimConfig(use_fused_transition=True)
    module = get_optimized_module("transition_block", channels=128, config=cfg)
"""

from __future__ import annotations

import torch.nn as nn

from benchmarks.utils import try_import
from deepfold3.model.config import OptimConfig

# Re-export for convenience
__all__ = ["OptimConfig", "is_available", "get_optimized_module", "print_availability"]


# Registry of available optimized modules
_AVAILABILITY_CACHE: dict[str, bool] = {}


def is_available(backend: str) -> bool:
    """Check if an optimization backend is importable."""
    if backend in _AVAILABILITY_CACHE:
        return _AVAILABILITY_CACHE[backend]

    checks = {
        "triton_swiglu": ("deepfold3.model.triton.swiglu", "FusedSwiGLU"),
        "triton_transition": ("deepfold3.model.triton.fused_transition", "FusedTransitionBlock"),
        "triton_evo_attention": ("deepfold3.model.triton.evo_attention", "fused_evo_attention"),
        "triton_layernorm_linear": ("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"),
        "torch_compile": ("torch", "compile"),
    }

    if backend not in checks:
        _AVAILABILITY_CACHE[backend] = False
        return False

    mod_path, attr = checks[backend]
    result = try_import(mod_path, attr) is not None
    _AVAILABILITY_CACHE[backend] = result
    return result


def get_optimized_module(
    module_type: str,
    config: OptimConfig,
    **kwargs,
) -> nn.Module:
    """Get a module with the best available optimization based on config.

    Args:
        module_type: One of "transition_block"
        config: Optimization configuration
        **kwargs: Module constructor arguments (e.g., channels, num_heads)

    Returns:
        The module instance (optimized or baseline)
    """
    import torch

    device = kwargs.pop("device", "cuda")
    dtype = kwargs.pop("dtype", torch.bfloat16)

    if module_type == "transition_block":
        channels = kwargs["channels"]
        factor = kwargs.get("expansion_factor", 4)

        if config.use_fused_transition and is_available("triton_transition"):
            FusedTransitionBlock = try_import(  # noqa: N806
                "deepfold3.model.triton.fused_transition", "FusedTransitionBlock"
            )
            return FusedTransitionBlock(channels, num_intermediate_factor=factor).to(device, dtype)

        from deepfold3.model.primitives import TransitionBlock

        return TransitionBlock(channels, num_intermediate_factor=factor).to(device, dtype)

    raise ValueError(f"Unknown module_type: {module_type}")


def print_availability():
    """Print which optimization backends are available."""
    backends = [
        ("triton_swiglu", "Triton FusedSwiGLU"),
        ("triton_transition", "Triton FusedTransition"),
        ("triton_evo_attention", "Triton FusedEvoAttention"),
        ("triton_layernorm_linear", "Triton FusedLayerNormLinear"),
        ("torch_compile", "torch.compile"),
    ]
    print("Optimization backends:")
    for key, name in backends:
        status = "available" if is_available(key) else "not found"
        print(f"  {name}: {status}")
