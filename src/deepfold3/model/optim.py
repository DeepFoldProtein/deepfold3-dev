"""Optimized kernel backends for DeepFold3 model modules.

Provides factory functions that return either a baseline PyTorch module or an
optimized Triton replacement based on OptimConfig flags. All replacements
are API-compatible drop-ins.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch.nn as nn

from deepfold3.model.config import OptimConfig

# Minimum channel dim for FusedTransitionBlock to outperform PyTorch baseline.
# Benchmarks show Triton overhead dominates at C=64 (MSA) and C=128 (pair).
_FUSED_TRANSITION_MIN_CHANNELS = 256


@lru_cache(maxsize=16)
def _try_import(module_path: str, attr: str):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def is_available(backend: str) -> bool:
    """Check if an optimization backend is importable."""
    checks = {
        "triton_transition": ("deepfold3.model.triton.fused_transition", "FusedTransitionBlock"),
        "triton_evo_attention": ("deepfold3.model.triton.evo_attention", "fused_evo_attention"),
        "triton_layernorm_linear": ("deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"),
        "torch_compile": ("torch", "compile"),
    }
    if backend not in checks:
        return False
    mod_path, attr = checks[backend]
    return _try_import(mod_path, attr) is not None


def make_transition_block(
    channels: int, num_intermediate_factor: int = 4, config: OptimConfig | None = None
) -> nn.Module:
    """Create a TransitionBlock, optionally using fused Triton kernels.

    Only uses FusedTransitionBlock when channels >= _FUSED_TRANSITION_MIN_CHANNELS,
    since Triton overhead dominates at smaller channel dimensions (C=64, C=128).
    """
    if config and config.use_fused_transition and channels >= _FUSED_TRANSITION_MIN_CHANNELS:
        cls = _try_import("deepfold3.model.triton.fused_transition", "FusedTransitionBlock")
        if cls is not None:
            return cls(channels, num_intermediate_factor=num_intermediate_factor)
    from deepfold3.model.primitives import TransitionBlock

    return TransitionBlock(channels, num_intermediate_factor=num_intermediate_factor)


def print_availability():
    """Print which optimization backends are available."""
    backends = [
        ("triton_transition", "Triton FusedTransition"),
        ("triton_evo_attention", "Triton FusedEvoAttention"),
        ("triton_layernorm_linear", "Triton FusedLayerNormLinear"),
        ("torch_compile", "torch.compile"),
    ]
    print("Optimization backends:")
    for key, name in backends:
        status = "available" if is_available(key) else "not found"
        print(f"  {name}: {status}")
