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

# Pair channel dim where fused transition is used (C=128 pair, M=N² → Triton wins).
_PAIR_TRANSITION_CHANNELS = (64, 128)
# Single/MSA channel dims where fused transition is slow (M=N, small → PyTorch wins).
_SINGLE_TRANSITION_CHANNELS = (384,)


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
        "triton_ada_transition": ("deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock"),
        "triton_triangle_mul_gate": ("deepfold3.model.triton.triangle_mul_gate", "FusedTriangleMulGate"),
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

    Uses separate flags for pair vs single transitions because they operate
    at vastly different M sizes:
    - Pair transitions: M = N² (up to 590k) — Triton wins at 1.2-1.3x
    - Single transitions: M = N (384-768) — PyTorch wins (Triton overhead dominates)
    """
    if config is not None:
        use_fused = False
        if config.use_fused_transition_pair and channels in _PAIR_TRANSITION_CHANNELS:
            use_fused = True
        elif config.use_fused_transition_single and channels in _SINGLE_TRANSITION_CHANNELS:
            use_fused = True
        if use_fused:
            cls = _try_import("deepfold3.model.triton.fused_transition", "FusedTransitionBlock")
            if cls is not None:
                return cls(channels, num_intermediate_factor=num_intermediate_factor)
    from deepfold3.model.primitives import TransitionBlock

    return TransitionBlock(channels, num_intermediate_factor=num_intermediate_factor)


def make_ada_transition_block(
    channels: int, cond_dim: int, num_intermediate_factor: int = 2, config: OptimConfig | None = None
) -> nn.Module:
    """Create an AdaTransitionBlock, optionally using fused Triton kernels."""
    if config and config.use_fused_ada_transition:
        cls = _try_import("deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock")
        if cls is not None:
            return cls(channels, cond_dim, num_intermediate_factor=num_intermediate_factor)
    from deepfold3.model.primitives import AdaTransitionBlock

    return AdaTransitionBlock(channels, cond_dim, num_intermediate_factor=num_intermediate_factor)


def print_availability():
    """Print which optimization backends are available."""
    backends = [
        ("triton_transition", "Triton FusedTransition"),
        ("triton_evo_attention", "Triton FusedEvoAttention"),
        ("triton_layernorm_linear", "Triton FusedLayerNormLinear"),
        ("triton_ada_transition", "Triton FusedAdaTransition"),
        ("triton_triangle_mul_gate", "Triton FusedTriangleMulGate"),
        ("torch_compile", "torch.compile"),
    ]
    print("Optimization backends:")
    for key, name in backends:
        status = "available" if is_available(key) else "not found"
        print(f"  {name}: {status}")
