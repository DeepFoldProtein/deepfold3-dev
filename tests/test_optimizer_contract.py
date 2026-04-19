"""Unit tests for ``WrappedOptimizer``'s precision contract assertions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler

from deepfold3.model.config import TrainingConfig
from deepfold3.training.distributed import DistInfo
from deepfold3.training.optim import WrappedOptimizer


def _dist() -> DistInfo:
    return DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


def _build_wrapped(*, param_dtype: torch.dtype = torch.float32) -> WrappedOptimizer:
    """Build a minimal WrappedOptimizer around a 1-param model.

    Uses a CPU AdamW (no fused kernel) so tests run on the CI host.
    """
    model = nn.Linear(4, 4, bias=False).to(param_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_cfg = TrainingConfig(lr=1e-3, accumulate_grad_batches=1, grad_norm_fp64=True)
    scaler = GradScaler("cpu", enabled=False)
    return WrappedOptimizer(
        optimizer=optimizer,
        scheduler=MagicMock(),
        scaler=scaler,
        model=model,
        dist_info=_dist(),
        cfg=train_cfg,
    )


def test_init_accepts_fp32_params():
    """fp32 params are the master-weight invariant; construction must succeed."""
    wrapped = _build_wrapped(param_dtype=torch.float32)
    assert wrapped is not None


def test_init_rejects_bf16_params():
    """bf16 params would force bf16 Adam moments → silent update underflow."""
    with pytest.raises(ValueError, match="fp32 master weights"):
        _build_wrapped(param_dtype=torch.bfloat16)


def test_init_rejects_fp16_params():
    with pytest.raises(ValueError, match="fp32 master weights"):
        _build_wrapped(param_dtype=torch.float16)


def test_state_dtype_check_is_idempotent():
    """``_assert_state_dtype`` is lazy + one-shot — second call is a no-op."""
    wrapped = _build_wrapped()
    # First call runs the walk (no Adam state yet → no-op, flag flips).
    wrapped._assert_state_dtype()
    assert wrapped._state_dtype_checked is True
    # Second call short-circuits even if we mutate the state to be bogus.
    wrapped._assert_state_dtype()


def test_state_dtype_check_rejects_bf16_moments():
    """Simulate a checkpoint load that downgraded Adam moments to bf16."""
    wrapped = _build_wrapped()
    # Manually populate Adam state with bf16 moments — mimics a bad resume.
    for group in wrapped.optimizer.param_groups:
        for param in group["params"]:
            wrapped.optimizer.state[param] = {
                "step": torch.tensor(1),
                "exp_avg": torch.zeros_like(param, dtype=torch.bfloat16),
                "exp_avg_sq": torch.zeros_like(param, dtype=torch.bfloat16),
            }

    with pytest.raises(ValueError, match="exp_avg.*must be fp32"):
        wrapped._assert_state_dtype()
