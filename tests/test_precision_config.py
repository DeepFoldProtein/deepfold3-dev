"""Unit tests for ``PrecisionConfig`` — new dtype axes + shorthand + validator."""

from __future__ import annotations

import pytest

from deepfold3.model.config import PrecisionConfig


def test_defaults_preserve_prerefactor_behavior():
    """Defaults: grad_acc=fp32, reduce=fp32, loss=fp32 — matches pre-refactor DF3."""
    p = PrecisionConfig()
    assert p.grad_acc_dtype == "fp32"
    assert p.reduce_dtype == "fp32"
    assert p.loss_dtype == "fp32"
    # Region-guard defaults unchanged.
    assert p.fp32_loss is True
    assert p.fp32_confidence_head is True


def test_validator_rejects_bad_grad_acc_dtype():
    with pytest.raises(ValueError, match="grad_acc_dtype"):
        PrecisionConfig(grad_acc_dtype="fp64")  # type: ignore[arg-type]


def test_validator_rejects_bad_reduce_dtype():
    with pytest.raises(ValueError, match="reduce_dtype"):
        PrecisionConfig(reduce_dtype="fp16")  # type: ignore[arg-type]


def test_validator_rejects_non_fp32_loss_dtype():
    with pytest.raises(ValueError, match="loss_dtype"):
        PrecisionConfig(loss_dtype="bf16")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shorthand, expected_reduce",
    [
        ("bf16-mixed", "bf16"),
        ("bf16-safe", "fp32"),
        ("fp16-mixed", "bf16"),
        ("fp32", "fp32"),
    ],
)
def test_from_shorthand_sets_reduce_dtype(shorthand, expected_reduce):
    p = PrecisionConfig.from_shorthand(shorthand)
    assert p.reduce_dtype == expected_reduce
    # grad_acc + loss always fp32 across all presets.
    assert p.grad_acc_dtype == "fp32"
    assert p.loss_dtype == "fp32"


def test_from_shorthand_rejects_unknown():
    with pytest.raises(ValueError, match="unknown precision shorthand"):
        PrecisionConfig.from_shorthand("bf32-quantum")


def test_from_shorthand_overrides_apply_last():
    """Overrides apply after the preset expansion."""
    p = PrecisionConfig.from_shorthand("bf16-safe", reduce_dtype="bf16", fp32_distogram_head=True)
    assert p.reduce_dtype == "bf16"
    assert p.fp32_distogram_head is True
    # Preset axes still intact where not overridden.
    assert p.grad_acc_dtype == "fp32"


def test_explicit_config_still_works():
    """Direct field assignment path — ensures the shorthand isn't required."""
    p = PrecisionConfig(grad_acc_dtype="fp32", reduce_dtype="bf16", fp32_confidence_head=False)
    assert p.reduce_dtype == "bf16"
    assert p.fp32_confidence_head is False
