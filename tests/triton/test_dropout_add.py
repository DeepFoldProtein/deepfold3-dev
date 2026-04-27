"""Unit tests for fused dropout+add (row-wise and column-wise)."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required for fused dropout tests", allow_module_level=True)

try:
    from deepfold3.model.triton.dropout_add import (
        _HAS_TRITON,
        dropout_add_columnwise,
        dropout_add_rowwise,
    )
except Exception as e:  # pragma: no cover
    pytest.skip(f"Could not import dropout_add module: {e}", allow_module_level=True)

if not _HAS_TRITON:
    pytest.skip("Triton not available", allow_module_level=True)


_SHAPES = [
    (1, 4, 4, 8),
    (1, 16, 16, 32),
    (2, 32, 32, 64),
    (1, 128, 128, 128),
    (2, 64, 64, 128),
]


def _reference_dropout_rowwise(x: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    """mask shape [B?, 1, C, D], broadcast along -3."""
    return x * mask / (1.0 - p)


def _reference_dropout_columnwise(x: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    """mask shape [B?, R, 1, D], broadcast along -2."""
    return x * mask / (1.0 - p)


# ----------------------------------------------------------------------------
# Category 1: Identity paths (training=False, p=0)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 8, 8, 16)])
def test_identity_when_not_training(shape):
    torch.manual_seed(0)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out_row = dropout_add_rowwise(residual, x, p_drop=0.25, training=False)
    out_col = dropout_add_columnwise(residual, x, p_drop=0.25, training=False)
    assert torch.allclose(out_row, residual + x)
    assert torch.allclose(out_col, residual + x)


@pytest.mark.parametrize("shape", [(1, 8, 8, 16)])
def test_identity_when_p_zero(shape):
    torch.manual_seed(0)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out_row = dropout_add_rowwise(residual, x, p_drop=0.0, training=True)
    out_col = dropout_add_columnwise(residual, x, p_drop=0.0, training=True)
    assert torch.allclose(out_row, residual + x)
    assert torch.allclose(out_col, residual + x)


# ----------------------------------------------------------------------------
# Category 2: Mask structure — row- and column-sharing
# ----------------------------------------------------------------------------


def _extract_mask(out: torch.Tensor, residual: torch.Tensor, x: torch.Tensor, p: float) -> torch.Tensor:
    """Recover the inverse-scaled dropout mask from a fused output.

    ``out = residual + x * mask / (1-p)``  →  ``mask = (out - residual) / (x / (1-p))``.
    Elements where ``x == 0`` give 0/0 → we avoid them by adding a constant
    offset and using a non-zero x.
    """
    diff = out.float() - residual.float()
    # Keep = diff != 0 (since x is non-zero); drop = diff == 0
    keep = (diff.abs() > 1e-6).float()
    return keep


@pytest.mark.parametrize("shape", _SHAPES)
def test_rowwise_mask_shared_along_rows(shape):
    """Rowwise: same keep-decision for all row indices at a given (col, chan)."""
    B, R, C, D = shape
    torch.manual_seed(42)
    # Non-zero x so we can recover the mask from the diff.
    x = torch.ones(shape, device="cuda", dtype=torch.float32) * 2.0
    residual = torch.zeros_like(x)
    out = dropout_add_rowwise(residual, x, p_drop=0.3, training=True)
    keep = _extract_mask(out, residual, x, 0.3)  # [B, R, C, D]
    # For rowwise: keep[:, r0, c, d] == keep[:, r1, c, d] for all r0, r1
    ref = keep[:, 0:1, :, :].expand_as(keep)
    assert torch.equal(keep, ref), "rowwise mask should be identical across rows"


@pytest.mark.parametrize("shape", _SHAPES)
def test_columnwise_mask_shared_along_cols(shape):
    """Columnwise: same keep-decision for all col indices at a given (row, chan)."""
    B, R, C, D = shape
    torch.manual_seed(42)
    x = torch.ones(shape, device="cuda", dtype=torch.float32) * 2.0
    residual = torch.zeros_like(x)
    out = dropout_add_columnwise(residual, x, p_drop=0.3, training=True)
    keep = _extract_mask(out, residual, x, 0.3)  # [B, R, C, D]
    # For columnwise: keep[:, r, c0, d] == keep[:, r, c1, d]
    ref = keep[:, :, 0:1, :].expand_as(keep)
    assert torch.equal(keep, ref), "columnwise mask should be identical across cols"


@pytest.mark.parametrize("shape", [(4, 16, 16, 32)])
def test_batch_mask_independent(shape):
    """Batch dim gets independent masks (not shared)."""
    B, R, C, D = shape
    torch.manual_seed(42)
    x = torch.ones(shape, device="cuda", dtype=torch.float32) * 2.0
    residual = torch.zeros_like(x)
    out_row = dropout_add_rowwise(residual, x, p_drop=0.3, training=True)
    out_col = dropout_add_columnwise(residual, x, p_drop=0.3, training=True)

    keep_row = _extract_mask(out_row, residual, x, 0.3)  # [B, R, C, D]
    keep_col = _extract_mask(out_col, residual, x, 0.3)

    # At least one (c, d) should differ between batch 0 and batch 1.
    # Probability that all CD cells coincide across two independent Bernoulli(0.7)
    # draws is 0.58^(CD) — astronomically small for C*D ≥ 32.
    row_b0 = keep_row[0, 0, :, :]
    row_b1 = keep_row[1, 0, :, :]
    assert not torch.equal(row_b0, row_b1), "rowwise: batches should have independent masks"

    col_b0 = keep_col[0, :, 0, :]
    col_b1 = keep_col[1, :, 0, :]
    assert not torch.equal(col_b0, col_b1), "columnwise: batches should have independent masks"


# ----------------------------------------------------------------------------
# Category 3: Numerical correctness (forward) — compare to manually-masked path
# ----------------------------------------------------------------------------


def _check_fused_matches_mask(out, residual, x, p, atol=1e-4):
    """Each element's diff = (out - residual) should be either 0 (dropped)
    or x * scale (kept). Allow small numerical tolerance for fp32 rounding.
    """
    scale = 1.0 / (1.0 - p)
    diff = out.float() - residual.float()
    x_scaled = x.float() * scale
    is_kept = (diff - x_scaled).abs() < atol
    is_dropped = diff.abs() < atol
    ok = is_kept | is_dropped
    bad = ~ok
    if bad.any():
        raise AssertionError(
            f"fused output inconsistent with dropout mask at "
            f"{bad.sum().item()}/{bad.numel()} elements; "
            f"max diff error = {(diff - x_scaled).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_rowwise_forward_matches_reference(shape, p):
    """Fused output == residual + dropout_rowwise(x) element-wise (with same mask)."""
    torch.manual_seed(0)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    out = dropout_add_rowwise(residual, x, p_drop=p, training=True)
    _check_fused_matches_mask(out, residual, x, p)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_columnwise_forward_matches_reference(shape, p):
    """Same as rowwise test but for columnwise."""
    torch.manual_seed(1)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    out = dropout_add_columnwise(residual, x, p_drop=p, training=True)
    _check_fused_matches_mask(out, residual, x, p)


# ----------------------------------------------------------------------------
# Category 4: Backward correctness
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 32, 32, 64)])
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_rowwise_backward_matches_forward_mask(shape, p):
    """grad_x should match the forward mask (regenerated from saved seed)."""
    torch.manual_seed(2)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    out = dropout_add_rowwise(residual.requires_grad_(True), x, p_drop=p, training=True)
    grad_out = torch.ones_like(out)
    grad_res, grad_x = torch.autograd.grad(out, [residual, x], grad_outputs=grad_out, retain_graph=True)

    # grad_residual = grad_out (addition is identity)
    assert torch.allclose(grad_res, grad_out)
    # grad_x = grad_out * mask * scale — scale = 1/(1-p)
    # Verify: grad_x / (scale * grad_out) ∈ {0, 1} everywhere
    scale = 1.0 / (1.0 - p)
    ratio = grad_x / (scale * grad_out)
    zeros = ratio.abs() < 1e-4
    ones = (ratio - 1.0).abs() < 1e-4
    assert (zeros | ones).all(), "grad_x not consistent with dropout mask"


@pytest.mark.parametrize("shape", [(1, 32, 32, 64)])
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_columnwise_backward_matches_forward_mask(shape, p):
    torch.manual_seed(3)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    out = dropout_add_columnwise(residual.requires_grad_(True), x, p_drop=p, training=True)
    grad_out = torch.ones_like(out)
    grad_res, grad_x = torch.autograd.grad(out, [residual, x], grad_outputs=grad_out, retain_graph=True)

    assert torch.allclose(grad_res, grad_out)
    scale = 1.0 / (1.0 - p)
    ratio = grad_x / (scale * grad_out)
    zeros = ratio.abs() < 1e-4
    ones = (ratio - 1.0).abs() < 1e-4
    assert (zeros | ones).all()


@pytest.mark.parametrize("rowwise", [True, False])
def test_backward_mask_matches_forward_mask_on_same_seed(rowwise):
    """Backward regenerates the same mask used in forward (seed consistency)."""
    shape = (1, 16, 16, 32)
    torch.manual_seed(4)
    x_val = torch.randn(shape, device="cuda", dtype=torch.float32)
    residual = torch.zeros(shape, device="cuda", dtype=torch.float32, requires_grad=False)

    x = x_val.clone().requires_grad_(True)
    fn = dropout_add_rowwise if rowwise else dropout_add_columnwise
    out = fn(residual, x, p_drop=0.3, training=True)

    grad_out = torch.ones_like(out)
    (grad_x,) = torch.autograd.grad(out, [x], grad_outputs=grad_out, retain_graph=True)

    # Forward mask (keep = where out != residual; handle x=0 edge).
    fwd_keep = ((out - residual).abs() > 1e-6).float()
    # Backward mask (grad_x > 0 → kept).
    bwd_keep = (grad_x.abs() > 1e-6).float()
    # Where x is non-zero, fwd and bwd should agree.
    agree_where_x_nonzero = x_val.abs() > 1e-6
    assert torch.equal(fwd_keep[agree_where_x_nonzero], bwd_keep[agree_where_x_nonzero])


# ----------------------------------------------------------------------------
# Category 5: Dtype support (bf16, fp16, fp32)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rowwise", [True, False])
def test_dtype_roundtrip(dtype, rowwise):
    shape = (1, 16, 16, 32)
    torch.manual_seed(5)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    fn = dropout_add_rowwise if rowwise else dropout_add_columnwise
    out = fn(residual, x, p_drop=0.25, training=True)
    assert out.dtype == dtype
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------------
# Category 6: Statistical — expected output magnitude
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("rowwise", [True, False])
@pytest.mark.parametrize("p", [0.1, 0.25])
def test_expected_output_preserves_mean(rowwise, p):
    """E[dropout(x) / (1-p)] = x  →  E[out] ≈ residual + x over many draws."""
    shape = (8, 32, 32, 64)
    torch.manual_seed(6)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    fn = dropout_add_rowwise if rowwise else dropout_add_columnwise

    accum = torch.zeros_like(x)
    n_trials = 32
    for _ in range(n_trials):
        accum += fn(residual, x, p_drop=p, training=True) - residual
    mean = accum / n_trials
    # mean should approximate x. Use loose tolerance since n=32 is small.
    err = (mean - x).abs().mean().item()
    assert err < 0.25, f"mean error {err:.3f} over {n_trials} trials too large (expected ≈ 0)"


# ----------------------------------------------------------------------------
# Category 7: Shape coverage (odd / non-power-of-two)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 17, 17, 24), (1, 1, 128, 64), (3, 384, 384, 128)])
@pytest.mark.parametrize("rowwise", [True, False])
def test_odd_shapes(shape, rowwise):
    torch.manual_seed(7)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    fn = dropout_add_rowwise if rowwise else dropout_add_columnwise
    out = fn(residual, x, p_drop=0.25, training=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
