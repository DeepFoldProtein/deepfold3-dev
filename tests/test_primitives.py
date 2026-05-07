"""Reference implementation tests for primitive building blocks.

Each test computes the expected output manually using the module's own weights,
then compares against the module's forward pass. This verifies both the
implementation and the weight loading.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ref_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Manual LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias."""
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def ref_linear_no_bias(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Manual LinearNoBias: y = x @ W^T."""
    return x @ weight.T


def ref_swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(a) * b where [a, b] = split(x)."""
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


class TestLayerNorm:
    """Verify LayerNorm matches manual computation."""

    def test_forward_matches_reference(self):
        from deepfold3.model.primitives import LayerNorm

        ln = LayerNorm(64)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y = ln(x)
            y_ref = ref_layernorm(x, ln.weight, ln.bias)
        torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)

    def test_with_af3_weights(self, model, device):
        """LayerNorm with loaded AF3 weights matches reference."""
        ln = model.evoformer.prev_embedding_layer_norm
        d = ln.weight.shape[0]
        x = torch.randn(16, d, device=device)
        with torch.no_grad():
            y = ln(x)
            y_ref = ref_layernorm(x, ln.weight, ln.bias)
        torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)

    def test_output_statistics(self):
        """LayerNorm output should have mean~0, std~1 (before affine)."""
        from deepfold3.model.primitives import LayerNorm

        ln = LayerNorm(128)
        # Set weight=1, bias=0 to see raw normalization
        ln.weight.data.fill_(1.0)
        ln.bias.data.fill_(0.0)
        x = torch.randn(32, 128) * 5 + 3
        with torch.no_grad():
            y = ln(x)
        assert y.mean().abs() < 0.1
        assert (y.std() - 1.0).abs() < 0.1

    def test_output_scale_multiplies_output(self):
        """LayerNorm(output_scale=k) should produce k× the unscaled output."""
        from deepfold3.model.primitives import LayerNorm

        torch.manual_seed(0)
        ln = LayerNorm(64)
        ln_scaled = LayerNorm(64, output_scale=0.25)
        ln_scaled.weight.data.copy_(ln.weight.data)
        ln_scaled.bias.data.copy_(ln.bias.data)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y = ln(x)
            y_scaled = ln_scaled(x)
        torch.testing.assert_close(y_scaled, y * 0.25, atol=1e-6, rtol=1e-5)

    def test_output_scale_default_is_unity(self):
        """Default output_scale=1.0 should match plain LayerNorm bit-for-bit."""
        from deepfold3.model.primitives import LayerNorm

        ln = LayerNorm(32)
        assert ln.output_scale == 1.0


class TestBlockResidualScale:
    """Per-layer / uniform Residual Output Scaling factor resolution."""

    def test_neither_returns_one(self):
        from deepfold3.model.primitives import block_residual_scale

        for i in range(5):
            assert block_residual_scale(i, 5, per_layer=False, uniform=False) == 1.0

    def test_per_layer(self):
        import math

        from deepfold3.model.primitives import block_residual_scale

        for i, n in [(0, 48), (1, 48), (5, 48), (47, 48)]:
            expected = 1.0 / math.sqrt(i + 1)
            assert block_residual_scale(i, n, per_layer=True, uniform=False) == expected

    def test_uniform_depth_flat(self):
        import math

        from deepfold3.model.primitives import block_residual_scale

        n = 48
        for i in range(n):
            assert block_residual_scale(i, n, per_layer=False, uniform=True) == 1.0 / math.sqrt(n)

    def test_mutual_exclusion(self):
        import pytest

        from deepfold3.model.primitives import block_residual_scale

        with pytest.raises(ValueError, match="mutually exclusive"):
            block_residual_scale(0, 4, per_layer=True, uniform=True)


class TestRmsNormalize:
    """Non-learnable RMSNorm over the last channel axis."""

    def test_unit_rms_output(self):
        from deepfold3.model.primitives import rms_normalize

        torch.manual_seed(0)
        x = torch.randn(2, 8, 64) * 5.7  # arbitrary non-unit scale
        y = rms_normalize(x)
        # Per-token RMS over channel axis ≈ 1.
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)

    def test_invariant_to_input_scale(self):
        from deepfold3.model.primitives import rms_normalize

        torch.manual_seed(1)
        x = torch.randn(4, 32)
        y_a = rms_normalize(x)
        y_b = rms_normalize(x * 137.0)
        torch.testing.assert_close(y_a, y_b, atol=1e-4, rtol=1e-4)

    def test_bf16_passthrough(self):
        from deepfold3.model.primitives import rms_normalize

        x = torch.randn(2, 4, 16, dtype=torch.bfloat16) * 2.5
        y = rms_normalize(x)
        assert y.dtype == torch.bfloat16
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-2, rtol=2e-2)

    def test_handles_near_zero_input(self):
        from deepfold3.model.primitives import rms_normalize

        # Zero input → output stays finite (eps prevents div-by-zero).
        x = torch.zeros(2, 16)
        y = rms_normalize(x)
        assert torch.isfinite(y).all()


class TestTransitionBlockLinearOutInit:
    """``linear_out_init_gain`` overrides the AF3 zeros init on linear2 with
    a scaled row-orthogonal init."""

    def test_default_is_zeros(self):
        from deepfold3.model.primitives import TransitionBlock

        tb = TransitionBlock(384, num_intermediate_factor=2)
        assert tb.linear2.weight.abs().max().item() == 0.0

    def test_orthogonal_init_gain(self):
        import torch

        from deepfold3.model.primitives import TransitionBlock

        gain = 0.01
        torch.manual_seed(0)
        tb = TransitionBlock(384, num_intermediate_factor=2, linear_out_init_gain=gain)
        W = tb.linear2.weight.detach()  # shape [384, 768]

        # Row-orthonormal scaled by gain: W W^T = gain² · I_384.
        gram = W @ W.transpose(0, 1)
        I_scaled = torch.eye(384) * (gain**2)
        torch.testing.assert_close(gram, I_scaled, atol=1e-6, rtol=1e-5)

        # All rows have norm ≈ gain.
        row_norms = W.norm(dim=1)
        torch.testing.assert_close(row_norms, torch.full((384,), gain), atol=1e-6, rtol=1e-5)

    def test_make_transition_block_passes_gain(self):
        from deepfold3.model.optim import make_transition_block

        block = make_transition_block(384, num_intermediate_factor=2, linear_out_init_gain=0.05)
        # Row-orthonormal × 0.05 → frobenius² = 384 · 0.05² = 0.96
        fro_sq = block.linear2.weight.pow(2).sum().item()
        assert abs(fro_sq - 384 * 0.05**2) < 1e-3

    def test_fused_transition_rejects_gain(self):
        import pytest

        from deepfold3.model.config import OptimConfig
        from deepfold3.model.optim import make_transition_block

        optim = OptimConfig(use_fused_transition_pair=True)
        with pytest.raises(ValueError, match="linear_out_init_gain"):
            make_transition_block(128, num_intermediate_factor=2, config=optim, linear_out_init_gain=0.01)


class TestLinearNoBias:
    """Verify LinearNoBias matches manual computation."""

    def test_forward_matches_reference(self):
        from deepfold3.model.primitives import LinearNoBias

        linear = LinearNoBias(64, 128)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y = linear(x)
            y_ref = ref_linear_no_bias(x, linear.weight)
        torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)

    def test_with_af3_weights(self, model, device):
        """LinearNoBias with AF3 weights matches reference."""
        linear = model.evoformer.left_single
        d_in = linear.weight.shape[1]
        x = torch.randn(16, d_in, device=device)
        with torch.no_grad():
            y = linear(x)
            y_ref = ref_linear_no_bias(x, linear.weight)
        torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)

    def test_no_bias(self):
        from deepfold3.model.primitives import LinearNoBias

        linear = LinearNoBias(32, 64)
        assert linear.bias is None


class TestSwiGLU:
    """Verify SwiGLU activation."""

    def test_matches_reference(self):
        from deepfold3.model.primitives import SwiGLU

        swiglu = SwiGLU()
        x = torch.randn(8, 128)
        y = swiglu(x)
        y_ref = ref_swiglu(x)
        torch.testing.assert_close(y, y_ref)

    def test_output_shape(self):
        from deepfold3.model.primitives import SwiGLU

        swiglu = SwiGLU()
        x = torch.randn(4, 64)
        y = swiglu(x)
        assert y.shape == (4, 32)  # halved


class TestTransitionBlock:
    """Verify SwiGLU TransitionBlock forward pass."""

    def test_output_shape(self):
        from deepfold3.model.primitives import TransitionBlock

        tb = TransitionBlock(64, num_intermediate_factor=4)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y = tb(x)
        assert y.shape == (8, 64)

    def test_matches_reference(self):
        """TransitionBlock = norm -> linear1 -> swiglu -> linear2 (no residual)."""
        from deepfold3.model.primitives import TransitionBlock

        tb = TransitionBlock(32, num_intermediate_factor=2)
        tb.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            y = tb(x)
            # Manual reference
            xn = ref_layernorm(x, tb.norm.weight, tb.norm.bias)
            h = ref_linear_no_bias(xn, tb.linear1.weight)
            h = ref_swiglu(h)
            y_ref = ref_linear_no_bias(h, tb.linear2.weight)
        torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)

    def test_finite_output(self):
        from deepfold3.model.primitives import TransitionBlock

        tb = TransitionBlock(64)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y = tb(x)
        assert torch.isfinite(y).all()


class TestAdaLayerNorm:
    """Verify AdaLayerNorm conditioned normalization."""

    def test_output_shape(self):
        from deepfold3.model.primitives import AdaLayerNorm

        aln = AdaLayerNorm(64, 128)
        x = torch.randn(8, 64)
        cond = torch.randn(8, 128)
        with torch.no_grad():
            y = aln(x, cond)
        assert y.shape == (8, 64)

    def test_matches_reference(self):
        """AdaLN: scale * norm(x) + bias, scale=sigmoid(cond_scale(cond_norm(c)))."""
        from deepfold3.model.primitives import AdaLayerNorm

        aln = AdaLayerNorm(32, 64)
        aln.eval()
        x = torch.randn(4, 32)
        cond = torch.randn(4, 64)
        with torch.no_grad():
            y = aln(x, cond)
            # Manual reference
            xn = F.layer_norm(x, [32])
            c = F.layer_norm(cond, [64]) * aln.cond_norm_weight
            scale = torch.sigmoid(F.linear(c, aln.cond_scale.weight, aln.cond_scale.bias))
            bias = ref_linear_no_bias(c, aln.cond_bias.weight)
            y_ref = scale * xn + bias
        torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
