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
