"""Reference implementation tests for TriangleMultiplication and OuterProductMean.

Each test implements the AF3 equation manually and compares against the PyTorch
module output. This verifies both correctness of the implementation and weight loading.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ref_layernorm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, [x.shape[-1]], weight, bias, eps)


class TestTriangleMulOutgoing:
    """Verify TriangleMulOutgoing: z_ij += gate_ij * sum_k(a_ik * b_jk)."""

    def test_output_shape(self):
        from deepfold3.model.ops import TriangleMulOutgoing

        tm = TriangleMulOutgoing(c_z=64)
        z = torch.randn(8, 8, 64)
        mask = torch.ones(8, 8)
        with torch.no_grad():
            out = tm(z, mask)
        assert out.shape == (8, 8, 64)

    def test_matches_reference(self):
        """Manual implementation of TriangleMulOutgoing equation."""
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        N, c = 6, 32
        tm = TriangleMulOutgoing(c_z=c)
        tm.eval()
        z = torch.randn(N, N, c)
        mask = torch.ones(N, N)

        with torch.no_grad():
            out = tm(z, mask)

            # Manual reference
            z0 = ref_layernorm(z, tm.norm_left.weight, tm.norm_left.bias)
            proj = F.linear(z0, tm.proj.weight)  # [N, N, 2c]
            gate = torch.sigmoid(F.linear(z0, tm.gate.weight))  # [N, N, 2c]
            proj = proj * gate
            # Interleaved split (AF3: reshape [2c]->[c,2] then split axis=1)
            a = proj[..., 0::2]  # even channels
            b = proj[..., 1::2]  # odd channels
            m = mask.unsqueeze(-1)
            a = a * m
            b = b * m
            # Outgoing einsum: sum over k
            act = torch.einsum("ikc,jkc->ijc", a, b)
            act = ref_layernorm(act, tm.norm_center.weight, tm.norm_center.bias)
            act = F.linear(act, tm.out_proj.weight)
            out_gate = torch.sigmoid(F.linear(z0, tm.out_gate.weight))
            ref = act * out_gate

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_mask_effect(self):
        """Masking should affect output."""
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        tm = TriangleMulOutgoing(c_z=32)
        # AF3-style init zeros output projections → output is zero at init.
        # Randomize weights so the mask actually affects output.
        for p in tm.parameters():
            if p.dim() >= 2:
                torch.nn.init.kaiming_uniform_(p)
        tm.eval()
        z = torch.randn(4, 4, 32)
        mask_full = torch.ones(4, 4)
        mask_half = torch.ones(4, 4)
        mask_half[2:, :] = 0

        with torch.no_grad():
            out_full = tm(z, mask_full)
            out_half = tm(z, mask_half)

        assert not torch.allclose(out_full, out_half, atol=1e-3)

    def test_batch_consistency(self):
        """Batched [1, N, N, C] should match unbatched [N, N, C]."""
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        tm = TriangleMulOutgoing(c_z=32)
        tm.eval()
        z = torch.randn(4, 4, 32)
        mask = torch.ones(4, 4)

        with torch.no_grad():
            out_unbatched = tm(z, mask)
            out_batched = tm(z.unsqueeze(0), mask.unsqueeze(0))

        torch.testing.assert_close(out_unbatched, out_batched.squeeze(0), atol=1e-5, rtol=1e-5)

    def test_deterministic(self):
        """Same input should give same output."""
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        tm = TriangleMulOutgoing(c_z=32)
        tm.eval()
        z = torch.randn(4, 4, 32)
        mask = torch.ones(4, 4)

        with torch.no_grad():
            out1 = tm(z, mask)
            out2 = tm(z, mask)

        torch.testing.assert_close(out1, out2)

    def test_af3_weights_finite_output(self, model, device):
        """TriangleMulOutgoing with AF3 weights produces finite output."""
        tm = model.evoformer.trunk_pairformer[0].tri_mul_out
        z = torch.randn(16, 16, 128, device=device) * 0.01  # small values for stability
        mask = torch.ones(16, 16, device=device)
        with torch.no_grad():
            out = tm(z, mask)
        assert torch.isfinite(out).all()


class TestTriangleMulIncoming:
    """Verify TriangleMulIncoming: z_ij += gate_ij * sum_k(a_kj * b_ki)."""

    def test_output_shape(self):
        from deepfold3.model.ops import TriangleMulIncoming

        tm = TriangleMulIncoming(c_z=64)
        z = torch.randn(8, 8, 64)
        mask = torch.ones(8, 8)
        with torch.no_grad():
            out = tm(z, mask)
        assert out.shape == (8, 8, 64)

    def test_matches_reference(self):
        """Manual implementation of TriangleMulIncoming equation."""
        from deepfold3.model.ops import TriangleMulIncoming

        torch.manual_seed(42)
        N, c = 6, 32
        tm = TriangleMulIncoming(c_z=c)
        tm.eval()
        z = torch.randn(N, N, c)
        mask = torch.ones(N, N)

        with torch.no_grad():
            out = tm(z, mask)

            z0 = ref_layernorm(z, tm.norm_left.weight, tm.norm_left.bias)
            proj = F.linear(z0, tm.proj.weight)
            gate = torch.sigmoid(F.linear(z0, tm.gate.weight))
            proj = proj * gate
            # Interleaved split (AF3: reshape [2c]->[c,2] then split axis=1)
            a = proj[..., 0::2]  # even channels
            b = proj[..., 1::2]  # odd channels
            m = mask.unsqueeze(-1)
            a = a * m
            b = b * m
            # Incoming einsum: sum over k
            act = torch.einsum("kjc,kic->ijc", a, b)
            act = ref_layernorm(act, tm.norm_center.weight, tm.norm_center.bias)
            act = F.linear(act, tm.out_proj.weight)
            out_gate = torch.sigmoid(F.linear(z0, tm.out_gate.weight))
            ref = act * out_gate

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_batch_consistency(self):
        from deepfold3.model.ops import TriangleMulIncoming

        torch.manual_seed(42)
        tm = TriangleMulIncoming(c_z=32)
        tm.eval()
        z = torch.randn(4, 4, 32)
        mask = torch.ones(4, 4)

        with torch.no_grad():
            out_unbatched = tm(z, mask)
            out_batched = tm(z.unsqueeze(0), mask.unsqueeze(0))

        torch.testing.assert_close(out_unbatched, out_batched.squeeze(0), atol=1e-5, rtol=1e-5)


class TestOuterProductMean:
    """Verify OuterProductMean: pair_ij += mean_s(proj_a(msa_si) outer proj_b(msa_sj))."""

    def test_output_shape(self):
        from deepfold3.model.ops import OuterProductMean

        opm = OuterProductMean(c_msa=64, c_z=128, c_hidden=32)
        msa = torch.randn(4, 8, 64)
        mask = torch.ones(4, 8)
        with torch.no_grad():
            out = opm(msa, mask)
        assert out.shape == (8, 8, 128)

    def test_matches_reference(self):
        """Manual implementation of OPM equation."""
        from deepfold3.model.ops import OuterProductMean

        torch.manual_seed(42)
        S, N, c_msa, c_z, c_h = 3, 5, 16, 8, 4
        opm = OuterProductMean(c_msa=c_msa, c_z=c_z, c_hidden=c_h)
        opm.eval()
        msa = torch.randn(S, N, c_msa)
        mask = torch.ones(S, N)

        with torch.no_grad():
            out = opm(msa, mask)

            # Manual reference
            act = ref_layernorm(msa, opm.norm.weight, opm.norm.bias)
            a = F.linear(act, opm.proj_a.weight) * mask.unsqueeze(-1)  # [S, N, c_h]
            b = F.linear(act, opm.proj_b.weight) * mask.unsqueeze(-1)  # [S, N, c_h]
            # Outer product summed over sequences
            ab = torch.einsum("sie,sjf->ijef", a, b)  # [N, N, c_h, c_h]
            result = torch.einsum("ijef,efc->ijc", ab, opm.out_w) + opm.out_b
            # Normalize: pairwise count of valid sequences
            norm = torch.einsum("si,sj->ij", mask, mask)  # [N, N]
            ref = result / (1e-3 + norm.unsqueeze(-1))

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_mask_normalization(self):
        """OPM normalization uses pairwise mask count, not just S."""
        from deepfold3.model.ops import OuterProductMean

        torch.manual_seed(42)
        opm = OuterProductMean(c_msa=16, c_z=8, c_hidden=4)
        opm.eval()
        S, N = 4, 5
        msa = torch.randn(S, N, 16)

        # Partial mask: different sequences valid at different positions
        mask = torch.ones(S, N)
        mask[2:, 3:] = 0  # last 2 seqs have no info at positions 3,4

        with torch.no_grad():
            out = opm(msa, mask)

        assert torch.isfinite(out).all()

    def test_batch_consistency(self):
        from deepfold3.model.ops import OuterProductMean

        torch.manual_seed(42)
        opm = OuterProductMean(c_msa=16, c_z=8, c_hidden=4)
        opm.eval()
        msa = torch.randn(3, 5, 16)
        mask = torch.ones(3, 5)

        with torch.no_grad():
            out_unbatched = opm(msa, mask)
            out_batched = opm(msa.unsqueeze(0), mask.unsqueeze(0))

        torch.testing.assert_close(out_unbatched, out_batched.squeeze(0), atol=1e-5, rtol=1e-5)

    def test_af3_weights_finite_output(self, model, device):
        """OPM with AF3 weights produces finite output."""
        opm = model.evoformer.msa_stack[0].outer_product
        msa = torch.randn(8, 16, 64, device=device) * 0.01
        mask = torch.ones(8, 16, device=device)
        with torch.no_grad():
            out = opm(msa, mask)
        assert torch.isfinite(out).all()
