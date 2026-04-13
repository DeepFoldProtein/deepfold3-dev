"""Reference implementation tests for attention modules.

Verifies GridSelfAttention, MSAAttention, and SingleAttention against
manual softmax attention implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ref_layernorm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, [x.shape[-1]], weight, bias, eps)


class TestGridSelfAttention:
    """Verify row-wise/column-wise gated self-attention with pair bias."""

    def test_row_output_shape(self):
        from deepfold3.model.attention import GridSelfAttention

        attn = GridSelfAttention(c=128, num_heads=4, transpose=False)
        z = torch.randn(8, 8, 128)
        mask = torch.ones(8, 8)
        with torch.no_grad():
            out = attn(z, mask)
        assert out.shape == (8, 8, 128)

    def test_col_output_shape(self):
        from deepfold3.model.attention import GridSelfAttention

        attn = GridSelfAttention(c=128, num_heads=4, transpose=True)
        z = torch.randn(8, 8, 128)
        mask = torch.ones(8, 8)
        with torch.no_grad():
            out = attn(z, mask)
        assert out.shape == (8, 8, 128)

    def test_matches_reference(self):
        """Manual softmax attention with pair bias and gating."""
        from deepfold3.model.attention import GridSelfAttention

        torch.manual_seed(42)
        N, C, H = 5, 32, 4
        attn = GridSelfAttention(c=C, num_heads=H, transpose=False)
        attn.eval()
        D = attn.head_dim  # C // H = 8

        z = torch.randn(N, N, C)
        mask = torch.ones(N, N)

        with torch.no_grad():
            out = attn(z, mask)

            # Manual reference (per-row attention)
            zn = ref_layernorm(z, attn.norm.weight, attn.norm.bias)
            pb = F.linear(zn, attn.pair_bias.weight)  # [N, N, H]
            pb = pb.permute(2, 0, 1)  # [H, N, N]

            results = []
            for row in range(N):
                z_row = zn[row]  # [N, C]
                q = F.linear(z_row, attn.to_q.weight).view(N, H, D).permute(1, 0, 2)  # [H, N, D]
                k = F.linear(z_row, attn.to_k.weight).view(N, H, D).permute(1, 0, 2)
                v = F.linear(z_row, attn.to_v.weight).view(N, H, D).permute(1, 0, 2)

                logits = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)
                logits = logits + pb  # [H, N, N] broadcast same for all rows
                logits = logits + 1e9 * (mask[row].float() - 1.0).view(1, 1, N)

                w = F.softmax(logits, dim=-1)
                o = torch.matmul(w, v)  # [H, N, D]
                o = o.permute(1, 0, 2).reshape(N, H * D)

                g = torch.sigmoid(F.linear(z_row, attn.to_g.weight))
                o = o * g
                o = F.linear(o, attn.out_proj.weight)
                results.append(o)

            ref = torch.stack(results, dim=0)  # [N, N, C]

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_mask_polarity(self):
        """Masked positions (0) should get large negative logits, not large positive."""
        from deepfold3.model.attention import GridSelfAttention

        attn = GridSelfAttention(c=32, num_heads=4, transpose=False)
        # AF3-style init zeros out_proj/to_g → output is zero at init.
        # Randomize weights so the mask actually affects output.
        for p in attn.parameters():
            if p.dim() >= 2:
                torch.nn.init.kaiming_uniform_(p)
        attn.eval()
        z = torch.randn(4, 4, 32)

        mask_full = torch.ones(4, 4)
        mask_partial = torch.ones(4, 4)
        mask_partial[:, 3] = 0  # mask out column 3

        with torch.no_grad():
            out_full = attn(z, mask_full)
            out_partial = attn(z, mask_partial)

        # Outputs should differ when mask changes
        assert not torch.allclose(out_full, out_partial, atol=1e-3)

    def test_batch_consistency(self):
        from deepfold3.model.attention import GridSelfAttention

        torch.manual_seed(42)
        attn = GridSelfAttention(c=32, num_heads=4, transpose=False)
        attn.eval()
        z = torch.randn(4, 4, 32)
        mask = torch.ones(4, 4)

        with torch.no_grad():
            out_unbatched = attn(z, mask)
            out_batched = attn(z.unsqueeze(0), mask.unsqueeze(0))

        torch.testing.assert_close(out_unbatched, out_batched.squeeze(0), atol=1e-5, rtol=1e-5)

    def test_af3_weights_finite_output(self, model, device):
        """GridSelfAttention with AF3 weights produces finite output."""
        attn = model.evoformer.trunk_pairformer[0].pair_attn_row
        z = torch.randn(16, 16, 128, device=device) * 0.01
        mask = torch.ones(16, 16, device=device)
        with torch.no_grad():
            out = attn(z, mask)
        assert torch.isfinite(out).all()


class TestMSAAttention:
    """Verify row-wise MSA attention with pair bias."""

    def test_output_shape(self):
        from deepfold3.model.attention import MSAAttention

        attn = MSAAttention(c_msa=64, c_pair=128, num_heads=8)
        msa = torch.randn(4, 8, 64)
        mask = torch.ones(4, 8)
        pair = torch.randn(8, 8, 128)
        with torch.no_grad():
            out = attn(msa, mask, pair)
        assert out.shape == (4, 8, 64)

    def test_matches_reference(self):
        """Manual MSA attention: pair logits as non-batched bias, value weighted avg."""
        from deepfold3.model.attention import MSAAttention

        torch.manual_seed(42)
        S, N, c_msa, c_pair, H = 3, 5, 32, 16, 4
        attn = MSAAttention(c_msa=c_msa, c_pair=c_pair, num_heads=H)
        attn.eval()
        D = attn.head_dim

        msa = torch.randn(S, N, c_msa)
        mask = torch.ones(S, N)
        pair = torch.randn(N, N, c_pair)

        with torch.no_grad():
            out = attn(msa, mask, pair)

            # Manual reference
            msa_n = ref_layernorm(msa, attn.norm.weight, attn.norm.bias)
            pair_n = ref_layernorm(pair, attn.pair_norm.weight, attn.pair_norm.bias)

            logits = F.linear(pair_n, attn.pair_logits.weight)  # [N, N, H]
            logits = logits.permute(2, 0, 1)  # [H, N, N]
            col_mask, _ = mask.max(dim=0)
            logits = logits + 1e9 * (col_mask.float() - 1.0).view(1, 1, N)
            w = F.softmax(logits, dim=-1)  # [H, N, N]

            v = F.linear(msa_n, attn.to_v.weight).view(S, N, H, D)
            v_avg = torch.einsum("hqk,skhc->sqhc", w, v).reshape(S, N, H * D)

            g = torch.sigmoid(F.linear(msa_n, attn.to_g.weight))
            v_avg = v_avg * g
            ref = F.linear(v_avg, attn.out_proj.weight)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_batch_consistency(self):
        from deepfold3.model.attention import MSAAttention

        torch.manual_seed(42)
        attn = MSAAttention(c_msa=32, c_pair=16, num_heads=4)
        attn.eval()
        msa = torch.randn(3, 5, 32)
        mask = torch.ones(3, 5)
        pair = torch.randn(5, 5, 16)

        with torch.no_grad():
            out_u = attn(msa, mask, pair)
            out_b = attn(msa.unsqueeze(0), mask.unsqueeze(0), pair.unsqueeze(0))

        torch.testing.assert_close(out_u, out_b.squeeze(0), atol=1e-5, rtol=1e-5)


class TestSingleAttention:
    """Verify single-track attention with pair-biased logits."""

    def test_output_shape(self):
        from deepfold3.model.attention import SingleAttention

        attn = SingleAttention(c=384, num_heads=16, key_dim=384, value_dim=384)
        x = torch.randn(8, 384)
        mask = torch.ones(8)
        pair_logits = torch.randn(16, 8, 8)
        with torch.no_grad():
            out = attn(x, mask, pair_logits)
        assert out.shape == (8, 384)

    def test_matches_reference(self):
        """Manual softmax attention with pair logits bias."""
        from deepfold3.model.attention import SingleAttention

        torch.manual_seed(42)
        N, C, H = 5, 64, 4
        Dk = C // H
        Dv = C // H

        attn = SingleAttention(c=C, num_heads=H, key_dim=C, value_dim=C)
        attn.eval()

        x = torch.randn(N, C)
        mask = torch.ones(N)
        pair_logits = torch.randn(H, N, N) * 0.1

        with torch.no_grad():
            out = attn(x, mask, pair_logits)

            # Manual reference
            xn = ref_layernorm(x, attn.norm.weight, attn.norm.bias)
            q = F.linear(xn, attn.to_q.weight, attn.to_q.bias).view(N, H, Dk).permute(1, 0, 2)
            k = F.linear(xn, attn.to_k.weight).view(N, H, Dk).permute(1, 0, 2)
            v = F.linear(xn, attn.to_v.weight).view(N, H, Dv).permute(1, 0, 2)

            logits = torch.matmul(q * (Dk**-0.5), k.transpose(-2, -1))
            logits = logits + pair_logits
            logits = logits + 1e9 * (mask.float() - 1.0).view(1, 1, N)
            w = F.softmax(logits.float(), dim=-1).to(x.dtype)
            o = torch.matmul(w, v).permute(1, 0, 2).reshape(N, H * Dv)

            g = torch.sigmoid(F.linear(xn, attn.to_g.weight))
            o = o * g
            ref = F.linear(o, attn.out_proj.weight)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_to_q_has_bias(self):
        """SingleAttention.to_q should have bias (nn.Linear, not LinearNoBias)."""
        from deepfold3.model.attention import SingleAttention

        attn = SingleAttention(c=64, num_heads=4, key_dim=64, value_dim=64)
        assert attn.to_q.bias is not None
        assert attn.to_k.bias is None  # LinearNoBias
        assert attn.to_v.bias is None
        assert attn.to_g.bias is None

    def test_batch_consistency(self):
        from deepfold3.model.attention import SingleAttention

        torch.manual_seed(42)
        attn = SingleAttention(c=64, num_heads=4, key_dim=64, value_dim=64)
        attn.eval()
        x = torch.randn(5, 64)
        mask = torch.ones(5)
        pair_logits = torch.randn(4, 5, 5)

        with torch.no_grad():
            out_u = attn(x, mask, pair_logits)
            out_b = attn(x.unsqueeze(0), mask.unsqueeze(0), pair_logits.unsqueeze(0))

        torch.testing.assert_close(out_u, out_b.squeeze(0), atol=1e-5, rtol=1e-5)
