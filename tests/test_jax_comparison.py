"""JAX comparison tests: verify PyTorch modules match AF3's JAX/Haiku implementation.

Each test implements AF3's reference computation in JAX (faithfully imitating
alphafold3/model/network/modules.py) and compares against the corresponding
PyTorch module output using identical weights.

Key finding: AF3 TriangleMultiplication uses INTERLEAVED split
(reshape [2c] -> [c,2] then split axis=1), not contiguous chunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _np(t: torch.Tensor) -> np.ndarray:
    """PyTorch tensor -> NumPy float32."""
    return t.detach().cpu().float().numpy()


def _haiku_w(pt_weight: torch.Tensor) -> np.ndarray:
    """PyTorch Linear weight [out, in] -> Haiku weight [in, out]."""
    return _np(pt_weight).T


def jax_layer_norm(x, scale, offset, eps=1e-5):
    """AF3 hm.LayerNorm: normalize last axis, apply scale + offset."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + offset


def jax_linear(x, w, b=None):
    """AF3 hm.Linear: x @ w [+ b]. w is [in, out] (Haiku layout)."""
    out = x @ w
    if b is not None:
        out = out + b
    return out


class TestLayerNormJax:
    def test_matches_jax(self):
        from deepfold3.model.primitives import LayerNorm

        torch.manual_seed(42)
        C = 128
        ln = LayerNorm(C)
        x = torch.randn(8, 8, C)
        out_pt = ln(x)

        out_jax = jax_layer_norm(
            jnp.array(_np(x)),
            jnp.array(_np(ln.weight)),
            jnp.array(_np(ln.bias)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-5, rtol=1e-5)


class TestTransitionBlockJax:
    def test_swiglu_matches_jax(self):
        """AF3 non-GLU path: Linear(2*mid) -> split -> swish(a)*b -> Linear(c)."""
        from deepfold3.model.primitives import TransitionBlock

        torch.manual_seed(42)
        C = 64
        tb = TransitionBlock(C, num_intermediate_factor=4)
        for p in tb.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        x = torch.randn(4, 4, C) * 0.1
        with torch.no_grad():
            out_pt = tb(x)

        # JAX reference (AF3 non-GLU path, contiguous split for TransitionBlock)
        xj = jnp.array(_np(x))
        act = jax_layer_norm(xj, jnp.array(_np(tb.norm.weight)), jnp.array(_np(tb.norm.bias)))
        act = jax_linear(act, jnp.array(_haiku_w(tb.linear1.weight)))
        a, b = jnp.split(act, 2, axis=-1)
        act = jax.nn.swish(a) * b
        out_jax = jax_linear(act, jnp.array(_haiku_w(tb.linear2.weight)))

        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-5, rtol=1e-5)


class TestTriangleMulJax:
    """AF3 TriangleMultiplication uses INTERLEAVED split (reshape [c,2] then split).

    Reference: modules.py lines 259-331.
    """

    @staticmethod
    def _extract_weights(mod):
        return {
            "ln_s": jnp.array(_np(mod.norm_left.weight)),
            "ln_o": jnp.array(_np(mod.norm_left.bias)),
            "proj_w": jnp.array(_haiku_w(mod.proj.weight)),
            "gate_w": jnp.array(_haiku_w(mod.gate.weight)),
            "cn_s": jnp.array(_np(mod.norm_center.weight)),
            "cn_o": jnp.array(_np(mod.norm_center.bias)),
            "out_w": jnp.array(_haiku_w(mod.out_proj.weight)),
            "gate_out_w": jnp.array(_haiku_w(mod.out_gate.weight)),
        }

    @staticmethod
    def _jax_ref(act, mask, equation, ln_s, ln_o, proj_w, gate_w, cn_s, cn_o, out_w, gate_out_w):
        """AF3 TriangleMultiplication — faithful JAX reference.

        Key: reshape(c, 2, N, N) then split on axis=1 = INTERLEAVED split.
        """
        c = act.shape[-1]

        # LayerNorm
        input_act = jax_layer_norm(act, ln_s, ln_o)

        # Projection + gating (non-GLU path, mathematically equivalent to GLU)
        projection = jax_linear(input_act, proj_w)  # [N, N, 2c]
        projection = jnp.transpose(projection, (2, 0, 1))  # [2c, N, N]
        projection *= mask[None, :, :]

        gate = jax_linear(input_act, gate_w)  # [N, N, 2c]
        gate = jnp.transpose(gate, (2, 0, 1))  # [2c, N, N]
        projection *= jax.nn.sigmoid(gate)

        # INTERLEAVED split: reshape [2c, N, N] -> [c, 2, N, N] -> split axis=1
        projection = projection.reshape(c, 2, *projection.shape[1:])
        a, b = jnp.split(projection, 2, axis=1)
        a = jnp.squeeze(a, axis=1)  # [c, N, N]
        b = jnp.squeeze(b, axis=1)  # [c, N, N]

        # Einsum (channel-first)
        result = jnp.einsum(equation, a, b)  # [c, N, N]

        # Center norm (LayerNorm axis=0, param_axis=0 on [c, N, N])
        mean = jnp.mean(result, axis=0, keepdims=True)
        var = jnp.var(result, axis=0, keepdims=True)
        result = (result - mean) / jnp.sqrt(var + 1e-5) * cn_s[:, None, None] + cn_o[:, None, None]

        # Back to channel-last
        result = jnp.transpose(result, (1, 2, 0))  # [N, N, c]
        result = jax_linear(result, out_w)

        # Output gating
        gate_out = jax.nn.sigmoid(jax_linear(input_act, gate_out_w))
        return result * gate_out

    def test_interleaved_vs_contiguous_differs(self):
        """Demonstrate that interleaved and contiguous splits differ."""
        torch.manual_seed(42)
        proj = torch.randn(6, 6, 64)

        a_cont, b_cont = proj.chunk(2, dim=-1)
        a_inter = proj[..., 0::2]
        b_inter = proj[..., 1::2]

        assert not torch.allclose(a_cont, a_inter, atol=1e-6)
        assert not torch.allclose(b_cont, b_inter, atol=1e-6)

    def test_outgoing_random(self):
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        C, N = 32, 6
        mod = TriangleMulOutgoing(C)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        z = torch.randn(N, N, C) * 0.1
        mask = torch.ones(N, N)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(
            jnp.array(_np(z)),
            jnp.array(_np(mask)),
            "cik,cjk->cij",
            **w,
        )
        np.testing.assert_allclose(
            _np(out_pt),
            np.array(out_jax),
            atol=1e-4,
            rtol=1e-4,
            err_msg="TriangleMulOutgoing does not match AF3 interleaved-split reference",
        )

    def test_incoming_random(self):
        from deepfold3.model.ops import TriangleMulIncoming

        torch.manual_seed(42)
        C, N = 32, 6
        mod = TriangleMulIncoming(C)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        z = torch.randn(N, N, C) * 0.1
        mask = torch.ones(N, N)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(
            jnp.array(_np(z)),
            jnp.array(_np(mask)),
            "ckj,cki->cij",
            **w,
        )
        np.testing.assert_allclose(
            _np(out_pt),
            np.array(out_jax),
            atol=1e-4,
            rtol=1e-4,
            err_msg="TriangleMulIncoming does not match AF3 interleaved-split reference",
        )

    def test_outgoing_af3_weights(self, model, device):
        """PairFormer layer 0 tri_mul_out with real AF3 weights."""
        mod = model.evoformer.trunk_pairformer[0].tri_mul_out
        torch.manual_seed(42)
        N = 8
        z = torch.randn(N, N, 128, device=device) * 0.1
        mask = torch.ones(N, N, device=device)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(
            jnp.array(_np(z)),
            jnp.array(_np(mask)),
            "cik,cjk->cij",
            **w,
        )
        np.testing.assert_allclose(
            _np(out_pt),
            np.array(out_jax),
            atol=1e-4,
            rtol=1e-4,
            err_msg="TriangleMulOutgoing (AF3 weights) mismatch",
        )

    def test_incoming_af3_weights(self, model, device):
        """PairFormer layer 0 tri_mul_in with real AF3 weights."""
        mod = model.evoformer.trunk_pairformer[0].tri_mul_in
        torch.manual_seed(42)
        N = 8
        z = torch.randn(N, N, 128, device=device) * 0.1
        mask = torch.ones(N, N, device=device)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(
            jnp.array(_np(z)),
            jnp.array(_np(mask)),
            "ckj,cki->cij",
            **w,
        )
        np.testing.assert_allclose(
            _np(out_pt),
            np.array(out_jax),
            atol=1e-4,
            rtol=1e-4,
            err_msg="TriangleMulIncoming (AF3 weights) mismatch",
        )


class TestOuterProductMeanJax:
    @staticmethod
    def _jax_ref(msa, mask, ln_s, ln_o, pa_w, pb_w, out_w, out_b):
        """AF3 OuterProductMean: modules.py lines 354-412."""
        msa = jax_layer_norm(msa, ln_s, ln_o)
        m = mask[..., None]
        left = m * jax_linear(msa, pa_w)
        right = m * jax_linear(msa, pb_w)
        ab = jnp.einsum("sie,sjf->ijef", left, right)
        act = jnp.einsum("ijef,efc->ijc", ab, out_w) + out_b
        norm = jnp.einsum("si,sj->ij", mask, mask)
        return act / (1e-3 + norm[..., None])

    def test_random_weights(self):
        from deepfold3.model.ops import OuterProductMean

        torch.manual_seed(42)
        c_msa, c_z, c_h = 64, 128, 32
        mod = OuterProductMean(c_msa, c_z, c_h)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        S, N = 4, 6
        msa = torch.randn(S, N, c_msa) * 0.1
        mask = torch.ones(S, N)
        with torch.no_grad():
            out_pt = mod(msa, mask)

        out_jax = self._jax_ref(
            jnp.array(_np(msa)),
            jnp.array(_np(mask)),
            jnp.array(_np(mod.norm.weight)),
            jnp.array(_np(mod.norm.bias)),
            jnp.array(_haiku_w(mod.proj_a.weight)),
            jnp.array(_haiku_w(mod.proj_b.weight)),
            jnp.array(_np(mod.out_w)),
            jnp.array(_np(mod.out_b)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)

    def test_af3_weights(self, model, device):
        """OPM from MSA stack layer 0 with AF3 weights."""
        mod = model.evoformer.msa_stack[0].outer_product
        torch.manual_seed(42)
        S, N = 4, 8
        msa = torch.randn(S, N, 64, device=device) * 0.1
        mask = torch.ones(S, N, device=device)
        with torch.no_grad():
            out_pt = mod(msa, mask)

        out_jax = self._jax_ref(
            jnp.array(_np(msa)),
            jnp.array(_np(mask)),
            jnp.array(_np(mod.norm.weight)),
            jnp.array(_np(mod.norm.bias)),
            jnp.array(_haiku_w(mod.proj_a.weight)),
            jnp.array(_haiku_w(mod.proj_b.weight)),
            jnp.array(_np(mod.out_w)),
            jnp.array(_np(mod.out_b)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)


class TestGridSelfAttentionJax:
    @staticmethod
    def _extract_weights(mod):
        return {
            "ln_s": jnp.array(_np(mod.norm.weight)),
            "ln_o": jnp.array(_np(mod.norm.bias)),
            "pb_w": jnp.array(_haiku_w(mod.pair_bias.weight)),
            "q_w": jnp.array(_haiku_w(mod.to_q.weight)),
            "k_w": jnp.array(_haiku_w(mod.to_k.weight)),
            "v_w": jnp.array(_haiku_w(mod.to_v.weight)),
            "g_w": jnp.array(_haiku_w(mod.to_g.weight)),
            "out_w": jnp.array(_haiku_w(mod.out_proj.weight)),
        }

    @staticmethod
    def _jax_ref(act, mask, ln_s, ln_o, pb_w, q_w, k_w, v_w, g_w, out_w, transpose=False):
        """AF3 GridSelfAttention: modules.py lines 131-242.

        All weights in Haiku [in, out] layout.
        """
        act = jax_layer_norm(act, ln_s, ln_o)

        # Pair bias: [N, N, H] -> [H, N, N]
        bias = jax_linear(act, pb_w)
        bias = jnp.transpose(bias, [2, 0, 1])

        if transpose:
            act = jnp.swapaxes(act, -2, -3)
            mask = jnp.swapaxes(mask, -1, -2)

        N_rows, N_cols, C = act.shape
        H = bias.shape[0]
        HD = q_w.shape[1]
        D = HD // H

        out_rows = []
        for r in range(N_rows):
            row = act[r]  # [N_cols, C]

            q = (row @ q_w).reshape(N_cols, H, D)
            k = (row @ k_w).reshape(N_cols, H, D)
            v = (row @ v_w).reshape(N_cols, H, D)

            q = jnp.transpose(q, [1, 0, 2])  # [H, N_cols, D]
            k = jnp.transpose(k, [1, 0, 2])
            v = jnp.transpose(v, [1, 0, 2])

            logits = jnp.einsum("hqd,hkd->hqk", q, k) * (D**-0.5)
            logits = logits + bias  # non-batched [H, N, N]

            m = 1e9 * (mask[r].astype(jnp.float32) - 1.0)
            logits = logits + m[None, None, :]

            w = jax.nn.softmax(logits, axis=-1)
            attn_out = jnp.einsum("hqk,hkd->hqd", w, v)
            attn_out = jnp.transpose(attn_out, [1, 0, 2]).reshape(N_cols, HD)

            g = jax.nn.sigmoid(row @ g_w)
            attn_out = attn_out * g
            attn_out = jax_linear(attn_out, out_w)
            out_rows.append(attn_out)

        result = jnp.stack(out_rows, axis=0)
        if transpose:
            result = jnp.swapaxes(result, -2, -3)
        return result

    def test_row_attn_random(self):
        from deepfold3.model.attention import GridSelfAttention

        torch.manual_seed(42)
        C, H, N = 32, 4, 6
        mod = GridSelfAttention(C, num_heads=H, transpose=False)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        z = torch.randn(N, N, C) * 0.1
        mask = torch.ones(N, N)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(jnp.array(_np(z)), jnp.array(_np(mask)), **w, transpose=False)
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)

    def test_col_attn_random(self):
        from deepfold3.model.attention import GridSelfAttention

        torch.manual_seed(42)
        C, H, N = 32, 4, 6
        mod = GridSelfAttention(C, num_heads=H, transpose=True)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        z = torch.randn(N, N, C) * 0.1
        mask = torch.ones(N, N)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(jnp.array(_np(z)), jnp.array(_np(mask)), **w, transpose=True)
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)

    def test_row_attn_af3_weights(self, model, device):
        mod = model.evoformer.trunk_pairformer[0].pair_attn_row
        torch.manual_seed(42)
        N = 8
        z = torch.randn(N, N, 128, device=device) * 0.1
        mask = torch.ones(N, N, device=device)
        with torch.no_grad():
            out_pt = mod(z, mask)

        w = self._extract_weights(mod)
        out_jax = self._jax_ref(jnp.array(_np(z)), jnp.array(_np(mask)), **w, transpose=False)
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-3, rtol=1e-3)


class TestMSAAttentionJax:
    @staticmethod
    def _jax_ref(msa, mask, pair, ln_s, ln_o, pn_s, pn_o, pl_w, v_w, g_w, out_w):
        """AF3 MSAAttention: modules.py lines 87-128."""
        msa = jax_layer_norm(msa, ln_s, ln_o)
        pair = jax_layer_norm(pair, pn_s, pn_o)

        logits = jax_linear(pair, pl_w)  # [N, N, H]
        logits = jnp.transpose(logits, [2, 0, 1])  # [H, N, N]

        col_mask = jnp.max(mask, axis=0)
        logits = logits + 1e9 * (col_mask - 1.0)[None, None, :]
        weights = jax.nn.softmax(logits, axis=-1)  # [H, q, k]

        S, N, c = msa.shape
        H = logits.shape[0]
        D = v_w.shape[1] // H

        v = (msa @ v_w).reshape(S, N, H, D)  # [S, N, H, D]
        v_avg = jnp.einsum("hqk,bkhc->bqhc", weights, v)
        v_avg = v_avg.reshape(S, N, H * D)

        gate = jax.nn.sigmoid(msa @ g_w)
        v_avg = v_avg * gate
        return jax_linear(v_avg, out_w)

    def test_random_weights(self):
        from deepfold3.model.attention import MSAAttention

        torch.manual_seed(42)
        c_msa, c_pair, H = 64, 128, 8
        mod = MSAAttention(c_msa, c_pair, num_heads=H)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        S, N = 4, 6
        msa = torch.randn(S, N, c_msa) * 0.1
        mask = torch.ones(S, N)
        pair = torch.randn(N, N, c_pair) * 0.1
        with torch.no_grad():
            out_pt = mod(msa, mask, pair)

        out_jax = self._jax_ref(
            jnp.array(_np(msa)),
            jnp.array(_np(mask)),
            jnp.array(_np(pair)),
            jnp.array(_np(mod.norm.weight)),
            jnp.array(_np(mod.norm.bias)),
            jnp.array(_np(mod.pair_norm.weight)),
            jnp.array(_np(mod.pair_norm.bias)),
            jnp.array(_haiku_w(mod.pair_logits.weight)),
            jnp.array(_haiku_w(mod.to_v.weight)),
            jnp.array(_haiku_w(mod.to_g.weight)),
            jnp.array(_haiku_w(mod.out_proj.weight)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)


class TestSingleAttentionJax:
    @staticmethod
    def _jax_ref(x, mask, pair_logits, ln_s, ln_o, q_w, q_b, k_w, v_w, g_w, out_w):
        """AF3 self_attention: diffusion_transformer.py lines 120-177.

        single_cond=None -> regular LayerNorm, plain Linear output.
        """
        x = jax_layer_norm(x, ln_s, ln_o)
        N, C = x.shape
        H = pair_logits.shape[0]
        Dk = q_w.shape[1] // H
        Dv = v_w.shape[1] // H

        q = jax_linear(x, q_w, q_b).reshape(N, H, Dk)
        k = jax_linear(x, k_w).reshape(N, H, Dk)
        v = jax_linear(x, v_w).reshape(N, H, Dv)

        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        logits = jnp.einsum("qhc,khc->hqk", q * (Dk**-0.5), k)
        logits = logits + pair_logits.astype(jnp.float32)
        bias = 1e9 * (mask.astype(jnp.float32) - 1.0)
        logits = logits + bias[None, None, :]

        weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
        weighted = jnp.einsum("hqk,khc->qhc", weights, v)
        weighted = weighted.reshape(N, H * Dv)

        gate = jax.nn.sigmoid(jax_linear(x, g_w))
        weighted = weighted * gate
        return jax_linear(weighted, out_w)

    def test_random_weights(self):
        from deepfold3.model.attention import SingleAttention

        torch.manual_seed(42)
        C, H = 64, 16
        mod = SingleAttention(C, num_heads=H, key_dim=64, value_dim=64)
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)

        N = 8
        x = torch.randn(N, C) * 0.1
        mask = torch.ones(N)
        pair_logits = torch.randn(H, N, N) * 0.01
        with torch.no_grad():
            out_pt = mod(x, mask, pair_logits)

        out_jax = self._jax_ref(
            jnp.array(_np(x)),
            jnp.array(_np(mask)),
            jnp.array(_np(pair_logits)),
            jnp.array(_np(mod.norm.weight)),
            jnp.array(_np(mod.norm.bias)),
            jnp.array(_haiku_w(mod.to_q.weight)),
            jnp.array(_np(mod.to_q.bias)),
            jnp.array(_haiku_w(mod.to_k.weight)),
            jnp.array(_haiku_w(mod.to_v.weight)),
            jnp.array(_haiku_w(mod.to_g.weight)),
            jnp.array(_haiku_w(mod.out_proj.weight)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-4, rtol=1e-4)


class TestPairFormerLayer0Jax:
    """Full first PairFormer layer: all sub-modules composed sequentially.

    tri_mul_out -> tri_mul_in -> pair_attn_row -> pair_attn_col -> pair_transition
    -> single_pair_logits -> single_attention -> single_transition
    """

    @staticmethod
    def _jax_transition(x, norm_w, norm_b, w1, w2):
        """JAX TransitionBlock: LN -> Linear(2*mid) -> SwiGLU -> Linear."""
        act = jax_layer_norm(x, norm_w, norm_b)
        act = jax_linear(act, w1)
        a, b = jnp.split(act, 2, axis=-1)
        act = jax.nn.swish(a) * b
        return jax_linear(act, w2)

    def test_first_layer(self, model, device):
        layer = model.evoformer.trunk_pairformer[0]

        torch.manual_seed(42)
        N = 8
        pair = torch.randn(N, N, 128, device=device) * 0.01
        single = torch.randn(N, 384, device=device) * 0.01
        pair_mask = torch.ones(N, N, device=device)
        seq_mask = torch.ones(N, device=device)

        with torch.no_grad():
            pair_out, single_out = layer(pair, single, pair_mask, seq_mask)

        # JAX reference: compose each sub-module
        p = jnp.array(_np(pair))
        s = jnp.array(_np(single))
        pm = jnp.array(_np(pair_mask))
        sm = jnp.array(_np(seq_mask))

        # 1) tri_mul_out
        w = TestTriangleMulJax._extract_weights(layer.tri_mul_out)
        p = p + TestTriangleMulJax._jax_ref(p, pm, "cik,cjk->cij", **w)

        # 2) tri_mul_in
        w = TestTriangleMulJax._extract_weights(layer.tri_mul_in)
        p = p + TestTriangleMulJax._jax_ref(p, pm, "ckj,cki->cij", **w)

        # 3) pair_attn_row
        w = TestGridSelfAttentionJax._extract_weights(layer.pair_attn_row)
        p = p + TestGridSelfAttentionJax._jax_ref(p, pm, **w, transpose=False)

        # 4) pair_attn_col
        w = TestGridSelfAttentionJax._extract_weights(layer.pair_attn_col)
        p = p + TestGridSelfAttentionJax._jax_ref(p, pm, **w, transpose=True)

        # 5) pair_transition
        tb = layer.pair_transition
        p = p + self._jax_transition(
            p,
            jnp.array(_np(tb.norm.weight)),
            jnp.array(_np(tb.norm.bias)),
            jnp.array(_haiku_w(tb.linear1.weight)),
            jnp.array(_haiku_w(tb.linear2.weight)),
        )

        # 6) single_pair_logits
        sp_norm = layer.single_to_pair_norm
        sp_proj = layer.single_to_pair
        pair_logits = jax_layer_norm(
            p,
            jnp.array(_np(sp_norm.weight)),
            jnp.array(_np(sp_norm.bias)),
        )
        pair_logits = jax_linear(pair_logits, jnp.array(_haiku_w(sp_proj.weight)))
        pair_logits = jnp.transpose(pair_logits, [2, 0, 1])  # [H, N, N]

        # 7) single_attention
        sa = layer.single_attn
        s = s + TestSingleAttentionJax._jax_ref(
            s,
            sm,
            pair_logits,
            jnp.array(_np(sa.norm.weight)),
            jnp.array(_np(sa.norm.bias)),
            jnp.array(_haiku_w(sa.to_q.weight)),
            jnp.array(_np(sa.to_q.bias)),
            jnp.array(_haiku_w(sa.to_k.weight)),
            jnp.array(_haiku_w(sa.to_v.weight)),
            jnp.array(_haiku_w(sa.to_g.weight)),
            jnp.array(_haiku_w(sa.out_proj.weight)),
        )

        # 8) single_transition
        st = layer.single_transition
        s = s + self._jax_transition(
            s,
            jnp.array(_np(st.norm.weight)),
            jnp.array(_np(st.norm.bias)),
            jnp.array(_haiku_w(st.linear1.weight)),
            jnp.array(_haiku_w(st.linear2.weight)),
        )

        # Tolerance loosened for composed layer: f32 accumulation across 8 sub-modules
        np.testing.assert_allclose(
            _np(pair_out),
            np.array(p),
            atol=5e-3,
            rtol=5e-3,
            err_msg="PairFormer layer 0: pair output mismatch",
        )
        np.testing.assert_allclose(
            _np(single_out),
            np.array(s),
            atol=5e-3,
            rtol=5e-3,
            err_msg="PairFormer layer 0: single output mismatch",
        )


# ---------------------------------------------------------------------------
# Diffusion transformer (AF3 diffusion_transformer.py)
#
# The diffusion head is 55% of AF3's parameters and had no JAX parity coverage:
# the tests above verify the trunk's building blocks, which leaves the module
# that actually turns the trunk's pair representation into coordinates unchecked.
# ---------------------------------------------------------------------------


def jax_adaptive_layernorm(x, single_cond, cond_norm_w, scale_w, scale_b, bias_w):
    """AF3 ``diffusion_transformer.adaptive_layernorm`` (lines 32-60).

    ``x`` is normalised with NO scale/offset; ``single_cond`` with scale only
    (``create_offset=False``); the pair of projections then give AdaLN's
    ``sigmoid(scale) * x + bias``.
    """
    xn = jax_layer_norm(x, 1.0, 0.0)
    cn = jax_layer_norm(single_cond, cond_norm_w, 0.0)
    single_scale = jax_linear(cn, scale_w, scale_b)
    single_bias = jax_linear(cn, bias_w)
    return jax.nn.sigmoid(single_scale) * xn + single_bias


def jax_adaptive_zero_init(x, single_cond, out_w, cond_w, cond_b):
    """AF3 ``adaptive_zero_init`` (lines 63-85) — AdaLN-zero output gate.

    NOTE the gate reads the RAW ``single_cond``, not the layer-normed copy the
    AdaLN above uses. Feeding it the normed one is a silent, plausible-looking
    error, which is exactly why it is pinned here.
    """
    output = jax_linear(x, out_w)
    cond = jax_linear(single_cond, cond_w, cond_b)
    return jax.nn.sigmoid(cond) * output


def _randomise(module, seed=0, std=0.3):
    """Give every parameter a non-degenerate value.

    AdaLN-zero ships several projections zero-initialised and the gate biases at
    -2, so a freshly built block is partly a no-op: leaving it at init would let
    a wrong wiring pass by multiplying zeros. Every test below randomises first.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)


class TestDiffusionAdaLayerNormJax:
    def test_matches_jax(self):
        from deepfold3.model.primitives import AdaLayerNorm

        torch.manual_seed(0)
        C, CS = 64, 48
        ada = AdaLayerNorm(C, CS)
        _randomise(ada, seed=1)

        x = torch.randn(3, 7, C)
        cond = torch.randn(3, 7, CS)
        with torch.no_grad():
            out_pt = ada(x, cond)

        out_jax = jax_adaptive_layernorm(
            jnp.array(_np(x)),
            jnp.array(_np(cond)),
            jnp.array(_np(ada.cond_norm_weight)),
            jnp.array(_haiku_w(ada.cond_scale.weight)),
            jnp.array(_np(ada.cond_scale.bias)),
            jnp.array(_haiku_w(ada.cond_bias.weight)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-5, rtol=1e-5)


class TestDiffusionTransitionJax:
    """AF3 ``transition_block`` with ``single_cond`` (lines 87-121)."""

    def test_matches_jax(self):
        from deepfold3.model.primitives import AdaTransitionBlock

        torch.manual_seed(0)
        C, CS, F = 64, 48, 2
        tb = AdaTransitionBlock(C, CS, num_intermediate_factor=F)
        _randomise(tb, seed=2, std=0.15)

        x = torch.randn(2, 5, C)
        cond = torch.randn(2, 5, CS)
        with torch.no_grad():
            out_pt = tb(x, cond)

        xj, cj = jnp.array(_np(x)), jnp.array(_np(cond))
        act = jax_adaptive_layernorm(
            xj, cj,
            jnp.array(_np(tb.cond_norm_weight)),
            jnp.array(_haiku_w(tb.cond_scale.weight)),
            jnp.array(_np(tb.cond_scale.bias)),
            jnp.array(_haiku_w(tb.cond_bias.weight)),
        )
        act = jax_linear(act, jnp.array(_haiku_w(tb.linear1.weight)))
        a, b = jnp.split(act, 2, axis=-1)          # contiguous split, not interleaved
        act = jax.nn.swish(a) * b
        out_jax = jax_adaptive_zero_init(
            act, cj,
            jnp.array(_haiku_w(tb.linear2.weight)),
            jnp.array(_haiku_w(tb.zero_cond.weight)),
            jnp.array(_np(tb.zero_cond.bias)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=1e-5, rtol=1e-5)


def _jax_diffusion_self_attention(x, mask, pair_logits, single_cond, blk, num_head):
    """AF3 ``diffusion_transformer.self_attention`` (lines 129-186), unbatched."""
    key_dim = blk.to_k.weight.shape[0] // num_head
    value_dim = blk.to_v.weight.shape[0] // num_head

    bias = (1e9 * (mask - 1.0))[..., None, None, :]
    xn = jax_adaptive_layernorm(
        x, single_cond,
        jnp.array(_np(blk.attn_cond_norm_w)),
        jnp.array(_haiku_w(blk.attn_cond_scale.weight)),
        jnp.array(_np(blk.attn_cond_scale.bias)),
        jnp.array(_haiku_w(blk.attn_cond_bias.weight)),
    )
    n = xn.shape[-2]
    q = (jax_linear(xn, jnp.array(_haiku_w(blk.to_q.weight)), jnp.array(_np(blk.to_q.bias)))
         .reshape(n, num_head, key_dim))
    k = jax_linear(xn, jnp.array(_haiku_w(blk.to_k.weight))).reshape(n, num_head, key_dim)
    v = jax_linear(xn, jnp.array(_haiku_w(blk.to_v.weight))).reshape(n, num_head, value_dim)

    logits = jnp.einsum("qhc,khc->hqk", q * key_dim ** (-0.5), k) + bias
    if pair_logits is not None:
        logits = logits + pair_logits
    weights = jax.nn.softmax(logits, axis=-1)
    weighted_avg = jnp.einsum("hqk,khc->qhc", weights, v).reshape(n, num_head * value_dim)

    # `gating_query` passes bias_init=1.0 but NOT use_bias=True, and hm.Linear
    # gates the parameter on use_bias (default False) — so there is no bias here
    # and LinearNoBias is right. The stray bias_init is dead in the reference.
    gate_logits = jax_linear(xn, jnp.array(_haiku_w(blk.to_g.weight)))
    weighted_avg = weighted_avg * jax.nn.sigmoid(gate_logits)

    return jax_adaptive_zero_init(
        weighted_avg, single_cond,
        jnp.array(_haiku_w(blk.out_proj.weight)),
        jnp.array(_haiku_w(blk.zero_cond.weight)),
        jnp.array(_np(blk.zero_cond.bias)),
    )


class TestDiffusionSelfAttentionJax:
    """Attention half of the block: mask bias, pair bias, gating, AdaLN-zero out."""

    @pytest.mark.parametrize("with_pair", [False, True])
    def test_matches_jax(self, with_pair):
        from deepfold3.model.diffusion import DiffusionTransformerBlock

        torch.manual_seed(0)
        C, CS, H, N = 64, 48, 4, 9
        blk = DiffusionTransformerBlock(C, CS, num_heads=H, num_intermediate_factor=2)
        _randomise(blk, seed=3, std=0.12)

        x = torch.randn(N, C)
        cond = torch.randn(N, CS)
        mask = torch.ones(N)
        mask[-2:] = 0.0                                    # exercise the 1e9 mask bias
        pair = torch.randn(H, N, N) * 0.5 if with_pair else None

        # Run only the attention half: FFN gate zeroed so its residual drops out.
        with torch.no_grad():
            blk.ffn_zero_cond.bias.fill_(-40.0)            # sigmoid(-40) ~ 0
            blk.ffn_zero_cond.weight.zero_()
            out_pt = blk(x.unsqueeze(0), mask.unsqueeze(0), cond.unsqueeze(0),
                         None if pair is None else pair.unsqueeze(0))[0]

        delta = _jax_diffusion_self_attention(
            jnp.array(_np(x)), jnp.array(_np(mask)),
            None if pair is None else jnp.array(_np(pair)),
            jnp.array(_np(cond)), blk, H,
        )
        out_jax = jnp.array(_np(x)) + delta
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=2e-5, rtol=2e-5)


class TestDiffusionTransformerBlockJax:
    """The assembled block: ``act += self_attention(act)``, then ``act += transition(act)``.

    Layer-level parity is not implied by the two halves passing — the residual
    order and which activation feeds the FFN are exactly the kind of wiring that
    a per-half test cannot see.
    """

    def test_matches_jax(self):
        from deepfold3.model.diffusion import DiffusionTransformerBlock

        torch.manual_seed(0)
        C, CS, H, N = 64, 48, 4, 9
        blk = DiffusionTransformerBlock(C, CS, num_heads=H, num_intermediate_factor=2)
        _randomise(blk, seed=4, std=0.12)

        x = torch.randn(N, C)
        cond = torch.randn(N, CS)
        mask = torch.ones(N)
        mask[-1] = 0.0
        pair = torch.randn(H, N, N) * 0.5

        with torch.no_grad():
            out_pt = blk(x.unsqueeze(0), mask.unsqueeze(0), cond.unsqueeze(0), pair.unsqueeze(0))[0]

        xj, cj = jnp.array(_np(x)), jnp.array(_np(cond))
        act = xj + _jax_diffusion_self_attention(
            xj, jnp.array(_np(mask)), jnp.array(_np(pair)), cj, blk, H
        )
        ffn = jax_adaptive_layernorm(
            act, cj,
            jnp.array(_np(blk.ffn_cond_norm_w)),
            jnp.array(_haiku_w(blk.ffn_cond_scale.weight)),
            jnp.array(_np(blk.ffn_cond_scale.bias)),
            jnp.array(_haiku_w(blk.ffn_cond_bias.weight)),
        )
        ffn = jax_linear(ffn, jnp.array(_haiku_w(blk.ffn_linear1.weight)))
        a, b = jnp.split(ffn, 2, axis=-1)
        ffn = jax.nn.swish(a) * b
        out_jax = act + jax_adaptive_zero_init(
            ffn, cj,
            jnp.array(_haiku_w(blk.ffn_linear2.weight)),
            jnp.array(_haiku_w(blk.ffn_zero_cond.weight)),
            jnp.array(_np(blk.ffn_zero_cond.bias)),
        )
        np.testing.assert_allclose(_np(out_pt), np.array(out_jax), atol=2e-5, rtol=2e-5)
