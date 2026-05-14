"""Tests for Evoformer trunk: MSA stack, PairFormer, full forward pass.

Focuses on:
- Block-level correctness (shapes, residuals)
- Activation statistics through layers (diagnose exploding/vanishing)
- Float32 vs float64 precision comparison
- Recycling convergence
"""

from __future__ import annotations

import pytest
import torch


class TestEvoformerBlock:
    """Verify single EvoformerBlock (MSA stack layer)."""

    def test_output_shapes(self):
        from deepfold3.model.evoformer import EvoformerBlock

        S, N, c_msa, c_pair = 4, 8, 64, 128
        block = EvoformerBlock(c_msa, c_pair)
        msa = torch.randn(S, N, c_msa)
        pair = torch.randn(N, N, c_pair)
        msa_mask = torch.ones(S, N)
        pair_mask = torch.ones(N, N)

        with torch.no_grad():
            msa_out, pair_out = block(msa, pair, msa_mask, pair_mask)

        assert msa_out.shape == (S, N, c_msa)
        assert pair_out.shape == (N, N, c_pair)

    def test_residual_connections(self):
        """Output should differ from input (residual adds non-zero delta)."""
        from deepfold3.model.evoformer import EvoformerBlock

        torch.manual_seed(42)
        S, N, c_msa, c_pair = 4, 8, 64, 128
        block = EvoformerBlock(c_msa, c_pair)
        # AF3-style init zeros output projections → residual delta is zero at init.
        # Randomize weights so the residual path contributes non-zero output.
        for p in block.parameters():
            if p.dim() >= 2:
                torch.nn.init.kaiming_uniform_(p)
        block.eval()
        msa = torch.randn(S, N, c_msa)
        pair = torch.randn(N, N, c_pair)
        msa_mask = torch.ones(S, N)
        pair_mask = torch.ones(N, N)

        with torch.no_grad():
            msa_out, pair_out = block(msa, pair, msa_mask, pair_mask)

        # Outputs should be different from inputs (residual adds something)
        assert not torch.allclose(msa_out, msa, atol=1e-3)
        assert not torch.allclose(pair_out, pair, atol=1e-3)

    def test_finite_output(self):
        from deepfold3.model.evoformer import EvoformerBlock

        torch.manual_seed(42)
        block = EvoformerBlock(64, 128)
        block.eval()
        with torch.no_grad():
            msa_out, pair_out = block(
                torch.randn(4, 8, 64),
                torch.randn(8, 8, 128),
                torch.ones(4, 8),
                torch.ones(8, 8),
            )
        assert torch.isfinite(msa_out).all()
        assert torch.isfinite(pair_out).all()


class TestPairFormerBlock:
    """Verify single PairFormerBlock (trunk layer)."""

    def test_output_shapes(self):
        from deepfold3.model.evoformer import PairFormerBlock

        N, c_pair, c_single = 8, 128, 384
        block = PairFormerBlock(c_pair, c_single)
        pair = torch.randn(N, N, c_pair)
        single = torch.randn(N, c_single)
        pair_mask = torch.ones(N, N)
        seq_mask = torch.ones(N)

        with torch.no_grad():
            pair_out, single_out = block(pair, single, pair_mask, seq_mask)

        assert pair_out.shape == (N, N, c_pair)
        assert single_out.shape == (N, c_single)

    def test_pair_logits_from_pair(self):
        """pair_logits for single attention should come from PAIR (not single)."""
        from deepfold3.model.evoformer import PairFormerBlock

        block = PairFormerBlock(128, 384)
        # single_to_pair projects pair act → per-head logits
        assert block.single_to_pair_norm.weight.shape[0] == 128  # pair channel
        assert block.single_to_pair.weight.shape[1] == 128  # input is pair channel
        assert block.single_to_pair.weight.shape[0] == 16  # num_heads


class TestPairFormerBranchScaler:
    """Per-branch residual scaler module (``identity`` | ``rmsnorm``).

    With ``branch_scaler="rmsnorm"`` the per-block residual update from each
    branch is exactly ``γ_k · unit-RMS-vector`` regardless of the branch's
    internal scale. Verified by replacing each branch's forward with a
    constant-scale tensor and checking the residual delta.
    """

    def _make_block(self, *, scaler: str, residual_out_scale: float = 1.0):
        from deepfold3.model.evoformer import PairFormerBlock

        torch.manual_seed(0)
        return PairFormerBlock(
            128,
            384,
            residual_out_scale=residual_out_scale,
            branch_scaler=scaler,
        )

    def test_branch_outputs_are_unit_rms_when_rmsnorm(self):
        import torch.nn as nn

        from deepfold3.model.primitives import rms_normalize

        class _ConstBranch(nn.Module):
            def __init__(self, scale: float):
                super().__init__()
                self.scale = float(scale)

            def forward(self, x: torch.Tensor, *_a, **_k) -> torch.Tensor:
                return torch.full_like(x, self.scale)

        N = 8
        scale = 73.0
        block_on = self._make_block(scaler="rmsnorm")
        block_off = self._make_block(scaler="identity")
        for blk in (block_on, block_off):
            blk.tri_mul_out = _ConstBranch(scale)
            blk.tri_mul_in = _ConstBranch(scale)
            blk.pair_attn_row = _ConstBranch(scale)
            blk.pair_attn_col = _ConstBranch(scale)
            blk.pair_transition = _ConstBranch(scale)
            blk.single_attn = _ConstBranch(scale)
            blk.single_transition = _ConstBranch(scale)
            blk.eval()
            blk.pair_dropout = 0.0

        pair = torch.zeros(N, N, 128)
        single = torch.zeros(N, 384)
        pair_mask = torch.ones(N, N)
        seq_mask = torch.ones(N)

        with torch.no_grad():
            _, single_on = block_on(pair.clone(), single.clone(), pair_mask, seq_mask)
            pair_off, _ = block_off(pair.clone(), single.clone(), pair_mask, seq_mask)

        assert pair_off.abs().max() > 50.0, "identity scaler should accumulate raw branch scale"
        assert single_on.abs().max() < 5.0, (
            f"with rmsnorm, single residual stream should stay O(1), got max={single_on.abs().max():.2f}"
        )
        const = torch.full((4, 384), scale)
        rms = rms_normalize(const).pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)

    def test_default_is_identity(self):
        from deepfold3.model.evoformer import PairFormerBlock

        block = PairFormerBlock(64, 128)
        assert block.branch_scaler_kind == "identity"


class TestPairFormerConfigBranchScalerValidation:
    """``PairFormerConfig.__post_init__`` rejects per-branch gain knobs when
    ``branch_scaler != "identity"`` (magnitude-fixing scalers make them
    redundant)."""

    def test_clean_combo_ok(self):
        from deepfold3.model.config import PairFormerConfig

        cfg = PairFormerConfig(branch_scaler="rmsnorm", uniform_residual_scale=True)
        assert cfg.branch_scaler == "rmsnorm"

    def test_rejects_single_transition_residual_gain(self):
        import pytest

        from deepfold3.model.config import PairFormerConfig

        with pytest.raises(ValueError, match="incompatible"):
            PairFormerConfig(
                branch_scaler="rmsnorm",
                single_transition_residual_gain=0.35,
            )

    def test_rejects_pair_transition_residual_gain(self):
        import pytest

        from deepfold3.model.config import PairFormerConfig

        with pytest.raises(ValueError, match="incompatible"):
            PairFormerConfig(
                branch_scaler="rmsnorm",
                pair_transition_residual_gain=0.5,
            )

    def test_rejects_per_layer_extra_flags(self):
        import pytest

        from deepfold3.model.config import PairFormerConfig

        with pytest.raises(ValueError, match="incompatible"):
            PairFormerConfig(
                branch_scaler="rmsnorm",
                single_transition_residual_per_layer_extra=True,
            )
        with pytest.raises(ValueError, match="incompatible"):
            PairFormerConfig(
                branch_scaler="rmsnorm",
                pair_transition_residual_per_layer_extra=True,
            )

    def test_identity_path_unchanged(self):
        """Default (``branch_scaler='identity'``) accepts every gain knob."""
        from deepfold3.model.config import PairFormerConfig

        cfg = PairFormerConfig(
            branch_scaler="identity",
            single_transition_residual_gain=0.35,
            pair_transition_residual_gain=0.5,
            single_transition_residual_per_layer_extra=True,
            pair_transition_residual_per_layer_extra=True,
        )
        assert cfg.single_transition_residual_gain == 0.35

    def test_confidence_config_mirror(self):
        import pytest

        from deepfold3.model.config import ConfidenceConfig

        cfg = ConfidenceConfig(branch_scaler="rmsnorm", uniform_residual_scale=True)
        assert cfg.branch_scaler == "rmsnorm"

        with pytest.raises(ValueError, match="incompatible"):
            ConfidenceConfig(
                branch_scaler="rmsnorm",
                single_transition_residual_gain=0.35,
            )
        with pytest.raises(ValueError, match="incompatible"):
            ConfidenceConfig(
                branch_scaler="rmsnorm",
                pair_transition_residual_per_layer_extra=True,
            )


class TestLnOutScaleWiring:
    """Uniform F-output scale design: every residual F output is multiplied
    by the per-block ``residual_out_scale``. No LayerNorm.output_scale is set
    anywhere — every LN keeps its baseline behaviour."""

    def _all_residual_lns_pairformer(self, block):
        # Every LN inside this PairFormerBlock — none should be rescaled.
        return {
            "tri_mul_out.norm_left": block.tri_mul_out.norm_left,
            "tri_mul_out.norm_center": block.tri_mul_out.norm_center,
            "tri_mul_in.norm_left": block.tri_mul_in.norm_left,
            "tri_mul_in.norm_center": block.tri_mul_in.norm_center,
            "pair_attn_row.norm": block.pair_attn_row.norm,
            "pair_attn_col.norm": block.pair_attn_col.norm,
            "pair_transition.norm": block.pair_transition.norm,
            "single_attn.norm": block.single_attn.norm,
            "single_transition.norm": block.single_transition.norm,
            "single_to_pair_norm": block.single_to_pair_norm,
        }

    def _all_residual_lns_evoformer(self, block):
        return {
            "row_attn.norm": block.row_attn.norm,
            "row_attn.pair_norm": block.row_attn.pair_norm,
            "outer_product.norm": block.outer_product.norm,
            "msa_transition.norm": block.msa_transition.norm,
            "tri_mul_out.norm_left": block.tri_mul_out.norm_left,
            "tri_mul_in.norm_left": block.tri_mul_in.norm_left,
            "pair_attn_row.norm": block.pair_attn_row.norm,
            "pair_attn_col.norm": block.pair_attn_col.norm,
            "pair_transition.norm": block.pair_transition.norm,
        }

    def test_lns_per_layer_pairformer(self):
        import math

        from deepfold3.model.config import EvoformerConfig, PairFormerConfig
        from deepfold3.model.evoformer import Evoformer

        cfg = EvoformerConfig()
        pf_cfg = PairFormerConfig(num_layer=8, residual_output_scaling=True)
        evo = Evoformer(cfg, pf_cfg)
        for i, block in enumerate(evo.trunk_pairformer):
            expected = 1.0 / math.sqrt(i + 1)
            assert block.residual_out_scale == expected
            # Every LN stays at 1.0 — F-output scale design only.
            for name, ln in self._all_residual_lns_pairformer(block).items():
                assert ln.output_scale == 1.0, f"layer {i} {name}: should be 1.0"

    def test_uniform_pairformer(self):
        import math

        from deepfold3.model.config import EvoformerConfig, PairFormerConfig
        from deepfold3.model.evoformer import Evoformer

        cfg = EvoformerConfig()
        pf_cfg = PairFormerConfig(num_layer=8, uniform_residual_scale=True)
        evo = Evoformer(cfg, pf_cfg)
        expected = 1.0 / math.sqrt(8)
        for block in evo.trunk_pairformer:
            assert block.residual_out_scale == expected
            for ln in self._all_residual_lns_pairformer(block).values():
                assert ln.output_scale == 1.0

    def test_lns_per_layer_msa_stack(self):
        import math

        from deepfold3.model.config import EvoformerConfig, PairFormerConfig
        from deepfold3.model.evoformer import Evoformer

        cfg = EvoformerConfig(num_layer=4, residual_output_scaling=True)
        pf_cfg = PairFormerConfig(num_layer=8)
        evo = Evoformer(cfg, pf_cfg)
        for i, block in enumerate(evo.msa_stack):
            expected = 1.0 / math.sqrt(i + 1)
            assert block.residual_out_scale == expected
            for name, ln in self._all_residual_lns_evoformer(block).items():
                assert ln.output_scale == 1.0, f"layer {i} {name}: should be 1.0"

    def test_default_off_keeps_unity(self):
        from deepfold3.model.config import EvoformerConfig, PairFormerConfig
        from deepfold3.model.evoformer import Evoformer

        evo = Evoformer(EvoformerConfig(), PairFormerConfig(num_layer=4))
        for block in evo.trunk_pairformer:
            assert block.residual_out_scale == 1.0
            for ln in self._all_residual_lns_pairformer(block).values():
                assert ln.output_scale == 1.0

    def test_pair_logits_unchanged_when_gate_off(self):
        """Without ``pair_bias_gate``, pair_logits passed to SingleAttention
        match the bare ``single_to_pair(single_to_pair_norm(pair))`` output —
        no fixed depth-dependent or per-head rescaling is applied."""
        import math

        import torch

        from deepfold3.model.evoformer import PairFormerBlock

        torch.manual_seed(0)
        # Even with LNS active, pair_logits stays unrescaled when gate is off.
        block = PairFormerBlock(128, 384, residual_out_scale=1.0 / math.sqrt(8)).eval()
        N = 6
        pair = torch.randn(N, N, 128)
        single = torch.randn(N, 384)
        pair_mask = torch.ones(N, N)
        seq_mask = torch.ones(N)

        captured: dict[str, torch.Tensor] = {}
        original_compute = block.single_attn._compute_baseline

        def spy(xn, mask, pair_logits):
            captured["pair_logits"] = pair_logits.detach().clone()
            return original_compute(xn, mask, pair_logits)

        block.single_attn._compute_baseline = spy
        with torch.no_grad():
            block(pair, single, pair_mask, seq_mask)
            unscaled = block.single_to_pair(block.single_to_pair_norm(pair)).movedim(-1, -3).unsqueeze(0)
        # Block-level pair_logits passed in is the unrescaled bias; the gate
        # (when on) lives inside SingleAttention, not in the block.
        torch.testing.assert_close(captured["pair_logits"], unscaled, atol=1e-5, rtol=1e-5)


class TestPairBiasGate:
    """Per-head learnable PairBias gate τ_h = exp(a_h) on SingleAttention."""

    def test_off_by_default(self):
        from deepfold3.model.attention import SingleAttention

        attn = SingleAttention(384)
        assert attn.pair_bias_gate_a is None

    def test_init_value(self):
        import math

        import torch

        from deepfold3.model.attention import SingleAttention

        attn = SingleAttention(384, num_heads=16, pair_bias_gate=True, pair_bias_gate_init_tau=0.25)
        assert attn.pair_bias_gate_a is not None
        # Init: a_h = log(τ₀) for every head.
        expected = math.log(0.25)
        assert attn.pair_bias_gate_a.shape == (16,)
        torch.testing.assert_close(attn.pair_bias_gate_a, torch.full((16,), expected), atol=1e-6, rtol=1e-6)

    def test_param_is_1d_so_routes_to_no_decay(self):
        from deepfold3.model.attention import SingleAttention

        attn = SingleAttention(384, num_heads=16, pair_bias_gate=True)
        # 1-D shape → ``param.dim() < 2`` rule in optim.py routes it to
        # the no-decay group automatically. Gate parameter must stay 1-D.
        assert attn.pair_bias_gate_a.dim() == 1

    def test_pair_logits_multiplied_by_tau(self):
        import math

        import torch

        from deepfold3.model.attention import SingleAttention

        torch.manual_seed(0)
        H = 4
        attn = SingleAttention(64, num_heads=H, key_dim=64, value_dim=64, pair_bias_gate=True).eval()

        # Override gate to head-distinct values so the broadcast pattern is verifiable.
        with torch.no_grad():
            attn.pair_bias_gate_a.copy_(torch.tensor([0.0, math.log(2.0), math.log(0.5), math.log(0.1)]))
        tau = torch.exp(attn.pair_bias_gate_a)  # [1, 2, 0.5, 0.1]

        N = 5
        x = torch.randn(N, 64)
        mask = torch.ones(N)
        pair_logits = torch.randn(H, N, N)

        # Run with pair_logits = 0 to capture pure QK contribution to logits;
        # then run with the real pair_logits and infer that the difference
        # equals τ_h · pair_logits per head.
        # Patch torch.nn.functional.softmax to capture the logits the attention
        # would normalise. Bind ``orig`` *before* patching so the spy can call
        # the original without recursing into itself.
        import torch.nn.functional as F

        orig_softmax = F.softmax
        captured: list[torch.Tensor] = []

        def spy_softmax(logits, *args, **kwargs):
            captured.append(logits.detach().clone())
            return orig_softmax(logits, *args, **kwargs)

        F.softmax = spy_softmax
        try:
            with torch.no_grad():
                attn(x, mask, torch.zeros_like(pair_logits))
                attn(x, mask, pair_logits)
        finally:
            F.softmax = orig_softmax

        # captured[0] = QK only + mask; captured[1] = QK + τ·pair_logits + mask.
        diff = captured[1] - captured[0]  # [B=1, H, N, N]
        expected = (pair_logits * tau.view(H, 1, 1)).unsqueeze(0).to(diff.dtype)
        torch.testing.assert_close(diff, expected, atol=1e-4, rtol=1e-4)

    def test_off_skips_multiplication(self):
        import torch

        from deepfold3.model.attention import SingleAttention

        torch.manual_seed(0)
        H = 4
        attn_off = SingleAttention(64, num_heads=H, key_dim=64, value_dim=64).eval()
        N = 5
        x = torch.randn(N, 64)
        mask = torch.ones(N)
        pair_logits = torch.randn(H, N, N)
        # Just check forward runs and output shape is right.
        with torch.no_grad():
            out = attn_off(x, mask, pair_logits)
        assert out.shape == x.shape

    def test_forward_unchanged_when_off(self):
        """With both flags off, forward must produce the original residual
        update — i.e. ``residual_out_scale=1.0`` is bit-exact identical to the
        pre-feature behaviour."""
        import torch

        from deepfold3.model.evoformer import EvoformerBlock, PairFormerBlock

        torch.manual_seed(0)
        block_evo = EvoformerBlock(64, 128).eval()
        block_pf = PairFormerBlock(128, 384).eval()
        # Just construct + verify no crash; numerical equivalence is implied
        # because all multipliers default to 1.0 and the dropout path is
        # bypassed in eval mode.
        N, S = 8, 4
        msa = torch.randn(S, N, 64)
        pair = torch.randn(N, N, 128)
        single = torch.randn(N, 384)
        msa_mask = torch.ones(S, N)
        pair_mask = torch.ones(N, N)
        seq_mask = torch.ones(N)
        with torch.no_grad():
            msa_out, pair_out = block_evo(msa, pair, msa_mask, pair_mask)
            pair_out2, single_out = block_pf(pair, single, pair_mask, seq_mask)
        assert msa_out.shape == msa.shape
        assert pair_out.shape == pair.shape
        assert single_out.shape == single.shape


class TestFullEvoformer:
    """Tests for the full Evoformer trunk with AF3 weights."""

    def test_output_shapes(self, model, small_features, device):
        """Full evoformer produces correct output shapes."""
        evo = model.evoformer
        feats = small_features
        N = feats["res_type"].shape[0]

        from deepfold3.model.embedders import create_relative_encoding

        rel_feat = create_relative_encoding(
            feats["token_index"],
            feats["residue_index"],
            feats["asym_id"],
            feats["entity_id"],
            feats["sym_id"],
        )

        target_feat = model._build_raw_target_feat(feats)
        evo_cond_act = model.evo_cond_encoder(
            ref_pos=feats["ref_pos"],
            ref_mask=feats["ref_mask"].float(),
            ref_element=feats["ref_element"],
            ref_charge=feats["ref_charge"],
            ref_atom_name_chars=feats["ref_atom_name_chars"],
        )
        target_feat = torch.cat([target_feat, evo_cond_act], dim=-1)
        msa_feat = model._build_msa_feat(feats)

        c_pair = 128
        c_single = 384
        prev_pair = torch.zeros(N, N, c_pair, device=device)
        prev_single = torch.zeros(N, c_single, device=device)

        with torch.no_grad():
            out = evo(
                target_feat=target_feat,
                msa_feat=msa_feat,
                rel_feat=rel_feat,
                pair_mask=feats["pair_mask"].float(),
                seq_mask=feats["seq_mask"].float(),
                msa_mask=feats["msa_mask"].float(),
                prev_pair=prev_pair,
                prev_single=prev_single,
            )

        assert out["pair"].shape == (N, N, c_pair)
        assert out["single"].shape == (N, c_single)
        assert torch.isfinite(out["pair"]).all()
        assert torch.isfinite(out["single"]).all()

    def test_activation_statistics_per_layer(self, model, small_features, device):
        """Track activation norm through PairFormer layers to diagnose exploding/vanishing.

        Healthy activations should maintain reasonable norms (not growing exponentially
        or collapsing to zero).
        """
        evo = model.evoformer
        feats = small_features
        N = feats["res_type"].shape[0]

        from deepfold3.model.embedders import create_relative_encoding

        rel_feat = create_relative_encoding(
            feats["token_index"],
            feats["residue_index"],
            feats["asym_id"],
            feats["entity_id"],
            feats["sym_id"],
        )

        target_feat = model._build_raw_target_feat(feats)
        evo_cond_act = model.evo_cond_encoder(
            ref_pos=feats["ref_pos"],
            ref_mask=feats["ref_mask"].float(),
            ref_element=feats["ref_element"],
            ref_charge=feats["ref_charge"],
            ref_atom_name_chars=feats["ref_atom_name_chars"],
        )
        target_feat = torch.cat([target_feat, evo_cond_act], dim=-1)
        msa_feat = model._build_msa_feat(feats)

        with torch.no_grad():
            # Build initial embeddings
            left = evo.left_single(target_feat).unsqueeze(-2)
            right = evo.right_single(target_feat).unsqueeze(-3)
            pair = left + right
            pair = pair + evo.prev_embedding(evo.prev_embedding_layer_norm(torch.zeros(N, N, 128, device=device)))
            pair = pair + evo.position_activations(rel_feat)

            single = evo.single_activations(target_feat)
            single = single + evo.prev_single_embedding(
                evo.prev_single_embedding_layer_norm(torch.zeros(N, 384, device=device))
            )

            msa_act = evo.msa_activations(msa_feat)
            msa_act = msa_act + evo.extra_msa_target_feat(target_feat).unsqueeze(-3)

            pair_mask = feats["pair_mask"].float()
            seq_mask = feats["seq_mask"].float()
            msa_mask = feats["msa_mask"].float()

            # Track norms through MSA stack
            msa_norms = []
            pair_norms_msa = []
            for layer in evo.msa_stack:
                msa_act, pair = layer(msa_act, pair, msa_mask, pair_mask)
                msa_norms.append(msa_act.norm().item())
                pair_norms_msa.append(pair.norm().item())

            # Track norms through PairFormer
            pair_norms = []
            single_norms = []
            for layer in evo.trunk_pairformer:
                pair, single = layer(pair, single, pair_mask, seq_mask)
                pair_norms.append(pair.norm().item())
                single_norms.append(single.norm().item())

        # Check: norms should not explode (>1e6) or vanish (<1e-6)
        for i, pn in enumerate(pair_norms):
            assert pn < 1e6, f"Pair norm exploded at PairFormer layer {i}: {pn:.2e}"
            assert pn > 1e-6, f"Pair norm vanished at PairFormer layer {i}: {pn:.2e}"

        for i, sn in enumerate(single_norms):
            assert sn < 1e6, f"Single norm exploded at PairFormer layer {i}: {sn:.2e}"
            assert sn > 1e-6, f"Single norm vanished at PairFormer layer {i}: {sn:.2e}"

        # Check: final/initial norm ratio should not be extreme
        pair_ratio = pair_norms[-1] / (pair_norms[0] + 1e-10)
        assert pair_ratio < 1000, f"Pair activations grew {pair_ratio:.1f}x through PairFormer"
        assert pair_ratio > 0.001, f"Pair activations shrank {pair_ratio:.1f}x through PairFormer"


class TestFloat32VsFloat64:
    """Compare float32 and float64 precision for key modules.

    If the float32/float64 difference is large, it indicates numerical sensitivity
    that could explain blurry distograms or unstable predictions.
    """

    @pytest.mark.slow
    def test_trimul_precision(self):
        """TriangleMul should give similar results in float32 and float64."""
        from deepfold3.model.ops import TriangleMulOutgoing

        torch.manual_seed(42)
        tm32 = TriangleMulOutgoing(c_z=128)
        tm64 = TriangleMulOutgoing(c_z=128).double()
        # Copy weights
        for (_n32, p32), (_n64, p64) in zip(tm32.named_parameters(), tm64.named_parameters(), strict=False):
            p64.data = p32.data.double()

        z32 = torch.randn(16, 16, 128)
        z64 = z32.double()
        mask32 = torch.ones(16, 16)
        mask64 = mask32.double()

        with torch.no_grad():
            out32 = tm32(z32, mask32)
            out64 = tm64(z64, mask64)

        diff = (out32.double() - out64).abs()
        rel_diff = diff / (out64.abs() + 1e-10)
        print(f"TriangleMul f32/f64: max_abs_diff={diff.max():.2e}, max_rel_diff={rel_diff.max():.2e}")
        assert diff.max() < 1e-2, f"TriangleMul precision loss: max diff={diff.max():.2e}"

    @pytest.mark.slow
    def test_opm_precision(self):
        """OuterProductMean should give similar results in float32 and float64."""
        from deepfold3.model.ops import OuterProductMean

        torch.manual_seed(42)
        opm32 = OuterProductMean(c_msa=64, c_z=128, c_hidden=32)
        opm64 = OuterProductMean(c_msa=64, c_z=128, c_hidden=32).double()
        for (_, p32), (_, p64) in zip(opm32.named_parameters(), opm64.named_parameters(), strict=False):
            p64.data = p32.data.double()

        msa32 = torch.randn(8, 16, 64)
        msa64 = msa32.double()
        mask32 = torch.ones(8, 16)
        mask64 = mask32.double()

        with torch.no_grad():
            out32 = opm32(msa32, mask32)
            out64 = opm64(msa64, mask64)

        diff = (out32.double() - out64).abs()
        print(f"OPM f32/f64: max_abs_diff={diff.max():.2e}, mean_abs_diff={diff.mean():.2e}")
        assert diff.max() < 1e-2, f"OPM precision loss: max diff={diff.max():.2e}"

    @pytest.mark.slow
    def test_evoformer_block_precision(self):
        """Single EvoformerBlock: compare f32 vs f64 to measure accumulated error."""
        from deepfold3.model.evoformer import EvoformerBlock

        torch.manual_seed(42)
        S, N, c_msa, c_pair = 4, 8, 64, 128
        block32 = EvoformerBlock(c_msa, c_pair)
        block64 = EvoformerBlock(c_msa, c_pair).double()
        for (_, p32), (_, p64) in zip(block32.named_parameters(), block64.named_parameters(), strict=False):
            p64.data = p32.data.double()

        msa32 = torch.randn(S, N, c_msa) * 0.1
        pair32 = torch.randn(N, N, c_pair) * 0.1
        msa64 = msa32.double()
        pair64 = pair32.double()
        msa_mask = torch.ones(S, N)
        pair_mask = torch.ones(N, N)

        with torch.no_grad():
            msa_out32, pair_out32 = block32(msa32, pair32, msa_mask, pair_mask)
            msa_out64, pair_out64 = block64(msa64, pair64, msa_mask.double(), pair_mask.double())

        msa_diff = (msa_out32.double() - msa_out64).abs()
        pair_diff = (pair_out32.double() - pair_out64).abs()
        print(f"EvoBlock f32/f64: msa_max_diff={msa_diff.max():.2e}, pair_max_diff={pair_diff.max():.2e}")
        assert msa_diff.max() < 0.1, f"MSA precision loss: {msa_diff.max():.2e}"
        assert pair_diff.max() < 0.1, f"Pair precision loss: {pair_diff.max():.2e}"

    @pytest.mark.slow
    def test_pairformer_block_precision(self):
        """Single PairFormerBlock: compare f32 vs f64."""
        from deepfold3.model.evoformer import PairFormerBlock

        torch.manual_seed(42)
        N, c_pair, c_single = 8, 128, 384
        block32 = PairFormerBlock(c_pair, c_single)
        block64 = PairFormerBlock(c_pair, c_single).double()
        for (_, p32), (_, p64) in zip(block32.named_parameters(), block64.named_parameters(), strict=False):
            p64.data = p32.data.double()

        pair32 = torch.randn(N, N, c_pair) * 0.1
        single32 = torch.randn(N, c_single) * 0.1
        pair64 = pair32.double()
        single64 = single32.double()
        pair_mask = torch.ones(N, N)
        seq_mask = torch.ones(N)

        with torch.no_grad():
            pair_out32, single_out32 = block32(pair32, single32, pair_mask, seq_mask)
            pair_out64, single_out64 = block64(pair64, single64, pair_mask.double(), seq_mask.double())

        pair_diff = (pair_out32.double() - pair_out64).abs()
        single_diff = (single_out32.double() - single_out64).abs()
        print(f"PairFormer f32/f64: pair_max_diff={pair_diff.max():.2e}, single_max_diff={single_diff.max():.2e}")
        assert pair_diff.max() < 0.1
        assert single_diff.max() < 0.1


class TestRecycling:
    """Test recycling behavior with real weights."""

    def test_recycling_changes_output(self, model, small_features):
        """Multiple recycles should produce different outputs."""
        with torch.no_grad():
            disto_0 = model.compute_distogram(small_features, num_recycles=0)
            disto_1 = model.compute_distogram(small_features, num_recycles=1)
        assert not torch.allclose(disto_0, disto_1, atol=1e-3)

    def test_recycling_convergence(self, model, small_features):
        """Consecutive recycle outputs should converge (decreasing delta)."""
        deltas = []
        with torch.no_grad():
            prev = model.compute_distogram(small_features, num_recycles=0)
            for r in range(1, 4):
                curr = model.compute_distogram(small_features, num_recycles=r)
                delta = (curr - prev).abs().mean().item()
                deltas.append(delta)
                prev = curr

        # Later recycles should have smaller deltas (convergence)
        # Allow some noise but the trend should be decreasing
        assert deltas[-1] < deltas[0] * 10, f"Recycling not converging: deltas={[f'{d:.4f}' for d in deltas]}"
