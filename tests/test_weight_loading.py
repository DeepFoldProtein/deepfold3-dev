"""Tests for AF3 weight loading and conversion correctness.

Verifies that all weights are loaded, shapes match, and the conversion
from Haiku format (scope/name → numpy array) to PyTorch state_dict is correct.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestWeightLoadingIntegrity:
    """Verify AF3 parameters load correctly into DeepFold3."""

    def test_no_unexpected_keys(self, af3_state_dict):
        """All converted keys should match DeepFold3 module parameters."""
        from deepfold3.model.model import DeepFold3

        model = DeepFold3()
        _, unexpected = model.load_state_dict(af3_state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}"

    def test_missing_keys_are_expected(self, model, af3_state_dict):
        """List any missing keys — they should only be template/optional modules."""
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(af3_state_dict.keys())
        missing = model_keys - loaded_keys
        # These are acceptable missing keys (template embedding not implemented,
        # fourier_weight/fourier_bias come from safetensors not haiku_to_torch)
        for key in missing:
            assert "template" in key or key.startswith("_") or "fourier_" in key, f"Unexpected missing key: {key}"

    def test_weight_shapes_match(self, model, af3_state_dict):
        """All converted weight shapes must match model parameter shapes."""
        model_sd = model.state_dict()
        mismatches = []
        for key in af3_state_dict:
            if key in model_sd:
                if af3_state_dict[key].shape != model_sd[key].shape:
                    mismatches.append(f"{key}: converted={af3_state_dict[key].shape} vs model={model_sd[key].shape}")
        assert not mismatches, "Shape mismatches:\n" + "\n".join(mismatches)

    def test_all_weights_finite(self, model):
        """All loaded weights must be finite (no NaN/Inf)."""
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"

    def test_all_weights_float32(self, model):
        """All parameters should be float32."""
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, f"{name} dtype={param.dtype}"

    def test_linear_no_bias_modules(self, model):
        """All LinearNoBias modules should have no bias."""
        from deepfold3.model.primitives import LinearNoBias

        for name, module in model.named_modules():
            if isinstance(module, LinearNoBias):
                assert module.bias is None, f"{name} has unexpected bias"


class TestWeightConversion:
    """Verify Haiku → PyTorch weight conversion is numerically correct."""

    def test_regular_linear_conversion(self, af3_raw_params, model):
        """Regular Linear: Haiku [in, out] -> PyTorch [out, in] (transpose)."""
        scope = "diffuser/evoformer/left_single"
        if scope not in af3_raw_params:
            pytest.skip(f"Scope {scope} not found")
        w_haiku = af3_raw_params[scope]["weights"]  # [447, 128]
        w_torch = model.evoformer.left_single.weight.detach().cpu().float().numpy()  # [128, 447]

        # Haiku stores [in, out]; PyTorch stores [out, in]
        np.testing.assert_allclose(w_torch, w_haiku.reshape(w_haiku.shape[0], -1).T, rtol=1e-6, atol=1e-6)

    def test_layernorm_conversion(self, af3_raw_params, model):
        """LayerNorm: Haiku scale -> weight, offset -> bias."""
        scope = "diffuser/evoformer/prev_embedding_layer_norm"
        if scope not in af3_raw_params:
            pytest.skip(f"Scope {scope} not found")
        scale = af3_raw_params[scope]["scale"].flatten()
        offset = af3_raw_params[scope]["offset"].flatten()

        ln = model.evoformer.prev_embedding_layer_norm
        np.testing.assert_allclose(ln.weight.detach().cpu().float().numpy(), scale, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ln.bias.detach().cpu().float().numpy(), offset, rtol=1e-6, atol=1e-6)

    def test_opm_output_w_conversion(self, af3_raw_params, model):
        """OPM output_w: raw parameter (no Linear wrapper), shape [c_h, c_h, c_z]."""
        # First MSA stack layer's outer_product
        scope = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean"
        if scope not in af3_raw_params:
            pytest.skip(f"Scope {scope} not found")
        ow_haiku = af3_raw_params[scope].get("output_w")
        if ow_haiku is None:
            pytest.skip("output_w not found in scope")
        # Stacked: first dim = num_layers. Take layer 0.
        ow_haiku_0 = ow_haiku[0]  # [c_h, c_h, c_z]
        ow_torch = model.evoformer.msa_stack[0].outer_product.out_w.detach().cpu().float().numpy()
        np.testing.assert_allclose(ow_torch, ow_haiku_0, rtol=1e-5, atol=1e-5)

    def test_distogram_head_conversion(self, af3_raw_params, model):
        """Distogram head: single LinearNoBias(128, 64)."""
        scope = "diffuser/distogram_head/half_logits"
        if scope not in af3_raw_params:
            pytest.skip(f"Scope {scope} not found")
        w_haiku = af3_raw_params[scope]["weights"]  # [128, 64]
        w_torch = model.distogram_head.half_logits.weight.detach().cpu().float().numpy()  # [64, 128]
        np.testing.assert_allclose(w_torch, w_haiku.T, rtol=1e-6, atol=1e-6)

    def test_trimul_weight_shapes(self, model):
        """TriangleMul weights should have expected shapes."""
        tm = model.evoformer.trunk_pairformer[0].tri_mul_out
        c_z = 128
        assert tm.norm_left.weight.shape == (c_z,)
        assert tm.proj.weight.shape == (c_z * 2, c_z)
        assert tm.gate.weight.shape == (c_z * 2, c_z)
        assert tm.norm_center.weight.shape == (c_z,)
        assert tm.out_proj.weight.shape == (c_z, c_z)
        assert tm.out_gate.weight.shape == (c_z, c_z)

    def test_confidence_head_shapes(self, model):
        """Confidence head shapes match config."""
        ch = model.confidence_head
        assert ch.left_target_feat_project.weight.shape[1] == 447  # target_feat_dim
        assert ch.right_target_feat_project.weight.shape[1] == 447
        assert ch.distogram_feat_project.weight.shape[1] == 39  # num_bins
        assert ch.plddt_logits.weight.shape[0] == 24 * 50  # max_atoms * num_plddt_bins

    def test_evoformer_stack_sizes(self, model):
        """Evoformer has correct number of layers."""
        assert len(model.evoformer.msa_stack) == 4  # MSA stack
        assert len(model.evoformer.trunk_pairformer) == 48  # PairFormer trunk
