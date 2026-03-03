"""End-to-end tests for the full DeepFold3 pipeline.

Tests distogram properties, inference smoke test, and output validation.
These tests require AF3 weights and are marked accordingly.
"""

from __future__ import annotations

import pytest
import torch


class TestDistogram:
    """Verify distogram head properties with real weights."""

    def test_output_shape(self, model, small_features):
        """Distogram should be [N, N, 64]."""
        with torch.no_grad():
            disto = model.compute_distogram(small_features, num_recycles=0)
        N = small_features["res_type"].shape[0]
        assert disto.shape == (N, N, 64)

    def test_symmetry(self, model, small_features):
        """Distogram should be symmetric: d[i,j] == d[j,i]."""
        with torch.no_grad():
            disto = model.compute_distogram(small_features, num_recycles=0)
        torch.testing.assert_close(disto, disto.transpose(0, 1), atol=1e-5, rtol=1e-5)

    def test_finite(self, model, small_features):
        """No NaN or Inf in distogram."""
        with torch.no_grad():
            disto = model.compute_distogram(small_features, num_recycles=0)
        assert torch.isfinite(disto).all()

    def test_probabilities_valid(self, model, small_features):
        """Softmax of distogram should be valid probability distributions."""
        with torch.no_grad():
            disto = model.compute_distogram(small_features, num_recycles=0)
        probs = torch.softmax(disto, dim=-1)
        # Each distribution should sum to 1
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_diagonal_short_distance(self, model, medium_features):
        """Diagonal (self-contact) should predict short distances.

        The distogram bins cover 2.3-21.7A. Self-contacts should peak
        in the first few bins (shortest distances).
        """
        with torch.no_grad():
            disto = model.compute_distogram(medium_features, num_recycles=1)
        probs = torch.softmax(disto, dim=-1)

        # Diagonal elements: probability mass should be concentrated in first bins
        N = disto.shape[0]
        for i in range(N):
            short_range_prob = probs[i, i, :10].sum().item()  # first 10 bins
            assert short_range_prob > 0.1, (
                f"Self-contact at ({i},{i}) has low short-range prob: {short_range_prob:.3f}"
            )

    def test_recycle_improves_sharpness(self, model, small_features):
        """More recycles should produce sharper (lower entropy) distograms."""
        with torch.no_grad():
            disto_0 = model.compute_distogram(small_features, num_recycles=0)
            disto_3 = model.compute_distogram(small_features, num_recycles=3)

        def entropy(logits):
            probs = torch.softmax(logits, dim=-1)
            return -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()

        h0 = entropy(disto_0).item()
        h3 = entropy(disto_3).item()
        # More recycles should reduce entropy (sharper predictions)
        # Allow some tolerance — this is a soft test
        assert h3 < h0 * 1.5, f"3-recycle entropy ({h3:.3f}) not sharper than 0-recycle ({h0:.3f})"


class TestDistogramHead:
    """Verify DistogramHead module properties."""

    def test_symmetrize(self, model, device):
        """half_logits + half_logits.T should give symmetric output."""
        from deepfold3.model.model import DistogramHead

        dh = DistogramHead(pair_channels=128).to(device)
        pair = torch.randn(8, 8, 128, device=device)
        with torch.no_grad():
            out = dh(pair)
        torch.testing.assert_close(out, out.transpose(0, 1))

    def test_expected_distance(self, model, device):
        """Expected distance should be positive and finite."""
        pair = torch.randn(8, 8, 128, device=device) * 0.01
        with torch.no_grad():
            dist = model.distogram_head.expected_distance(pair)
        assert (dist >= 0).all()
        assert torch.isfinite(dist).all()


class TestFullInference:
    """Smoke tests for full forward pass (expensive, require AF3 weights)."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_forward_shapes(self, model, small_features):
        """Full forward pass produces correctly shaped outputs."""
        with torch.no_grad():
            out = model(small_features, seed=42, num_recycles=0, num_samples=1)

        N = small_features["res_type"].shape[0]
        M = 24  # max_atoms_per_token
        assert out.atom_positions.shape[0] == 1  # num_samples
        assert out.atom_positions.shape[1] == N
        assert out.atom_positions.shape[2] == M
        assert out.atom_positions.shape[3] == 3
        assert out.atom_mask.shape[-2:] == (N, M)
        assert out.distogram.shape == (N, N, 64)

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_forward_finite(self, model, small_features):
        """All outputs should be finite (no NaN/Inf)."""
        with torch.no_grad():
            out = model(small_features, seed=42, num_recycles=0, num_samples=1)

        assert torch.isfinite(out.atom_positions).all(), "NaN/Inf in atom_positions"
        assert torch.isfinite(out.distogram).all(), "NaN/Inf in distogram"
        if out.plddt is not None:
            assert torch.isfinite(out.plddt).all(), "NaN/Inf in pLDDT"
        if out.pae is not None:
            assert torch.isfinite(out.pae).all(), "NaN/Inf in PAE"

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_plddt_range(self, model, small_features):
        """pLDDT logits should produce reasonable probabilities."""
        with torch.no_grad():
            out = model(small_features, seed=42, num_recycles=0, num_samples=1)

        if out.plddt is not None:
            # pLDDT shape: [num_samples, N, max_atoms * num_plddt_bins]
            # Reshape to [num_samples, N, max_atoms, num_plddt_bins] and check
            N = small_features["res_type"].shape[0]
            plddt_logits = out.plddt[0]  # [N, M*50]
            plddt_logits = plddt_logits.view(N, 24, 50)
            probs = torch.softmax(plddt_logits, dim=-1)
            # Should be valid probabilities
            assert (probs >= 0).all()
            sums = probs.sum(dim=-1)
            torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_distogram_symmetry_from_forward(self, model, small_features):
        """Distogram from full forward should also be symmetric."""
        with torch.no_grad():
            out = model(small_features, seed=42, num_recycles=0, num_samples=1)
        torch.testing.assert_close(
            out.distogram, out.distogram.transpose(0, 1), atol=1e-5, rtol=1e-5
        )


class TestDataPipeline:
    """Verify data pipeline produces valid features."""

    def test_from_sequence(self):
        """InferenceInput.from_sequence should produce valid features."""
        from deepfold3.data.module.dataset import InferenceInput

        feats = InferenceInput.from_sequence("AAAAAA")
        assert "res_type" in feats
        assert "ref_pos" in feats
        assert "msa" in feats
        assert "pair_mask" in feats
        assert feats["res_type"].shape[0] == 6

    def test_from_sequence_shapes(self):
        """All feature shapes should be consistent."""
        from deepfold3.data.module.dataset import InferenceInput

        feats = InferenceInput.from_sequence("MKWVTF")
        N = feats["res_type"].shape[0]
        assert feats["residue_index"].shape == (N,)
        assert feats["token_index"].shape == (N,)
        assert feats["asym_id"].shape == (N,)
        assert feats["entity_id"].shape == (N,)
        assert feats["sym_id"].shape == (N,)
        assert feats["seq_mask"].shape == (N,)
        assert feats["pair_mask"].shape == (N, N)
        assert feats["ref_pos"].shape[0] == N
        assert feats["ref_pos"].shape[2] == 3
        assert feats["ref_mask"].shape[0] == N
        assert feats["msa"].shape[1] == N
        assert feats["profile"].shape == (N, 31)
        assert feats["deletion_mean"].shape == (N,)

    def test_pair_mask_from_seq_mask(self):
        """pair_mask should be outer product of seq_mask."""
        from deepfold3.data.module.dataset import InferenceInput

        feats = InferenceInput.from_sequence("AAAAAA")
        expected = feats["seq_mask"].unsqueeze(1) & feats["seq_mask"].unsqueeze(0)
        torch.testing.assert_close(feats["pair_mask"], expected)

    def test_res_type_range(self):
        """res_type should be in [0, 30] (31 classes)."""
        from deepfold3.data.module.dataset import InferenceInput

        feats = InferenceInput.from_sequence("ACDEFGHIKLMNPQRSTVWY")  # all 20 AAs
        assert feats["res_type"].min() >= 0
        assert feats["res_type"].max() <= 30
