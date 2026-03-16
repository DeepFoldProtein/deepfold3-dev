"""Tests for input feature embeddings and construction.

Verifies relative encoding, target_feat, and MSA feat match AF3 specifications.
"""

from __future__ import annotations

import math

import torch


class TestRelativeEncoding:
    """Verify create_relative_encoding produces correct features."""

    def test_output_shape(self):
        from deepfold3.model.embedders import create_relative_encoding

        N = 10
        rel = create_relative_encoding(
            token_index=torch.arange(N),
            residue_index=torch.arange(N),
            asym_id=torch.zeros(N, dtype=torch.long),
            entity_id=torch.ones(N, dtype=torch.long),
            sym_id=torch.ones(N, dtype=torch.long),
        )
        # 66 (rel_pos) + 66 (rel_token) + 1 (entity_same) + 6 (rel_chain) = 139
        assert rel.shape == (N, N, 139)

    def test_diagonal_properties(self):
        """On diagonal (i=j): rel_pos=0, rel_token=0, entity_same=1."""
        from deepfold3.model.embedders import create_relative_encoding

        N = 8
        rel = create_relative_encoding(
            token_index=torch.arange(N),
            residue_index=torch.arange(N),
            asym_id=torch.zeros(N, dtype=torch.long),
            entity_id=torch.ones(N, dtype=torch.long),
            sym_id=torch.ones(N, dtype=torch.long),
        )
        # rel_pos one-hot: position 32 (offset=0 + max_rel_idx=32)
        diag = rel[torch.arange(N), torch.arange(N)]
        # Should be one-hot at position 32 in first 66 dims
        rel_pos_diag = diag[:, :66]
        assert (rel_pos_diag.argmax(dim=-1) == 32).all()

        # entity_same should be 1 on diagonal (same chain)
        entity_same = diag[:, 132]  # index 66+66=132
        assert (entity_same == 1.0).all()

    def test_cross_chain(self):
        """Cross-chain pairs should use the extra bin for rel_pos."""
        from deepfold3.model.embedders import create_relative_encoding

        N = 6
        asym_id = torch.tensor([0, 0, 0, 1, 1, 1])
        rel = create_relative_encoding(
            token_index=torch.arange(N),
            residue_index=torch.arange(N),
            asym_id=asym_id,
            entity_id=asym_id + 1,
            sym_id=torch.ones(N, dtype=torch.long),
        )
        # Cross-chain pairs (0,3), (1,4), etc. should use extra bin (index 65)
        cross_pair_rel_pos = rel[0, 3, :66]
        assert cross_pair_rel_pos.argmax() == 65  # extra bin

    def test_one_hot_normalization(self):
        """Each one-hot slice should sum to 1."""
        from deepfold3.model.embedders import create_relative_encoding

        N = 8
        rel = create_relative_encoding(
            token_index=torch.arange(N),
            residue_index=torch.arange(N),
            asym_id=torch.zeros(N, dtype=torch.long),
            entity_id=torch.ones(N, dtype=torch.long),
            sym_id=torch.ones(N, dtype=torch.long),
        )
        # rel_pos [0:66] and rel_token [66:132] should each sum to 1
        for i in range(N):
            for j in range(N):
                assert abs(rel[i, j, :66].sum().item() - 1.0) < 1e-5
                assert abs(rel[i, j, 66:132].sum().item() - 1.0) < 1e-5
                assert abs(rel[i, j, 133:139].sum().item() - 1.0) < 1e-5

    def test_batch_support(self):
        """Should work with batched [B, N] inputs."""
        from deepfold3.model.embedders import create_relative_encoding

        B, N = 2, 5
        rel = create_relative_encoding(
            token_index=torch.arange(N).expand(B, -1),
            residue_index=torch.arange(N).expand(B, -1),
            asym_id=torch.zeros(B, N, dtype=torch.long),
            entity_id=torch.ones(B, N, dtype=torch.long),
            sym_id=torch.ones(B, N, dtype=torch.long),
        )
        assert rel.shape == (B, N, N, 139)


class TestTargetFeat:
    """Verify raw target_feat construction [N, 63]."""

    def test_output_shape(self, model, small_features):
        with torch.no_grad():
            raw = model._build_raw_target_feat(small_features)
        N = small_features["res_type"].shape[0]
        assert raw.shape == (N, 63)  # aatype(31) + profile(31) + deletion_mean(1)

    def test_aatype_one_hot(self, model, small_features):
        """First 31 dims should be one-hot encoding of res_type."""
        with torch.no_grad():
            raw = model._build_raw_target_feat(small_features)
        aatype_1hot = raw.detach().cpu()[:, :31]
        # Each row should sum to 1 (one-hot)
        torch.testing.assert_close(aatype_1hot.sum(dim=-1), torch.ones(aatype_1hot.shape[0]))
        # Argmax should match res_type
        assert (aatype_1hot.argmax(dim=-1) == small_features["res_type"].detach().cpu()).all()

    def test_profile_dims(self, model, small_features):
        """Profile should be 31-dim."""
        with torch.no_grad():
            raw = model._build_raw_target_feat(small_features)
        profile = raw[:, 31:62]
        assert profile.shape[-1] == 31

    def test_deletion_mean_dim(self, model, small_features):
        """Last dim should be deletion_mean."""
        with torch.no_grad():
            raw = model._build_raw_target_feat(small_features)
        del_mean = raw[:, 62]
        assert del_mean.shape == (small_features["res_type"].shape[0],)


class TestMSAFeat:
    """Verify MSA feature construction [S, N, 34]."""

    def test_output_shape(self, model, small_features):
        with torch.no_grad():
            msa_feat = model._build_msa_feat(small_features)
        S, N = small_features["msa"].shape
        assert msa_feat.shape == (S, N, 34)  # one_hot(32) + has_del(1) + del_val(1)

    def test_one_hot_component(self, model, small_features):
        """First 32 dims should be one-hot encoding of MSA."""
        with torch.no_grad():
            msa_feat = model._build_msa_feat(small_features)
        one_hot = msa_feat[:, :, :32]
        # Each position should sum to 1 (one-hot)
        sums = one_hot.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums))

    def test_deletion_features(self, model):
        """Deletion features: has_del=clip(del, 0, 1), del_val=arctan(del/3)*2/pi."""
        S, N = 4, 8
        features = {
            "msa": torch.randint(0, 22, (S, N)),
            "deletion_matrix": torch.tensor([[0, 1, 3, 5, 0, 0, 0, 0]] * S, dtype=torch.long),
        }
        with torch.no_grad():
            msa_feat = model._build_msa_feat(features)

        has_del = msa_feat[:, :, 32]  # clamp(del, 0, 1)
        del_val = msa_feat[:, :, 33]  # arctan(del/3) * 2/pi

        # Check has_deletion: 0 → 0, 1+ → 1
        expected_has_del = features["deletion_matrix"].float().clamp(0, 1)
        torch.testing.assert_close(has_del, expected_has_del, atol=1e-5, rtol=1e-5)

        # Check deletion_value: arctan(del/3) * 2/pi
        del_float = features["deletion_matrix"].float()
        expected_del_val = torch.arctan(del_float / 3.0) * (2.0 / math.pi)
        torch.testing.assert_close(del_val, expected_del_val, atol=1e-5, rtol=1e-5)
