"""Tests for batch dimension support.

Verifies that collated batches flow through the model correctly,
producing outputs with the expected batch dimension.
"""

from __future__ import annotations

import torch

from deepfold3.data.module.collate import collate_fn
from deepfold3.model.config import (
    DiffusionConfig,
    DiffusionTransformerConfig,
    EvoformerConfig,
    ModelConfig,
    PairFormerConfig,
)
from deepfold3.model.model import DeepFold3


def _tiny_config() -> ModelConfig:
    """Minimal model config for fast batch testing.

    Keeps full channel sizes (384/128) since atom cross-attention hardcodes them,
    but uses minimal layers throughout.
    """
    return ModelConfig(
        evoformer=EvoformerConfig(num_layer=1),
        pairformer=PairFormerConfig(num_layer=1),
        diffusion=DiffusionConfig(
            steps=2,
            num_samples=1,
            transformer=DiffusionTransformerConfig(num_blocks=4, super_block_size=2),
        ),
        num_recycles=0,
        msa_crop_size=8,
    )


def _make_features(N: int, M: int = 24, S: int = 4, seed: int = 42) -> dict[str, torch.Tensor]:
    """Create a synthetic unbatched feature dict."""
    torch.manual_seed(seed)
    ref_mask = torch.zeros(N, M, dtype=torch.bool)
    ref_mask[:, :5] = True
    return {
        "res_type": torch.randint(0, 20, (N,)),
        "residue_index": torch.arange(N),
        "token_index": torch.arange(N),
        "asym_id": torch.zeros(N, dtype=torch.long),
        "entity_id": torch.ones(N, dtype=torch.long),
        "sym_id": torch.zeros(N, dtype=torch.long),
        "seq_mask": torch.ones(N, dtype=torch.bool),
        "pair_mask": torch.ones(N, N, dtype=torch.bool),
        "ref_pos": torch.randn(N, M, 3) * ref_mask.unsqueeze(-1).float(),
        "ref_mask": ref_mask,
        "ref_element": torch.randint(1, 8, (N, M)) * ref_mask.long(),
        "ref_charge": torch.zeros(N, M),
        "ref_atom_name_chars": torch.zeros(N, M, 4, dtype=torch.long),
        "msa": torch.randint(0, 22, (S, N)),
        "msa_mask": torch.ones(S, N, dtype=torch.bool),
        "deletion_matrix": torch.zeros(S, N, dtype=torch.long),
        "profile": torch.zeros(N, 31),
        "deletion_mean": torch.zeros(N),
    }


class TestCollation:
    """Test that collate_fn produces correct batch shapes."""

    def test_same_size(self):
        feats = [_make_features(8, seed=i) for i in range(3)]
        batch = collate_fn(feats)
        assert batch["res_type"].shape == (3, 8)
        assert batch["msa"].shape == (3, 4, 8)
        assert batch["ref_pos"].shape == (3, 8, 24, 3)
        assert batch["pair_mask"].shape == (3, 8, 8)

    def test_different_sizes_padded(self):
        feats = [_make_features(6, seed=0), _make_features(10, seed=1)]
        batch = collate_fn(feats)
        assert batch["res_type"].shape == (2, 10)
        assert batch["pair_mask"].shape == (2, 10, 10)
        assert batch["ref_pos"].shape == (2, 10, 24, 3)
        # First sample's padding should be zero
        assert batch["seq_mask"][0, 6:].sum() == 0


class TestBatchedEvoformer:
    """Test evoformer trunk with batched inputs."""

    def test_evoformer_batch_shapes(self):
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        batch = collate_fn([_make_features(8, seed=i) for i in range(2)])

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(batch, num_recycles=0)

        B, N = 2, 8
        assert embeddings["single"].shape == (B, N, cfg.evoformer.seq_channel)
        assert embeddings["pair"].shape == (B, N, N, cfg.evoformer.pair_channel)
        assert target_feat.shape == (B, N, 447)
        assert rel_feat.shape == (B, N, N, 139)
        assert seq_mask.shape == (B, N)

    def test_evoformer_padded_batch(self):
        """Batched evoformer with different-length sequences (padded)."""
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        batch = collate_fn([_make_features(6, seed=0), _make_features(10, seed=1)])
        N_max = 10

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(batch, num_recycles=0)

        assert embeddings["single"].shape == (2, N_max, cfg.evoformer.seq_channel)
        assert embeddings["pair"].shape == (2, N_max, N_max, cfg.evoformer.pair_channel)

    def test_evoformer_unbatched_unchanged(self):
        """Unbatched forward still works after batch support changes."""
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        feats = _make_features(8)

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(feats, num_recycles=0)

        N = 8
        assert embeddings["single"].shape == (N, cfg.evoformer.seq_channel)
        assert embeddings["pair"].shape == (N, N, cfg.evoformer.pair_channel)


class TestBatchedDenoise:
    """Test diffusion denoise with batched inputs."""

    def test_denoise_batch(self):
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        B, N, M = 2, 8, 24
        batch = collate_fn([_make_features(N, seed=i) for i in range(B)])

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(batch, num_recycles=0)

            # Create noisy positions
            noisy_pos = torch.randn(B, N, M, 3)
            noise_level = torch.tensor(1.0)

            denoised = model.diffusion_head.denoise(
                noisy_pos=noisy_pos,
                noise_level=noise_level,
                embeddings=embeddings,
                ref_pos=batch["ref_pos"],
                ref_mask=batch["ref_mask"].float(),
                ref_element=batch["ref_element"],
                ref_charge=batch["ref_charge"].float(),
                ref_atom_name_chars=batch["ref_atom_name_chars"],
                seq_mask=seq_mask,
                rel_feat=rel_feat,
            )

        assert denoised.shape == (B, N, M, 3)
        assert torch.isfinite(denoised).all()

    def test_denoise_per_sample_noise(self):
        """Test denoise with per-sample noise levels [B]."""
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        B, N, M = 2, 8, 24
        batch = collate_fn([_make_features(N, seed=i) for i in range(B)])

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(batch, num_recycles=0)

            noisy_pos = torch.randn(B, N, M, 3)
            noise_level = torch.tensor([1.0, 0.5])  # per-sample

            denoised = model.diffusion_head.denoise(
                noisy_pos=noisy_pos,
                noise_level=noise_level,
                embeddings=embeddings,
                ref_pos=batch["ref_pos"],
                ref_mask=batch["ref_mask"].float(),
                ref_element=batch["ref_element"],
                ref_charge=batch["ref_charge"].float(),
                ref_atom_name_chars=batch["ref_atom_name_chars"],
                seq_mask=seq_mask,
                rel_feat=rel_feat,
            )

        assert denoised.shape == (B, N, M, 3)
        assert torch.isfinite(denoised).all()

    def test_denoise_unbatched_unchanged(self):
        """Unbatched denoise still works."""
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        N, M = 8, 24
        feats = _make_features(N)

        with torch.no_grad():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(feats, num_recycles=0)

            noisy_pos = torch.randn(N, M, 3)
            noise_level = torch.tensor(1.0)

            denoised = model.diffusion_head.denoise(
                noisy_pos=noisy_pos,
                noise_level=noise_level,
                embeddings=embeddings,
                ref_pos=feats["ref_pos"],
                ref_mask=feats["ref_mask"].float(),
                ref_element=feats["ref_element"],
                ref_charge=feats["ref_charge"].float(),
                ref_atom_name_chars=feats["ref_atom_name_chars"],
                seq_mask=seq_mask,
                rel_feat=rel_feat,
            )

        assert denoised.shape == (N, M, 3)
        assert torch.isfinite(denoised).all()


class TestBatchedDistogram:
    """Test distogram head with batched inputs."""

    def test_distogram_batch(self):
        cfg = _tiny_config()
        model = DeepFold3(cfg)
        model.eval()

        B, N = 2, 8
        batch = collate_fn([_make_features(N, seed=i) for i in range(B)])

        with torch.no_grad():
            embeddings, *_ = model.run_evoformer(batch, num_recycles=0)
            disto = model.distogram_head(embeddings["pair"])

        assert disto.shape == (B, N, N, 64)
        # Should be symmetric
        torch.testing.assert_close(disto, disto.transpose(-3, -2), atol=1e-5, rtol=1e-5)
