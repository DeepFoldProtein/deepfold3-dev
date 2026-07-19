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

    def test_bucket_pad_rounds_to_next_bucket(self):
        """With bucket_pad=True the N dim is rounded up to the next bucket size."""
        from deepfold3.model.bucket import next_bucket

        # N=100 → next bucket is 128
        feats = [_make_features(100, seed=0), _make_features(50, seed=1)]
        batch = collate_fn(feats, bucket_pad=True)
        expected_n = next_bucket(100)  # 128
        assert expected_n == 128
        assert batch["res_type"].shape == (2, expected_n)
        assert batch["pair_mask"].shape == (2, expected_n, expected_n)
        assert batch["ref_pos"].shape == (2, expected_n, 24, 3)
        assert batch["msa"].shape == (2, 4, expected_n)
        # All masks must be 0 in the padded slots — both for the smaller
        # sample (50→128) and the larger (100→128).
        assert batch["seq_mask"][0, 100:].sum() == 0
        assert batch["seq_mask"][1, 50:].sum() == 0
        assert batch["pair_mask"][0, 100:, :].sum() == 0
        assert batch["pair_mask"][0, :, 100:].sum() == 0
        assert batch["pair_mask"][1, 50:, :].sum() == 0
        assert batch["pair_mask"][1, :, 50:].sum() == 0
        assert batch["msa_mask"][0, :, 100:].sum() == 0
        assert batch["msa_mask"][1, :, 50:].sum() == 0
        # ref_mask is dim 0 = N, dim 1 = M (atoms per token)
        assert batch["ref_mask"][0, 100:, :].sum() == 0
        assert batch["ref_mask"][1, 50:, :].sum() == 0

    def test_bucket_pad_mask_loss_weight_kept_in_lockstep(self):
        """``mask_loss_weight`` (per-token) must be padded along N so it
        stays aligned with ``atom_gt_mask`` after bucketing. Regression for
        the diffusion-loss shape mismatch in df3-20260429-skip5k-v2."""
        from deepfold3.model.bucket import next_bucket

        feat_a = _make_features(192, seed=0)
        feat_a["atom_gt_positions"] = torch.zeros(192, 24, 3)
        feat_a["atom_gt_mask"] = torch.ones(192, 24, dtype=torch.bool)
        feat_a["mask_loss_weight"] = torch.ones(192, dtype=torch.float32)

        feat_b = _make_features(50, seed=1)
        feat_b["atom_gt_positions"] = torch.zeros(50, 24, 3)
        feat_b["atom_gt_mask"] = torch.ones(50, 24, dtype=torch.bool)
        feat_b["mask_loss_weight"] = torch.ones(50, dtype=torch.float32)

        batch = collate_fn([feat_a, feat_b], bucket_pad=True)
        expected_n = next_bucket(192)
        assert batch["atom_gt_mask"].shape[1] == expected_n
        assert batch["mask_loss_weight"].shape == (2, expected_n)
        # Padded slots are 0 (excluded from loss — matches seq_mask=0 there).
        assert batch["mask_loss_weight"][0, 192:].sum() == 0
        assert batch["mask_loss_weight"][1, 50:].sum() == 0
        # Real slots are unchanged.
        assert batch["mask_loss_weight"][0, :192].min() == 1.0
        assert batch["mask_loss_weight"][1, :50].min() == 1.0

    def test_bucket_pad_token_to_center_atom_kept_in_lockstep(self):
        """``token_to_center_atom`` (per-token rep-atom slot for pLDDT) must be
        padded along N so it stays aligned with ``mol_type`` / ``atom_gt_mask``
        after bucketing. Regression for the FT3 plddt_loss shape mismatch
        (``mol_type`` [384] vs ``atom_mask`` [210]) — the feature was emitted by
        the featurizer but missing from ``DIM0_KEYS``, so it passed through
        unpadded while every other token feature was padded to the bucket."""
        from deepfold3.model.bucket import next_bucket

        feat = _make_features(200, seed=0)
        feat["mol_type"] = torch.zeros(200, dtype=torch.long)
        feat["token_to_rep_atom"] = torch.ones(200, dtype=torch.long)
        feat["token_to_center_atom"] = torch.ones(200, dtype=torch.long)

        batch = collate_fn([feat], bucket_pad=True)
        expected_n = next_bucket(200)
        assert batch["token_to_center_atom"].shape == (1, expected_n)
        # Stays lockstep with the other per-token features.
        assert batch["token_to_center_atom"].shape[1] == batch["mol_type"].shape[1]
        assert batch["token_to_center_atom"].shape[1] == batch["token_to_rep_atom"].shape[1]
        # Real slots unchanged; padded slots 0 (masked out via seq_mask=0 there).
        assert batch["token_to_center_atom"][0, :200].min() == 1
        assert batch["token_to_center_atom"][0, 200:].sum() == 0

    def test_every_featurizer_key_registered_in_bucket(self):
        """Every per-example feature the featurizer emits must be registered in
        one of ``bucket.py``'s pad-lists (DIM0/DIM01/DIM1/passthrough). An
        unregistered key silently passes through ``pad_features`` unpadded,
        desyncing its N dim from the bucketed features — the root cause of the
        FT3 ``token_to_center_atom`` crash. Static guard against the next one."""
        import re
        from pathlib import Path

        import deepfold3.model.bucket as bucket

        src = Path(bucket.__file__).parent.parent / "data" / "feature" / "featurizer.py"
        emitted = set(re.findall(r'"([a-z_0-9]+)":\s*torch\.', src.read_text()))
        registered = bucket.DIM0_KEYS | bucket.DIM01_KEYS | bucket.DIM1_KEYS | bucket.PAD_PASSTHROUGH_KEYS
        missing = emitted - registered
        assert not missing, (
            f"featurizer emits keys absent from bucket.py pad-lists: {sorted(missing)}. "
            "Add each to DIM0_KEYS/DIM01_KEYS/DIM1_KEYS (if N-dependent) or "
            "PAD_PASSTHROUGH_KEYS (if not), or it will be padded incorrectly."
        )

    def test_bucket_pad_off_by_default(self):
        """Without bucket_pad, behaviour is unchanged from the legacy collate."""
        feats = [_make_features(100, seed=0)]
        batch_default = collate_fn(feats)
        batch_no_bucket = collate_fn(feats, bucket_pad=False)
        assert batch_default["res_type"].shape == (1, 100)
        assert batch_no_bucket["res_type"].shape == (1, 100)

    def test_bucket_pad_trunk_invariance(self):
        """The Evoformer trunk produces the same pair/single embeddings on
        the unpadded slice whether or not the input was bucket-padded.

        This is the deterministic part of the contract: every mask in the
        trunk must zero out padded slots so the embedding values for the
        real tokens are identical.  The diffusion head is stochastic so
        comparing ``atom_positions`` directly is unreliable; the trunk
        embeddings are not.
        """
        cfg = _tiny_config()
        torch.manual_seed(0)
        model = DeepFold3(cfg).eval()

        # Single sample, N=10 → bucket-pad rounds to 128 (smallest bucket).
        feat = _make_features(10, seed=11)

        batch_unpadded = collate_fn([feat], bucket_pad=False)
        batch_bucket = collate_fn([feat], bucket_pad=True)
        N = feat["res_type"].shape[0]

        with torch.no_grad():
            emb_unpadded, *_ = model.run_evoformer(batch_unpadded, num_recycles=0, seed=0)
            emb_bucket, *_ = model.run_evoformer(batch_bucket, num_recycles=0, seed=0)

        pair_unpadded = emb_unpadded["pair"]
        pair_bucket = emb_bucket["pair"][..., :N, :N, :]
        single_unpadded = emb_unpadded["single"]
        single_bucket = emb_bucket["single"][..., :N, :]

        assert pair_unpadded.shape == pair_bucket.shape
        assert single_unpadded.shape == single_bucket.shape

        pair_diff = (pair_unpadded - pair_bucket).abs().max().item()
        single_diff = (single_unpadded - single_bucket).abs().max().item()
        assert pair_diff < 1e-4, (
            f"trunk pair embedding diverges after bucket pad: max abs diff {pair_diff:.2e} — "
            f"a mask is leaking somewhere in the Evoformer"
        )
        assert single_diff < 1e-4, (
            f"trunk single embedding diverges after bucket pad: max abs diff {single_diff:.2e} — "
            f"a mask is leaking somewhere in the Evoformer"
        )


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


# --- feature-dict contract: a key must be present on every sample -----------
#
# collate_fn cannot repair a key that only some samples carry. For a 0-dim
# (scalar) key it drops the key from the WHOLE batch, so even the samples that
# had it lose it. Datasets therefore emit such keys unconditionally, with an
# explicit sentinel for "unknown".


def test_collate_drops_scalar_key_missing_from_any_sample():
    """Documents the hazard that forces the unconditional-emission contract."""
    batch = [
        {"res_type": torch.zeros(4, dtype=torch.long), "resolution": torch.tensor(2.5)},
        {"res_type": torch.zeros(4, dtype=torch.long)},  # scalar key absent
    ]
    out = collate_fn(batch)
    assert "resolution" not in out, "collate no longer drops partial scalars — update the contract docs"


def test_collate_keeps_scalar_key_present_on_every_sample():
    """With the contract honoured, each sample keeps its own scalar."""
    from deepfold3.data import const

    batch = [
        {"res_type": torch.zeros(4, dtype=torch.long), "resolution": torch.tensor(2.5)},
        {"res_type": torch.zeros(4, dtype=torch.long), "resolution": torch.tensor(const.RESOLUTION_UNKNOWN)},
    ]
    out = collate_fn(batch)
    assert out["resolution"].tolist() == [2.5, const.RESOLUTION_UNKNOWN]


def test_resolution_unknown_sentinel_is_gated_out():
    """The sentinel must fall outside the confidence-loss resolution window.

    ``RESOLUTION_UNKNOWN`` marks a record with no metadata; the AF3 §4.3 gate
    must exclude it rather than wave it through.
    """
    from deepfold3.data import const
    from deepfold3.model.config import TrainingConfig

    cfg = TrainingConfig()
    assert not (cfg.min_resolution <= const.RESOLUTION_UNKNOWN <= cfg.max_resolution)
