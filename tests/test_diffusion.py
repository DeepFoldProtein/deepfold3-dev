"""Tests for diffusion head components: noise schedule, Fourier embedding, augmentation."""

from __future__ import annotations

import math

import torch


class TestNoiseSchedule:
    """Verify the AF3 noise schedule."""

    def test_boundary_values(self):
        from deepfold3.model.diffusion import noise_schedule

        # t=0 → sigma = sigma_data * smax = 16 * 160 = 2560
        sigma_0 = noise_schedule(torch.tensor(0.0))
        assert abs(sigma_0.item() - 2560.0) < 1.0

        # t=1 → sigma = sigma_data * smin = 16 * 0.0004 = 0.0064
        sigma_1 = noise_schedule(torch.tensor(1.0))
        assert abs(sigma_1.item() - 0.0064) < 0.001

    def test_monotonically_decreasing(self):
        from deepfold3.model.diffusion import noise_schedule

        t = torch.linspace(0, 1, 201)
        sigma = noise_schedule(t)
        # Noise should decrease as t goes from 0 to 1
        diffs = sigma[1:] - sigma[:-1]
        assert (diffs <= 0).all(), "Noise schedule should be monotonically decreasing"

    def test_positive(self):
        from deepfold3.model.diffusion import noise_schedule

        t = torch.linspace(0, 1, 201)
        sigma = noise_schedule(t)
        assert (sigma > 0).all(), "All noise levels should be positive"


class TestFourierEmbedding:
    """Verify Fourier noise embedding (via DiffusionConditioningBlock buffers)."""

    def _make_block(self):
        """Create a DiffusionConditioningBlock with random Fourier buffers."""
        from deepfold3.model.diffusion import DiffusionConditioningBlock

        block = DiffusionConditioningBlock(
            c_pair_in=128, c_single_in=384, rel_feat_dim=139, c_pair_out=128, c_single_out=384
        )
        block.fourier_weight = torch.randn(256)
        block.fourier_bias = torch.rand(256)
        return block

    def test_output_shape(self):
        block = self._make_block()
        sigma = torch.tensor(1.0)
        emb = block._fourier_noise_embedding(sigma)
        assert emb.shape == (256,)

    def test_batch_shape(self):
        block = self._make_block()
        sigma = torch.tensor([1.0, 10.0, 100.0])
        emb = block._fourier_noise_embedding(sigma)
        assert emb.shape == (3, 256)

    def test_bounded_output(self):
        """Fourier embedding uses cos(), so output should be in [-1, 1]."""
        block = self._make_block()
        sigma = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0, 1000.0])
        emb = block._fourier_noise_embedding(sigma)
        assert emb.min() >= -1.0
        assert emb.max() <= 1.0

    def test_formula(self):
        """Verify: cos(2*pi*(log(sigma)/4 * w + b))."""
        block = self._make_block()
        sigma = torch.tensor(5.0)
        emb = block._fourier_noise_embedding(sigma)

        # Manual computation
        x = 0.25 * torch.log(sigma)
        emb_manual = x * block.fourier_weight + block.fourier_bias
        emb_manual = torch.cos(2 * math.pi * emb_manual)

        torch.testing.assert_close(emb, emb_manual, atol=1e-6, rtol=1e-5)

    def test_different_sigma_different_embedding(self):
        block = self._make_block()
        e1 = block._fourier_noise_embedding(torch.tensor(1.0))
        e2 = block._fourier_noise_embedding(torch.tensor(10.0))
        assert not torch.allclose(e1, e2, atol=1e-3)

    def test_returns_fp32_even_for_bf16_input(self):
        """AF3 hardcodes Fourier weights/biases as fp32; we upcast sigma and
        buffers to fp32 regardless of ambient precision so the embedding is
        numerically stable under bf16 autocast."""
        block = self._make_block()
        sigma_bf16 = torch.tensor(5.0, dtype=torch.bfloat16)
        emb = block._fourier_noise_embedding(sigma_bf16)
        assert emb.dtype == torch.float32

        # Fp32 output should match a fully-fp32 reference within fp32 tolerance.
        sigma_f32 = sigma_bf16.float()
        x = 0.25 * torch.log(sigma_f32)
        ref = torch.cos(2 * math.pi * (x * block.fourier_weight.float() + block.fourier_bias.float()))
        torch.testing.assert_close(emb, ref, atol=1e-6, rtol=1e-5)


class TestRandomAugmentation:
    """Verify random rigid augmentation (center → rotate → translate)."""

    def test_output_shape(self):
        from deepfold3.model.diffusion import _random_augmentation

        pos = torch.randn(10, 24, 3)
        mask = torch.ones(10, 24)
        gen = torch.Generator().manual_seed(42)
        out = _random_augmentation(pos, mask, gen)
        assert out.shape == (10, 24, 3)

    def test_masked_atoms_zeroed(self):
        """Masked atoms should be exactly zero after augmentation."""
        from deepfold3.model.diffusion import _random_augmentation

        pos = torch.randn(10, 24, 3)
        mask = torch.zeros(10, 24)
        mask[:, :5] = 1
        gen = torch.Generator().manual_seed(42)
        out = _random_augmentation(pos, mask, gen)
        assert (out[:, 5:] == 0).all()

    def test_preserves_distances(self):
        """Rotation + translation should preserve pairwise distances."""
        from deepfold3.model.diffusion import _random_augmentation

        torch.manual_seed(42)
        pos = torch.randn(5, 10, 3)
        mask = torch.ones(5, 10)
        gen = torch.Generator().manual_seed(42)
        out = _random_augmentation(pos, mask, gen)

        # Compute pairwise distances for first token's atoms
        d_before = torch.cdist(pos[0, :5], pos[0, :5])
        d_after = torch.cdist(out[0, :5], out[0, :5])

        # Distances should be preserved (rigid transformation)
        torch.testing.assert_close(d_before, d_after, atol=1e-4, rtol=1e-4)

    def test_deterministic_with_generator(self):
        from deepfold3.model.diffusion import _random_augmentation

        pos = torch.randn(5, 10, 3)
        mask = torch.ones(5, 10)

        gen1 = torch.Generator().manual_seed(42)
        out1 = _random_augmentation(pos, mask, gen1)

        gen2 = torch.Generator().manual_seed(42)
        out2 = _random_augmentation(pos, mask, gen2)

        torch.testing.assert_close(out1, out2)

    def test_bf16_input_preserves_dtype(self):
        """Gram-Schmidt runs in fp32 internally (AF3 Precision.HIGHEST),
        but the output is cast back to the input dtype so downstream
        computation stays in the ambient precision."""
        from deepfold3.model.diffusion import _random_augmentation_from_vectors

        torch.manual_seed(0)
        pos = torch.randn(5, 10, 3, dtype=torch.bfloat16)
        mask = torch.ones(5, 10, dtype=torch.bfloat16)
        v0 = torch.randn(3, dtype=torch.bfloat16)
        v1 = torch.randn(3, dtype=torch.bfloat16)
        tr = torch.zeros(3, dtype=torch.bfloat16)  # zero translation for exactness
        out = _random_augmentation_from_vectors(pos, mask, v0, v1, tr)
        assert out.dtype == torch.bfloat16

        # Fp32 Gram-Schmidt should keep pairwise distances accurate despite
        # the bf16 input — far tighter than a naive all-bf16 rotation.
        d_before = torch.cdist(pos[0].float(), pos[0].float())
        d_after = torch.cdist(out[0].float(), out[0].float())
        torch.testing.assert_close(d_before, d_after, atol=0.05, rtol=0.02)


class TestSkipAmp:
    """Verify the Protenix-style ``skip_amp`` options on DiffusionConfig."""

    def test_flags_default_false(self):
        """AF3 default: both flags off — rely on selective fp32 upcasts."""
        from deepfold3.model.config import DiffusionConfig

        cfg = DiffusionConfig()
        assert cfg.skip_amp_sample is False
        assert cfg.skip_amp_sample_training is False

    def test_flags_roundtrip(self):
        from deepfold3.model.config import DiffusionConfig

        cfg = DiffusionConfig(skip_amp_sample=True, skip_amp_sample_training=True)
        assert cfg.skip_amp_sample is True
        assert cfg.skip_amp_sample_training is True

    def test_upcast_fp32_tensor(self):
        from deepfold3.model.diffusion import _upcast_fp32

        bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        assert _upcast_fp32(bf16).dtype == torch.float32

        int_t = torch.tensor([1, 2], dtype=torch.int64)
        assert _upcast_fp32(int_t).dtype == torch.int64  # integer types untouched

        bool_t = torch.tensor([True, False])
        assert _upcast_fp32(bool_t).dtype == torch.bool

    def test_upcast_fp32_handles_nested_structures(self):
        from deepfold3.model.diffusion import _upcast_fp32

        bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
        int_t = torch.tensor([1], dtype=torch.int64)
        nested = {
            "embeddings": {"single": bf16, "pair": bf16},
            "ref_element": int_t,
            "optional": None,
            "seq": [bf16, int_t],
        }
        out = _upcast_fp32(nested)
        assert out["embeddings"]["single"].dtype == torch.float32
        assert out["embeddings"]["pair"].dtype == torch.float32
        assert out["ref_element"].dtype == torch.int64
        assert out["optional"] is None
        assert out["seq"][0].dtype == torch.float32
        assert out["seq"][1].dtype == torch.int64

    def test_skip_amp_sample_training_disables_autocast_in_impl(self):
        """With skip_amp_sample_training=True, the autocast state inside
        ``_sample_training_impl`` is disabled even when the caller is
        running under an active autocast context."""
        from deepfold3.model.config import DiffusionConfig, OptimConfig
        from deepfold3.model.diffusion import DiffusionHead

        cfg = DiffusionConfig(skip_amp_sample_training=True)
        head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig())

        captured = {}

        def fake_impl(*args, **kwargs):
            captured["cuda_enabled"] = torch.is_autocast_enabled("cuda")
            captured["cpu_enabled"] = torch.is_autocast_enabled("cpu")
            N, M, B, S = 2, 2, 1, 1
            return (
                torch.zeros(B, S, N, M, 3),
                torch.zeros(B, S, N, M, 3),
                torch.zeros(B, S),
            )

        head._sample_training_impl = fake_impl  # type: ignore[method-assign]

        # Build minimally-sized inputs.
        N, M, B = 2, 2, 1
        kwargs = {
            "gt_positions": torch.randn(B, N, M, 3),
            "gt_mask": torch.ones(B, N, M),
            "embeddings": {
                "single": torch.randn(B, N, 384),
                "pair": torch.randn(B, N, N, 128),
                "target_feat": torch.randn(B, N, 447),
            },
            "ref_pos": torch.randn(B, N, M, 3),
            "ref_mask": torch.ones(B, N, M),
            "ref_element": torch.zeros(B, N, M, dtype=torch.int64),
            "ref_charge": torch.zeros(B, N, M),
            "ref_atom_name_chars": torch.zeros(B, N, M, 4, dtype=torch.int64),
            "seq_mask": torch.ones(B, N),
            "rel_feat": torch.randn(B, N, N, 8),
            "num_samples": 1,
        }

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            head.sample_training(**kwargs)
        assert captured["cpu_enabled"] is False, "skip_amp_sample_training should disable CPU autocast"
        assert captured["cuda_enabled"] is False

    def test_skip_amp_sample_training_off_keeps_autocast(self):
        """With the flag off (AF3 default), the caller's autocast context
        remains active inside the impl."""
        from deepfold3.model.config import DiffusionConfig, OptimConfig
        from deepfold3.model.diffusion import DiffusionHead

        cfg = DiffusionConfig()  # skip_amp_sample_training defaults False
        head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig())

        captured = {}

        def fake_impl(*args, **kwargs):
            captured["cpu_enabled"] = torch.is_autocast_enabled("cpu")
            N, M, B, S = 2, 2, 1, 1
            return (
                torch.zeros(B, S, N, M, 3),
                torch.zeros(B, S, N, M, 3),
                torch.zeros(B, S),
            )

        head._sample_training_impl = fake_impl  # type: ignore[method-assign]

        N, M, B = 2, 2, 1
        kwargs = {
            "gt_positions": torch.randn(B, N, M, 3),
            "gt_mask": torch.ones(B, N, M),
            "embeddings": {
                "single": torch.randn(B, N, 384),
                "pair": torch.randn(B, N, N, 128),
                "target_feat": torch.randn(B, N, 447),
            },
            "ref_pos": torch.randn(B, N, M, 3),
            "ref_mask": torch.ones(B, N, M),
            "ref_element": torch.zeros(B, N, M, dtype=torch.int64),
            "ref_charge": torch.zeros(B, N, M),
            "ref_atom_name_chars": torch.zeros(B, N, M, 4, dtype=torch.int64),
            "seq_mask": torch.ones(B, N),
            "rel_feat": torch.randn(B, N, N, 8),
            "num_samples": 1,
        }

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            head.sample_training(**kwargs)
        assert captured["cpu_enabled"] is True, "autocast should remain active without skip_amp"
