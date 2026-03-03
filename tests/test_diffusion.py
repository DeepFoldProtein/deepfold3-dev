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
