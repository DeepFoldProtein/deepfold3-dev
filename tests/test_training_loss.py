"""Tests for training loss functions."""

from __future__ import annotations

import torch

from deepfold3.training.loss import (
    _pairwise_binned_loss,
    diffusion_mse_loss,
    distogram_loss,
    exp_resolved_loss,
    pae_loss,
    pde_loss,
    plddt_loss,
    smooth_lddt_loss,
    training_loss,
)

N, M, S, NUM_BINS = 8, 4, 2, 16


def _randn(*shape, grad=False):
    torch.manual_seed(42)
    return torch.randn(*shape, requires_grad=grad)


def _masks():
    atom_mask = torch.ones(N, M, dtype=torch.bool)
    seq_mask = torch.ones(N, dtype=torch.bool)
    return atom_mask, seq_mask


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


class TestSmoothLddtLoss:
    def test_scalar_output(self):
        pred = _randn(S, N, M, 3)
        gt = _randn(N, M, 3)
        atom_mask, _ = _masks()
        loss = smooth_lddt_loss(pred, gt, atom_mask)
        assert loss.shape == ()
        assert loss.isfinite()

    def test_gradient_flow(self):
        pred = _randn(S, N, M, 3, grad=True)
        gt = _randn(N, M, 3)
        atom_mask, _ = _masks()
        loss = smooth_lddt_loss(pred, gt, atom_mask)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.isfinite().all()


class TestDiffusionMseLoss:
    def test_scalar_output(self):
        pred = _randn(S, N, M, 3)
        gt = _randn(S, N, M, 3)
        atom_mask, _ = _masks()
        noise_levels = torch.rand(S)
        loss = diffusion_mse_loss(pred, gt, atom_mask, noise_levels)
        assert loss.shape == ()
        assert loss.isfinite()

    def test_zero_when_identical(self):
        gt = _randn(S, N, M, 3)
        atom_mask, _ = _masks()
        noise_levels = torch.ones(S)
        loss = diffusion_mse_loss(gt, gt.clone(), atom_mask, noise_levels)
        assert loss.item() < 1e-6

    def test_gradient_flow(self):
        pred = _randn(S, N, M, 3, grad=True)
        gt = _randn(S, N, M, 3)
        atom_mask, _ = _masks()
        noise_levels = torch.rand(S)
        loss = diffusion_mse_loss(pred, gt, atom_mask, noise_levels)
        loss.backward()
        assert pred.grad is not None


class TestDistogramLoss:
    def test_scalar_output(self):
        logits = _randn(N, N, NUM_BINS, grad=True)
        gt_pos = _randn(N, 3)
        _, seq_mask = _masks()
        loss = distogram_loss(logits, gt_pos, seq_mask, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None


class TestPairwiseBinnedLoss:
    def test_bin_edges_consistent_with_distogram(self):
        """Verify _pairwise_binned_loss uses num_bins-1 edges (matching distogram_loss)."""
        logits = _randn(S, N, N, NUM_BINS)
        error = torch.rand(S, N, N) * 10
        pair_mask = torch.ones(N, N)

        # Should not crash — if it used num_bins edges, bucketize would produce
        # values up to num_bins which would be out of range after clamp
        loss = _pairwise_binned_loss(logits, error, pair_mask, max_error=31.0, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()

    def test_gradient_flow(self):
        logits = _randn(S, N, N, NUM_BINS, grad=True)
        error = torch.rand(S, N, N) * 10
        pair_mask = torch.ones(N, N)
        loss = _pairwise_binned_loss(logits, error, pair_mask, max_error=31.0, num_bins=NUM_BINS)
        loss.backward()
        assert logits.grad is not None


class TestPaeLoss:
    def test_scalar_output(self):
        logits = _randn(S, N, N, NUM_BINS, grad=True)
        pred_pos = _randn(S, N, 3)
        gt_pos = _randn(N, 3)
        _, seq_mask = _masks()
        loss = pae_loss(logits, pred_pos, gt_pos, seq_mask, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None


class TestPdeLoss:
    def test_scalar_output(self):
        logits = _randn(S, N, N, NUM_BINS, grad=True)
        pred_pos = _randn(S, N, 3)
        gt_pos = _randn(N, 3)
        _, seq_mask = _masks()
        loss = pde_loss(logits, pred_pos, gt_pos, seq_mask, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None


class TestPlddtLoss:
    def test_scalar_output(self):
        logits = _randn(S, N, M * NUM_BINS, grad=True)
        pred_pos = _randn(S, N, M, 3)
        gt_pos = _randn(N, M, 3)
        atom_mask, _ = _masks()
        loss = plddt_loss(logits, pred_pos, gt_pos, atom_mask, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None


class TestExpResolvedLoss:
    def test_scalar_output(self):
        logits = _randn(S, N, M * 2, grad=True)
        atom_mask, _ = _masks()
        loss = exp_resolved_loss(logits, atom_mask)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None

    def test_correct_with_perfect_prediction(self):
        """When logits perfectly predict the mask, loss should be low."""
        atom_mask = torch.ones(N, M, dtype=torch.bool)
        # Class 1 (resolved) with high logit
        logits = torch.zeros(S, N, M * 2)
        logits_r = logits.reshape(S, N, M, 2)
        logits_r[..., 1] = 10.0  # high confidence for "resolved"
        logits_r[..., 0] = -10.0
        loss = exp_resolved_loss(logits_r.reshape(S, N, M * 2), atom_mask)
        assert loss.item() < 0.01


class TestTrainingLoss:
    def test_combined_loss(self):
        from deepfold3.model.config import TrainingConfig

        atom_mask, seq_mask = _masks()
        gt_pos = _randn(N, M, 3)
        rep_idx = torch.ones(N, dtype=torch.long)  # CA index

        pred = {
            "denoised": _randn(S, N, M, 3),
            "gt_augmented": _randn(S, N, M, 3),
            "noise_levels": torch.rand(S),
            "distogram": _randn(N, N, 64),
            "plddt": _randn(S, N, M * 50),
            "pae": _randn(S, N, N, 64),
            "pde": _randn(S, N, N, 64),
            "exp_resolved": _randn(S, N, M * 2),
            "mini_rollout_pos": _randn(S, N, M, 3),
        }
        features = {
            "atom_gt_positions": gt_pos,
            "atom_gt_mask": atom_mask,
            "seq_mask": seq_mask.float(),
            "token_to_rep_atom": rep_idx,
        }

        cfg = TrainingConfig()
        total, losses = training_loss(pred, features, cfg)
        assert total.shape == ()
        assert total.isfinite()
        assert "total" in losses
        assert "diffusion_mse" in losses
        assert "smooth_lddt" in losses
        assert "distogram" in losses
