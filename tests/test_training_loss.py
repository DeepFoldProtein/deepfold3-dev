"""Tests for training loss functions."""

from __future__ import annotations

import torch

from deepfold3.training.loss import (
    _pairwise_binned_loss,
    diffusion_mse_loss,
    distogram_loss,
    exp_resolved_loss,
    express_coordinates_in_frame,
    pae_loss,
    pde_loss,
    plddt_loss,
    smooth_lddt_loss,
    training_loss,
    weighted_rigid_align,
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


def _make_pae_mask(frames_mask: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    return (frames_mask.bool().unsqueeze(-1) & pair_mask.bool()).float()


class TestWeightedRigidAlign:
    """Verify Kabsch-style weighted rigid alignment (AF3 §3.7.1 eq. 2)."""

    def test_self_align_is_identity(self):
        torch.manual_seed(0)
        x = torch.randn(4, 20, 3)
        w = torch.ones(4, 20)
        out = weighted_rigid_align(x, x, w)
        torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-4)

    def test_recovers_known_rotation(self):
        """GT = R·pred + t → align should recover R·pred + t up to numerical error."""
        torch.manual_seed(1)
        n = 32
        pred = torch.randn(n, 3)
        # Build a known rotation from Gram-Schmidt of two random vectors
        v0 = torch.randn(3)
        v1 = torch.randn(3)
        e0 = v0 / v0.norm()
        e1 = v1 - (v1 @ e0) * e0
        e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(e0, e1)
        R = torch.stack([e0, e1, e2], dim=0)
        t = torch.tensor([1.5, -0.7, 2.3])

        gt = pred @ R.T + t  # ground truth is pred transformed
        # weighted_rigid_align maps gt -> pred, so we feed (gt, pred) and
        # expect the result to match pred.
        w = torch.ones(n)
        gt_aligned = weighted_rigid_align(gt, pred, w)
        torch.testing.assert_close(gt_aligned, pred, atol=1e-4, rtol=1e-4)

    def test_weight_zero_atoms_ignored(self):
        """Atoms with weight=0 must not affect the alignment — verified by
        pinning high weight on a subset that is already aligned."""
        torch.manual_seed(2)
        n = 40
        pred = torch.randn(n, 3)
        gt = pred.clone()
        # Add big noise to the last 10 atoms; zero their weight.
        gt[-10:] += torch.randn(10, 3) * 10.0
        w = torch.ones(n)
        w[-10:] = 0.0
        out = weighted_rigid_align(gt, pred, w)
        # The unmasked atoms should map back almost exactly to pred.
        torch.testing.assert_close(out[:-10], pred[:-10], atol=1e-4, rtol=1e-4)

    def test_batched(self):
        """Works with a leading batch dim (per-sample alignment)."""
        torch.manual_seed(3)
        S, n = 4, 16
        pred = torch.randn(S, n, 3)
        # Each sample gets a different random rigid transform on GT
        gt = pred + torch.randn(S, 1, 3)  # translation only per sample
        w = torch.ones(S, n)
        out = weighted_rigid_align(gt, pred, w)
        torch.testing.assert_close(out, pred, atol=1e-4, rtol=1e-4)

    def test_bf16_input_preserves_dtype_and_accuracy(self):
        """SVD is forced to fp32 internally; caller's dtype is preserved."""
        torch.manual_seed(4)
        n = 24
        pred = torch.randn(n, 3)
        gt = pred + torch.tensor([0.5, -0.2, 1.0])
        w = torch.ones(n)
        out32 = weighted_rigid_align(gt, pred, w)
        out_bf = weighted_rigid_align(gt.bfloat16(), pred.bfloat16(), w.bfloat16())
        assert out_bf.dtype == torch.bfloat16
        torch.testing.assert_close(out_bf.float(), out32, atol=0.05, rtol=0.02)

    def test_stop_gradient_default(self):
        """Default ``stop_gradient=True`` returns a detached, no-grad result."""
        torch.manual_seed(5)
        pred = torch.randn(10, 3, requires_grad=True)
        gt = torch.randn(10, 3, requires_grad=True)
        w = torch.ones(10)
        out = weighted_rigid_align(gt, pred, w)
        assert not out.requires_grad

    def test_diffusion_mse_flag(self):
        """``apply_rigid_align`` flag gates the call: True should reduce
        the MSE vs. a pure-rotation-only perturbed GT."""
        torch.manual_seed(6)
        S, N, M = 2, 8, 4
        pred = torch.randn(S, N, M, 3)
        # GT is pred rotated by a random rigid transform per sample.
        # Without rigid_align, MSE is large; with rigid_align, ~0.
        gt = pred.clone()
        for s in range(S):
            v0, v1 = torch.randn(3), torch.randn(3)
            e0 = v0 / v0.norm()
            e1 = v1 - (v1 @ e0) * e0
            e1 = e1 / e1.norm()
            e2 = torch.linalg.cross(e0, e1)
            R = torch.stack([e0, e1, e2], dim=0)
            gt[s] = pred[s] @ R.T + torch.randn(3)
        atom_mask = torch.ones(N, M, dtype=torch.bool)
        noise_levels = torch.tensor([0.5, 2.0])

        loss_off = diffusion_mse_loss(
            pred, gt, atom_mask, noise_levels, apply_rigid_align=False
        )
        loss_on = diffusion_mse_loss(
            pred, gt, atom_mask, noise_levels, apply_rigid_align=True
        )
        assert loss_on < loss_off * 0.01, (
            f"rigid_align should collapse rotation-only error: {loss_on=} vs {loss_off=}"
        )


class TestExpressCoordinatesInFrame:
    """Verify the rigid-frame transform runs in fp32 regardless of input dtype.

    AF3 uses `jax.lax.Precision.HIGHEST` for geometric transforms; bf16
    degrades orthonormality of the frame basis and skews PAE targets.
    """

    def test_preserves_input_dtype(self):
        torch.manual_seed(0)
        n = 8
        rep = torch.randn(n, 3, dtype=torch.bfloat16)
        frames = torch.randn(n, 3, 3, dtype=torch.bfloat16)
        out = express_coordinates_in_frame(rep, frames)
        assert out.dtype == torch.bfloat16, "output dtype must match input dtype"

    def test_identity_point_is_origin(self):
        """Placing the rep atom at ``b`` (frame origin) must give zero in
        every local frame — verifies the matmul is not corrupted."""
        torch.manual_seed(0)
        n = 6
        frames = torch.randn(n, 3, 3)
        rep = frames[:, 1, :]  # rep[i] == b[i]
        out = express_coordinates_in_frame(rep, frames)
        # Diagonal entries (i, i) are trivially zero; the cross-pair entries
        # measure rep[j] - b[i] projected into frame i. For rep == b this
        # collapses to b[j] - b[i], not zero — only diagonal tests origin.
        diag = out[torch.arange(n), torch.arange(n)]
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_bf16_accuracy_matches_fp32(self):
        """bf16 inputs must produce the same transform as fp32 to within
        ambient bf16 tolerance — the internal Gram-Schmidt is fp32, so the
        basis itself is exact; only the final cast back loses precision."""
        torch.manual_seed(1)
        n = 8
        rep_f = torch.randn(n, 3)
        frames_f = torch.randn(n, 3, 3)
        out_f = express_coordinates_in_frame(rep_f, frames_f)
        out_b = express_coordinates_in_frame(rep_f.bfloat16(), frames_f.bfloat16())
        torch.testing.assert_close(out_f, out_b.float(), atol=0.05, rtol=0.02)


class TestPaeLoss:
    def test_scalar_output(self):
        logits = _randn(S, N, N, NUM_BINS, grad=True)
        pred_rep = _randn(S, N, 3)
        pred_frames = _randn(S, N, 3, 3)
        gt_rep = _randn(N, 3)
        gt_frames = _randn(N, 3, 3)
        _, seq_mask = _masks()
        pair_mask = seq_mask.unsqueeze(1) & seq_mask.unsqueeze(0)
        pae_mask = _make_pae_mask(seq_mask, pair_mask)
        loss = pae_loss(logits, pred_rep, pred_frames, gt_rep, gt_frames, pae_mask, num_bins=NUM_BINS)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None

    def test_identity_case(self):
        """When pred == gt the frame-aligned error is 0 for all pairs.

        With uniform-random logits, the loss should match
        ``-log_softmax[..., 0].mean()`` exactly.
        """
        torch.manual_seed(0)
        n = 16
        num_bins = 64
        rep = torch.randn(n, 3)
        frames = torch.randn(n, 3, 3)
        pred_rep = rep.unsqueeze(0)
        pred_frames = frames.unsqueeze(0)
        logits = torch.randn(1, n, n, num_bins, requires_grad=True)
        pae_mask = torch.ones(n, n)

        loss = pae_loss(logits, pred_rep, pred_frames, rep, frames, pae_mask, num_bins=num_bins)

        log_probs = torch.nn.functional.log_softmax(logits.detach().float(), dim=-1)
        expected = -log_probs[..., 0].mean()
        assert torch.allclose(loss, expected, atol=1e-5)
        import math

        assert abs(loss.item() - math.log(num_bins)) < 1.5

    def test_monotonic_in_perturbation(self):
        """Perturbing the prediction should not decrease the loss when
        logits concentrate mass on bin 0 (identity target)."""
        torch.manual_seed(1)
        n = 8
        num_bins = 64
        rep = torch.randn(n, 3)
        frames = torch.randn(n, 3, 3)
        pae_mask = torch.ones(n, n)

        logits = torch.zeros(1, n, n, num_bins)
        logits[..., 0] = 10.0

        loss_identity = pae_loss(
            logits, rep.unsqueeze(0), frames.unsqueeze(0), rep, frames, pae_mask, num_bins=num_bins
        )
        loss_perturbed = pae_loss(
            logits,
            rep.unsqueeze(0) + 3.0,
            frames.unsqueeze(0) + 3.0,
            rep,
            frames,
            pae_mask,
            num_bins=num_bins,
        )
        assert loss_perturbed.item() > loss_identity.item()


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
        valid_mask = torch.ones(N, M, dtype=torch.bool)
        loss = exp_resolved_loss(logits, atom_mask, valid_mask)
        assert loss.shape == ()
        assert loss.isfinite()
        loss.backward()
        assert logits.grad is not None

    def test_correct_with_perfect_prediction(self):
        """When logits perfectly predict the mask, loss should be low."""
        atom_mask = torch.ones(N, M, dtype=torch.bool)
        valid_mask = torch.ones(N, M, dtype=torch.bool)
        # Class 1 (resolved) with high logit
        logits = torch.zeros(S, N, M * 2)
        logits_r = logits.reshape(S, N, M, 2)
        logits_r[..., 1] = 10.0  # high confidence for "resolved"
        logits_r[..., 0] = -10.0
        loss = exp_resolved_loss(logits_r.reshape(S, N, M * 2), atom_mask, valid_mask)
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
