"""Chain permutation is shared with every GT-based loss, not just diffusion MSE.

AF3 §4.3 resolves the chain assignment once (from the mini-rollout) and scores
all GT-based losses against the permuted GT. This checks the end-to-end wiring
in ``diffusion_only_loss``: with a swapped homodimer prediction, the smooth-lDDT
component is computed against the permuted GT (low), not the raw GT (high).
"""

from __future__ import annotations

import torch

from deepfold3.data import const
from deepfold3.model.config import TrainingConfig
from deepfold3.training.loss import diffusion_only_loss, smooth_lddt_loss


def _swapped_asym_homodimer():
    """Homodimer (one entity) whose two copies have *different* shapes, with the
    prediction folding them swapped relative to the GT labelling.

    Different shapes make the swap distance-sensitive (a symmetric homodimer's
    swap is distance-invariant, so smooth-lDDT wouldn't see it). chain 0 is a
    unit-spaced line near the origin; chain 1 a 2-spaced line far along x.
    """
    N, M = 6, 1
    gt = torch.zeros(N, M, 3)
    gt[0:3, 0] = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])  # chain 0: spacing 1
    gt[3:6, 0] = torch.tensor([[50.0, 0, 0], [50, 2, 0], [50, 4, 0]])  # chain 1: spacing 2, far

    # Prediction (denoised == mini-rollout) has the copies swapped.
    denoised = torch.zeros(1, N, M, 3)
    denoised[0, 0:3, 0] = gt[3:6, 0]  # pred chain 0 ← GT chain 1
    denoised[0, 3:6, 0] = gt[0:3, 0]  # pred chain 1 ← GT chain 0

    features = {
        "atom_gt_positions": gt,
        "atom_gt_mask": torch.ones(N, M, dtype=torch.bool),
        "seq_mask": torch.ones(N, dtype=torch.bool),
        "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        "entity_id": torch.zeros(N, dtype=torch.long),
        "sym_token_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        "token_to_rep_atom": torch.zeros(N, dtype=torch.long),
        "mol_type": torch.full((N,), const.MOL_PROTEIN, dtype=torch.long),
    }
    pred = {
        "denoised": denoised,
        "gt_augmented": gt.unsqueeze(0),  # [1, N, M, 3]
        "noise_levels": torch.tensor([1.0]),
        "mini_rollout_pos": denoised,  # drives the chain assignment → {0:1, 1:0}
    }
    return pred, features


def test_smooth_lddt_uses_permuted_gt_in_diffusion_only_loss():
    pred, features = _swapped_asym_homodimer()
    cfg = TrainingConfig()  # defaults: alpha_diffusion>0, alpha_smooth_lddt=1, bond/chir/fape off

    _total, comp = diffusion_only_loss(pred, features, cfg)

    # The shared chain permutation reorders GT so the swapped prediction lines
    # up → smooth-lDDT hits its best-possible value.
    shared = comp["smooth_lddt"].item()

    kw = dict(
        cutoffs=cfg.lddt_cutoffs,
        inclusion_radius=cfg.lddt_inclusion_radius,
        nucleotide_inclusion_radius=cfg.lddt_nucleotide_inclusion_radius,
        min_distance=cfg.lddt_min_distance,
        mol_type=features["mol_type"],
    )
    # Without the permutation (scoring against the raw GT) the mismatched intra-
    # chain shapes give a substantially larger smooth-lDDT loss.
    raw = smooth_lddt_loss(pred["denoised"], features["atom_gt_positions"], features["atom_gt_mask"], **kw).item()
    # The best attainable value: prediction vs itself (smooth-lDDT has a nonzero
    # floor from its sigmoid relaxation, so "perfect" is this, not 0).
    floor = smooth_lddt_loss(pred["denoised"], pred["denoised"][0], features["atom_gt_mask"], **kw).item()

    assert shared < raw - 0.05, f"shared-perm smooth_lddt ({shared:.3f}) should beat raw ({raw:.3f})"
    assert abs(shared - floor) < 1e-4, f"permuted GT should be a perfect match (floor {floor:.3f}), got {shared:.3f}"


def test_identity_assignment_leaves_losses_unchanged():
    """No swap (pred matches GT labelling) → permutation is identity → the
    smooth-lDDT equals the raw computation."""
    pred, features = _swapped_asym_homodimer()
    # Un-swap the prediction so the optimal assignment is identity.
    gt = features["atom_gt_positions"]
    pred["denoised"] = gt.unsqueeze(0).clone()
    pred["mini_rollout_pos"] = gt.unsqueeze(0).clone()
    pred["gt_augmented"] = gt.unsqueeze(0).clone()
    cfg = TrainingConfig()

    _total, comp = diffusion_only_loss(pred, features, cfg)
    raw = smooth_lddt_loss(
        pred["denoised"],
        gt,
        features["atom_gt_mask"],
        cutoffs=cfg.lddt_cutoffs,
        inclusion_radius=cfg.lddt_inclusion_radius,
        nucleotide_inclusion_radius=cfg.lddt_nucleotide_inclusion_radius,
        min_distance=cfg.lddt_min_distance,
        mol_type=features["mol_type"],
    ).item()
    assert abs(comp["smooth_lddt"].item() - raw) < 1e-5
