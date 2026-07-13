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
from deepfold3.training.loss import (
    chain_perm_from_mini_rollout,
    diffusion_only_loss,
    resolve_gt_atom_symmetry,
    smooth_lddt_loss,
)


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


def test_shared_gt_resolves_symmetric_side_chain_atoms():
    """Intra-residue symmetric atoms (e.g. ASP OD1/OD2) are relabelled on the
    shared GT to match the mini-rollout — benefiting the per-atom GT losses,
    not only the diffusion MSE."""
    N, M = 1, 6  # one residue: N, CA, C, O anchor the frame; slots 4/5 are the symmetric pair
    gt = torch.zeros(N, M, 3)
    gt[0, 0] = torch.tensor([0.0, 0, 0])
    gt[0, 1] = torch.tensor([1.0, 0, 0])
    gt[0, 2] = torch.tensor([2.0, 0, 0])
    gt[0, 3] = torch.tensor([2.0, 1, 0])
    gt[0, 4] = torch.tensor([3.0, 0, 0])  # OD1
    gt[0, 5] = torch.tensor([3.0, 2, 0])  # OD2

    # Mini-rollout: same backbone, OD1/OD2 swapped relative to the GT labelling.
    mini = gt.clone().unsqueeze(0)  # [1, N, M, 3]
    mini[0, 0, 4] = gt[0, 5]
    mini[0, 0, 5] = gt[0, 4]

    atom_mask = torch.ones(N, M, dtype=torch.bool)
    features = {
        "atom_swap_pairs": torch.tensor([[0, 4, 5]], dtype=torch.long),
        "token_to_rep_atom": torch.tensor([1], dtype=torch.long),  # CA
    }

    # ref = the mini-rollout structure [N, M, 3]; gt is relabelled to match it.
    resolved = resolve_gt_atom_symmetry(mini[0], features, gt, atom_mask)

    # GT's symmetric atoms are now labelled to match the prediction.
    assert torch.allclose(resolved[0, 4], mini[0, 0, 4])  # slot 4 ← GT OD2
    assert torch.allclose(resolved[0, 5], mini[0, 0, 5])  # slot 5 ← GT OD1
    # Backbone untouched.
    assert torch.allclose(resolved[0, :4], gt[0, :4])


def test_smooth_lddt_always_sees_atom_resolved_gt():
    """AF3 §4.2 (always on): smooth-lDDT scores against the mini-rollout
    atom-symmetry-resolved GT. A monomer ASP with OD1/OD2 swapped in the
    prediction → the loss is much lower than against the raw (unresolved) GT."""
    N, M = 1, 6
    gt = torch.zeros(N, M, 3)
    gt[0, 0] = torch.tensor([0.0, 0, 0])
    gt[0, 1] = torch.tensor([1.0, 0, 0])
    gt[0, 2] = torch.tensor([2.0, 0, 0])
    gt[0, 3] = torch.tensor([2.0, 1, 0])
    gt[0, 4] = torch.tensor([3.0, 0, 0])  # OD1
    gt[0, 5] = torch.tensor([3.0, 2, 0])  # OD2
    denoised = gt.clone().unsqueeze(0)
    denoised[0, 0, 4] = gt[0, 5]  # prediction swapped OD1/OD2
    denoised[0, 0, 5] = gt[0, 4]

    features = {
        "atom_gt_positions": gt,
        "atom_gt_mask": torch.ones(N, M, dtype=torch.bool),
        "seq_mask": torch.ones(N, dtype=torch.bool),
        "mol_type": torch.full((N,), const.MOL_PROTEIN, dtype=torch.long),
        "token_to_rep_atom": torch.tensor([1], dtype=torch.long),
        "atom_swap_pairs": torch.tensor([[0, 4, 5]], dtype=torch.long),
    }
    pred = {
        "denoised": denoised,
        "gt_augmented": gt.unsqueeze(0),
        "noise_levels": torch.tensor([1.0]),
        "mini_rollout_pos": denoised,
    }

    cfg = TrainingConfig()
    _t, comp = diffusion_only_loss(pred, features, cfg)
    raw = smooth_lddt_loss(denoised, gt, features["atom_gt_mask"], mol_type=features["mol_type"]).item()
    assert comp["smooth_lddt"].item() < raw - 0.05


def test_resolve_gt_atom_symmetry_noop_without_tables():
    """No swap tables → GT returned unchanged."""
    N, M = 1, 6
    gt = torch.randn(N, M, 3)
    mask = torch.ones(N, M, dtype=torch.bool)
    assert torch.allclose(resolve_gt_atom_symmetry(gt.clone(), {}, gt, mask), gt)


def test_inter_chain_bonds_from_features_groups_ligand_attachment():
    """A cross-chain token bond with a non-polymer endpoint groups the two
    chains into one covalent permutation component (item 11 wiring)."""
    from deepfold3.training.loss import _inter_chain_bonds_from_features

    N = 4
    asym = torch.tensor([0, 0, 1, 2])  # chain0 protein, chain1 ligand/glycan, chain2 protein
    mol = torch.tensor(
        [const.MOL_PROTEIN, const.MOL_PROTEIN, const.MOL_NONPOLYMER, const.MOL_PROTEIN]
    )
    bm = torch.zeros(N, N, 1)
    bm[1, 2, 0] = bm[2, 1, 0] = 1.0  # protein(chain0) — glycan(chain1)
    out = _inter_chain_bonds_from_features({"bond_matrix": bm, "asym_id": asym, "mol_type": mol})
    assert out == [(0, 1)]


def test_inter_chain_bonds_excludes_polymer_polymer_disulfide():
    """An inter-chain polymer–polymer bond (disulfide) must NOT group the chains
    — OF3 strips those in prefilter; the non-polymer-endpoint rule matches that."""
    from deepfold3.training.loss import _inter_chain_bonds_from_features

    asym = torch.tensor([0, 1])
    mol = torch.tensor([const.MOL_PROTEIN, const.MOL_PROTEIN])
    bm = torch.zeros(2, 2, 1)
    bm[0, 1, 0] = bm[1, 0, 0] = 1.0
    assert _inter_chain_bonds_from_features({"bond_matrix": bm, "asym_id": asym, "mol_type": mol}) is None


def test_inter_chain_bonds_none_without_bond_matrix():
    from deepfold3.training.loss import _inter_chain_bonds_from_features

    assert _inter_chain_bonds_from_features({"asym_id": torch.tensor([0, 1])}) is None


# ---------------------------------------------------------------------------
# Chain-permutation stability gate (K-rollout unanimity)
# ---------------------------------------------------------------------------


def test_stability_gate_unanimous_applies_permutation():
    """K identical well-separated draws all vote the same swap → gate applies it."""
    pred, features = _swapped_asym_homodimer()
    swapped = pred["mini_rollout_pos"][0]  # drives assignment {0: 1, 1: 0}
    pred["mini_rollout_pos"] = swapped.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, N, M, 3]
    assert chain_perm_from_mini_rollout(pred, features, stability_gate_k=3) == {0: 1, 1: 0}


def test_stability_gate_disagreement_falls_back_to_identity():
    """Draws that disagree on the assignment (coin-flip) → identity (un-permuted)."""
    pred, features = _swapped_asym_homodimer()
    swapped = pred["mini_rollout_pos"][0]  # -> {0: 1, 1: 0}
    identity_match = features["atom_gt_positions"]  # pred == GT -> {0: 0, 1: 1}
    pred["mini_rollout_pos"] = torch.stack([swapped, identity_match], dim=0)
    assert chain_perm_from_mini_rollout(pred, features, stability_gate_k=2) == {}


def test_stability_gate_k1_uses_first_sample_only():
    """K=1 (gate off) ignores extra samples and matches legacy single-sample behaviour."""
    pred, features = _swapped_asym_homodimer()
    swapped = pred["mini_rollout_pos"][0]
    pred["mini_rollout_pos"] = torch.stack([swapped, features["atom_gt_positions"]], dim=0)
    assert chain_perm_from_mini_rollout(pred, features, stability_gate_k=1) == {0: 1, 1: 0}


def test_stability_gate_reads_dedicated_gate_positions():
    """Production wiring: the K-sample vote uses ``chain_perm_gate_pos`` while
    ``mini_rollout_pos`` keeps its own (confidence-facing) sample count."""
    pred, features = _swapped_asym_homodimer()
    swapped = pred["mini_rollout_pos"][0]  # -> {0: 1, 1: 0}
    # mini_rollout_pos left at 1 sample (as confidence/tm_true see it); the vote
    # comes from the separate K-sample gate rollout (all swapped -> unanimous).
    pred["mini_rollout_pos"] = features["atom_gt_positions"].unsqueeze(0)  # identity-looking
    pred["chain_perm_gate_pos"] = swapped.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, N, M, 3]
    assert chain_perm_from_mini_rollout(pred, features, stability_gate_k=3) == {0: 1, 1: 0}
