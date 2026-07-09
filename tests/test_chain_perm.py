"""Chain-permutation alignment for the diffusion MSE loss (AF3 §4.2).

Covers the token-layout entry point ``compute_chain_permutation_tokens`` and
the fixes from the OF3 audit:
  * representative (token-center) atom alignment,
  * ligand-only entities deprioritised as the anchor,
  * ``seq_mask`` padding filter so zero-padded ``asym_id == 0`` tokens do not
    pollute chain 0's block,
  * ``sym_token_index`` matching so symmetric chains a crop cut to unequal
    token counts are still permuted instead of silently skipped (gap-1).
"""

from __future__ import annotations

import torch

from deepfold3.data import const
from deepfold3.data.symmetry.chain_perm import (
    apply_chain_permutation_tokens,
    compute_chain_permutation_tokens,
)


def _homodimer_swapped(M: int = 2):
    """A 2-chain, same-entity scene where pred places the copies swapped.

    GT chain 0 sits near the origin (x = 0,1,2); GT chain 1 near x = 10,11,12.
    The prediction puts chain 0 where GT chain 1 is and vice versa, so the
    correct assignment is ``{0: 1, 1: 0}``.
    """
    N = 6  # 3 tokens per chain
    gt = torch.zeros(N, M, 3)
    pred = torch.zeros(N, M, 3)

    gt_chain0 = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    gt_chain1 = torch.tensor([[10.0, 0, 0], [11, 0, 0], [12, 0, 0]])
    gt[0:3, 0] = gt_chain0
    gt[3:6, 0] = gt_chain1

    # pred chain 0 ≈ GT chain 1 location; pred chain 1 ≈ GT chain 0 location.
    pred[0:3, 0] = gt_chain1
    pred[3:6, 0] = gt_chain0

    atom_mask = torch.zeros(N, M, dtype=torch.bool)
    atom_mask[:, 0] = True  # only the rep atom is resolved
    asym_id = torch.tensor([0, 0, 0, 1, 1, 1])
    entity_id = torch.zeros(N, dtype=torch.long)  # both chains one entity
    token_to_rep_atom = torch.zeros(N, dtype=torch.long)  # rep = slot 0
    mol_type = torch.full((N,), const.MOL_PROTEIN, dtype=torch.long)
    seq_mask = torch.ones(N, dtype=torch.bool)
    return pred, gt, atom_mask, asym_id, entity_id, token_to_rep_atom, mol_type, seq_mask


def test_swap_recovered_rep_atom():
    pred, gt, mask, asym, ent, rep, mol, seq = _homodimer_swapped()
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    assert assign == {0: 1, 1: 0}


def test_apply_swap_reorders_gt():
    pred, gt, mask, asym, ent, rep, mol, seq = _homodimer_swapped()
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    new_gt, new_mask = apply_chain_permutation_tokens(gt, mask, asym, assign)
    # After reorder, GT rep atoms line up with the prediction (≈ 0 residual).
    resid = ((new_gt[:, 0] - pred[:, 0]) ** 2).sum().item()
    assert resid < 1e-6
    assert new_mask.shape == mask.shape


def test_identity_for_singleton_entities():
    """Distinct entities (no symmetric copies) → identity mapping."""
    pred, gt, mask, asym, _ent, rep, mol, seq = _homodimer_swapped()
    entity_id = torch.tensor([0, 0, 0, 1, 1, 1])  # two *different* entities
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, entity_id, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    assert assign == {0: 0, 1: 1}


def test_padding_tokens_do_not_pollute_chain0():
    """Zero-padded tokens (asym_id 0, seq_mask False) must not break chain 0.

    Without the ``seq_mask`` filter the padding inflates chain 0's token count,
    which can drop the real chain 0 as an anchor candidate and flip the swap to
    a wrong identity mapping. With the filter the swap is still recovered.
    """
    pred, gt, mask, asym, ent, rep, mol, seq = _homodimer_swapped()
    n_pad = 4
    pad_pred = torch.full((n_pad, pred.shape[1], 3), 99.0)
    pad_gt = torch.full((n_pad, gt.shape[1], 3), -99.0)
    pred = torch.cat([pred, pad_pred], dim=0)
    gt = torch.cat([gt, pad_gt], dim=0)
    mask = torch.cat([mask, torch.zeros(n_pad, mask.shape[1], dtype=torch.bool)], dim=0)
    asym = torch.cat([asym, torch.zeros(n_pad, dtype=torch.long)], dim=0)  # pad asym_id == 0
    ent = torch.cat([ent, torch.zeros(n_pad, dtype=torch.long)], dim=0)
    rep = torch.cat([rep, torch.zeros(n_pad, dtype=torch.long)], dim=0)
    mol = torch.cat([mol, torch.full((n_pad,), const.MOL_PROTEIN, dtype=torch.long)], dim=0)
    seq = torch.cat([seq, torch.zeros(n_pad, dtype=torch.bool)], dim=0)

    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    assert assign == {0: 1, 1: 0}


def test_ligand_entity_deprioritised_but_polymer_swap_recovered():
    """A single-copy ligand entity must not become the anchor.

    The ligand is the least-ambiguous entity by chain count (1 copy), so
    without the polymer-first rule it would be picked as the anchor. Marking it
    via ``mol_type`` keeps the polymer homodimer swap correct.
    """
    pred, gt, mask, asym, ent, rep, mol, seq = _homodimer_swapped()
    # Append one ligand token (its own chain / entity) far from the dimer.
    lig_pred = torch.tensor([[[50.0, 0, 0], [50, 0, 0]]])  # [1, M, 3]
    lig_gt = torch.tensor([[[50.0, 0, 0], [50, 0, 0]]])
    pred = torch.cat([pred, lig_pred], dim=0)
    gt = torch.cat([gt, lig_gt], dim=0)
    mask = torch.cat([mask, torch.tensor([[True, False]])], dim=0)
    asym = torch.cat([asym, torch.tensor([2])], dim=0)
    ent = torch.cat([ent, torch.tensor([1])], dim=0)  # ligand = its own entity
    rep = torch.cat([rep, torch.tensor([0])], dim=0)
    mol = torch.cat([mol, torch.tensor([const.MOL_NONPOLYMER])], dim=0)
    seq = torch.cat([seq, torch.tensor([True])], dim=0)

    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    assert assign[0] == 1 and assign[1] == 0
    assert assign[2] == 2  # singleton ligand maps to itself


def test_all_atom_fallback_still_runs():
    """No ``token_to_rep_atom`` → legacy all-heavy-atom flatten path."""
    pred, gt, mask, asym, ent, _rep, _mol, _seq = _homodimer_swapped()
    # Make every atom resolved so the flattened block is well-defined.
    mask = torch.ones_like(mask)
    gt[:, 1] = gt[:, 0] + 0.1
    pred[:, 1] = pred[:, 0] + 0.1
    assign = compute_chain_permutation_tokens(pred, gt, mask, asym, ent)
    assert assign == {0: 1, 1: 0}


# ---------------------------------------------------------------------------
# gap-1: partial crop cuts symmetric chains to UNEQUAL token counts
# ---------------------------------------------------------------------------


def _unequal_crop_swapped(M: int = 2):
    """Homodimer (entity length 5) that a crop cut to unequal counts.

    Surviving: chain 0 keeps sym 0..3 (4 tokens), chain 1 keeps sym 0..2 (3).
    GT chain 0 sits near the origin, GT chain 1 near x = 100. The prediction
    folds the copies swapped, so the correct assignment is ``{0: 1, 1: 0}`` —
    but positional (equal-count) matching cannot even align 4 vs 3 tokens.
    """
    N = 7
    gt = torch.zeros(N, M, 3)
    pred = torch.zeros(N, M, 3)

    # rep atom (slot 0) x-coords keyed by sym_token_index
    gt[0:4, 0, 0] = torch.tensor([0.0, 1, 2, 3])  # GT chain 0 @ origin, sym 0..3
    gt[4:7, 0, 0] = torch.tensor([100.0, 101, 102])  # GT chain 1 @ x=100, sym 0..2
    pred[0:4, 0, 0] = torch.tensor([100.0, 101, 102, 103])  # pred chain 0 ≈ GT chain 1 (+extra sym 3)
    pred[4:7, 0, 0] = torch.tensor([0.0, 1, 2])  # pred chain 1 ≈ GT chain 0

    atom_mask = torch.zeros(N, M, dtype=torch.bool)
    atom_mask[:, 0] = True
    asym_id = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    entity_id = torch.zeros(N, dtype=torch.long)
    sym_token_index = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    token_to_rep_atom = torch.zeros(N, dtype=torch.long)
    mol_type = torch.full((N,), const.MOL_PROTEIN, dtype=torch.long)
    seq_mask = torch.ones(N, dtype=torch.bool)
    return pred, gt, atom_mask, asym_id, entity_id, sym_token_index, token_to_rep_atom, mol_type, seq_mask


def test_unequal_crop_swap_recovered_with_sym_index():
    pred, gt, mask, asym, ent, sti, rep, mol, seq = _unequal_crop_swapped()
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq, sym_token_index=sti
    )
    assert assign == {0: 1, 1: 0}


def test_unequal_crop_skipped_without_sym_index():
    """Documents the gap: without sym_token_index the 4-vs-3 pair can't be
    matched (equal-count candidate filter) → identity fallback."""
    pred, gt, mask, asym, ent, _sti, rep, mol, seq = _unequal_crop_swapped()
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq
    )
    assert assign == {0: 0, 1: 1}


def test_apply_sym_index_reorders_and_masks_off_no_counterpart():
    pred, gt, mask, asym, ent, sti, rep, mol, seq = _unequal_crop_swapped()
    assign = compute_chain_permutation_tokens(
        pred, gt, mask, asym, ent, token_to_rep_atom=rep, mol_type=mol, seq_mask=seq, sym_token_index=sti
    )
    new_gt, new_mask = apply_chain_permutation_tokens(gt, mask, asym, assign, sym_token_index=sti)
    # pred chain 0 sym 0..2 pull GT chain 1 (x≈100..102); sym 3 has no GT
    # counterpart in chain 1 (only sym 0..2) → masked off.
    assert new_mask[0, 0] and new_mask[1, 0] and new_mask[2, 0]
    assert not new_mask[3, 0]  # sym 3 token: no aligned GT → mask zeroed
    assert torch.allclose(new_gt[0:3, 0, 0], torch.tensor([100.0, 101, 102]))
    # pred chain 1 sym 0..2 pull GT chain 0 (x≈0..2).
    assert torch.allclose(new_gt[4:7, 0, 0], torch.tensor([0.0, 1, 2]))
    assert new_mask[4, 0] and new_mask[5, 0] and new_mask[6, 0]
