"""Chain-permutation alignment for the diffusion MSE loss (AF3 §4.2).

Covers the token-layout entry point ``compute_chain_permutation_tokens`` and
the fixes from the OF3 audit:
  * representative (token-center) atom alignment (품질 3),
  * ligand-only entities deprioritised as the anchor (품질 2),
  * ``seq_mask`` padding filter so zero-padded ``asym_id == 0`` tokens do not
    pollute chain 0's block.
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
