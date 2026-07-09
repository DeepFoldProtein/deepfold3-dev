"""Whole-permutation ligand symmetry resolution (AF3 §4.2 / OF3 parity).

The pairwise ``apply_ligand_token_swap`` only handles 2-cycles; these cover the
new ``compute_ligand_token_permutations`` / ``apply_ligand_token_permutation``
path that scores each full CCD automorphism jointly and so resolves 3-fold+
symmetries (e.g. PO4's three equivalent oxygens).
"""

from __future__ import annotations

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.symmetry.atom_perms import (
    apply_ligand_token_permutation,
    compute_ligand_token_permutations,
)
from deepfold3.data.types import ATOM_DTYPE, BOND_DTYPE, CHAIN_DTYPE, RESIDUE_DTYPE, TOKEN_DTYPE, Structure

# --- apply: the OF3-parity win -------------------------------------------------


def _lig3(pred_x, gt_x=(0.0, 1.0, 2.0)):
    """One 3-atom ligand (3 single-atom tokens) at slot 0, on the x-axis."""
    N, M = 3, 1
    gt = torch.zeros(1, N, M, 3)
    pred = torch.zeros(1, N, M, 3)
    gt[0, :, 0, 0] = torch.tensor(gt_x)
    pred[0, :, 0, 0] = torch.tensor(pred_x)
    mask = torch.ones(N, M, dtype=torch.bool)
    return pred, gt, mask


# Full S3 non-identity permutations of 3 atoms, as moved-atom rows
# (ligand_id, perm_id, dst_token, src_token). gt_new[dst] = gt[src].
_S3_ROWS = np.array(
    [
        (0, 0, 0, 1), (0, 0, 1, 2), (0, 0, 2, 0),  # 3-cycle (0 1 2)
        (0, 1, 0, 2), (0, 1, 1, 0), (0, 1, 2, 1),  # 3-cycle (0 2 1)
        (0, 2, 0, 1), (0, 2, 1, 0),                # swap (0 1)
        (0, 3, 1, 2), (0, 3, 2, 1),                # swap (1 2)
        (0, 4, 0, 2), (0, 4, 2, 0),                # swap (0 2)
    ],
    dtype=np.int32,
)


def test_apply_recovers_3cycle():
    # pred is GT rotated by the 3-cycle (0->1->2->0): pred[k] == gt[perm[k]].
    pred, gt, mask = _lig3(pred_x=(1.0, 2.0, 0.0))
    out = apply_ligand_token_permutation(pred, gt, mask, _S3_ROWS)
    # The winning permutation reorders GT to exactly match pred (cost 0) — a
    # single transposition (all the pairwise path could do) cannot.
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([1.0, 2.0, 0.0]))


def test_apply_identity_when_already_aligned():
    pred, gt, mask = _lig3(pred_x=(0.0, 1.0, 2.0))  # pred == gt
    out = apply_ligand_token_permutation(pred, gt, mask, _S3_ROWS)
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_apply_recovers_single_transposition():
    # Only atoms 1,2 swapped → the (1 2) transposition wins.
    pred, gt, mask = _lig3(pred_x=(0.0, 2.0, 1.0))
    out = apply_ligand_token_permutation(pred, gt, mask, _S3_ROWS)
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([0.0, 2.0, 1.0]))


def test_apply_skips_permutation_with_unresolved_atom():
    # A swap that would win, but one atom is unresolved → permutation skipped,
    # GT left unchanged (never write a garbage coord into a resolved slot).
    N, M = 2, 1
    gt = torch.zeros(1, N, M, 3)
    gt[0, :, 0, 0] = torch.tensor([0.0, 5.0])
    pred = torch.zeros(1, N, M, 3)
    pred[0, :, 0, 0] = torch.tensor([5.0, 0.0])  # swapped → swap would win
    mask = torch.tensor([[True], [False]])  # atom 1 unresolved
    table = np.array([(0, 0, 0, 1), (0, 0, 1, 0)], dtype=np.int32)
    out = apply_ligand_token_permutation(pred, gt, mask, table)
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([0.0, 5.0]))  # unchanged


def test_apply_empty_and_padding_only():
    pred, gt, mask = _lig3(pred_x=(1.0, 2.0, 0.0))
    assert torch.allclose(apply_ligand_token_permutation(pred, gt, mask, np.zeros((0, 4), np.int32)), gt)
    pad = np.zeros((4, 4), dtype=np.int32)  # collate zero-padding (dst == src)
    assert torch.allclose(apply_ligand_token_permutation(pred, gt, mask, pad), gt)


# --- compute: moved-atom extraction + crop restriction -------------------------


def _one_ligand_structure(n_atoms: int, ccd: str):
    atoms = np.zeros(n_atoms, dtype=ATOM_DTYPE)
    atoms["is_present"] = True
    atoms["element"] = 6
    residues = np.zeros(1, dtype=RESIDUE_DTYPE)
    residues[0]["name"] = ccd.encode()
    residues[0]["atom_idx"] = 0
    residues[0]["atom_num"] = n_atoms
    residues[0]["is_present"] = True
    chains = np.zeros(1, dtype=CHAIN_DTYPE)
    chains[0]["name"] = b"L"
    chains[0]["mol_type"] = const.MOL_NONPOLYMER
    chains[0]["asym_id"] = 0
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 1
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_atoms
    return Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)


def _ligand_tokens(local_indices):
    toks = np.zeros(len(local_indices), dtype=TOKEN_DTYPE)
    for i, li in enumerate(local_indices):
        toks[i]["token_idx"] = i
        toks[i]["mol_type"] = const.MOL_NONPOLYMER
        toks[i]["asym_id"] = 0
        toks[i]["atom_num"] = 1
        toks[i]["atom_in_res"] = li
    return toks


def test_compute_moved_atoms_only_identity_omitted():
    struct = _one_ligand_structure(3, "XYZ")
    toks = _ligand_tokens([0, 1, 2])
    cache = {"XYZ": [[0, 1, 2], [1, 2, 0], [0, 2, 1]]}  # identity + 3-cycle + (1 2) swap
    rows = compute_ligand_token_permutations(struct, toks, cache)
    got = {tuple(int(x) for x in r) for r in rows}
    expected = {
        (0, 1, 0, 1), (0, 1, 1, 2), (0, 1, 2, 0),  # 3-cycle, all 3 moved
        (0, 2, 1, 2), (0, 2, 2, 1),                # (1 2) swap, atoms 1,2 moved
    }
    assert got == expected  # identity (perm_id 0) emits no rows


def test_compute_drops_permutation_touching_cropped_atom():
    # Atom 2 fell outside the crop → any permutation moving it is invalid.
    struct = _one_ligand_structure(3, "XYZ")
    toks = _ligand_tokens([0, 1])  # only atoms 0,1 survive
    cache = {"XYZ": [[0, 1, 2], [1, 2, 0], [1, 0, 2]]}  # 3-cycle touches atom 2; (0 1) swap does not
    rows = compute_ligand_token_permutations(struct, toks, cache)
    got = {tuple(int(x) for x in r) for r in rows}
    assert got == {(0, 2, 0, 1), (0, 2, 1, 0)}  # only the (0 1) swap survives
