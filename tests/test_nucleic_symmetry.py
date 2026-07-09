"""Nucleic-acid phosphate symmetry (OP1/OP2) via the residue-symmetry table.

The two non-bridging phosphate oxygens are chemically equivalent — a true 180°
symmetry that DF3's protein-only table used to miss (OF3 catches it via RDKit
automorphism). These check the token-level swap emitter now handles nucleotides
under both the v3 (OP1/OP2) and legacy v2 (O1P/O2P) naming.
"""

from __future__ import annotations

import numpy as np

from deepfold3.data import const
from deepfold3.data.symmetry.atom_perms import (
    RESIDUE_ATOM_SYMMETRY_GROUPS,
    compute_token_atom_symmetry_pairs,
)
from deepfold3.data.types import ATOM_DTYPE, BOND_DTYPE, CHAIN_DTYPE, RESIDUE_DTYPE, TOKEN_DTYPE, Structure


def _nucleotide_structure_and_tokens(atom_names, res_type):
    n = len(atom_names)
    atoms = np.zeros(n, dtype=ATOM_DTYPE)
    for i, nm in enumerate(atom_names):
        atoms[i]["name"] = nm.encode()
        atoms[i]["element"] = 8 if nm.startswith("O") else 6
        atoms[i]["is_present"] = True
    residues = np.zeros(1, dtype=RESIDUE_DTYPE)
    residues[0]["name"] = const.RES_TYPE_INDEX_TO_NAME[res_type].encode()
    residues[0]["res_type"] = res_type
    residues[0]["atom_idx"] = 0
    residues[0]["atom_num"] = n
    residues[0]["is_present"] = True
    chains = np.zeros(1, dtype=CHAIN_DTYPE)
    chains[0]["name"] = b"A"
    chains[0]["asym_id"] = 0
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 1
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n
    struct = Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)

    toks = np.zeros(1, dtype=TOKEN_DTYPE)
    toks[0]["token_idx"] = 0
    toks[0]["res_type"] = res_type
    toks[0]["mol_type"] = const.MOL_RNA if res_type in (22, 23, 24, 25) else const.MOL_DNA
    toks[0]["asym_id"] = 0
    toks[0]["atom_idx"] = 0
    toks[0]["atom_num"] = n
    return struct, toks


def test_table_covers_nucleic():
    for res in ("A", "G", "C", "U", "DA", "DG", "DC", "DT"):
        assert res in RESIDUE_ATOM_SYMMETRY_GROUPS


def test_op1_op2_swap_emitted_v3_naming():
    # A (res_type 22) with atoms P, OP1, OP2, O5', C5'
    struct, toks = _nucleotide_structure_and_tokens(["P", "OP1", "OP2", "O5'", "C5'"], res_type=22)
    pairs = compute_token_atom_symmetry_pairs(struct, toks)
    got = {tuple(int(x) for x in r) for r in pairs}
    assert got == {(0, 1, 2)}  # token 0, slots of OP1 and OP2


def test_o1p_o2p_swap_emitted_legacy_naming():
    # DG (res_type 27) with legacy v2 phosphate names O1P/O2P
    struct, toks = _nucleotide_structure_and_tokens(["P", "O1P", "O2P", "O5'", "C5'"], res_type=27)
    pairs = compute_token_atom_symmetry_pairs(struct, toks)
    got = {tuple(int(x) for x in r) for r in pairs}
    assert got == {(0, 1, 2)}


def test_no_phosphate_no_pair():
    # 5'-terminal without a phosphate → nothing to swap.
    struct, toks = _nucleotide_structure_and_tokens(["O5'", "C5'", "C4'", "O4'"], res_type=22)
    pairs = compute_token_atom_symmetry_pairs(struct, toks)
    assert len(pairs) == 0
