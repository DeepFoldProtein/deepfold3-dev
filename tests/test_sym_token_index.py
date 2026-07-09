"""``sym_token_index`` — per-chain token ordinal shared across symmetric copies.

It is the cross-copy key that lets chain-permutation match partially-cropped
symmetric chains (gap-1). These tests check the tokenizer assigns it correctly
and that it survives cropping when copies are cut to unequal token counts.
"""

from __future__ import annotations

import numpy as np

from deepfold3.data.crop.cropper import Cropper
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    Input,
    Structure,
)

_ATOM_NAMES = ("N", "CA", "C")


def _homodimer_structure(res_per_chain: int = 3) -> Structure:
    """Two protein chains of one entity (sym_id 0 / 1), identical residues."""
    n_chains = 2
    apr = len(_ATOM_NAMES)
    total_res = n_chains * res_per_chain
    total_atoms = total_res * apr

    atoms = np.zeros(total_atoms, dtype=ATOM_DTYPE)
    residues = np.zeros(total_res, dtype=RESIDUE_DTYPE)
    chains = np.zeros(n_chains, dtype=CHAIN_DTYPE)

    ri = 0
    ai = 0
    for c in range(n_chains):
        for r in range(res_per_chain):
            for k, nm in enumerate(_ATOM_NAMES):
                atoms[ai + k]["x"] = float(c * 100 + r)
                atoms[ai + k]["y"] = float(k)
                atoms[ai + k]["z"] = 0.0
                atoms[ai + k]["element"] = 6
                atoms[ai + k]["is_present"] = True
                atoms[ai + k]["name"] = nm.encode()
            residues[ri]["name"] = b"ALA"
            residues[ri]["res_type"] = 0
            residues[ri]["mol_type"] = 0
            residues[ri]["res_id"] = r + 1  # same numbering on both copies
            residues[ri]["atom_idx"] = ai
            residues[ri]["atom_num"] = apr
            residues[ri]["is_standard"] = True
            residues[ri]["is_present"] = True
            ri += 1
            ai += apr

        chains[c]["name"] = f"C{c}".encode()
        chains[c]["mol_type"] = 0
        chains[c]["entity_id"] = 0  # SAME entity → symmetric copies
        chains[c]["sym_id"] = c
        chains[c]["asym_id"] = c
        chains[c]["res_idx"] = c * res_per_chain
        chains[c]["res_num"] = res_per_chain
        chains[c]["atom_idx"] = c * res_per_chain * apr
        chains[c]["atom_num"] = res_per_chain * apr

    return Structure(
        atoms=atoms,
        bonds=np.empty(0, dtype=BOND_DTYPE),
        residues=residues,
        chains=chains,
    )


def test_sym_token_index_resets_per_chain_and_shared_across_copies():
    tok = Tokenizer().tokenize(Input(structure=_homodimer_structure(3), paired_msa={}, unpaired_msa={}))
    sti = tok.tokens["sym_token_index"]
    asym = tok.tokens["asym_id"]
    # Per-chain 0-based ordinal, reset each chain → identical on both copies.
    assert list(sti) == [0, 1, 2, 0, 1, 2]
    assert list(asym) == [0, 0, 0, 1, 1, 1]
    # Corresponding tokens of the two copies share the key.
    assert list(sti[asym == 0]) == list(sti[asym == 1])


def test_sym_token_index_survives_unequal_crop():
    tok = Tokenizer().tokenize(Input(structure=_homodimer_structure(3), paired_msa={}, unpaired_msa={}))
    # Keep all of chain 0 (sym 0,1,2) but only sym 1,2 of chain 1 → unequal
    # counts, the exact gap-1 partial-crop case.
    cropped = Cropper()._reindex(tok, np.array([0, 1, 2, 4, 5], dtype=np.int64))
    sti = cropped.tokens["sym_token_index"]
    asym = cropped.tokens["asym_id"]
    assert list(sti) == [0, 1, 2, 1, 2]
    assert list(asym) == [0, 0, 0, 1, 1]
    # Intersection of surviving sym indices (what chain-perm will match on).
    common = set(sti[asym == 0].tolist()) & set(sti[asym == 1].tolist())
    assert common == {1, 2}
