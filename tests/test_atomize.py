"""Tests for stochastic atomization of standard residues (training regularizer).

Covers the ``atomize_tokens`` transform (token expansion, msa/mask expansion,
bond remap + CCD intra-residue bond synthesis), the inter-token symmetry
emitter, and end-to-end featurization of atomized tokens (single-atom-polymer
path — which also fixes the modified-residue ``ref_pos`` bug).
"""

from __future__ import annotations

import numpy as np

from deepfold3.data import const
from deepfold3.data.crop.cropper import Cropper
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.masking import MASK_REGION, TokenMask
from deepfold3.data.symmetry.atom_perms import compute_atomized_residue_token_swap_pairs
from deepfold3.data.tokenize.atomize import atomize_tokens, select_residues_to_atomize
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    Input,
    Structure,
)

# ALA: N CA C O CB ; ASP: N CA C O CB CG OD1 OD2
_ALA_ATOMS = [("N", 7), ("CA", 6), ("C", 6), ("O", 8), ("CB", 6)]
_ASP_ATOMS = [("N", 7), ("CA", 6), ("C", 6), ("O", 8), ("CB", 6), ("CG", 6), ("OD1", 8), ("OD2", 8)]


def _build_ala_asp_structure() -> Structure:
    """A one-chain protein with a standard ALA then a standard ASP residue."""
    specs = [("ALA", const.RES_TYPE_ORDER["ALA"], _ALA_ATOMS), ("ASP", const.RES_TYPE_ORDER["ASP"], _ASP_ATOMS)]
    total = sum(len(a) for _, _, a in specs)
    atoms = np.zeros(total, dtype=ATOM_DTYPE)
    residues = np.zeros(len(specs), dtype=RESIDUE_DTYPE)

    ai = 0
    rng = np.random.default_rng(0)
    for ri, (name, rtype, atom_list) in enumerate(specs):
        residues[ri]["name"] = name.encode()
        residues[ri]["res_type"] = rtype
        residues[ri]["mol_type"] = const.MOL_PROTEIN
        residues[ri]["res_id"] = ri + 1
        residues[ri]["atom_idx"] = ai
        residues[ri]["atom_num"] = len(atom_list)
        residues[ri]["is_standard"] = True
        residues[ri]["is_present"] = True
        for aname, elem in atom_list:
            atoms[ai]["name"] = aname.encode()
            atoms[ai]["element"] = elem
            atoms[ai]["is_present"] = True
            # distinct, non-colinear coords so frames / centering are well-posed
            atoms[ai]["x"], atoms[ai]["y"], atoms[ai]["z"] = rng.standard_normal(3) + ri * 5.0
            ai += 1

    chains = np.zeros(1, dtype=CHAIN_DTYPE)
    chains[0]["name"] = b"A"
    chains[0]["mol_type"] = const.MOL_PROTEIN
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = len(specs)
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = total
    return Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)


def _tokenize(struct: Structure):
    return Tokenizer().tokenize(Input(structure=struct, paired_msa={}, unpaired_msa={}))


def test_atomize_expands_selected_residue():
    tok = _tokenize(_build_ala_asp_structure())
    assert len(tok.tokens) == 2  # 1 ALA token + 1 ASP token
    assert list(tok.msa_col_idx) == [0, 1]

    out = atomize_tokens(tok, np.array([1]))  # atomize ASP (token 1)

    # 1 (ALA) + 8 (ASP atoms) = 9 tokens
    assert len(out.tokens) == 9
    asp = out.tokens[1:]
    assert np.all(asp["atom_num"] == 1)
    # res_type keeps the real ASP identity (not UNK)
    assert np.all(asp["res_type"] == const.RES_TYPE_ORDER["ASP"])
    # residue_index shared across the residue's atom tokens
    assert len(np.unique(asp["residue_index"])) == 1
    assert asp[0]["residue_index"] == 2
    # token_idx is sequential 0..8
    assert list(out.tokens["token_idx"]) == list(range(9))
    # MSA columns: ALA=0, all 8 ASP atoms share col 1
    assert list(out.msa_col_idx) == [0, 1, 1, 1, 1, 1, 1, 1, 1]
    # atoms untouched (structure shared)
    assert out.structure is tok.structure


def test_atomize_synthesizes_ccd_intra_residue_bonds():
    tok = _tokenize(_build_ala_asp_structure())
    out = atomize_tokens(tok, np.array([1]))
    # ASP intra bonds present (N-CA, CA-C, C-O, CA-CB, CB-CG, CG-OD1, CG-OD2);
    # OXT bond is skipped because the residue has no OXT atom.
    assert len(out.bonds) == 7
    # Every bond is between two ASP atom tokens (indices 1..8), slot 0.
    for b in out.bonds:
        assert 1 <= int(b["token_1"]) <= 8 and 1 <= int(b["token_2"]) <= 8
        assert int(b["atom_1"]) == 0 and int(b["atom_2"]) == 0


def test_atomize_expands_mask_channels():
    tok = _tokenize(_build_ala_asp_structure())
    region = np.array([0, 5], dtype=np.int16)  # ASP labelled region 5
    tok = tok.with_mask(TokenMask({MASK_REGION: region}))
    out = atomize_tokens(tok, np.array([1]))
    expanded = out.token_mask.get(MASK_REGION)
    assert list(expanded) == [0, 5, 5, 5, 5, 5, 5, 5, 5]


def test_atomize_empty_is_identity():
    tok = _tokenize(_build_ala_asp_structure())
    out = atomize_tokens(tok, np.array([], dtype=np.int64))
    assert out is tok


def test_select_region_source_only_picks_labelled():
    tok = _tokenize(_build_ala_asp_structure())
    region = np.array([0, 5], dtype=np.int16)
    tok = tok.with_mask(TokenMask({MASK_REGION: region}))
    rng = np.random.default_rng(0)
    picked = select_residues_to_atomize(tok, rng, prob=1.0, source="region")
    assert list(picked) == [1]  # only the ASP (region>=1) residue


def test_symmetry_emitter_pairs_asp_carboxyl_oxygens():
    tok = _tokenize(_build_ala_asp_structure())
    out = atomize_tokens(tok, np.array([1]))
    pairs = compute_atomized_residue_token_swap_pairs(out.structure, out.tokens)
    assert len(pairs) == 1
    # The pair must be OD1 / OD2 token indices.
    names = {out.structure.atoms[int(out.tokens[t]["atom_idx"])]["name"].decode().strip() for t in pairs[0]}
    assert names == {"OD1", "OD2"}


def test_featurize_atomized_keeps_tokens_in_masks():
    tok = _tokenize(_build_ala_asp_structure())
    out = atomize_tokens(tok, np.array([1]))
    feats = Featurizer(max_msa_seqs=8).process(out, training=True, random_rot=False)

    N = len(out.tokens)
    assert feats["res_type"].shape[0] == N
    # Atomized tokens must retain a valid representative atom → seq_mask True.
    assert bool(feats["seq_mask"].all())
    # Rep atom for atomized tokens is slot 0.
    assert list(feats["token_to_rep_atom"][1:].tolist()) == [0] * 8
    # Single-atom polymer tokens: only ref slot 0 is present.
    ref_mask = feats["ref_mask"].numpy()
    assert ref_mask[1:, 0].all()
    assert not ref_mask[1:, 1:].any()
    # GT positions for atomized atoms live at slot 0.
    gt_mask = feats["atom_gt_mask"].numpy()
    assert gt_mask[1:, 0].all()
    assert not gt_mask[1:, 1:].any()
    # Frames off for single-atom tokens (no local rigid frame).
    assert not bool(feats["frames_mask"][1:].any())
    # Symmetric ASP oxygens surfaced into the (ligand) swap-pair channel.
    assert len(feats["ligand_swap_pairs"]) == 1


def _build_ala_plus_modified_structure() -> Structure:
    """One standard ALA + one *modified* (non-standard) residue of 3 atoms.

    The tokenizer atomizes the modified residue (1 token/atom). Before the fix,
    each single-atom token wrongly received the whole residue's reference
    template; it must now carry only its own atom in ref slot 0.
    """
    ala = ("ALA", const.RES_TYPE_ORDER["ALA"], _ALA_ATOMS, True)
    mod = ("SEP", 20, [("N", 7), ("CA", 6), ("CB", 6)], False)
    specs = [ala, mod]
    total = sum(len(a) for _, _, a, _ in specs)
    atoms = np.zeros(total, dtype=ATOM_DTYPE)
    residues = np.zeros(len(specs), dtype=RESIDUE_DTYPE)
    ai = 0
    rng = np.random.default_rng(1)
    for ri, (name, rtype, atom_list, is_std) in enumerate(specs):
        residues[ri]["name"] = name.encode()
        residues[ri]["res_type"] = rtype
        residues[ri]["mol_type"] = const.MOL_PROTEIN
        residues[ri]["res_id"] = ri + 1
        residues[ri]["atom_idx"] = ai
        residues[ri]["atom_num"] = len(atom_list)
        residues[ri]["is_standard"] = is_std
        residues[ri]["is_present"] = True
        for aname, elem in atom_list:
            atoms[ai]["name"] = aname.encode()
            atoms[ai]["element"] = elem
            atoms[ai]["is_present"] = True
            atoms[ai]["x"], atoms[ai]["y"], atoms[ai]["z"] = rng.standard_normal(3) + ri * 5.0
            ai += 1
    chains = np.zeros(1, dtype=CHAIN_DTYPE)
    chains[0]["name"] = b"A"
    chains[0]["mol_type"] = const.MOL_PROTEIN
    chains[0]["res_num"] = len(specs)
    chains[0]["atom_num"] = total
    return Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)


def test_modified_residue_refpos_bug_fixed():
    """Regression: a modified-residue atom token gets only its own atom in ref."""
    tok = _tokenize(_build_ala_plus_modified_structure())
    # ALA token (1) + 3 atomized modified-residue tokens = 4 tokens.
    assert len(tok.tokens) == 4
    feats = Featurizer(max_msa_seqs=8).process(tok, training=True, random_rot=False)
    ref_mask = feats["ref_mask"].numpy()
    # Each modified-residue single-atom token: only ref slot 0 present
    # (previously the whole-residue template filled multiple slots).
    assert ref_mask[1:, 0].all()
    assert not ref_mask[1:, 1:].any()
    # And they stay in the masks with rep atom = slot 0.
    assert bool(feats["seq_mask"].all())
    assert list(feats["token_to_rep_atom"][1:].tolist()) == [0, 0, 0]


def test_featurize_baseline_unchanged_without_atomize():
    """atomize is opt-in: an un-atomized structure featurizes as 2 residue tokens."""
    tok = _tokenize(_build_ala_asp_structure())
    feats = Featurizer(max_msa_seqs=8).process(tok, training=True, random_rot=False)
    assert feats["res_type"].shape[0] == 2
    assert bool(feats["seq_mask"].all())
    # ASP (residue token) rep atom = CB (slot 4).
    assert int(feats["token_to_rep_atom"][1]) == 4


def _build_ala_chain_structure(n_res: int, n_chains: int = 2) -> Structure:
    """``n_res`` ALA residues distributed over ``n_chains`` (for crop tests)."""
    atoms = np.zeros(n_res * len(_ALA_ATOMS), dtype=ATOM_DTYPE)
    residues = np.zeros(n_res, dtype=RESIDUE_DTYPE)
    rng = np.random.default_rng(2)
    chain_of = np.linspace(0, n_chains, n_res, endpoint=False).astype(int)
    ai = 0
    for ri in range(n_res):
        residues[ri]["name"] = b"ALA"
        residues[ri]["res_type"] = const.RES_TYPE_ORDER["ALA"]
        residues[ri]["mol_type"] = const.MOL_PROTEIN
        residues[ri]["res_id"] = ri + 1
        residues[ri]["atom_idx"] = ai
        residues[ri]["atom_num"] = len(_ALA_ATOMS)
        residues[ri]["is_standard"] = True
        residues[ri]["is_present"] = True
        for aname, elem in _ALA_ATOMS:
            atoms[ai]["name"] = aname.encode()
            atoms[ai]["element"] = elem
            atoms[ai]["is_present"] = True
            atoms[ai]["x"], atoms[ai]["y"], atoms[ai]["z"] = rng.standard_normal(3) + ri * 3.0
            ai += 1
    chains = np.zeros(n_chains, dtype=CHAIN_DTYPE)
    for c in range(n_chains):
        m = chain_of == c
        f, ln = int(np.where(m)[0][0]), int(np.where(m)[0][-1])
        chains[c]["name"] = f"C{c}".encode()
        chains[c]["mol_type"] = const.MOL_PROTEIN
        chains[c]["entity_id"] = c
        chains[c]["asym_id"] = c
        chains[c]["res_idx"] = f
        chains[c]["res_num"] = ln - f + 1
        chains[c]["atom_idx"] = int(residues[f]["atom_idx"])
        chains[c]["atom_num"] = (ln - f + 1) * len(_ALA_ATOMS)
    # asym_id per residue is derived from its chain; the tokenizer reads it from
    # the chain rows, so set residue mol_type only (chain drives asym).
    return Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)


def test_crop_reindexes_residue_hierarchy():
    """Regression: cropping re-indexes residues/chains to the cropped atoms.

    Before the fix, ``_reindex`` kept the full residues/chains arrays with
    original offsets while atom indices were remapped, so ``_find_res_name``
    resolved a cropped residue to the WRONG name (kept ASP → "ALA").
    """
    from deepfold3.data.feature.featurizer import Featurizer, _find_res_name

    tok = _tokenize(_build_ala_asp_structure())
    # Keep only the ASP residue (token 1); its atoms remap to positions 0..7.
    cropped = Cropper()._reindex(tok, np.array([1]))

    names = [r["name"].decode().strip() for r in cropped.structure.residues]
    assert names == ["ASP"]
    # residues / chains atom ranges are consistent with the cropped atoms.
    assert int(cropped.structure.residues[0]["atom_idx"]) == 0
    assert int(cropped.structure.residues[0]["atom_num"]) == len(_ASP_ATOMS)
    assert int(cropped.structure.chains[0]["res_idx"]) == 0
    assert int(cropped.structure.chains[0]["res_num"]) == 1
    assert int(cropped.structure.chains[0]["atom_num"]) == len(_ASP_ATOMS)
    # _find_res_name now resolves the correct residue for the token's atom.
    assert _find_res_name(cropped, int(cropped.tokens[0]["atom_idx"])) == "ASP"

    # And featurization gives ASP's CB rep atom (slot 4), not ALA's.
    feats = Featurizer(max_msa_seqs=8).process(cropped, training=True, random_rot=False)
    assert int(feats["token_to_rep_atom"][0]) == 4
    # ASP's full 8-atom reference template is present (5 for ALA would be wrong).
    assert int(feats["ref_mask"][0].sum()) == len(_ASP_ATOMS)


def test_full_pipeline_atomize_then_crop_then_featurize():
    """Real _load_sample ordering: tokenize → atomize → crop → featurize."""
    struct = _build_ala_chain_structure(n_res=12, n_chains=2)
    tok = _tokenize(struct)
    assert len(tok.tokens) == 12

    rng = np.random.default_rng(0)
    picked = select_residues_to_atomize(tok, rng, prob=0.5, source="random")
    atomized = atomize_tokens(tok, picked)
    # Each atomized residue adds 4 tokens (5 atoms − 1 residue token).
    assert len(atomized.tokens) == 12 + 4 * len(picked)

    # Crop tighter than the atomized token count so _reindex actually runs over
    # a mix of residue + atom tokens.
    cropper = Cropper()
    cropped = cropper.crop(atomized, max_tokens=10, max_atoms=10_000, seed=0)
    assert len(cropped.tokens) <= 10
    # token_idx re-numbered contiguously, msa_col_idx aligned to token count.
    assert list(cropped.tokens["token_idx"]) == list(range(len(cropped.tokens)))
    assert len(cropped.msa_col_idx) == len(cropped.tokens)

    feats = Featurizer(max_msa_seqs=8).process(cropped, training=True, random_rot=True)
    assert feats["res_type"].shape[0] == len(cropped.tokens)
    assert bool(feats["seq_mask"].all())
