"""Tests for ligand inference pipeline.

Tests CCD molecule loading, SMILES conversion, JSON input parsing,
structure building, and featurization with ligand chains.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
import torch

from deepfold3.data import const

# ---------------------------------------------------------------------------
# CCD molecule builder tests
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_ccd_molecule_atp():
    """Test loading ATP from CCD (requires network)."""
    from deepfold3.data.ccd import load_ccd_molecule

    mol = load_ccd_molecule("ATP")
    assert len(mol.atom_names) > 20, f"ATP should have >20 heavy atoms, got {len(mol.atom_names)}"
    assert mol.positions.shape == (len(mol.atom_names), 3)
    assert len(mol.bonds) > 0, "ATP should have bonds"
    assert mol.res_name == "ATP"
    # All bond indices should be valid
    n = len(mol.atom_names)
    for a1, a2, order in mol.bonds:
        assert 0 <= a1 < n
        assert 0 <= a2 < n
        assert order in (1, 2, 3, 4)


def test_load_ccd_molecule_invalid():
    """Test that invalid CCD code raises ValueError."""
    from deepfold3.data.ccd import load_ccd_molecule

    with pytest.raises(ValueError):
        load_ccd_molecule("ZZZZNOTACODE")


def test_smiles_to_molecule():
    """Test SMILES conversion (requires RDKit)."""
    pytest.importorskip("rdkit")
    from deepfold3.data.ccd import smiles_to_molecule

    mol = smiles_to_molecule("CCO", res_name="ETH")
    assert len(mol.atom_names) == 3  # C, C, O (heavy atoms only)
    assert mol.positions.shape == (3, 3)
    assert mol.res_name == "ETH"
    assert len(mol.bonds) == 2  # C-C, C-O


def test_smiles_invalid():
    """Test invalid SMILES raises ValueError."""
    pytest.importorskip("rdkit")
    from deepfold3.data.ccd import smiles_to_molecule

    with pytest.raises(ValueError):
        smiles_to_molecule("this_is_not_smiles")


# ---------------------------------------------------------------------------
# JSON input parsing tests
# ---------------------------------------------------------------------------


def test_parse_protenix_ligand():
    """Test parsing Protenix-format JSON with ligand entry."""
    from deepfold3.data.parse.json_input import parse_json_input

    job_data = [
        {
            "name": "test_ligand",
            "sequences": [
                {"proteinChain": {"sequence": "MKTII", "count": 1}},
                {"ligand": {"ligand": "CCD_ATP", "count": 1}},
            ],
            "modelSeeds": [42],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(job_data, f)
        f.flush()
        jobs = parse_json_input(f.name)

    assert len(jobs) == 1
    job = jobs[0]
    assert len(job.protein_chains) == 1
    assert len(job.ligand_chains) == 1
    assert job.ligand_chains[0].ccd_ids == ["ATP"]


def test_parse_af3_ligand_ccd():
    """Test parsing AF3-format JSON with CCD ligand."""
    from deepfold3.data.parse.json_input import parse_json_input

    job_data = [
        {
            "name": "test_af3_ligand",
            "dialect": "alphafold3",
            "version": 2,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTII"}},
                {"ligand": {"id": "B", "ccdCodes": ["ATP"]}},
            ],
            "modelSeeds": [42],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(job_data, f)
        f.flush()
        jobs = parse_json_input(f.name)

    assert len(jobs) == 1
    job = jobs[0]
    assert len(job.protein_chains) == 1
    assert len(job.ligand_chains) == 1
    assert job.ligand_chains[0].ccd_ids == ["ATP"]


def test_parse_af3_ligand_smiles():
    """Test parsing AF3-format JSON with SMILES ligand."""
    from deepfold3.data.parse.json_input import parse_json_input

    job_data = [
        {
            "name": "test_smiles",
            "dialect": "alphafold3",
            "version": 2,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTII"}},
                {"ligand": {"id": "B", "smiles": "CCO"}},
            ],
            "modelSeeds": [42],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(job_data, f)
        f.flush()
        jobs = parse_json_input(f.name)

    job = jobs[0]
    assert len(job.ligand_chains) == 1
    assert job.ligand_chains[0].smiles == "CCO"


# ---------------------------------------------------------------------------
# Structure builder tests
# ---------------------------------------------------------------------------


def test_build_ligand_chain():
    """Test building a ligand chain from CcdMolecule."""
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.module.dataset import _build_ligand_chain_atoms_residues

    mol = CcdMolecule(
        atom_names=["C1", "C2", "O1"],
        positions=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        elements=["C", "C", "O"],
        charges=[0.0, 0.0, 0.0],
        bonds=[(0, 1, 1), (1, 2, 1)],
        res_name="ETH",
    )

    atoms, bonds, residues, n_atoms = _build_ligand_chain_atoms_residues(mol, atom_offset=10)
    assert n_atoms == 3
    assert len(atoms) == 3
    assert len(bonds) == 2
    assert len(residues) == 1
    # Check bond offsets
    assert bonds[0] == (10, 11, 1)
    assert bonds[1] == (11, 12, 1)
    # Check residue
    assert residues[0][2] == const.MOL_NONPOLYMER  # mol_type


def test_build_complex_with_ligands():
    """Test building a complex structure with both polymer and ligand chains."""
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.module.dataset import _build_complex_structure_with_ligands

    mol = CcdMolecule(
        atom_names=["C1", "O1"],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        elements=["C", "O"],
        charges=[0.0, 0.0],
        bonds=[(0, 1, 1)],
        res_name="CO",
    )

    structure = _build_complex_structure_with_ligands(
        chains=[("MKT", const.MOL_PROTEIN, "A")],
        ligand_chains=[(mol, "B")],
    )

    assert len(structure.chains) == 2
    # Check ligand chain
    lig_chain = structure.chains[1]
    assert int(lig_chain["mol_type"]) == const.MOL_NONPOLYMER
    assert int(lig_chain["atom_num"]) == 2
    # Check bonds exist
    assert len(structure.bonds) > 0


# ---------------------------------------------------------------------------
# Featurization test
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_featurize_protein_plus_ligand():
    """Test featurization of a protein + CCD ligand complex (requires network for CCD)."""
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.feature.featurizer import Featurizer
    from deepfold3.data.module.dataset import _build_complex_structure_with_ligands, _sequence_to_msa
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input

    # Build a small protein + simple ligand
    mol = CcdMolecule(
        atom_names=["C1", "O1"],
        positions=np.array([[1.0, 2.0, 3.0], [4.5, 5.0, 6.0]], dtype=np.float32),
        elements=["C", "O"],
        charges=[0.0, 0.0],
        bonds=[(0, 1, 1)],
        res_name="CO",
    )

    structure = _build_complex_structure_with_ligands(
        chains=[("AAA", const.MOL_PROTEIN, "A")],
        ligand_chains=[(mol, "B")],
    )

    msa = _sequence_to_msa("AAA", const.MOL_PROTEIN)
    inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa})

    tokenizer = Tokenizer()
    tokenized = tokenizer.tokenize(inp)

    featurizer = Featurizer(max_msa_seqs=16)
    feats = featurizer.process(tokenized, training=False, random_rot=False)

    # 3 protein tokens + 2 ligand tokens = 5
    N = feats["res_type"].shape[0]
    assert N == 5, f"Expected 5 tokens, got {N}"

    # Check ref_space_uid exists
    assert "ref_space_uid" in feats
    assert feats["ref_space_uid"].shape == (N, featurizer.max_atoms)

    # Ligand tokens should share the same ref_space_uid
    lig_uids = feats["ref_space_uid"][3:, 0]  # tokens 3 and 4
    assert lig_uids[0] == lig_uids[1], "Ligand tokens should share ref_space_uid"

    # Protein tokens should have independent ref_space_uid
    prot_uids = feats["ref_space_uid"][:3, 0]
    assert len(set(prot_uids.tolist())) == 3

    # Bond matrix should be present (ligand has bonds)
    assert "bond_matrix" in feats
    assert feats["bond_matrix"].shape == (N, N, 1)

    # Check frames_mask: protein tokens True; 2-atom ligand chain → still
    # False because a ligand frame needs ≥3 chain atoms (AF3 §4.3.2).
    assert feats["frames_mask"][:3].all()
    assert not feats["frames_mask"][3:].any()

    # frame_token_index should default to (i, i, i) for polymer tokens.
    assert (feats["frame_token_index"][:3] == torch.arange(3).unsqueeze(-1)).all()

    # Check ref_pos for ligand tokens has non-zero positions
    assert feats["ref_pos"][3, 0].abs().sum() > 0, "Ligand token should have non-zero ref_pos"


def _featurize_with_ligand(ligand_positions: np.ndarray, ligand_names: list[str]):
    """Helper: featurise a minimal (protein ``AAA`` + 3-atom ligand) complex.

    Returns the feature dict. Needs network for protein CCD (same as
    ``test_featurize_protein_plus_ligand``).
    """
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.feature.featurizer import Featurizer
    from deepfold3.data.module.dataset import _build_complex_structure_with_ligands, _sequence_to_msa
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input

    n = len(ligand_names)
    mol = CcdMolecule(
        atom_names=ligand_names,
        positions=ligand_positions.astype(np.float32),
        elements=["C"] * n,
        charges=[0.0] * n,
        bonds=[(i, i + 1, 1) for i in range(n - 1)],
        res_name="LIG",
    )
    structure = _build_complex_structure_with_ligands(
        chains=[("AAA", const.MOL_PROTEIN, "A")],
        ligand_chains=[(mol, "B")],
    )
    msa = _sequence_to_msa("AAA", const.MOL_PROTEIN)
    inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa})
    tokenized = Tokenizer().tokenize(inp)
    return Featurizer(max_msa_seqs=16).process(tokenized, training=False, random_rot=False)


@pytest.mark.network
def test_featurize_ligand_frame_triangle():
    """3-atom ligand in triangle → AF3 §4.3.2 frame is valid; per-token
    frame_token_index points to nearest / 2nd-nearest neighbours."""
    import torch

    # Triangle in 3D; token 0 (ligand offset 3) is at origin; distances 1, 1.414 respectively
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    feats = _featurize_with_ligand(pos, ["C1", "C2", "C3"])

    N = feats["res_type"].shape[0]
    assert N == 6  # 3 protein + 3 ligand

    # Protein tokens unchanged
    assert feats["frames_mask"][:3].all()
    assert (feats["frame_token_index"][:3] == torch.arange(3).unsqueeze(-1)).all()

    # All 3 ligand tokens get valid frames (non-collinear triangle).
    assert feats["frames_mask"][3:].all(), feats["frames_mask"]

    # For token i = 3 (C1 at origin): nearest = C2 (dist 1.0), 2nd = C3 (dist 1.0 also
    # — tie; either is acceptable, but both are ligand tokens, not the i itself).
    frame_tok = feats["frame_token_index"]
    # Middle column is always i itself (origin b of the frame).
    assert frame_tok[3, 1].item() == 3
    assert frame_tok[4, 1].item() == 4
    assert frame_tok[5, 1].item() == 5
    # a / c picks should be other ligand tokens in the same chain.
    for i in (3, 4, 5):
        a_tok, _, c_tok = frame_tok[i].tolist()
        assert a_tok in {3, 4, 5} and a_tok != i
        assert c_tok in {3, 4, 5} and c_tok != i
        assert a_tok != c_tok


@pytest.mark.network
def test_featurize_ligand_frame_collinear():
    """3 collinear atoms → frame is invalid (<25° deviation from colinear)."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    feats = _featurize_with_ligand(pos, ["C1", "C2", "C3"])

    # Protein tokens still have valid frames.
    assert feats["frames_mask"][:3].all()
    # All 3 ligand tokens rejected — collinearity check fires for each.
    assert not feats["frames_mask"][3:].any()


@pytest.mark.network
def test_featurize_single_atom_ion():
    """Single-atom ligand chain (ion-like) → frame invalid: <3 chain atoms."""
    pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    feats = _featurize_with_ligand(pos, ["NA"])

    assert feats["frames_mask"][:3].all()  # protein unchanged
    assert not feats["frames_mask"][3:].any()  # ion gets no frame


@pytest.mark.network
def test_featurize_ligand_frame_near_collinear_rejected():
    """Arrangement 24° off colinear is rejected; 26° off is accepted."""
    # 24°: just inside the 25° cutoff → rejected.
    ang_rad = np.radians(180.0 - 24.0)
    pos_24 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.cos(ang_rad), np.sin(ang_rad), 0.0],
        ],
        dtype=np.float32,
    )
    feats_24 = _featurize_with_ligand(pos_24, ["C1", "C2", "C3"])
    # Token 3 (at origin) sees neighbours along two directions within 24°
    # of anti-parallel → collinearity rejects this frame.
    assert not feats_24["frames_mask"][3].item()

    # 26°: outside the 25° cutoff → accepted.
    ang_rad = np.radians(180.0 - 26.0)
    pos_26 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.cos(ang_rad), np.sin(ang_rad), 0.0],
        ],
        dtype=np.float32,
    )
    feats_26 = _featurize_with_ligand(pos_26, ["C1", "C2", "C3"])
    assert feats_26["frames_mask"][3].item()
