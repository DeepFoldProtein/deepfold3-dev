"""Tests for ligand inference pipeline.

Tests CCD molecule loading, SMILES conversion, JSON input parsing,
structure building, and featurization with ligand chains.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

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

    # Check frames_mask: protein tokens True, ligand tokens False
    assert feats["frames_mask"][:3].all()
    assert not feats["frames_mask"][3:].any()

    # Check ref_pos for ligand tokens has non-zero positions
    assert feats["ref_pos"][3, 0].abs().sum() > 0, "Ligand token should have non-zero ref_pos"
