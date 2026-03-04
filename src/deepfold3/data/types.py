"""Data types for the DeepFold3 data pipeline.

Numpy structured-array dtypes + Python dataclasses.
Structure follows Boltz; dtypes follow plan spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ATOM_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("element", np.int8),  # atomic number, clipped to [0, 127]
        ("charge", np.float16),  # formal charge
        ("name", "S4"),  # atom name, max 4 chars
        ("is_present", np.bool_),  # resolved in structure
    ]
)

BOND_DTYPE = np.dtype(
    [
        ("atom_1", np.int32),
        ("atom_2", np.int32),
        ("type", np.int8),  # bond order: 1=single,2=double,3=triple,4=aromatic
    ]
)

RESIDUE_DTYPE = np.dtype(
    [
        ("name", "S5"),  # residue name (3-letter AA or nucleic code)
        ("res_type", np.int16),  # index in RES_TYPE_ORDER (0–30)
        ("mol_type", np.int8),  # MOL_PROTEIN/DNA/RNA/NONPOLYMER
        ("res_id", np.int32),  # PDB sequence number
        ("atom_idx", np.int32),  # index of first atom in atoms array
        ("atom_num", np.int16),  # number of atoms
        ("is_standard", np.bool_),  # is a standard residue
        ("is_present", np.bool_),  # has resolved coordinates
    ]
)

CHAIN_DTYPE = np.dtype(
    [
        ("name", "S4"),  # chain name (PDB chain ID)
        ("mol_type", np.int8),
        ("entity_id", np.int32),  # entity group (same sequence → same entity)
        ("sym_id", np.int32),  # 0-based copy index within entity
        ("asym_id", np.int32),  # unique sequential chain index in this structure
        ("res_idx", np.int32),  # index of first residue in residues array
        ("res_num", np.int32),  # number of residues
        ("atom_idx", np.int32),  # index of first atom in atoms array
        ("atom_num", np.int32),  # total atoms in chain
    ]
)

TOKEN_DTYPE = np.dtype(
    [
        ("token_idx", np.int32),  # sequential token index
        ("residue_index", np.int32),  # original PDB residue number (or atom index for ligands)
        ("res_type", np.int16),  # index in RES_TYPE_ORDER
        ("mol_type", np.int8),
        ("asym_id", np.int32),
        ("entity_id", np.int32),
        ("sym_id", np.int32),
        ("atom_idx", np.int32),  # index in atoms array of this token's atoms
        ("atom_num", np.int16),  # number of atoms belonging to this token
        ("center_x", np.float32),  # representative atom x (for cropping)
        ("center_y", np.float32),
        ("center_z", np.float32),
    ]
)

TOKEN_BOND_DTYPE = np.dtype(
    [
        ("token_1", np.int32),
        ("token_2", np.int32),
    ]
)


@dataclass
class Structure:
    """Parsed biomolecular structure."""

    atoms: np.ndarray  # ATOM_DTYPE
    bonds: np.ndarray  # BOND_DTYPE
    residues: np.ndarray  # RESIDUE_DTYPE
    chains: np.ndarray  # CHAIN_DTYPE

    def save(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            atoms=self.atoms,
            bonds=self.bonds,
            residues=self.residues,
            chains=self.chains,
        )

    @classmethod
    def load(cls, path: Path) -> Structure:
        data = np.load(path, allow_pickle=False)
        return cls(
            atoms=data["atoms"].view(ATOM_DTYPE),
            bonds=data["bonds"].view(BOND_DTYPE),
            residues=data["residues"].view(RESIDUE_DTYPE),
            chains=data["chains"].view(CHAIN_DTYPE),
        )


@dataclass
class MSA:
    """Multiple sequence alignment for one chain.

    sequences: int8 [S, N]  — MSA vocabulary indices
    mask:      bool [S, N]  — valid positions (False = gap inserted by alignment)
    deletion_matrix: int8 [S, N]  — number of insertions before position i
    """

    sequences: np.ndarray  # int8 [S, N]
    mask: np.ndarray  # bool [S, N]
    deletion_matrix: np.ndarray  # int8 [S, N]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            sequences=self.sequences,
            mask=self.mask,
            deletion_matrix=self.deletion_matrix,
        )

    @classmethod
    def load(cls, path: Path) -> MSA:
        data = np.load(path, allow_pickle=False)
        return cls(
            sequences=data["sequences"],
            mask=data["mask"],
            deletion_matrix=data["deletion_matrix"],
        )


@dataclass
class Input:
    """Input to the tokenizer: structure + per-chain MSAs.

    For multimer MSA pairing, paired and unpaired MSAs are stored separately.
    Paired MSAs have cross-chain co-evolutionary signal: row i from each chain's
    paired MSA corresponds to the same species and should be placed on the same
    full-width MSA row. Unpaired MSAs are chain-specific and go into
    block-diagonal rows.
    """

    structure: Structure
    paired_msa: dict[int, MSA]  # keyed by asym_id — paired (cross-chain) MSA
    unpaired_msa: dict[int, MSA]  # keyed by asym_id — unpaired (chain-specific) MSA

    @classmethod
    def from_msa(cls, structure: Structure, msa: dict[int, MSA]) -> Input:
        """Convenience constructor: treat all MSAs as unpaired (single-chain compat)."""
        return cls(structure=structure, paired_msa={}, unpaired_msa=msa)


@dataclass(frozen=True)
class Tokenized:
    """Output of tokenization / after cropping."""

    tokens: np.ndarray  # TOKEN_DTYPE [N_tokens]
    bonds: np.ndarray  # TOKEN_BOND_DTYPE [N_bonds]
    structure: Structure
    paired_msa: dict[int, MSA]  # keyed by asym_id
    unpaired_msa: dict[int, MSA]  # keyed by asym_id


@dataclass
class ChainInfo:
    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: str | int
    msa_id: str | int
    num_residues: int
    entity_id: str | int | None = None


@dataclass
class InterfaceInfo:
    chain_1: int
    chain_2: int


@dataclass
class Record:
    """Manifest record: pointer to stored structure + MSA files."""

    id: str
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
