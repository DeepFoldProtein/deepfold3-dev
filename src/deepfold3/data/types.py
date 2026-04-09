"""Data types for the DeepFold3 data pipeline.

Numpy structured-array dtypes + Python dataclasses.
Structure follows Boltz; dtypes follow plan spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deepfold3.data.masking import TokenMask

ATOM_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("element", np.int8),  # atomic number, clipped to [0, 127]
        ("charge", np.float16),  # formal charge
        ("name", "S4"),  # atom name, max 4 chars
        ("is_present", np.bool_),  # resolved in structure
        ("b_factor", np.float32),  # B-factor or pLDDT confidence
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
    name: str = field(default="unset")

    def save(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            atoms=self.atoms,
            bonds=self.bonds,
            residues=self.residues,
            chains=self.chains,
            name=np.array(self.name),
        )

    @classmethod
    def load(cls, path: Path) -> Structure:
        data = np.load(path, allow_pickle=False)
        atoms_raw = data["atoms"]
        # Backward compatibility: old files lack the b_factor field
        if atoms_raw.dtype != ATOM_DTYPE:
            old_atoms = atoms_raw.view(np.dtype([f for f in ATOM_DTYPE.descr if f[0] != "b_factor"]))
            new_atoms = np.zeros(len(old_atoms), dtype=ATOM_DTYPE)
            for field_name in old_atoms.dtype.names:
                new_atoms[field_name] = old_atoms[field_name]
            atoms_raw = new_atoms
        else:
            atoms_raw = atoms_raw.view(ATOM_DTYPE)
        name = str(data["name"]) if "name" in data else "unset"
        return cls(
            atoms=atoms_raw,
            bonds=data["bonds"].view(BOND_DTYPE),
            residues=data["residues"].view(RESIDUE_DTYPE),
            chains=data["chains"].view(CHAIN_DTYPE),
            name=name,
        )

    def save_split(self, directory: Path) -> None:
        """Save structure as split files: coords.npz + meta.json."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # coords.npz: x, y, z, b_factor
        np.savez_compressed(
            str(directory / "coords.npz"),
            x=self.atoms["x"],
            y=self.atoms["y"],
            z=self.atoms["z"],
            b_factor=self.atoms["b_factor"],
        )

        # meta.json: name + table data (excluding coordinates)
        def _bytes_to_str(v: object) -> object:
            if isinstance(v, bytes):
                return v.decode("ascii", errors="replace")
            if isinstance(v, np.generic):
                return v.item()
            return v

        def _row_to_dict(row: np.void) -> dict:
            return {name: _bytes_to_str(row[name]) for name in row.dtype.names}

        meta = {
            "name": self.name,
            "chains": [_row_to_dict(r) for r in self.chains],
            "residues": [_row_to_dict(r) for r in self.residues],
            "bonds": [_row_to_dict(r) for r in self.bonds],
            "atoms": [
                {
                    "element": int(a["element"]),
                    "charge": float(a["charge"]),
                    "name": a["name"].decode("ascii", errors="replace"),
                    "is_present": bool(a["is_present"]),
                }
                for a in self.atoms
            ],
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load_split(cls, directory: Path) -> Structure:
        """Load structure from split files: coords.npz + meta.json."""
        directory = Path(directory)
        coords = np.load(directory / "coords.npz", allow_pickle=False)
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        n_atoms = len(coords["x"])
        atoms = np.zeros(n_atoms, dtype=ATOM_DTYPE)
        atoms["x"] = coords["x"]
        atoms["y"] = coords["y"]
        atoms["z"] = coords["z"]
        atoms["b_factor"] = coords["b_factor"]
        atom_meta = meta["atoms"]
        if atom_meta:
            atoms["element"] = [a["element"] for a in atom_meta]
            atoms["charge"] = [a["charge"] for a in atom_meta]
            atoms["name"] = [a["name"].encode("ascii") for a in atom_meta]
            atoms["is_present"] = [a["is_present"] for a in atom_meta]

        chains = (
            np.array(
                [
                    tuple(c["name"].encode("ascii") if k == "name" else c[k] for k in CHAIN_DTYPE.names)
                    for c in meta["chains"]
                ],
                dtype=CHAIN_DTYPE,
            )
            if meta["chains"]
            else np.empty(0, dtype=CHAIN_DTYPE)
        )

        residues = (
            np.array(
                [
                    tuple(r["name"].encode("ascii") if k == "name" else r[k] for k in RESIDUE_DTYPE.names)
                    for r in meta["residues"]
                ],
                dtype=RESIDUE_DTYPE,
            )
            if meta["residues"]
            else np.empty(0, dtype=RESIDUE_DTYPE)
        )

        bonds = (
            np.array(
                [tuple(b[k] for k in BOND_DTYPE.names) for b in meta["bonds"]],
                dtype=BOND_DTYPE,
            )
            if meta["bonds"]
            else np.empty(0, dtype=BOND_DTYPE)
        )

        return cls(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            name=meta.get("name", "unset"),
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
    token_mask: TokenMask | None = None

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
    token_mask: TokenMask | None = None

    def with_mask(self, token_mask: TokenMask) -> Tokenized:
        """Return a copy with the given token mask attached."""
        from dataclasses import replace

        return replace(self, token_mask=token_mask)


@dataclass
class ChainInfo:
    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: str | int
    msa_id: str | int
    num_residues: int
    entity_id: str | int | None = None
    paired_msa_id: str | int | None = None  # separate file for paired (cross-chain) MSA
    valid: bool = True
    auth_chain_name: str | None = None  # original chain name from input CIF (before assembly renaming)


@dataclass
class InterfaceInfo:
    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass
class StructureInfo:
    """Metadata about the experimental structure."""

    resolution: float | None = None
    method: str | None = None
    deposited: str | None = None
    released: str | None = None
    num_chains: int | None = None
    num_interfaces: int | None = None


@dataclass
class Record:
    """Manifest record: pointer to stored structure + MSA files."""

    id: str
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    structure_info: StructureInfo | None = None

    def to_dict(self) -> dict:
        d: dict = {"id": self.id}
        d["chains"] = [
            {
                "chain_id": c.chain_id,
                "chain_name": c.chain_name,
                "mol_type": c.mol_type,
                "cluster_id": c.cluster_id,
                "msa_id": c.msa_id,
                "num_residues": c.num_residues,
                "entity_id": c.entity_id,
                "paired_msa_id": c.paired_msa_id,
                "valid": c.valid,
                **({"auth_chain_name": c.auth_chain_name} if c.auth_chain_name is not None else {}),
            }
            for c in self.chains
        ]
        d["interfaces"] = [{"chain_1": i.chain_1, "chain_2": i.chain_2, "valid": i.valid} for i in self.interfaces]
        if self.structure_info is not None:
            si = self.structure_info
            d["structure_info"] = {
                "resolution": si.resolution,
                "method": si.method,
                "deposited": si.deposited,
                "released": si.released,
                "num_chains": si.num_chains,
                "num_interfaces": si.num_interfaces,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Record:
        chains = [ChainInfo(**c) for c in d["chains"]]
        interfaces = [InterfaceInfo(**i) for i in d["interfaces"]]
        si_data = d.get("structure_info")
        si = StructureInfo(**si_data) if si_data else None
        return cls(id=d["id"], chains=chains, interfaces=interfaces, structure_info=si)
