"""CCD molecule builder.

Build molecule data from CCD codes (downloaded from RCSB) or SMILES strings
(via RDKit).  Heavy atoms only — hydrogens are excluded.
"""

from __future__ import annotations

import logging
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_CCD_DIR: Path | None = None

# Bond order mapping from CCD _chem_comp_bond.value_order
_BOND_ORDER: dict[str, int] = {"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}


def _get_ccd_dir() -> Path:
    global _CCD_DIR
    if _CCD_DIR is None:
        _CCD_DIR = Path(tempfile.gettempdir()) / "deepfold3_ccd_cache"
        _CCD_DIR.mkdir(exist_ok=True)
    return _CCD_DIR


@dataclass
class CcdMolecule:
    """Heavy-atom molecule data from CCD or SMILES.

    Attributes:
        atom_names: PDB atom names (e.g. ["C1", "C2", "O1"]).
        positions: Ideal 3D coordinates, shape [n_atoms, 3].
        elements: Element symbols (e.g. ["C", "C", "O"]).
        charges: Formal charges per atom.
        bonds: List of (atom_idx_1, atom_idx_2, bond_order) tuples.
        res_name: Residue name (CCD code or "LIG" for SMILES).
    """

    atom_names: list[str]
    positions: np.ndarray  # [n_atoms, 3] float32
    elements: list[str]
    charges: list[float]
    bonds: list[tuple[int, int, int]] = field(default_factory=list)
    res_name: str = "LIG"


# Cache: ccd_code -> CcdMolecule | None
_MOL_CACHE: dict[str, CcdMolecule | None] = {}


def load_ccd_molecule(ccd_code: str) -> CcdMolecule:
    """Loads a molecule from the CCD by its component code.

    Downloads the CIF from RCSB on first use and caches locally
    in ``/tmp/deepfold3_ccd_cache/``.  Skips hydrogen atoms.

    Args:
        ccd_code: CCD component identifier (e.g. "ATP", "HEM").

    Returns:
        CcdMolecule with heavy atoms, ideal positions, and bonds.

    Raises:
        ValueError: If the CCD code cannot be fetched or parsed.
    """
    ccd_code = ccd_code.strip().upper()
    if ccd_code in _MOL_CACHE:
        mol = _MOL_CACHE[ccd_code]
        if mol is None:
            raise ValueError(f"CCD code {ccd_code!r} not found or has no heavy atoms")
        return mol

    try:
        mol = _parse_ccd_cif(ccd_code)
    except Exception as e:
        _MOL_CACHE[ccd_code] = None
        raise ValueError(f"Failed to load CCD code {ccd_code!r}: {e}") from e

    _MOL_CACHE[ccd_code] = mol
    return mol


def _parse_ccd_cif(ccd_code: str) -> CcdMolecule:
    """Downloads and parses a CCD CIF file from RCSB."""
    import gemmi

    ccd_dir = _get_ccd_dir()
    cif_path = ccd_dir / f"{ccd_code}.cif"

    if not cif_path.exists():
        url = f"https://files.rcsb.org/ligands/view/{ccd_code}.cif"
        urllib.request.urlretrieve(url, str(cif_path))

    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]

    # Parse atoms
    atom_table = block.find(
        [
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        ]
    )

    atom_names: list[str] = []
    positions_list: list[list[float]] = []
    elements: list[str] = []
    charges: list[float] = []
    name_to_idx: dict[str, int] = {}

    for row in atom_table:
        name = row[0].strip().strip('"')
        elem = row[1].strip().strip('"')
        if elem == "H":
            continue
        try:
            x, y, z = float(row[3]), float(row[4]), float(row[5])
        except (ValueError, TypeError):
            x, y, z = 0.0, 0.0, 0.0

        name_to_idx[name] = len(atom_names)
        atom_names.append(name)
        positions_list.append([x, y, z])
        elements.append(elem)
        try:
            charges.append(float(row[2]))
        except (ValueError, TypeError):
            charges.append(0.0)

    if not atom_names:
        raise ValueError(f"No heavy atoms found for {ccd_code}")

    # Parse bonds
    bonds: list[tuple[int, int, int]] = []
    bond_table = block.find(
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
        ]
    )
    for row in bond_table:
        a1 = row[0].strip().strip('"')
        a2 = row[1].strip().strip('"')
        order_str = row[2].strip().strip('"')
        if a1 not in name_to_idx or a2 not in name_to_idx:
            continue  # skip bonds involving H
        order = _BOND_ORDER.get(order_str, 1)
        bonds.append((name_to_idx[a1], name_to_idx[a2], order))

    return CcdMolecule(
        atom_names=atom_names,
        positions=np.array(positions_list, dtype=np.float32),
        elements=elements,
        charges=charges,
        bonds=bonds,
        res_name=ccd_code,
    )


def smiles_to_molecule(smiles: str, res_name: str = "LIG") -> CcdMolecule:
    """Builds a molecule from a SMILES string using RDKit.

    Adds hydrogens, generates a 3D conformer via ETKDG, optimizes
    with MMFF, then strips all hydrogens to keep only heavy atoms.

    Args:
        smiles: SMILES string describing the molecule.
        res_name: Residue name to assign (default "LIG").

    Returns:
        CcdMolecule with 3D coordinates and bonds.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If SMILES parsing or conformer generation fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise ImportError("RDKit is required for SMILES input: pip install rdkit") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Retry without distance geometry constraints
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            raise ValueError(f"Failed to generate 3D conformer for SMILES: {smiles!r}")

    AllChem.MMFFOptimizeMolecule(mol)

    # Remove non-heavy atoms
    mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer()
    atom_names: list[str] = []
    positions_list: list[list[float]] = []
    elements: list[str] = []
    charges: list[float] = []

    # Generate unique atom names
    elem_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
        name = f"{sym}{elem_counts[sym]}"
        atom_names.append(name)
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions_list.append([pos.x, pos.y, pos.z])
        elements.append(sym)
        charges.append(float(atom.GetFormalCharge()))

    bonds: list[tuple[int, int, int]] = []
    _RDKIT_BOND_ORDER = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 2,
    }
    for bond in mol.GetBonds():
        order = _RDKIT_BOND_ORDER.get(bond.GetBondType(), 1)
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order))

    return CcdMolecule(
        atom_names=atom_names,
        positions=np.array(positions_list, dtype=np.float32),
        elements=elements,
        charges=charges,
        bonds=bonds,
        res_name=res_name,
    )
