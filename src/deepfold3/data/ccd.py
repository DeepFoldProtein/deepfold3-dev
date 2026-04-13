"""CCD molecule builder.

Build molecule data from CCD codes (downloaded from RCSB) or SMILES strings
(via RDKit).  Heavy atoms only — hydrogens are excluded.

CCD lookups can use either:
1. A monolithic ``components.cif`` file (recommended for training; ~110 MB,
   loaded once via gemmi into memory).
2. Individual per-compound downloads from RCSB (on-demand, cached in
   ``/tmp/deepfold3_ccd_cache/``).

Call :func:`set_ccd_components_path` before any lookups to use the bulk file.
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

# Global components.cif document (loaded once, shared across ccd.py and ref_atoms.py)
_COMPONENTS_DOC = None  # gemmi.cif.Document | None
_COMPONENTS_PATH: Path | None = None

_COMPONENTS_CIF_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def set_ccd_components_path(path: str | Path) -> None:
    """Point CCD lookups at a local ``components.cif`` (or ``.cif.gz``).

    This avoids per-compound HTTP downloads and is strongly recommended
    for batch processing (training data preparation).  The file is loaded
    lazily on first CCD lookup.
    """
    global _COMPONENTS_PATH, _COMPONENTS_DOC
    _COMPONENTS_PATH = Path(path)
    _COMPONENTS_DOC = None  # reset — will be loaded on next access


def get_ccd_block(ccd_code: str):
    """Look up a CCD block from the loaded components.cif, or return None.

    Returns a ``gemmi.cif.Block`` if ``components.cif`` is configured and the
    code exists, otherwise ``None`` (caller should fall back to per-file download).
    """
    global _COMPONENTS_DOC
    if _COMPONENTS_PATH is None:
        return None

    if _COMPONENTS_DOC is None:
        import gemmi

        log.info("Loading CCD components from %s (this may take a minute)...", _COMPONENTS_PATH)
        _COMPONENTS_DOC = gemmi.cif.read(str(_COMPONENTS_PATH))
        log.info("CCD components loaded: %d blocks", len(_COMPONENTS_DOC))

    return _COMPONENTS_DOC.find_block(ccd_code.strip().upper())


def fetch_components_cif(dest: str | Path) -> Path:
    """Download the full CCD ``components.cif.gz`` from wwPDB.

    Args:
        dest: Directory to save the file in.  The file will be named
              ``components.cif.gz``.

    Returns:
        Path to the downloaded (or already-existing) file.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "components.cif.gz"
    if out_path.exists():
        log.info("CCD components already present: %s", out_path)
        return out_path
    log.info("Downloading CCD components.cif.gz from wwPDB (~100 MB)...")
    urllib.request.urlretrieve(_COMPONENTS_CIF_URL, str(out_path))
    log.info("Downloaded to %s", out_path)
    return out_path


def _get_ccd_dir() -> Path:
    global _CCD_DIR
    if _CCD_DIR is None:
        _CCD_DIR = Path(tempfile.gettempdir()) / "deepfold3_ccd_cache"
        _CCD_DIR.mkdir(exist_ok=True)
    return _CCD_DIR


def fetch_ccd_block(ccd_code: str):
    """Resolve a CCD code to a gemmi.cif.Block.

    Tries bulk components.cif first, then falls back to per-compound
    RCSB download.  Returns ``None`` if the code cannot be resolved.
    """
    import gemmi

    ccd_code = ccd_code.strip().upper()

    # Try bulk components.cif first
    block = get_ccd_block(ccd_code)
    if block is not None:
        return block

    # Fall back to per-compound download
    ccd_dir = _get_ccd_dir()
    cif_path = ccd_dir / f"{ccd_code}.cif"

    if not cif_path.exists():
        url = f"https://files.rcsb.org/ligands/view/{ccd_code}.cif"
        try:
            urllib.request.urlretrieve(url, str(cif_path))
        except Exception:
            return None

    try:
        doc = gemmi.cif.read(str(cif_path))
        return doc[0]
    except Exception:
        return None


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
    """Parse a CCD CIF entry. Uses components.cif if available, else downloads."""
    ccd_code = ccd_code.strip().upper()
    block = fetch_ccd_block(ccd_code)
    if block is None:
        raise ValueError(f"CCD code {ccd_code!r} not found")

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


def _embed_conformer(smiles: str, seed: int = 42):
    """Shared RDKit ETKDGv3 + MMFF conformer generation.

    Returns (heavy_atom_mol, conformer) or (None, None) on failure.
    Requires RDKit — caller must handle ImportError.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            return None, None

    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol, mol.GetConformer()


def _extract_heavy_atoms(mol, conf) -> tuple[list[str], list[list[float]], list[str], list[float]]:
    """Extract atom names, positions, elements, charges from an RDKit mol+conformer."""
    atom_names: list[str] = []
    positions_list: list[list[float]] = []
    elements: list[str] = []
    charges: list[float] = []
    elem_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
        atom_names.append(f"{sym}{elem_counts[sym]}")
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions_list.append([pos.x, pos.y, pos.z])
        elements.append(sym)
        charges.append(float(atom.GetFormalCharge()))
    return atom_names, positions_list, elements, charges


def smiles_to_molecule(smiles: str, res_name: str = "LIG") -> CcdMolecule:
    """Builds a molecule from a SMILES string using RDKit.

    Adds hydrogens, generates a 3D conformer via ETKDGv3, optimizes
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
    except ImportError as e:
        raise ImportError("RDKit is required for SMILES input: pip install rdkit") from e

    mol, conf = _embed_conformer(smiles)
    if mol is None:
        raise ValueError(f"Failed to generate 3D conformer for SMILES: {smiles!r}")

    atom_names, positions_list, elements, charges = _extract_heavy_atoms(mol, conf)

    _RDKIT_BOND_ORDER = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 2,
    }
    bonds: list[tuple[int, int, int]] = []
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


# ---------------------------------------------------------------------------
# ETKDGv3 conformer generation from CCD SMILES (AF3 §2.8 ref_pos)
# ---------------------------------------------------------------------------


def ccd_smiles_from_block(block) -> str | None:
    """Extract canonical SMILES from a CCD gemmi block."""
    table = block.find(["_pdbx_chem_comp_descriptor.type", "_pdbx_chem_comp_descriptor.descriptor"])
    for row in table:
        dtype = row[0].strip().strip('"')
        if dtype == "SMILES_CANONICAL":
            smiles = row[1].strip().strip('"')
            if smiles and smiles != "?":
                return smiles
    for row in table:
        dtype = row[0].strip().strip('"')
        if dtype == "SMILES":
            smiles = row[1].strip().strip('"')
            if smiles and smiles != "?":
                return smiles
    return None


def etkdg_conformer(smiles: str, seed: int = 42) -> CcdMolecule | None:
    """Generate a 3D conformer from SMILES via RDKit ETKDGv3 (AF3 §2.8).

    Returns CcdMolecule with heavy-atom positions from force-field-optimised
    conformer, or None if generation fails.
    """
    try:
        mol, conf = _embed_conformer(smiles, seed)
    except ImportError:
        return None
    if mol is None:
        return None

    atom_names, positions_list, elements, charges = _extract_heavy_atoms(mol, conf)
    if not atom_names:
        return None

    return CcdMolecule(
        atom_names=atom_names,
        positions=np.array(positions_list, dtype=np.float32),
        elements=elements,
        charges=charges,
        bonds=[],
        res_name="LIG",
    )


# ---------------------------------------------------------------------------
# CCD atom name cache for validation during structure processing
# ---------------------------------------------------------------------------


def _generate_ref_pos_for_code(args: tuple[str, str]) -> tuple[str, list[list[float]] | None]:
    """Worker: generate ETKDGv3 ref_pos for a single CCD code from its SMILES."""
    code, smiles = args
    mol = etkdg_conformer(smiles)
    if mol is None:
        return code, None
    return code, mol.positions.tolist()


def build_atom_name_cache(
    components_path: str | Path,
    output_path: str | Path,
    *,
    ref_pos: bool = False,
    workers: int = 1,
) -> int:
    """Build a JSON cache mapping CCD code → atom info.

    Reads the full ``components.cif`` (or ``.cif.gz``) and extracts per-component:
    - ``atoms``: list of valid heavy-atom names
    - ``leaving``: list of heavy-atom names with ``pdbx_leaving_atom_flag = Y``
    - ``ref_pos`` (optional): ETKDGv3-generated 3D coordinates ``[[x,y,z], ...]``
      for each heavy atom (AF3 §2.8). Requires RDKit.

    Args:
        components_path: Path to ``components.cif`` or ``components.cif.gz``.
        output_path: Where to write the JSON cache.
        ref_pos: If True, generate ETKDGv3 conformers via RDKit for ref_pos.
        workers: Number of parallel workers for ref_pos generation.

    Returns:
        Number of CCD entries written.
    """
    import json

    import gemmi

    log.info("Reading %s (this may take a minute)...", components_path)
    doc = gemmi.cif.read(str(components_path))
    cache: dict[str, dict] = {}
    smiles_map: dict[str, str] = {}  # code → SMILES (for ref_pos generation)

    for block in doc:
        code = block.name
        table = block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.pdbx_leaving_atom_flag",
            ]
        )
        names: list[str] = []
        leaving: list[str] = []
        for row in table:
            elem = row[1].strip().strip('"')
            if elem.upper() == "H":
                continue
            name = row[0].strip().strip('"')
            names.append(name)
            flag = row[2].strip().strip('"').upper() if len(row) > 2 else ""
            if flag == "Y":
                leaving.append(name)
        if names:
            entry: dict = {"atoms": names}
            if leaving:
                entry["leaving"] = leaving
            cache[code] = entry

        # Collect SMILES for ref_pos generation
        if ref_pos:
            smiles = ccd_smiles_from_block(block)
            if smiles:
                smiles_map[code] = smiles

    # Generate ETKDGv3 ref_pos from SMILES
    if ref_pos and smiles_map:
        log.info("Generating ETKDGv3 ref_pos for %d CCD entries (workers=%d)...", len(smiles_map), workers)
        generated = 0

        failed_codes: list[str] = []
        n_total = len(smiles_map)

        skipped_no_atoms: list[str] = []

        def _apply_result(code: str, positions: list[list[float]] | None) -> None:
            nonlocal generated
            if positions is None:
                failed_codes.append(code)
            elif code not in cache:
                skipped_no_atoms.append(code)
            else:
                cache[code]["ref_pos"] = positions
                generated += 1

        if workers <= 1:
            for i, (code, smiles) in enumerate(smiles_map.items()):
                result = _generate_ref_pos_for_code((code, smiles))
                _apply_result(code, result[1])
                if (i + 1) % 5000 == 0:
                    log.info("  ref_pos: %d/%d done (%d ok, %d failed)", i + 1, n_total, generated, len(failed_codes))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            items = list(smiles_map.items())
            done = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_generate_ref_pos_for_code, item): item[0] for item in items}
                for future in as_completed(futures):
                    done += 1
                    code = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        failed_codes.append(code)
                        continue
                    _apply_result(result[0], result[1])
                    if done % 5000 == 0:
                        log.info(
                            "  ref_pos: %d/%d done (%d ok, %d failed)", done, n_total, generated, len(failed_codes)
                        )

        log.info(
            "Generated ref_pos for %d/%d CCD entries (%d conformer failed, %d skipped no atoms)",
            generated, n_total, len(failed_codes), len(skipped_no_atoms),
        )
        if failed_codes:
            for code in failed_codes[:50]:
                log.warning("  ref_pos conformer failed: %s (SMILES: %.60s)", code, smiles_map.get(code, "?"))
            if len(failed_codes) > 50:
                log.warning("  ... and %d more", len(failed_codes) - 50)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f, separators=(",", ":"))

    log.info("Wrote CCD cache: %d entries → %s", len(cache), output_path)
    return len(cache)


def load_ccd_cache(
    path: str | Path,
) -> tuple[dict[str, set[str]], dict[str, np.ndarray], dict[str, set[str]]]:
    """Load all CCD cache data in a single JSON parse.

    Returns:
        (atom_names, ref_pos, leaving_atoms) — three dicts keyed by CCD code.
    """
    import json

    with open(path) as f:
        raw = json.load(f)

    atom_names: dict[str, set[str]] = {}
    ref_pos: dict[str, np.ndarray] = {}
    leaving: dict[str, set[str]] = {}

    for k, v in raw.items():
        if isinstance(v, list):
            atom_names[k] = set(v)  # backward compat: old format is list of names
        else:
            atom_names[k] = set(v["atoms"])
            if "ref_pos" in v:
                ref_pos[k] = np.array(v["ref_pos"], dtype=np.float32)
            if "leaving" in v:
                leaving[k] = set(v["leaving"])

    return atom_names, ref_pos, leaving


def load_atom_name_cache(path: str | Path) -> dict[str, set[str]]:
    """Load atom names only. Prefer :func:`load_ccd_cache` to avoid re-parsing."""
    return load_ccd_cache(path)[0]


def load_ref_pos_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load ref_pos only. Prefer :func:`load_ccd_cache` to avoid re-parsing."""
    return load_ccd_cache(path)[1]


def load_leaving_atom_cache(path: str | Path) -> dict[str, set[str]]:
    """Load leaving atoms only. Prefer :func:`load_ccd_cache` to avoid re-parsing."""
    return load_ccd_cache(path)[2]
