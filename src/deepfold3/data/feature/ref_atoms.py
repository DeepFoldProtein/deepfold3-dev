"""Reference atom data lookup.

Uses CCD for ideal atom coordinates when available (via bulk ``components.cif``
or per-compound RCSB download); falls back to hardcoded ideal CCD coordinates
in const.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from deepfold3.data import const
from deepfold3.data.ccd import fetch_ccd_block

log = logging.getLogger(__name__)

_CCD_CACHE: dict[str, _CCDEntry | None] = {}
# Optional ETKDGv3-generated ref_pos cache (AF3 §2.8). Set via set_ref_pos_cache().
_REF_POS_CACHE: dict[str, np.ndarray] | None = None


def set_ref_pos_cache(cache: dict[str, np.ndarray]) -> None:
    """Set ETKDGv3-generated reference positions for CCD codes.

    When set, ref_pos are taken from this cache instead of CCD ideal
    coordinates.  Generate with ``deepfold3 build-ccd-cache --ref-pos``.
    """
    global _REF_POS_CACHE
    _REF_POS_CACHE = cache


@dataclass
class _CCDEntry:
    atom_names: list[str]
    positions: np.ndarray  # [n_atoms, 3] float32 ideal positions
    elements: list[int]  # atomic numbers
    charges: list[float]  # formal charges


def _load_ccd_entry_gemmi(res_name: str) -> _CCDEntry | None:
    """Load CCD entry via shared fetch_ccd_block (bulk or per-compound)."""
    block = fetch_ccd_block(res_name)
    if block is None:
        return None

    try:
        table = block.find(
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
        elements: list[int] = []
        charges: list[float] = []

        for row in table:
            name = row[0].strip().strip('"')
            elem = row[1].strip().strip('"')
            if elem == "H":
                continue  # skip hydrogen
            try:
                x = float(row[3])
                y = float(row[4])
                z = float(row[5])
            except (ValueError, TypeError):
                continue

            atom_names.append(name)
            positions_list.append([x, y, z])
            elements.append(const.ELEMENT_TO_INT.get(elem.upper(), 0))
            try:
                charges.append(float(row[2]))
            except (ValueError, TypeError):
                charges.append(0.0)

        if not atom_names:
            return None

        return _CCDEntry(
            atom_names=atom_names,
            positions=np.array(positions_list, dtype=np.float32),
            elements=elements,
            charges=charges,
        )
    except Exception:
        return None


def _get_ccd_entry(res_name: str) -> _CCDEntry | None:
    if res_name not in _CCD_CACHE:
        _CCD_CACHE[res_name] = _load_ccd_entry_gemmi(res_name)
    return _CCD_CACHE[res_name]


@dataclass
class RefAtomData:
    """Reference atom data for one token, padded to max_atoms."""

    positions: np.ndarray  # [max_atoms, 3] float32 — ideal positions
    mask: np.ndarray  # [max_atoms] bool — present atoms
    element: np.ndarray  # [max_atoms] int64 — atomic numbers
    charge: np.ndarray  # [max_atoms] float32 — formal charges
    atom_name_chars: np.ndarray  # [max_atoms, 4] int64 — ord(c)-32 encoding


def _encode_atom_name(name: str) -> list[int]:
    """Encode atom name as 4 ints: ord(c)-32 for c in name.ljust(4)[:4].

    Clamped to ``[0, 63]`` so the downstream ``F.one_hot(..., 64)`` in the atom
    encoder can never index out of bounds. Uppercase names stay < 64 (``ord('Z')
    - 32 == 58``); any lowercase / high-codepoint char (e.g. a mixed-case ligand
    atom name) would otherwise produce an index >= 64 and crash the whole batch.
    """
    padded = name.ljust(4)[:4]
    return [min(63, max(0, ord(c) - 32)) for c in padded]


def decode_atom_name_chars(chars) -> str:
    """Decode the ``ord(c)-32`` 4-int encoding back to a stripped atom name.

    Inverse of :func:`_encode_atom_name`. Accepts any iterable of ints
    (numpy array, torch tensor, list).
    """
    return "".join(chr(int(c) + 32) for c in chars).strip()


def get_ref_atoms(
    res_name: str,
    mol_type: int,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
    atom_names_override: list[str] | None = None,
) -> RefAtomData:
    """Get reference atom data for a residue.

    Tries CCD first; falls back to hardcoded ideal coordinates in const.py.

    Parameters
    ----------
    res_name :            3-letter residue code (or short nucleic code)
    mol_type :            MOL_PROTEIN / MOL_RNA / MOL_DNA / MOL_NONPOLYMER
    max_atoms :           Pad/truncate to this many atoms
    atom_names_override : If given, use these atom names (for single-atom ligand tokens)

    Returns
    -------
    RefAtomData padded to max_atoms
    """
    positions = np.zeros((max_atoms, 3), dtype=np.float32)
    mask = np.zeros(max_atoms, dtype=bool)
    element = np.zeros(max_atoms, dtype=np.int64)
    charge = np.zeros(max_atoms, dtype=np.float32)
    atom_name_chars = np.zeros((max_atoms, 4), dtype=np.int64)

    # OXT is a C-terminal artifact in the CCD monomer definition.
    # For polymer residues, skip it (only C-terminal should have OXT, and
    # the model predicts positions without requiring OXT in reference atoms).
    is_polymer = mol_type in (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA)

    if atom_names_override is not None:
        # Single atom with given name
        atom_names = atom_names_override
        ccd_positions = None
        ccd_elements = None
        ccd_charges = None
    else:
        # Try CCD lookup (downloads from RCSB PDB on first use)
        entry = _get_ccd_entry(res_name)
        if entry is not None:
            atom_names = entry.atom_names
            ccd_positions = entry.positions
            ccd_elements = entry.elements
            ccd_charges = entry.charges

            # AF3 §2.8: prefer ETKDGv3-generated ref_pos when available
            if _REF_POS_CACHE is not None and res_name in _REF_POS_CACHE:
                etkdg_pos = _REF_POS_CACHE[res_name]
                if len(etkdg_pos) == len(atom_names):
                    ccd_positions = etkdg_pos

            # Filter OXT for polymer residues
            if is_polymer and "OXT" in atom_names:
                idx = atom_names.index("OXT")
                atom_names = atom_names[:idx] + atom_names[idx + 1 :]
                ccd_positions = np.delete(ccd_positions, idx, axis=0)
                ccd_elements = ccd_elements[:idx] + ccd_elements[idx + 1 :]
                ccd_charges = ccd_charges[:idx] + ccd_charges[idx + 1 :]
        else:
            # Fall back to hardcoded ideal coordinates from const.py
            canonical_names = const.REF_ATOMS.get(res_name, const.REF_ATOMS.get("UNK", ["CA"]))
            canonical_pos = const.REF_ATOM_POSITIONS.get(res_name, const.REF_ATOM_POSITIONS.get("UNK"))
            atom_names = canonical_names
            if canonical_pos is not None:
                ccd_positions = np.array(canonical_pos, dtype=np.float32)
            else:
                ccd_positions = None
            ccd_elements = None
            ccd_charges = None

    n = min(len(atom_names), max_atoms)
    mask[:n] = True

    for i in range(n):
        name = atom_names[i]
        atom_name_chars[i] = _encode_atom_name(name)

        if ccd_positions is not None and i < len(ccd_positions):
            positions[i] = ccd_positions[i]

        if ccd_elements is not None:
            element[i] = int(np.clip(ccd_elements[i], 0, 127))
        else:
            # Guess element from atom name
            sym = _atom_name_to_element(name)
            element[i] = int(np.clip(const.ELEMENT_TO_INT.get(sym, 6), 0, 127))  # default Carbon

        if ccd_charges is not None:
            charge[i] = float(ccd_charges[i])

    return RefAtomData(
        positions=positions,
        mask=mask,
        element=element,
        charge=charge,
        atom_name_chars=atom_name_chars,
    )


def _atom_name_to_element(atom_name: str) -> str:
    """Guess element symbol from PDB atom name (heuristic)."""
    # PDB convention: columns 13-14 are element, strip leading digit/space
    name = atom_name.strip()
    if not name:
        return "C"
    # Remove trailing digits
    sym = name.rstrip("0123456789'\"*")
    if len(sym) >= 2:
        # Try two-char element
        elem2 = sym[:2].capitalize()
        if elem2 in const.ELEMENT_TO_INT:
            return elem2
    if sym:
        elem1 = sym[0].upper()
        if elem1 in const.ELEMENT_TO_INT:
            return elem1
    return "C"
