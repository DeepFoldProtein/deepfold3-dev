"""Ligand atom symmetry via RDKit automorphism enumeration.

For each unique ligand CCD code, the heavy-atom subgraph automorphism
group enumerates all ways the molecule can be mapped to itself —
equivalently, the set of "symmetry-equivalent" atom labellings the
training loss must be invariant to. Without this, RDKit's deterministic
atom ordering picks one arbitrary labelling per ligand and the
prediction is penalised whenever the model places equivalent atoms in
any other valid order (consider benzene: 6 carbons in 12 indistinguishable
positions; without this filter only 1 of 12 valid placements is
"correct").

The automorphism group is computed once per **CCD code** (not per
NPZ) — it's a property of the molecular graph, identical across every
record that contains this ligand. Cached on disk as a single JSON file
because RDKit's automorphism call is O(seconds) for big ligands.

Implementation is clean-room: the algorithm (build RDKit Mol from CCD
heavy atoms + bonds, run ``GetSubstructMatches`` on (mol, mol)) is
publicly documented and uses RDKit's standard API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepfold3.data.ccd import CcdMolecule

logger = logging.getLogger(__name__)


# Atomic-number lookup for the elements that show up in CCD heavy atoms.
# Anything outside this table is queried via RDKit's PeriodicTable.
_ELEMENT_TO_ATOMIC_NUM: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}

# Sentinel returned by ``compute_ligand_automorphisms`` for ligands the
# function couldn't process. Cached so a known-bad ligand isn't retried.
_UNAVAILABLE: tuple[tuple[int, ...], ...] = ()


def _build_rdkit_mol(mol: CcdMolecule):
    """Build an editable RDKit ``Mol`` from a ``CcdMolecule``'s heavy-atom graph.

    Returns ``None`` if RDKit isn't installed or the molecule has zero
    bonds (atom equivalence is then trivial).
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.debug("rdkit not installed; ligand automorphism unavailable")
        return None

    if not mol.atom_names:
        return None

    rwmol = Chem.RWMol()
    pt = Chem.GetPeriodicTable()
    for elem in mol.elements:
        atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(elem)
        if atomic_num is None:
            try:
                atomic_num = pt.GetAtomicNumber(elem.title())
            except Exception:
                return None
        rwmol.AddAtom(Chem.Atom(atomic_num))

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }
    for a, b, order in mol.bonds:
        rwmol.AddBond(int(a), int(b), bond_type_map.get(order, Chem.BondType.SINGLE))

    return rwmol.GetMol()


def compute_ligand_automorphisms(mol: CcdMolecule) -> tuple[tuple[int, ...], ...]:
    """Enumerate the heavy-atom automorphism group of one CCD molecule.

    Returns a tuple of permutations, each represented as a tuple of length
    ``len(mol.atom_names)`` whose i-th entry gives the new index for atom
    i. The first permutation is always the identity. Returns an empty
    tuple if RDKit isn't installed or the molecule can't be processed
    (caller should treat that as "no symmetry beyond identity").
    """
    rmol = _build_rdkit_mol(mol)
    if rmol is None:
        return _UNAVAILABLE
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return _UNAVAILABLE
    try:
        matches = rmol.GetSubstructMatches(rmol, useChirality=False, uniquify=False, maxMatches=1024)
    except Exception:
        logger.debug("RDKit failed to enumerate automorphisms for %s", mol.res_name, exc_info=True)
        return _UNAVAILABLE
    if not matches:
        return _UNAVAILABLE
    return tuple(tuple(int(i) for i in m) for m in matches)


# In-process memo: avoid re-running RDKit for ligands seen before in the
# same process. Disk cache is the responsibility of the CLI builder.
_AUTOMORPHISM_MEMO: dict[str, tuple[tuple[int, ...], ...]] = {}


def get_ligand_automorphisms(ccd_code: str) -> tuple[tuple[int, ...], ...]:
    """Memoised wrapper around :func:`compute_ligand_automorphisms`.

    Loads the molecule via ``ccd.load_ccd_molecule`` (uses the existing
    components.cif cache) and caches the result in-process. Returns an
    empty tuple if the CCD code can't be loaded.
    """
    code = ccd_code.strip().upper()
    if code in _AUTOMORPHISM_MEMO:
        return _AUTOMORPHISM_MEMO[code]
    from deepfold3.data.ccd import load_ccd_molecule

    try:
        mol = load_ccd_molecule(code)
    except Exception:
        _AUTOMORPHISM_MEMO[code] = _UNAVAILABLE
        return _UNAVAILABLE
    perms = compute_ligand_automorphisms(mol)
    _AUTOMORPHISM_MEMO[code] = perms
    return perms


def prime_automorphism_memo(cache: dict[str, list[list[int]]]) -> None:
    """Bulk-load a precomputed automorphism cache into the in-process memo.

    The CLI builder writes its output as ``{ccd_code: [[perm], [perm], …]}``
    JSON; the dataloader calls this once at start-up to skip per-record
    RDKit calls. Permutations are converted to tuples to match the in-memory
    representation returned by :func:`get_ligand_automorphisms`.
    """
    for code, perms in cache.items():
        _AUTOMORPHISM_MEMO[code.upper()] = tuple(tuple(int(i) for i in p) for p in perms)
