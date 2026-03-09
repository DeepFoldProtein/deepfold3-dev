"""Quality filters for training data preprocessing.

Applied per-chain after mmCIF → Structure conversion. Chains that fail
a filter are marked as ``valid=False`` in the Record (not removed).
"""

from __future__ import annotations

import logging

import numpy as np

from deepfold3.data import const
from deepfold3.data.types import ChainInfo, Record, Structure

logger = logging.getLogger(__name__)


def _get_chain_residues(structure: Structure, chain_info: ChainInfo) -> np.ndarray:
    """Get the residue array slice for a chain."""
    chain = structure.chains[chain_info.chain_id]
    res_start = int(chain["res_idx"])
    res_num = int(chain["res_num"])
    return structure.residues[res_start : res_start + res_num]


def _resolved_count(structure: Structure, chain_info: ChainInfo) -> int:
    """Count resolved (is_present=True, atom_num>0) residues in a chain."""
    residues = _get_chain_residues(structure, chain_info)
    return int((residues["is_present"] & (residues["atom_num"] > 0)).sum())


def filter_min_length(
    structure: Structure,
    chain_info: ChainInfo,
    min_residues: int = 4,
    max_residues: int = 5000,
) -> bool:
    """Reject chains with too few or too many resolved residues."""
    if chain_info.mol_type == const.MOL_NONPOLYMER:
        return True  # ligands always pass
    resolved = _resolved_count(structure, chain_info)
    return min_residues <= resolved <= max_residues


def filter_unknown(structure: Structure, chain_info: ChainInfo) -> bool:
    """Reject chains where all residues are UNK/unknown."""
    if chain_info.mol_type == const.MOL_NONPOLYMER:
        return True
    residues = _get_chain_residues(structure, chain_info)
    res_types = residues["res_type"]
    # UNK=20 for protein, N=30 for nucleic
    if chain_info.mol_type == const.MOL_PROTEIN:
        return not np.all(res_types == 20)
    else:
        return not np.all(res_types == 30)


def filter_consecutive_ca(
    structure: Structure,
    chain_info: ChainInfo,
    max_dist: float = 10.0,
) -> bool:
    """Reject protein chains with consecutive CA atoms > max_dist apart."""
    if chain_info.mol_type != const.MOL_PROTEIN:
        return True

    residues = _get_chain_residues(structure, chain_info)

    ca_coords = []
    for res in residues:
        if not res["is_present"] or res["atom_num"] == 0:
            continue
        atom_start = int(res["atom_idx"])
        atom_end = atom_start + int(res["atom_num"])
        atoms = structure.atoms[atom_start:atom_end]
        # Find CA
        for atom in atoms:
            name = atom["name"].decode("ascii", errors="replace").strip()
            if name == "CA":
                ca_coords.append([float(atom["x"]), float(atom["y"]), float(atom["z"])])
                break

    if len(ca_coords) < 2:
        return True

    ca = np.array(ca_coords)
    dists = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    return bool(np.all(dists <= max_dist))


def filter_clashing_chains(
    structure: Structure,
    record: Record,
    freq_threshold: float = 0.3,
    dist_threshold: float = 1.7,
) -> set[int]:
    """Find chains with >freq_threshold fraction of atoms clashing.

    Returns set of chain_ids (asym_id) that should be marked invalid.
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return set()

    # Build per-chain coordinate arrays
    chain_coords: dict[int, np.ndarray] = {}
    for ci in record.chains:
        chain = structure.chains[ci.chain_id]
        a_start = int(chain["atom_idx"])
        a_num = int(chain["atom_num"])
        if a_num == 0:
            continue
        atoms = structure.atoms[a_start : a_start + a_num]
        coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]]).astype(np.float64)
        chain_coords[ci.chain_id] = coords

    # Pre-build all KDTrees once
    chain_ids = sorted(chain_coords.keys())
    trees: dict[int, KDTree] = {cid: KDTree(chain_coords[cid]) for cid in chain_ids}

    clash_count: dict[int, int] = {cid: 0 for cid in chain_ids}
    atom_count: dict[int, int] = {cid: len(chain_coords[cid]) for cid in chain_ids}

    for i, c1 in enumerate(chain_ids):
        for c2 in chain_ids[i + 1 :]:
            n_clash = trees[c1].count_neighbors(trees[c2], dist_threshold)
            if n_clash > 0:
                clash_count[c1] += n_clash
                clash_count[c2] += n_clash

    invalid = set()
    for cid in chain_ids:
        if atom_count[cid] > 0 and clash_count[cid] / atom_count[cid] > freq_threshold:
            invalid.add(cid)

    return invalid


def filter_structure(
    structure: Structure,
    record: Record,
    min_residues: int = 4,
    max_residues: int = 5000,
    max_ca_dist: float = 10.0,
    clash_freq: float = 0.3,
    clash_dist: float = 1.7,
) -> Record:
    """Apply all filters to a Record, marking invalid chains.

    Mutates ``record`` in-place, setting ``valid=False`` on failing chains
    and interfaces, and returns it.
    """
    # Per-chain filters
    for ci in record.chains:
        if not filter_min_length(structure, ci, min_residues, max_residues):
            logger.debug("%s chain %s: failed min_length filter", record.id, ci.chain_name)
            ci.valid = False
        elif not filter_unknown(structure, ci):
            logger.debug("%s chain %s: all residues unknown", record.id, ci.chain_name)
            ci.valid = False
        elif not filter_consecutive_ca(structure, ci, max_ca_dist):
            logger.debug("%s chain %s: consecutive CA > %.1f Å", record.id, ci.chain_name, max_ca_dist)
            ci.valid = False

    # Clashing chains
    clashing = filter_clashing_chains(structure, record, clash_freq, clash_dist)
    for ci in record.chains:
        if ci.chain_id in clashing:
            logger.debug("%s chain %s: clashing", record.id, ci.chain_name)
            ci.valid = False

    # Mark interfaces invalid if either chain is invalid
    valid_chain_ids = {ci.chain_id for ci in record.chains if ci.valid}
    for iface in record.interfaces:
        if iface.chain_1 not in valid_chain_ids or iface.chain_2 not in valid_chain_ids:
            iface.valid = False

    # Update metadata
    if record.structure_info is not None:
        record.structure_info.num_chains = sum(1 for c in record.chains if c.valid)
        record.structure_info.num_interfaces = sum(1 for i in record.interfaces if i.valid)

    return record
