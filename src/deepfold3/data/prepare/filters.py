"""Quality filters for training data preprocessing.

Two levels of filtering:

1. **Structure-level** (``filter_structure_metadata``): reject entire structures
   based on metadata (resolution, method, date, chain count). These run before
   parsing atoms and are cheap.

2. **Chain-level** (``filter_structure``): mark individual chains as
   ``valid=False`` based on geometry (length, unknown residues, CA gaps,
   clashing). Applied after mmCIF → Structure conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from deepfold3.data import const
from deepfold3.data.types import ChainInfo, Record, Structure

logger = logging.getLogger(__name__)


@dataclass
class StructureFilterConfig:
    """Configuration for structure-level and chain-level filters.

    Structure-level filters (reject entire structure):
        max_resolution: Maximum resolution in Å. None = disabled.
        min_resolution: Minimum resolution in Å. None = disabled.
        allowed_methods: Set of allowed experimental methods, or None = any.
        max_release_date: Reject structures released after this date (YYYY-MM-DD). None = disabled.
        min_release_date: Reject structures released before this date (YYYY-MM-DD). None = disabled.
        max_chains: Maximum number of chains. Enabled by default (300).
        max_total_residues: Maximum total residues. Enabled by default (16384).
        min_polymer_chains: Minimum polymer chains. Enabled by default (1).

    Chain-level filters (mark individual chains invalid):
        min_residues: Minimum resolved residues per polymer chain.
        max_residues: Maximum resolved residues per polymer chain.
        max_ca_dist: Maximum consecutive CA distance in Å.
        clash_freq: Fraction of atoms clashing to reject a chain.
        clash_dist: Distance threshold in Å for clashing.
        max_unk_fraction: Max fraction of UNK residues allowed per polymer
            chain. None = only reject 100%-UNK chains (legacy behaviour).
            A value like 0.1 rejects any chain where >10% of resolved
            residues are unknown (typical PATCHR offenders sit at 1–4%;
            set higher to keep those but reject truly uninterpretable
            chains).

    Structure-quality filter (reject the whole record when any threshold
    is exceeded; all chains are marked invalid):
        max_tight_clashes: Max allowed heavy-atom pairs below
            ``tight_clash_dist``. None disables. These are true geometric
            corruptions (atoms at <0.8 Å indicate modelling errors or
            assembly bugs), not chain-level clashes from the docking
            filter above.
        tight_clash_dist: Distance threshold (Å) for the tight-clash
            count. Default 0.8 — non-bonded heavy atoms at this distance
            are never chemically meaningful.
    """

    # Structure-level (disabled by default unless noted)
    max_resolution: float | None = None
    min_resolution: float | None = None
    allowed_methods: frozenset[str] | None = None
    max_release_date: str | None = None
    min_release_date: str | None = None
    max_chains: int = 300
    max_total_residues: int = 16384
    min_polymer_chains: int = 1

    # Chain-level
    min_residues: int = 4
    max_residues: int = 5000
    max_ca_dist: float = 10.0
    clash_freq: float = 0.3
    clash_dist: float = 1.7
    max_unk_fraction: float | None = None

    # Structure-quality (reject whole record)
    max_tight_clashes: int | None = None
    tight_clash_dist: float = 0.8


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


def _unk_fraction(structure: Structure, chain_info: ChainInfo) -> float:
    """Fraction of resolved residues that are UNK in a polymer chain."""
    residues = _get_chain_residues(structure, chain_info)
    resolved = residues["is_present"] & (residues["atom_num"] > 0)
    n_resolved = int(resolved.sum())
    if n_resolved == 0:
        return 0.0
    unk_code = 20 if chain_info.mol_type == const.MOL_PROTEIN else 30
    n_unk = int(((residues["res_type"] == unk_code) & resolved).sum())
    return n_unk / n_resolved


def filter_unknown_fraction(
    structure: Structure,
    chain_info: ChainInfo,
    max_fraction: float,
) -> bool:
    """Reject polymer chains whose UNK fraction exceeds ``max_fraction``."""
    if chain_info.mol_type == const.MOL_NONPOLYMER:
        return True
    return _unk_fraction(structure, chain_info) <= max_fraction


def count_tight_clashes(structure: Structure, dist: float = 0.8) -> int:
    """Count heavy-atom pairs closer than ``dist`` Å across the whole structure.

    Excludes hydrogens and unresolved atoms. Excludes directly bonded
    atoms (some CCD entries have legitimately short reference bond
    lengths for metals / aromatic systems — bonding info from
    ``structure.bonds`` masks those pairs so a normal covalent bond
    doesn't count as a clash).

    Unlike :func:`filter_clashing_chains`, which flags chains that clash
    with each other at ~1.7 Å (typical interface-clash threshold), this
    detects true atomic overlap (e.g. <0.8 Å) — a sign of structural
    corruption, not just a docking issue. A single such pair in the loss
    produces wildly large gradients.
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return 0

    atoms = structure.atoms
    present = atoms["is_present"]
    heavy = atoms["element"] != 1  # skip H
    mask = present & heavy
    if not mask.any():
        return 0

    coords = np.column_stack([atoms["x"][mask], atoms["y"][mask], atoms["z"][mask]]).astype(np.float64)
    # Map filtered indices back to original atom indices so we can
    # dedupe directly-bonded pairs from ``structure.bonds``.
    orig_idx = np.flatnonzero(mask)
    inv: dict[int, int] = {int(o): i for i, o in enumerate(orig_idx)}

    tree = KDTree(coords)
    pairs = tree.query_pairs(dist, output_type="set")  # (i, j) with i < j
    if not pairs:
        return 0

    # Mask bonded pairs (either direction).
    bonded: set[tuple[int, int]] = set()
    for b in structure.bonds:
        a1, a2 = int(b["atom_1"]), int(b["atom_2"])
        i, j = inv.get(a1), inv.get(a2)
        if i is None or j is None:
            continue
        bonded.add((i, j) if i < j else (j, i))

    return len(pairs - bonded)


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

    Uses AF3 §2.5.4 iterative 3-level tie-breaking: at each step remove
    the single worst chain by (1) clash fraction, (2) number of clashing
    atoms, (3) chain_id (highest removed first).  Clash counts are
    recomputed after each removal since removing one chain can reduce
    clashes for its neighbours.

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

    remaining = set(chain_coords.keys())
    atom_count: dict[int, int] = {cid: len(chain_coords[cid]) for cid in remaining}
    invalid: set[int] = set()

    # Build KDTrees once — trees don't change when a chain is removed
    trees: dict[int, KDTree] = {cid: KDTree(chain_coords[cid]) for cid in remaining}

    while True:
        chain_ids = sorted(remaining)

        # Compute per-chain clash counts against all other remaining chains
        clash_count: dict[int, int] = {cid: 0 for cid in chain_ids}
        for i, c1 in enumerate(chain_ids):
            for c2 in chain_ids[i + 1 :]:
                n_clash = trees[c1].count_neighbors(trees[c2], dist_threshold)
                if n_clash > 0:
                    clash_count[c1] += n_clash
                    clash_count[c2] += n_clash

        # Find the single worst chain above threshold
        worst_cid = None
        worst_key = (-1.0, -1, -1)  # (clash_frac, clash_count, chain_id)
        for cid in chain_ids:
            if atom_count[cid] == 0:
                continue
            frac = clash_count[cid] / atom_count[cid]
            if frac > freq_threshold:
                key = (frac, clash_count[cid], cid)
                if key > worst_key:
                    worst_key = key
                    worst_cid = cid

        if worst_cid is None:
            break  # no chain exceeds threshold

        invalid.add(worst_cid)
        remaining.discard(worst_cid)
        del trees[worst_cid]

    return invalid


def select_interface_chains(
    record: Record,
    max_polymer_chains: int = 20,
) -> set[int]:
    """If >max_polymer_chains polymer chains, greedily keep those maximising interfaces.

    AF3 §2.5.4: when a structure has more than 20 polymer chains, select
    the 20 that maximise the number of cross-chain interfaces.  Uses a
    greedy algorithm: repeatedly pick the chain that adds the most new
    interfaces with already-selected chains.

    Returns set of chain_ids (asym_ids) to mark invalid (the ones NOT kept).
    """
    polymer_chains = [ci for ci in record.chains if ci.valid and ci.mol_type != const.MOL_NONPOLYMER]
    if len(polymer_chains) <= max_polymer_chains:
        return set()

    polymer_ids = {ci.chain_id for ci in polymer_chains}

    # Build adjacency from interfaces: chain_id → set of neighbouring chain_ids
    neighbours: dict[int, set[int]] = {cid: set() for cid in polymer_ids}
    for iface in record.interfaces:
        if not iface.valid:
            continue
        c1, c2 = iface.chain_1, iface.chain_2
        if c1 in polymer_ids and c2 in polymer_ids:
            neighbours[c1].add(c2)
            neighbours[c2].add(c1)

    # Greedy selection: start with the chain having most interfaces, then
    # iteratively pick the chain that adds the most new interfaces to the
    # selected set.
    selected: set[int] = set()
    remaining = set(polymer_ids)

    for _ in range(min(max_polymer_chains, len(polymer_ids))):
        best_cid = max(remaining, key=lambda c: len(neighbours[c] & selected) if selected else len(neighbours[c]))
        selected.add(best_cid)
        remaining.discard(best_cid)

    return polymer_ids - selected


def filter_structure_metadata(
    record: Record,
    cfg: StructureFilterConfig | None = None,
) -> str | None:
    """Check structure-level metadata filters.

    Returns a skip reason string, or None if the structure passes.
    These run before chain-level filters and are cheap (no atom parsing).
    """
    if cfg is None:
        cfg = StructureFilterConfig()

    info = record.structure_info

    # Resolution
    if info is not None and info.resolution is not None:
        if cfg.max_resolution is not None and info.resolution > cfg.max_resolution:
            return f"resolution {info.resolution:.2f} > {cfg.max_resolution:.1f} Å"
        if cfg.min_resolution is not None and info.resolution < cfg.min_resolution:
            return f"resolution {info.resolution:.2f} < {cfg.min_resolution:.1f} Å"

    # Experimental method
    if cfg.allowed_methods is not None and info is not None and info.method is not None:
        if info.method.lower() not in cfg.allowed_methods:
            return f"method '{info.method}' not in allowed set"

    # Release date cutoff
    if info is not None and info.released is not None:
        if cfg.max_release_date is not None and info.released > cfg.max_release_date:
            return f"released {info.released} > cutoff {cfg.max_release_date}"
        if cfg.min_release_date is not None and info.released < cfg.min_release_date:
            return f"released {info.released} < cutoff {cfg.min_release_date}"

    # Chain count
    n_chains = len(record.chains)
    if n_chains > cfg.max_chains:
        return f"{n_chains} chains > max {cfg.max_chains}"

    # Total residues
    total_res = sum(c.num_residues for c in record.chains)
    if total_res > cfg.max_total_residues:
        return f"{total_res} total residues > max {cfg.max_total_residues}"

    # Minimum polymer chains
    n_polymer = sum(1 for c in record.chains if c.mol_type in (const.MOL_PROTEIN, const.MOL_DNA, const.MOL_RNA))
    if n_polymer < cfg.min_polymer_chains:
        return f"{n_polymer} polymer chains < min {cfg.min_polymer_chains}"

    return None


def filter_structure(
    structure: Structure,
    record: Record,
    cfg: StructureFilterConfig | None = None,
) -> tuple[Record, dict[str, str]]:
    """Apply chain-level filters to a Record, marking invalid chains.

    Mutates ``record`` in-place, setting ``valid=False`` on failing chains
    and interfaces.

    Returns:
        Tuple of (record, reasons) where reasons maps chain_name to the
        filter that rejected it (empty dict if all chains pass).
    """
    if cfg is None:
        cfg = StructureFilterConfig()

    reasons: dict[str, str] = {}

    # Per-chain filters
    for ci in record.chains:
        if not filter_min_length(structure, ci, cfg.min_residues, cfg.max_residues):
            resolved = _resolved_count(structure, ci)
            reasons[ci.chain_name] = f"length {resolved} not in [{cfg.min_residues}, {cfg.max_residues}]"
            ci.valid = False
        elif not filter_unknown(structure, ci):
            reasons[ci.chain_name] = "all residues unknown"
            ci.valid = False
        elif cfg.max_unk_fraction is not None and not filter_unknown_fraction(structure, ci, cfg.max_unk_fraction):
            frac = _unk_fraction(structure, ci)
            reasons[ci.chain_name] = f"UNK fraction {frac:.2f} > {cfg.max_unk_fraction:.2f}"
            ci.valid = False
        elif not filter_consecutive_ca(structure, ci, cfg.max_ca_dist):
            reasons[ci.chain_name] = f"consecutive CA > {cfg.max_ca_dist:.0f} Å"
            ci.valid = False

    # Clashing chains
    clashing = filter_clashing_chains(structure, record, cfg.clash_freq, cfg.clash_dist)
    for ci in record.chains:
        if ci.chain_id in clashing:
            reasons[ci.chain_name] = "chain clashing"
            ci.valid = False

    # Record-level tight-clash gate: atoms closer than ~0.8 Å are not
    # chemically meaningful. Even a single such pair can produce
    # non-finite gradients through the smooth-LDDT / distogram paths on
    # early training. Reject the whole record by marking every chain
    # invalid.
    if cfg.max_tight_clashes is not None:
        n_tight = count_tight_clashes(structure, cfg.tight_clash_dist)
        if n_tight > cfg.max_tight_clashes:
            for ci in record.chains:
                if ci.valid:
                    reasons[ci.chain_name] = (
                        f"record tight-clash {n_tight} > max {cfg.max_tight_clashes} (< {cfg.tight_clash_dist:.2f} Å)"
                    )
                    ci.valid = False

    # NOTE: chain selection (>20 polymer chains → keep 20 maximising interfaces)
    # is NOT applied here.  It belongs at training time (sampler/dataset) because
    # the limit varies by stage (initial=20, fine-tune=larger).
    # Use select_interface_chains() from this module at load time.

    # Mark interfaces invalid if either chain is invalid
    valid_chain_ids = {ci.chain_id for ci in record.chains if ci.valid}
    for iface in record.interfaces:
        if iface.chain_1 not in valid_chain_ids or iface.chain_2 not in valid_chain_ids:
            iface.valid = False

    # Update metadata
    if record.structure_info is not None:
        record.structure_info.num_chains = sum(1 for c in record.chains if c.valid)
        record.structure_info.num_interfaces = sum(1 for i in record.interfaces if i.valid)

    for name, reason in reasons.items():
        logger.debug("%s chain %s: %s", record.id, name, reason)

    return record, reasons
