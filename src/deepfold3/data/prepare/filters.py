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
import random
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
    ca_res_ids = []
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
                ca_res_ids.append(int(res["res_id"]))
                break

    if len(ca_coords) < 2:
        return True

    ca = np.array(ca_coords)
    res_ids = np.array(ca_res_ids)
    dists = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    # Only test *sequence-adjacent* residues (res_id diff == 1). A large CA-CA
    # gap across an unmodeled internal loop (e.g. res 50 → 101) is a missing
    # region, not broken backbone geometry, and must not reject the chain
    # (AF3 §2.5.4 masks numbering discontinuities).
    adjacent = (res_ids[1:] - res_ids[:-1]) == 1
    return bool(np.all(dists[adjacent] <= max_dist))


def filter_clashing_chains(
    structure: Structure,
    record: Record,
    freq_threshold: float = 0.3,
    dist_threshold: float = 1.7,
) -> set[int]:
    """Find chains with >freq_threshold fraction of atoms clashing.

    A chain clashes when more than ``freq_threshold`` of its atoms lie within
    ``dist_threshold`` of another chain (AF3 §2.5.4). At each step the single
    worst chain is removed, tie-broken by (1) clash fraction, (2) fewer total
    atoms, (3) larger chain_id. Clashing-atom sets are recomputed after each
    removal since removing one chain can un-clash its neighbours.

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

        # AF3 §2.5.4: a chain clashes if >30% of ITS atoms lie within 1.7 Å of
        # another chain. Count the *unique clashing atoms* of each chain (a true
        # ≤1 fraction), not the number of within-threshold atom pairs — the pair
        # count double-counts (added to both chains) and routinely exceeds the
        # atom count, over-removing chains.
        clash_atoms: dict[int, set[int]] = {cid: set() for cid in chain_ids}
        for i, c1 in enumerate(chain_ids):
            for c2 in chain_ids[i + 1 :]:
                nbr = trees[c1].query_ball_tree(trees[c2], dist_threshold)
                for a1, matches in enumerate(nbr):
                    if matches:
                        clash_atoms[c1].add(a1)
                        clash_atoms[c2].update(matches)

        # Remove the single worst chain above threshold. Tie-break (AF3): higher
        # clash fraction, then fewer total atoms, then larger chain id.
        worst_cid = None
        worst_key = (-1.0, 0, -1)  # (clash_frac, -atom_count, chain_id)
        for cid in chain_ids:
            if atom_count[cid] == 0:
                continue
            frac = len(clash_atoms[cid]) / atom_count[cid]
            if frac > freq_threshold:
                key = (frac, -atom_count[cid], cid)
                if key > worst_key:
                    worst_key = key
                    worst_cid = cid

        if worst_cid is None:
            break  # no chain exceeds threshold

        invalid.add(worst_cid)
        remaining.discard(worst_cid)
        del trees[worst_cid]

    return invalid


def _chain_resolved_coords(structure: Structure) -> dict[int, np.ndarray]:
    """Map each chain's asym_id → (K, 3) array of its resolved-atom coords.

    Uses the chain's contiguous ``[atom_idx, atom_idx + atom_num)`` slice of the
    atom array and keeps only ``is_present`` atoms (unresolved / missing-coord
    atoms are dropped so they can't seed false zero-distance contacts). Chains
    with no resolved atoms map to an empty ``(0, 3)`` array.
    """
    atoms = structure.atoms
    out: dict[int, np.ndarray] = {}
    for ch in structure.chains:
        aid = int(ch["asym_id"])
        a0 = int(ch["atom_idx"])
        an = int(ch["atom_num"])
        if an <= 0:
            out[aid] = np.empty((0, 3), dtype=np.float64)
            continue
        sl = atoms[a0 : a0 + an]
        m = sl["is_present"]
        if not m.any():
            out[aid] = np.empty((0, 3), dtype=np.float64)
            continue
        out[aid] = np.column_stack([sl["x"][m], sl["y"][m], sl["z"][m]]).astype(np.float64)
    return out


def _interface_coords(coords_a: np.ndarray, coords_b: np.ndarray, distance: float) -> np.ndarray:
    """Resolved atoms of chains a/b that lie within ``distance`` of the other chain.

    Mirrors OF3 ``get_interface_atoms`` restricted to the preferred chain pair:
    returns the union of a-atoms near b and b-atoms near a. Empty ``(0, 3)`` if
    there is no contact (or scipy is unavailable).
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.empty((0, 3), dtype=np.float64)
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return np.empty((0, 3), dtype=np.float64)
    tree_a, tree_b = KDTree(coords_a), KDTree(coords_b)
    a_near = [i for i, nbrs in enumerate(tree_a.query_ball_tree(tree_b, distance)) if nbrs]
    b_near = [i for i, nbrs in enumerate(tree_b.query_ball_tree(tree_a, distance)) if nbrs]
    parts: list[np.ndarray] = []
    if a_near:
        parts.append(coords_a[a_near])
    if b_near:
        parts.append(coords_b[b_near])
    return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.float64)


def _pick_reference_atom(
    coords_by_chain: dict[int, np.ndarray],
    preferred: list[int],
    anchor_iface: tuple[int, int] | None,
    interface_distance: float,
    nonligand_ids: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    """Random resolved reference atom, biased to the preferred chain/interface.

    Mirrors OF3 ``get_resolved_and_preferred_token_center_atoms`` +
    ``get_interface_atoms``: draw a random resolved atom from the preferred
    interface subregion if an interface anchor is given, otherwise from the
    preferred chain(s); fall back to all non-ligand resolved atoms when the
    preferred region has none. Returns ``None`` only if nothing is resolved.
    """
    # Interface anchor: prefer the contact subregion between the two chains.
    if anchor_iface is not None:
        ca = coords_by_chain.get(anchor_iface[0], np.empty((0, 3)))
        cb = coords_by_chain.get(anchor_iface[1], np.empty((0, 3)))
        iface = _interface_coords(ca, cb, interface_distance)
        if len(iface) > 0:
            return iface[rng.randrange(len(iface))]

    pool_ids = preferred if preferred else nonligand_ids
    parts = [coords_by_chain[a] for a in pool_ids if len(coords_by_chain.get(a, np.empty((0, 3)))) > 0]
    if not parts:
        # Nothing resolved in the preferred region → any resolved atom anywhere.
        parts = [c for c in coords_by_chain.values() if len(c) > 0]
        if not parts:
            return None
    allc = np.concatenate(parts)
    return allc[rng.randrange(len(allc))]


def _reinclude_ligands(
    coords_by_chain: dict[int, np.ndarray],
    ligand_ids: list[int],
    kept: set[int],
    distance: float,
) -> set[int]:
    """Ligand chains with any resolved atom within ``distance`` of a kept chain.

    Mirrors OF3 ``crop_chainwise`` step 10 (``ligand_inclusion_distance``): re-add
    ligands proximal to the retained chains regardless of the chain limit.
    """
    if not ligand_ids:
        return set()
    kept_parts = [coords_by_chain[a] for a in kept if len(coords_by_chain.get(a, np.empty((0, 3)))) > 0]
    if not kept_parts:
        return set()
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return set()
    tree = KDTree(np.concatenate(kept_parts))
    out: set[int] = set()
    for lid in ligand_ids:
        lc = coords_by_chain.get(lid, np.empty((0, 3)))
        if len(lc) == 0:
            continue
        if any(len(n) > 0 for n in tree.query_ball_point(lc, distance)):
            out.add(lid)
    return out


def select_interface_chains(
    structure: Structure,
    anchor: tuple[str, int, int] | None = None,
    max_chains: int = 20,
    interface_distance: float = 15.0,
    ligand_inclusion_distance: float = 5.0,
    rng: random.Random | None = None,
) -> set[int]:
    """Spatial chain "pre-crop" for large assemblies (AF3 §2.5.4 / OF3 ``crop_chainwise``).

    When a structure has more than ``max_chains`` *effective* (non-ligand)
    chains, keep the ``max_chains`` spatially-closest chains around a random
    reference atom drawn from the sampled ``anchor`` (preferred chain or
    interface), then re-include any ligand within ``ligand_inclusion_distance``
    of a kept chain. The preferred chain/interface is *always* kept — this fixes
    the prior greedy-interface pre-crop, which was anchor-independent and could
    drop the very chain/interface the sampler drew.

    Unlike the previous metadata-only greedy selector, this operates on the
    resolved atom geometry and the drawn anchor (so it must run per sample, not
    once per record). Chains with no resolved atoms are never selected as
    "closest" (matching OF3, which ranks only resolved token-center atoms) but
    an anchor chain is still force-kept.

    Parameters mirror OF3 ``crop_chainwise``. ``anchor`` is
    ``(kind, anchor_a, anchor_b)`` from the weighted sampler:
    ``kind="chain"`` → preferred chain ``anchor_a``; ``kind="interface"`` →
    preferred interface ``(anchor_a, anchor_b)``.

    Returns the set of asym_ids to **KEEP**. When the assembly is already within
    the chain limit, every chain is kept.

    Adaptation notes vs OF3:
      * DF3 ranks by minimum distance to any *resolved atom* of a chain rather
        than to its token-center atom; for chain-proximity ranking this is
        essentially equivalent (interfaces are defined by close contacts) and
        avoids re-tokenizing the full assembly just to pre-crop it.
      * Ligand re-inclusion uses resolved atoms (OF3 uses the raw atom array);
        this prevents unresolved (zero-coord) atoms from faking contacts.
    """
    if rng is None:
        rng = random.Random()

    chains = structure.chains
    if len(chains) == 0:
        return set()

    all_ids = [int(c["asym_id"]) for c in chains]
    mol_type = {int(c["asym_id"]): int(c["mol_type"]) for c in chains}
    ligand_ids = [aid for aid in all_ids if mol_type[aid] == const.MOL_NONPOLYMER]
    nonligand_ids = [aid for aid in all_ids if mol_type[aid] != const.MOL_NONPOLYMER]

    # Only pre-crop when there are too many *effective* (non-ligand) chains.
    if len(nonligand_ids) <= max_chains:
        return set(all_ids)

    coords_by_chain = _chain_resolved_coords(structure)

    # Preferred chain / interface from the sampler anchor.
    preferred: list[int] = []
    anchor_iface: tuple[int, int] | None = None
    if anchor is not None:
        kind, a, b = anchor
        if kind == "chain":
            preferred = [int(a)]
        elif kind == "interface":
            preferred = [int(a), int(b)]
            anchor_iface = (int(a), int(b))

    ref = _pick_reference_atom(coords_by_chain, preferred, anchor_iface, interface_distance, nonligand_ids, rng)

    if ref is None:
        # No resolved atoms at all — cannot rank spatially; keep the first
        # ``max_chains`` non-ligand chains plus the (force-kept) preferred set.
        kept = set(nonligand_ids[:max_chains]) | set(preferred)
    else:
        # Rank non-ligand chains by minimum resolved-atom distance to the
        # reference atom; unresolved chains are not ranking candidates.
        dist_by_chain: dict[int, float] = {}
        for aid in nonligand_ids:
            cc = coords_by_chain.get(aid)
            if cc is None or len(cc) == 0:
                continue
            dist_by_chain[aid] = float(np.min(np.linalg.norm(cc - ref, axis=1)))
        ranked = sorted(dist_by_chain, key=lambda a: dist_by_chain[a])
        kept = set(ranked[:max_chains]) | set(preferred)

    # Re-include proximal ligands (never counted toward the chain limit).
    kept |= _reinclude_ligands(coords_by_chain, ligand_ids, kept, ligand_inclusion_distance)

    return kept


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

    # NOTE: large-assembly chain "pre-crop" (>max_chains effective chains →
    # keep the spatially-closest max_chains around the sampled anchor) is NOT
    # applied here. It belongs at training time (sampler/dataset) because it is
    # anchor- and geometry-dependent and the limit varies by stage (initial=20,
    # fine-tune=larger). Use select_interface_chains() from this module per sample.

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
