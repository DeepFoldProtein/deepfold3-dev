"""Convert OpenFold 3 preprocessed data to DeepFold3 format.

OpenFold 3 (Apache 2.0, AlQuraishi Laboratory) stores preprocessed
structures as biotite AtomArray NPZ files with flat per-atom annotation
arrays.  This module converts those files into DeepFold3's structured
numpy-array format (ATOM_DTYPE, RESIDUE_DTYPE, CHAIN_DTYPE, BOND_DTYPE)
and produces a DeepFold3 manifest.

Only the *conversion logic* is original — no OpenFold 3 code is copied.
The NPZ field names are treated as a data format specification.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from deepfold3.data import const
from deepfold3.data.io_cache import read_npz
from deepfold3.data.prepare.structure import _compute_interfaces, _hash_sequence
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    ChainInfo,
    Record,
    Structure,
    StructureInfo,
)

logger = logging.getLogger(__name__)

# OpenFold 3 MoleculeType → DeepFold3 mol_type.
# OF3: PROTEIN=0, RNA=1, DNA=2, LIGAND=3
# DF3: PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3
_OF3_MOL_TO_DF3 = {0: const.MOL_PROTEIN, 1: const.MOL_RNA, 2: const.MOL_DNA, 3: const.MOL_NONPOLYMER}


# Residue-type index -> 1-letter code, used to rebuild a chain sequence from the
# already-resolved ``res_type`` rather than from raw residue names. Going through
# ``res_type`` is what makes modified residues (SEP -> S), unknown protein (UNK ->
# X) and unknown nucleic (N / DN -> N) come out as single characters; the previous
# name-keyed maps emitted the raw 2-character "DN" and inflated the sequence.
_RES_TYPE_TO_LETTER: dict[int, str] = {}
for _n, _i in const.RES_TYPE_ORDER.items():
    if _i <= 20:
        _RES_TYPE_TO_LETTER.setdefault(_i, const.THREE_LETTER_TO_ONE.get(_n, "X"))
    elif _i == 21:
        _RES_TYPE_TO_LETTER[_i] = "-"
    elif _i <= 25:
        _RES_TYPE_TO_LETTER.setdefault(_i, _n)  # RNA A/G/C/U
    elif _i <= 29:
        _RES_TYPE_TO_LETTER.setdefault(_i, _n[1:])  # DNA DA/DG/DC/DT -> A/G/C/T
    else:
        _RES_TYPE_TO_LETTER[_i] = "N"  # unknown nucleic


def _chain_sequence(res_types: np.ndarray, mol_type: int) -> str:
    """Reconstruct a 1-letter sequence from a chain's ``res_type`` indices."""
    if mol_type == const.MOL_NONPOLYMER:
        return ""
    default = "X" if mol_type == const.MOL_PROTEIN else "N"
    return "".join(_RES_TYPE_TO_LETTER.get(int(t), default) for t in res_types)


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode a biotite annotation array to stripped ``<U`` strings."""
    return np.char.strip(arr.astype(str))


def _segment_starts(*keys: np.ndarray) -> np.ndarray:
    """Indices where any of ``keys`` changes value (segment boundaries)."""
    n = len(keys[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    boundary = np.zeros(n, dtype=bool)
    boundary[0] = True
    for k in keys:
        boundary[1:] |= k[1:] != k[:-1]
    return np.flatnonzero(boundary).astype(np.int64)


def convert_of3_atom_array(
    npz_path: Path,
    pdb_id: str | None = None,
) -> tuple[Structure, Record, list[str]]:
    """Convert an OpenFold 3 biotite-NPZ file to DeepFold3 Structure + Record.

    The OpenFold 3 NPZ is a serialised biotite ``AtomArray``: ``coord`` plus one
    flat per-atom annotation array per category. The ones consumed here:

    - ``coord`` [N, 3] float32 — **NaN on unresolved atoms** (OF3 materialises the
      full reference atom set of every polymer residue, resolved or not).
    - ``chain_id`` [N] ``<U`` — OF3's chain key. Note this is a *string of an
      integer* (``"1"``, ``"3"``), and it is the key the dataset cache indexes
      chains by — not a chain letter. ``label_asym_id`` carries the letter.
    - ``res_id`` [N] int — residue sequence number within the chain.
    - ``res_name`` / ``atom_name`` / ``element`` [N] ``<U``.
    - ``molecule_type_id`` [N] int — OF3 ``MoleculeType`` (0 protein, 1 RNA,
      2 DNA, 3 ligand).
    - ``is_atomized`` [N] bool — OF3's own tokenization decision: True for every
      atom of a residue that is tokenized **per atom** (ligands *and* modified /
      non-standard polymer residues). This is the authoritative source for
      DeepFold3's ``is_standard``; re-deriving it from the residue name maps
      modified residues onto their standard parent (SEP -> SER) and would emit a
      single SER token over SEP's atom set, which the featurizer then slots
      against SER's reference atoms.
    - ``occupancy`` [N] float, ``charge`` [N] int, ``entity_id`` [N] int,
      ``label_asym_id`` / ``auth_asym_id`` [N] ``<U``.
    - ``bonds`` [B, 3] uint32 — biotite ``BondList`` (atom_1, atom_2, bond_type).

    Args:
        npz_path: Path to an OpenFold 3 structure NPZ file.
        pdb_id: Override PDB ID (defaults to filename stem).

    Returns:
        ``(Structure, Record, chain_keys)`` where ``chain_keys[asym_id]`` is the
        OF3 ``chain_id`` string of that chain — the join key into the OF3 dataset
        cache's per-structure ``chains`` dict.
    """
    if pdb_id is None:
        pdb_id = npz_path.stem.split(".")[0].lower()

    data = read_npz(npz_path)

    coord = np.asarray(data["coord"], dtype=np.float32)
    n_atoms = coord.shape[0]

    chain_key = _decode(data["chain_id"])
    res_ids = np.asarray(data["res_id"]).astype(np.int64)
    res_names = _decode(data["res_name"])
    atom_names = _decode(data["atom_name"])
    elements = np.char.upper(_decode(data["element"]))
    ins_codes = _decode(data["ins_code"]) if "ins_code" in data else np.zeros(n_atoms, dtype="<U1")

    of3_mol = np.asarray(data["molecule_type_id"]).astype(np.int64) if "molecule_type_id" in data else None
    occupancy = np.asarray(data["occupancy"], dtype=np.float64) if "occupancy" in data else np.ones(n_atoms)
    charges = np.asarray(data["charge"], dtype=np.float64) if "charge" in data else np.zeros(n_atoms)
    is_atomized = np.asarray(data["is_atomized"], dtype=bool) if "is_atomized" in data else None

    if "entity_id" in data:
        entity_ids = np.asarray(data["entity_id"]).astype(np.int64)
    elif "label_entity_id" in data:
        entity_ids = np.asarray(data["label_entity_id"]).astype(np.int64)
    else:
        entity_ids = np.zeros(n_atoms, dtype=np.int64)

    label_ids = _decode(data["label_asym_id"]) if "label_asym_id" in data else None
    auth_ids = _decode(data["auth_asym_id"]) if "auth_asym_id" in data else None

    # DeepFold3's Structure requires each chain's atoms (and residues) to be
    # contiguous. OF3 files already are, but a stable sort by first-occurrence
    # rank of the chain key guarantees it without reordering within a chain.
    uniq_keys, first_idx = np.unique(chain_key, return_index=True)
    chain_order = uniq_keys[np.argsort(first_idx)]
    rank = {k: i for i, k in enumerate(chain_order)}
    chain_rank = np.array([rank[k] for k in chain_key], dtype=np.int64)
    if np.any(np.diff(chain_rank) < 0):
        perm = np.argsort(chain_rank, kind="stable")
        inv_perm = np.empty(n_atoms, dtype=np.int64)
        inv_perm[perm] = np.arange(n_atoms, dtype=np.int64)
        coord, res_ids, res_names, atom_names = coord[perm], res_ids[perm], res_names[perm], atom_names[perm]
        elements, ins_codes, occupancy, charges = elements[perm], ins_codes[perm], occupancy[perm], charges[perm]
        entity_ids, chain_rank, chain_key = entity_ids[perm], chain_rank[perm], chain_key[perm]
        if of3_mol is not None:
            of3_mol = of3_mol[perm]
        if is_atomized is not None:
            is_atomized = is_atomized[perm]
        if label_ids is not None:
            label_ids = label_ids[perm]
        if auth_ids is not None:
            auth_ids = auth_ids[perm]
    else:
        inv_perm = None

    # --- per-atom arrays -------------------------------------------------
    # Unresolved atoms carry NaN coordinates in OF3; DeepFold3's convention is
    # zeros + ``is_present=False`` (the featurizer masks them out of the GT
    # losses). Leaving the NaNs in poisons the crop's distance computation and
    # every downstream loss.
    finite = np.isfinite(coord).all(axis=1)
    coord = np.where(finite[:, None], coord, 0.0).astype(np.float32)
    is_present = (occupancy > 0.0) & finite

    elem_lut = {e: const.element_name_to_int(e) for e in np.unique(elements)}
    elem_int = np.array([elem_lut[e] for e in elements], dtype=np.int8)

    atoms_arr = np.zeros(n_atoms, dtype=ATOM_DTYPE)
    atoms_arr["x"], atoms_arr["y"], atoms_arr["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    atoms_arr["element"] = elem_int
    atoms_arr["charge"] = charges.astype(np.float16)
    atoms_arr["name"] = np.char.encode(np.char.ljust(atom_names, 4).astype("<U4"), "ascii")
    atoms_arr["is_present"] = is_present
    atoms_arr["b_factor"] = 0.0

    # --- residues --------------------------------------------------------
    # A residue is a maximal run of atoms sharing (chain, res_id, ins_code,
    # res_name). Segmenting on runs (rather than grouping by res_id) preserves
    # the file's atom order, which multi-residue ligands and the bond indices
    # both depend on.
    res_starts = _segment_starts(chain_rank, res_ids, ins_codes, res_names)
    res_counts = np.diff(np.append(res_starts, n_atoms)).astype(np.int64)
    n_res = len(res_starts)

    mol_lut = np.full(4, const.MOL_NONPOLYMER, dtype=np.int8)
    for _k, _v in _OF3_MOL_TO_DF3.items():
        mol_lut[_k] = _v
    if of3_mol is not None:
        res_mol = mol_lut[np.clip(of3_mol[res_starts], 0, 3)]
    else:
        res_mol = np.full(n_res, const.MOL_PROTEIN, dtype=np.int8)

    res_name_first = res_names[res_starts]
    type_lut = {
        (nm, int(mt)): const.get_res_type(nm, int(mt))
        for nm, mt in {(a, b) for a, b in zip(res_name_first, res_mol, strict=True)}
    }
    res_type = np.array(
        [type_lut[(nm, int(mt))] for nm, mt in zip(res_name_first, res_mol, strict=True)], dtype=np.int16
    )

    if is_atomized is not None:
        # OF3's own decision, taken verbatim (see docstring).
        res_standard = ~is_atomized[res_starts]
    else:
        res_standard = (res_mol != const.MOL_NONPOLYMER) & (res_type != 20)
    res_standard &= res_mol != const.MOL_NONPOLYMER

    res_present = np.add.reduceat(is_present.astype(np.int64), res_starts) > 0 if n_res else np.zeros(0, dtype=bool)

    residues_arr = np.zeros(n_res, dtype=RESIDUE_DTYPE)
    residues_arr["name"] = np.char.encode(res_name_first.astype("<U5"), "ascii")
    residues_arr["res_type"] = res_type
    residues_arr["mol_type"] = res_mol
    residues_arr["res_id"] = res_ids[res_starts].astype(np.int32)
    residues_arr["atom_idx"] = res_starts.astype(np.int32)
    residues_arr["atom_num"] = res_counts.astype(np.int16)
    residues_arr["is_standard"] = res_standard
    residues_arr["is_present"] = res_present

    # --- chains ----------------------------------------------------------
    res_chain_rank = chain_rank[res_starts]
    ch_res_starts = _segment_starts(res_chain_rank)
    ch_res_counts = np.diff(np.append(ch_res_starts, n_res)).astype(np.int64)
    n_chains = len(ch_res_starts)

    ch_atom_starts = res_starts[ch_res_starts] if n_chains else np.zeros(0, dtype=np.int64)
    ch_atom_counts = np.diff(np.append(ch_atom_starts, n_atoms)).astype(np.int64)

    chain_keys = [str(chain_key[int(a)]) for a in ch_atom_starts]
    ch_entity = entity_ids[ch_atom_starts].astype(np.int32) if n_chains else np.zeros(0, dtype=np.int32)
    ch_mol = res_mol[ch_res_starts] if n_chains else np.zeros(0, dtype=np.int8)

    if label_ids is not None:
        ch_names = [str(label_ids[int(a)])[:4] for a in ch_atom_starts]
    elif auth_ids is not None:
        ch_names = [str(auth_ids[int(a)])[:4] for a in ch_atom_starts]
    else:
        ch_names = [chr(ord("A") + i) if i < 26 else str(i) for i in range(n_chains)]
    ch_auth = [str(auth_ids[int(a)])[:4] for a in ch_atom_starts] if auth_ids is not None else ch_names

    sym_counter: Counter = Counter()
    sym_ids = np.empty(n_chains, dtype=np.int32)
    for i, eid in enumerate(ch_entity):
        sym_ids[i] = sym_counter[int(eid)]
        sym_counter[int(eid)] += 1

    chains_arr = np.zeros(n_chains, dtype=CHAIN_DTYPE)
    chains_arr["name"] = np.array([n.encode("ascii", "replace") for n in ch_names], dtype="S4")
    chains_arr["mol_type"] = ch_mol
    chains_arr["entity_id"] = ch_entity
    chains_arr["sym_id"] = sym_ids
    chains_arr["asym_id"] = np.arange(n_chains, dtype=np.int32)
    chains_arr["res_idx"] = ch_res_starts.astype(np.int32)
    chains_arr["res_num"] = ch_res_counts.astype(np.int32)
    chains_arr["atom_idx"] = ch_atom_starts.astype(np.int32)
    chains_arr["atom_num"] = ch_atom_counts.astype(np.int32)

    # --- bonds -----------------------------------------------------------
    raw_bonds = data["bonds"] if "bonds" in data else None
    if raw_bonds is not None and raw_bonds.ndim == 2 and raw_bonds.shape[0] > 0 and raw_bonds.shape[1] >= 2:
        b = np.asarray(raw_bonds).astype(np.int64)
        a1, a2 = b[:, 0], b[:, 1]
        btype = b[:, 2] if b.shape[1] > 2 else np.ones(len(b), dtype=np.int64)
        keep = (a1 >= 0) & (a1 < n_atoms) & (a2 >= 0) & (a2 < n_atoms)
        a1, a2, btype = a1[keep], a2[keep], btype[keep]
        if inv_perm is not None:
            a1, a2 = inv_perm[a1], inv_perm[a2]
        bonds_arr = np.zeros(len(a1), dtype=BOND_DTYPE)
        bonds_arr["atom_1"], bonds_arr["atom_2"] = a1.astype(np.int32), a2.astype(np.int32)
        bonds_arr["type"] = np.clip(btype, 0, 127).astype(np.int8)
    else:
        bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    structure = Structure(atoms=atoms_arr, bonds=bonds_arr, residues=residues_arr, chains=chains_arr, name=pdb_id)

    # --- record ----------------------------------------------------------
    chain_infos: list[ChainInfo] = []
    for ci in range(n_chains):
        mol_type = int(ch_mol[ci])
        rs, rn = int(ch_res_starts[ci]), int(ch_res_counts[ci])
        seq = _chain_sequence(res_type[rs : rs + rn], mol_type)
        msa_id = _hash_sequence(seq) if seq else f"{pdb_id}_{ch_names[ci]}"
        chain_infos.append(
            ChainInfo(
                chain_id=ci,
                chain_name=ch_names[ci],
                mol_type=mol_type,
                cluster_id=msa_id,
                msa_id=msa_id,
                num_residues=rn,
                entity_id=int(ch_entity[ci]),
                auth_chain_name=ch_auth[ci],
                res_name=str(res_name_first[rs]) if mol_type == const.MOL_NONPOLYMER and rn else None,
            )
        )

    chain_atoms = {
        i: np.stack(
            [
                atoms_arr["x"][int(ch_atom_starts[i]) : int(ch_atom_starts[i]) + int(ch_atom_counts[i])],
                atoms_arr["y"][int(ch_atom_starts[i]) : int(ch_atom_starts[i]) + int(ch_atom_counts[i])],
                atoms_arr["z"][int(ch_atom_starts[i]) : int(ch_atom_starts[i]) + int(ch_atom_counts[i])],
            ],
            axis=1,
        )
        for i in range(n_chains)
    }
    interfaces = _compute_interfaces(chain_atoms)

    record = Record(id=pdb_id, chains=chain_infos, interfaces=interfaces)
    return structure, record, chain_keys


def convert_of3_structure(npz_path: Path, pdb_id: str | None = None) -> tuple[Structure, Record]:
    """``convert_of3_atom_array`` without the OF3 chain-key list.

    Kept as the entry point for callers that address chains by DeepFold3
    ``asym_id`` (the monomer ``format: of3`` path and ``deepfold3 import-of3``).
    """
    structure, record, _ = convert_of3_atom_array(npz_path, pdb_id=pdb_id)
    return structure, record


def convert_of3_structure_with_cache(
    npz_path: Path,
    cache_entry: dict | None,
    pdb_id: str | None = None,
) -> tuple[Structure, Record]:
    """Convert an OF3 NPZ with optional dataset-cache metadata.

    If a ``cache_entry`` dict is provided (from the OF3 dataset_cache.json
    ``structure_data[pdb_id]`` entry), resolution, experimental method, and
    cluster IDs are extracted and added to the Record's StructureInfo and
    ChainInfo.

    Args:
        npz_path: Path to an OpenFold 3 structure NPZ.
        cache_entry: Optional dict from OF3 dataset_cache.json for this PDB ID.
        pdb_id: Override PDB ID.

    Returns:
        (Structure, Record) tuple.
    """
    structure, record = convert_of3_structure(npz_path, pdb_id=pdb_id)

    if cache_entry is None:
        return structure, record

    resolution = cache_entry.get("resolution")
    method = cache_entry.get("experimental_method")
    release_date = cache_entry.get("release_date")
    record.structure_info = StructureInfo(
        resolution=float(resolution) if resolution is not None else None,
        method=str(method) if method else None,
        released=str(release_date) if release_date else None,
        num_chains=len(record.chains),
        num_interfaces=len(record.interfaces),
    )

    cache_chains = cache_entry.get("chains", {})
    for chain_info in record.chains:
        cname = chain_info.chain_name
        if cname in cache_chains:
            cc = cache_chains[cname]
            cid = cc.get("cluster_id")
            if cid is not None:
                chain_info.cluster_id = int(cid) if isinstance(cid, (int, float)) else cid

    return structure, record


def load_of3_dataset_cache(cache_path: Path) -> dict[str, dict]:
    """Load an OpenFold 3 dataset_cache.json and return per-PDB metadata.

    Returns a dict mapping lowercase PDB ID to the cache entry dict.
    Handles both raw JSON format and the ``_type``-prefixed format that OF3
    uses for its dataclass hierarchy.
    """
    with open(cache_path) as f:
        raw = json.load(f)

    if "structure_data" in raw:
        return {k.lower(): v for k, v in raw["structure_data"].items()}
    return {k.lower(): v for k, v in raw.items()}


def scan_of3_structures(structures_dir: Path) -> list[tuple[str, Path]]:
    """Scan an OpenFold 3 target_structures_directory for NPZ files.

    Supports both layouts:
    - ``{pdb_id}/{pdb_id}.npz`` (standard OF3)
    - ``{pdb_id}.npz`` (flat)

    Returns list of (pdb_id, npz_path) tuples.
    """
    results: list[tuple[str, Path]] = []

    for subdir in sorted(structures_dir.iterdir()):
        if subdir.is_dir():
            npz = subdir / f"{subdir.name}.npz"
            if npz.exists():
                results.append((subdir.name.lower(), npz))
            else:
                npz_alt = subdir / "structure.npz"
                if npz_alt.exists():
                    results.append((subdir.name.lower(), npz_alt))

    if not results:
        for npz in sorted(structures_dir.glob("*.npz")):
            pdb_id = npz.stem.split(".")[0].lower()
            results.append((pdb_id, npz))

    return results
