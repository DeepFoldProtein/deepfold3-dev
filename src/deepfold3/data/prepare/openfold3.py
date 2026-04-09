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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from deepfold3.data import const
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

# Reverse maps for sequence reconstruction (module-level to avoid per-call rebuild)
_RNA_RES_TO_LETTER: dict[str, str] = (
    {v: k for k, v in const.RNA_LETTER_TO_RES.items()} if hasattr(const, "RNA_LETTER_TO_RES") else {}
)
_DNA_RES_TO_LETTER: dict[str, str] = (
    {v: k for k, v in const.DNA_LETTER_TO_RES.items()} if hasattr(const, "DNA_LETTER_TO_RES") else {}
)


def _chain_sequence(res_names: list[str], mol_type: int) -> str:
    """Reconstruct a 1-letter sequence from a list of 3-letter residue names."""
    if mol_type == const.MOL_PROTEIN:
        return "".join(const.THREE_LETTER_TO_ONE.get(n, "X") for n in res_names)
    elif mol_type == const.MOL_RNA:
        return "".join(_RNA_RES_TO_LETTER.get(n, n) for n in res_names)
    elif mol_type == const.MOL_DNA:
        return "".join(_DNA_RES_TO_LETTER.get(n, n) for n in res_names)
    return ""


def convert_of3_structure(npz_path: Path, pdb_id: str | None = None) -> tuple[Structure, Record]:
    """Convert an OpenFold 3 biotite-NPZ file to DeepFold3 Structure + Record.

    The OpenFold 3 NPZ contains flat per-atom arrays:
    - coord [N, 3]: atomic coordinates
    - chain_id [N]: integer chain identifiers
    - res_id [N]: residue sequence numbers
    - res_name [N]: 3-letter residue names (Unicode strings)
    - atom_name [N]: atom names (Unicode strings)
    - element [N]: element symbols (Unicode strings)
    - molecule_type_id [N]: 0=protein, 1=RNA, 2=DNA, 3=ligand
    - occupancy [N]: occupancy (optional)
    - charge [N]: formal charge (optional)
    - bonds: BondList array (optional) [B, 3] (atom1, atom2, bond_type)

    Args:
        npz_path: Path to an OpenFold 3 structure NPZ file.
        pdb_id: Override PDB ID (defaults to filename stem).

    Returns:
        (Structure, Record) tuple in DeepFold3 format.
    """
    if pdb_id is None:
        pdb_id = npz_path.stem.split(".")[0].lower()

    data = np.load(npz_path, allow_pickle=False)

    coord = data["coord"]  # [N, 3]
    n_atoms = coord.shape[0]

    chain_ids = data["chain_id"]
    res_ids = data["res_id"]
    res_names = np.array([s.strip() for s in data["res_name"].astype(str)])
    atom_names = np.array([s.strip() for s in data["atom_name"].astype(str)])
    elements = np.array([s.strip().upper() for s in data["element"].astype(str)])

    mol_type_ids = data["molecule_type_id"].astype(int) if "molecule_type_id" in data else np.zeros(n_atoms, dtype=int)
    occupancy = data["occupancy"].astype(float) if "occupancy" in data else np.ones(n_atoms, dtype=float)
    charges = data["charge"].astype(float) if "charge" in data else np.zeros(n_atoms, dtype=float)

    if "entity_id" in data:
        entity_ids_arr = data["entity_id"].astype(int)
    elif "label_entity_id" in data:
        entity_ids_arr = data["label_entity_id"].astype(int)
    else:
        entity_ids_arr = chain_ids.copy()

    # Bond list: OF3 stores BondList as [B, 3] array (atom1, atom2, bond_type)
    if "bonds" in data:
        raw_bonds = data["bonds"]
        if raw_bonds.ndim == 2 and raw_bonds.shape[1] >= 2:
            bond_count = raw_bonds.shape[0]
        else:
            bond_count = 0
    else:
        raw_bonds = None
        bond_count = 0

    # Decode chain name arrays once (outside per-chain loop)
    label_ids = data["label_asym_id"].astype(str) if "label_asym_id" in data else None
    auth_ids = data["auth_asym_id"].astype(str) if "auth_asym_id" in data else None

    # Identify unique chains preserving order of first occurrence
    seen_chains: dict[int, int] = {}
    chain_order: list[int] = []
    for cid in chain_ids:
        cid = int(cid)
        if cid not in seen_chains:
            seen_chains[cid] = len(chain_order)
            chain_order.append(cid)

    # Group atom indices by (chain_id, res_id)
    chain_to_residue_atoms: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in range(n_atoms):
        chain_to_residue_atoms[int(chain_ids[i])][int(res_ids[i])].append(i)

    atoms_list: list[tuple] = []
    residues_list: list[tuple] = []
    chains_list: list[tuple] = []
    old_to_new = np.empty(n_atoms, dtype=np.int32)

    atom_offset = 0
    res_offset = 0
    entity_sym_counter: Counter = Counter()  # entity_id → next sym_id

    for asym_id, cid in enumerate(chain_order):
        residue_map = chain_to_residue_atoms[cid]
        sorted_res_ids = sorted(residue_map.keys())

        chain_atom_start = atom_offset
        chain_res_start = res_offset
        chain_mol_type = None
        chain_entity_id = None
        chain_res_names: list[str] = []

        for rid in sorted_res_ids:
            atom_indices = residue_map[rid]
            n_res_atoms = len(atom_indices)

            first = atom_indices[0]
            rname = str(res_names[first])
            of3_mol = int(mol_type_ids[first])
            df3_mol = _OF3_MOL_TO_DF3.get(of3_mol, const.MOL_NONPOLYMER)
            eid = int(entity_ids_arr[first])

            if chain_mol_type is None:
                chain_mol_type = df3_mol
            if chain_entity_id is None:
                chain_entity_id = eid

            rtype = const.get_res_type(rname, df3_mol)
            is_standard = rtype != 20 or df3_mol == const.MOL_NONPOLYMER
            chain_res_names.append(rname)

            any_present = any(occupancy[j] > 0.0 for j in atom_indices)

            for j in atom_indices:
                elem_int = const.element_name_to_int(str(elements[j]))
                aname = str(atom_names[j])[:4]
                atoms_list.append(
                    (
                        float(coord[j, 0]),
                        float(coord[j, 1]),
                        float(coord[j, 2]),
                        np.int8(elem_int),
                        np.float16(charges[j]),
                        aname.encode("ascii"),
                        bool(occupancy[j] > 0.0),
                        np.float32(0.0),
                    )
                )
                old_to_new[j] = atom_offset
                atom_offset += 1

            residues_list.append(
                (
                    rname[:5].encode("ascii"),
                    np.int16(rtype),
                    np.int8(df3_mol),
                    np.int32(rid),
                    np.int32(chain_atom_start + (atom_offset - chain_atom_start - n_res_atoms)),
                    np.int16(n_res_atoms),
                    bool(is_standard),
                    bool(any_present),
                )
            )
            res_offset += 1

        chain_atom_count = atom_offset - chain_atom_start
        chain_res_count = res_offset - chain_res_start

        if chain_mol_type is None:
            chain_mol_type = const.MOL_PROTEIN
        if chain_entity_id is None:
            chain_entity_id = asym_id

        sym_id = entity_sym_counter[chain_entity_id]
        entity_sym_counter[chain_entity_id] += 1

        # Chain name from labels
        if label_ids is not None:
            first_chain_atom = residue_map[sorted_res_ids[0]][0]
            cname = str(label_ids[first_chain_atom]).strip()[:4]
        elif auth_ids is not None:
            first_chain_atom = residue_map[sorted_res_ids[0]][0]
            cname = str(auth_ids[first_chain_atom]).strip()[:4]
        else:
            cname = chr(ord("A") + asym_id) if asym_id < 26 else str(asym_id)

        chains_list.append(
            (
                cname.encode("ascii"),
                np.int8(chain_mol_type),
                np.int32(chain_entity_id),
                np.int32(sym_id),
                np.int32(asym_id),
                np.int32(chain_res_start),
                np.int32(chain_res_count),
                np.int32(chain_atom_start),
                np.int32(chain_atom_count),
            )
        )

    atoms_arr = np.array(atoms_list, dtype=ATOM_DTYPE) if atoms_list else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(residues_list, dtype=RESIDUE_DTYPE) if residues_list else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(chains_list, dtype=CHAIN_DTYPE) if chains_list else np.empty(0, dtype=CHAIN_DTYPE)

    # Convert bonds (remap indices)
    if bond_count > 0:
        bonds_out = []
        for b in range(bond_count):
            a1 = int(raw_bonds[b, 0])
            a2 = int(raw_bonds[b, 1])
            btype = int(raw_bonds[b, 2]) if raw_bonds.shape[1] > 2 else 1
            if a1 < n_atoms and a2 < n_atoms:
                bonds_out.append((old_to_new[a1], old_to_new[a2], np.int8(btype)))
        bonds_arr = np.array(bonds_out, dtype=BOND_DTYPE) if bonds_out else np.empty(0, dtype=BOND_DTYPE)
    else:
        bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    structure = Structure(atoms=atoms_arr, bonds=bonds_arr, residues=residues_arr, chains=chains_arr, name=pdb_id)

    # Build Record
    chain_infos: list[ChainInfo] = []
    for ci, chain_row in enumerate(chains_arr):
        mol_type = int(chain_row["mol_type"])
        chain_name = chain_row["name"].decode("ascii").strip()
        n_res = int(chain_row["res_num"])
        entity_id = int(chain_row["entity_id"])

        res_start = int(chain_row["res_idx"])
        res_slice = residues_arr[res_start : res_start + n_res]
        res_name_list = [r["name"].decode("ascii").strip() for r in res_slice]
        seq = _chain_sequence(res_name_list, mol_type)
        msa_id = _hash_sequence(seq) if seq else f"{pdb_id}_{chain_name}"

        chain_infos.append(
            ChainInfo(
                chain_id=ci,
                chain_name=chain_name,
                mol_type=mol_type,
                cluster_id=msa_id,
                msa_id=msa_id,
                num_residues=n_res,
                entity_id=entity_id,
            )
        )

    # Detect interfaces using shared KDTree-based implementation
    def _get_chain_coords(chain_row: np.void) -> np.ndarray:
        start = int(chain_row["atom_idx"])
        end = start + int(chain_row["atom_num"])
        return np.stack([atoms_arr["x"][start:end], atoms_arr["y"][start:end], atoms_arr["z"][start:end]], axis=1)

    chain_atoms = {i: _get_chain_coords(chains_arr[i]) for i in range(len(chains_arr))}
    interfaces = _compute_interfaces(chain_atoms)

    record = Record(id=pdb_id, chains=chain_infos, interfaces=interfaces)
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
