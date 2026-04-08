"""Convert PDB mmCIF → DeepFold3 Structure with biological assembly expansion.

Reads a raw mmCIF file, expands biological assembly 1 using gemmi,
and converts to the DeepFold3 Structure format (numpy structured arrays).
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import zstandard as zstd

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for mmCIF preprocessing: pip install gemmi") from e

from deepfold3.data import const
from deepfold3.data.parse.mmcif import mol_type_from_gemmi
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    ChainInfo,
    InterfaceInfo,
    Record,
    Structure,
    StructureInfo,
)

logger = logging.getLogger(__name__)

_ELEMENT_H = gemmi.Element("H")
_ELEMENT_D = gemmi.Element("D")

# Interface detection cutoff (Å)
_INTERFACE_DIST = 5.0


def _read_cif_text(path: Path) -> str:
    """Read a CIF file as text, transparently decompressing ``.gz`` and ``.zst``."""
    s = str(path)
    if s.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            return dctx.decompress(f.read()).decode("utf-8")
    elif s.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.read()
    else:
        return path.read_text()


def _strip_duplicate_loops(text: str) -> gemmi.cif.Document:
    """Parse CIF text, stripping duplicate loop blocks that gemmi rejects."""
    # Remove duplicate loop_ blocks: keep the first occurrence of each
    # category prefix (e.g. _chem_comp.), strip later duplicates.
    seen_categories: set[str] = set()
    result_parts: list[str] = []
    in_dup = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("loop_"):
            in_dup = False  # reset
            result_parts.append(line)
            continue
        m = re.match(r"(_\w+)\.", stripped)
        if m and result_parts and result_parts[-1].strip() == "loop_":
            cat = m.group(1)
            if cat in seen_categories:
                # Duplicate category — skip this entire loop
                in_dup = True
                result_parts.pop()  # remove the loop_ line
                continue
            seen_categories.add(cat)
        if not in_dup:
            result_parts.append(line)
        elif stripped.startswith("#"):
            in_dup = False

    return gemmi.cif.read_string("".join(result_parts))


def _read_cif_lenient(path: Path) -> gemmi.cif.Document:
    """Read a CIF file, stripping duplicate loop blocks that gemmi rejects."""
    return _strip_duplicate_loops(_read_cif_text(path))


def _extract_metadata(block: gemmi.cif.Block, resolution: float | None) -> StructureInfo:
    """Extract resolution, method, dates from a CIF block."""
    method = None
    deposited = None
    released = None
    try:
        method_val = block.find_value("_exptl.method")
        if method_val:
            method = gemmi.cif.as_string(method_val)
        dep_val = block.find_value("_pdbx_database_status.recvd_initial_deposition_date")
        if dep_val:
            deposited = gemmi.cif.as_string(dep_val)
        rel_val = block.find_value("_pdbx_audit_revision_history.revision_date")
        if rel_val:
            released = gemmi.cif.as_string(rel_val)
    except Exception:
        logger.debug("Failed to extract some metadata fields", exc_info=True)
    return StructureInfo(
        resolution=resolution,
        method=method,
        deposited=deposited,
        released=released,
    )


def _expand_assembly(st: gemmi.Structure, assembly_id: str = "1") -> gemmi.Model:
    """Expand biological assembly using gemmi symmetry operations.

    Falls back to asymmetric unit (model 0) if no assemblies defined.
    """
    if not st.assemblies:
        logger.debug("%s: no assemblies defined, using asymmetric unit", st.name)
        return st[0]

    # Find requested assembly
    assembly = None
    for a in st.assemblies:
        if a.name == assembly_id:
            assembly = a
            break

    if assembly is None:
        logger.debug("%s: assembly %s not found, using first assembly", st.name, assembly_id)
        assembly = st.assemblies[0]

    return gemmi.make_assembly(assembly, st[0], gemmi.HowToNameCopiedChain.Short)


def _hash_sequence(seq: str) -> str:
    """SHA256 hash of sequence string (used as MSA ID)."""
    return hashlib.sha256(seq.encode()).hexdigest()


def _chain_sequence(residues: list[tuple[str, int, bool]]) -> str:
    """Build canonical sequence string from residue list for entity/MSA dedup."""
    return "|".join(res_name for res_name, _, _ in residues)


def _compute_interfaces(
    chain_atoms: dict[int, np.ndarray],
    cutoff: float = _INTERFACE_DIST,
) -> list[InterfaceInfo]:
    """Detect chain-chain interfaces by atom proximity."""
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return []

    chain_ids = sorted(chain_atoms.keys())
    interfaces = []
    for i, c1 in enumerate(chain_ids):
        coords1 = chain_atoms[c1]
        if len(coords1) == 0:
            continue
        tree1 = KDTree(coords1)
        for c2 in chain_ids[i + 1 :]:
            coords2 = chain_atoms[c2]
            if len(coords2) == 0:
                continue
            # Check if any atom pair is within cutoff
            dists = tree1.query(coords2, k=1)[0]
            if np.any(dists < cutoff):
                interfaces.append(InterfaceInfo(chain_1=c1, chain_2=c2))
    return interfaces


def process_mmcif(
    path: Path,
    use_assembly: bool = True,
    assembly_id: str = "1",
) -> tuple[Structure, Record]:
    """Process a PDB mmCIF file into a DeepFold3 Structure and Record.

    Args:
        path: Path to .cif, .cif.gz, or .cif.zst file.
        use_assembly: If True, expand biological assembly (default: assembly 1).
        assembly_id: Which assembly to expand (default "1").

    Returns:
        Tuple of (Structure, Record).
    """
    # Read CIF text once, then parse. Re-uses the cached text on retry so
    # compressed files (.gz/.zst) are only decompressed once.
    # Some CIF files (e.g. PATCHR Atlas ModelCIF) have duplicate _chem_comp
    # loops; gemmi rejects these, so we strip duplicates on retry.
    cif_text = _read_cif_text(path)
    try:
        doc = gemmi.cif.read_string(cif_text)
    except RuntimeError as exc:
        if "duplicate tag" in str(exc):
            doc = _strip_duplicate_loops(cif_text)
        else:
            raise
    st = gemmi.make_structure_from_block(doc[0])
    file_stem = Path(path).stem.split(".")[0].lower()
    # Use entry ID from CIF, but fall back to filename for generic IDs
    # (e.g. ModelCIF files use "_entry.id model")
    pdb_id = st.name.lower() if st.name and st.name.lower() != "model" else file_stem

    resolution = st.resolution if st.resolution > 0 else None
    metadata = _extract_metadata(doc[0], resolution)

    # Build entity → mol_type mapping
    entity_mol_type: dict[str, int] = {}
    for entity in st.entities:
        mt = mol_type_from_gemmi(entity)
        if mt is not None:
            entity_mol_type[entity.name] = mt

    # Expand assembly or use asymmetric unit
    if use_assembly:
        model = _expand_assembly(st, assembly_id)
    else:
        model = st[0]

    # Pre-build subchain label → entity name mapping (O(1) lookup)
    subchain_to_entity: dict[str, str] = {}
    for entity in st.entities:
        for sc in entity.subchains:
            subchain_to_entity[sc] = entity.name

    # Parse model into arrays
    all_atoms: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    # Entity deduplication: sequence → (entity_id, next_sym_id)
    seq_to_entity: dict[str, list[int]] = {}
    next_entity_id = 1

    atom_offset = 0
    res_offset = 0
    asym_id = 0

    chain_infos: list[ChainInfo] = []
    chain_atom_coords: dict[int, np.ndarray] = {}  # asym_id → coords for interface detection

    for chain in model:
        # Get subchains as ResidueSpan objects (works for both raw and assembly models)
        subchain_spans = chain.subchains()
        if not subchain_spans:
            continue

        for sc_span in subchain_spans:
            if len(sc_span) == 0:
                continue

            # Get subchain label from first residue
            sc_label = sc_span[0].subchain
            if not sc_label:
                sc_label = chain.name

            # Resolve entity via pre-built mapping
            # Assembly labels are like "C:A" — the part after ":" is the original subchain
            orig_label = sc_label.split(":")[-1] if ":" in sc_label else sc_label
            entity_name = subchain_to_entity.get(orig_label, orig_label)

            if entity_name not in entity_mol_type:
                continue  # skip water, unknown

            mol_type = entity_mol_type[entity_name]

            # Build residue data for this subchain
            chain_res_data: list[tuple[str, int, bool]] = []  # (res_name, res_id, is_standard)
            chain_atom_data: list[tuple] = []  # full atom tuples
            chain_coords: list[list[float]] = []

            for res in sc_span:
                if res.entity_type == gemmi.EntityType.Water:
                    continue

                res_name = res.name[:5]
                res_type = const.get_res_type(res.name, mol_type)
                try:
                    res_id = res.seqid.num
                except Exception:
                    res_id = res_offset + len(chain_res_data)

                is_standard = mol_type != const.MOL_NONPOLYMER and res_type < 20
                res_atom_start = atom_offset + len(chain_atom_data)
                res_atom_count = 0

                for atom in res:
                    if atom.element in (_ELEMENT_H, _ELEMENT_D):
                        continue

                    pos = atom.pos
                    element = const.element_name_to_int(atom.element.name)
                    charge = float(atom.charge) if hasattr(atom, "charge") else 0.0
                    atom_name = atom.name[:4].ljust(4)
                    b_factor = float(atom.b_iso) if hasattr(atom, "b_iso") else 0.0

                    chain_atom_data.append(
                        (
                            float(pos.x),
                            float(pos.y),
                            float(pos.z),
                            element,
                            charge,
                            atom_name.encode("ascii"),
                            True,
                            b_factor,
                        )
                    )
                    chain_coords.append([float(pos.x), float(pos.y), float(pos.z)])
                    res_atom_count += 1

                chain_res_data.append((res_name, res_id, is_standard))

                all_residues.append(
                    (
                        res_name.encode("ascii"),
                        res_type,
                        mol_type,
                        int(res_id),
                        res_atom_start,
                        res_atom_count,
                        is_standard,
                        res_atom_count > 0,
                    )
                )

            if not chain_res_data:
                continue

            # Entity/sym assignment
            seq_key = _chain_sequence(chain_res_data)
            if seq_key in seq_to_entity:
                entity_id = seq_to_entity[seq_key][0]
                sym_id = seq_to_entity[seq_key][1]
                seq_to_entity[seq_key][1] += 1
            else:
                entity_id = next_entity_id
                sym_id = 0
                seq_to_entity[seq_key] = [entity_id, 1]
                next_entity_id += 1

            chain_atom_count = len(chain_atom_data)
            chain_res_count = len(chain_res_data)

            all_atoms.extend(chain_atom_data)

            # Use the parent chain name as the structural chain identifier.
            # For assembly-expanded models this is the short renamed id (A, B, C, D, ...).
            # Multiple subchains within the same chain share this name; they are
            # differentiated by asym_id.
            chain_label = chain.name
            all_chains.append(
                (
                    chain_label[:4].encode("ascii"),
                    mol_type,
                    entity_id,
                    sym_id,
                    asym_id,
                    res_offset,
                    chain_res_count,
                    atom_offset,
                    chain_atom_count,
                )
            )

            # Store coords for interface detection
            if chain_coords:
                chain_atom_coords[asym_id] = np.array(chain_coords, dtype=np.float64)
            else:
                chain_atom_coords[asym_id] = np.empty((0, 3), dtype=np.float64)

            # Build ChainInfo
            seq_hash = _hash_sequence(seq_key)
            chain_infos.append(
                ChainInfo(
                    chain_id=asym_id,
                    chain_name=chain_label[:4],
                    mol_type=mol_type,
                    cluster_id=seq_hash,  # placeholder until clustering
                    msa_id=seq_hash,
                    num_residues=chain_res_count,
                    entity_id=entity_id,
                )
            )

            atom_offset += chain_atom_count
            res_offset += chain_res_count
            asym_id += 1

    # Build numpy arrays
    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    structure = Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
        name=pdb_id,
    )

    # Compute interfaces
    interfaces = _compute_interfaces(chain_atom_coords)

    metadata.num_chains = len(chain_infos)
    metadata.num_interfaces = len(interfaces)

    record = Record(
        id=pdb_id,
        chains=chain_infos,
        interfaces=interfaces,
        structure_info=metadata,
    )

    return structure, record
