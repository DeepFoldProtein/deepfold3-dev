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
from deepfold3.data.prepare.canonical import get_canonical_one_letter, get_canonical_three_letter
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
    """Read a CIF file. Tries the fast path first; falls back to stripping
    duplicate loop blocks only if gemmi rejects the file.

    Most wwPDB mirror files parse cleanly, so the duplicate-loop scrub
    should rarely run.
    """
    text = _read_cif_text(path)
    try:
        return gemmi.cif.read_string(text)
    except (RuntimeError, ValueError):
        return _strip_duplicate_loops(text)


def _cif_loop_or_value(block: gemmi.cif.Block, tag: str) -> list[str]:
    """Return all values of a CIF tag as a list, handling key-value or loop form.

    ``gemmi.cif.Block.find_value`` returns ``None`` for loop columns and
    ``find_loop`` returns an empty Table for key-value pairs, so neither one
    handles both forms. This helper returns a list of strings (possibly empty)
    in either case.
    """
    loop = list(block.find_loop(tag))
    if loop:
        return [gemmi.cif.as_string(v) for v in loop]
    val = block.find_value(tag)
    return [gemmi.cif.as_string(val)] if val else []


# Resolution priority order from AlphaFold 3 (references/alphafold3/.../structure/mmcif.py:240-242)
_RESOLUTION_KEYS = (
    "_refine.ls_d_res_high",
    "_em_3d_reconstruction.resolution",
    "_reflns.d_resolution_high",
)


def _get_resolution(block: gemmi.cif.Block) -> float | None:
    """Resolution from the first parseable mmCIF tag, AF3-style priority."""
    for key in _RESOLUTION_KEYS:
        for raw in _cif_loop_or_value(block, key):
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _get_method(block: gemmi.cif.Block) -> str | None:
    """Comma-joined lowercased ``_exptl.method`` list (matches AlphaFold 3)."""
    methods = _cif_loop_or_value(block, "_exptl.method")
    return ",".join(m.lower() for m in methods) if methods else None


def _get_release_date(block: gemmi.cif.Block) -> str | None:
    """Earliest ``_pdbx_audit_revision_history.revision_date`` (initial release).

    Revision rows are stored as ISO 8601 ``YYYY-MM-DD``, so lexicographic
    ``min`` is also chronological.
    """
    rows = _cif_loop_or_value(block, "_pdbx_audit_revision_history.revision_date")
    return min(rows) if rows else None


def _get_deposition_date(block: gemmi.cif.Block) -> str | None:
    """``_pdbx_database_status.recvd_initial_deposition_date`` (single value)."""
    rows = _cif_loop_or_value(block, "_pdbx_database_status.recvd_initial_deposition_date")
    return rows[0] if rows else None


def _extract_metadata(block: gemmi.cif.Block) -> StructureInfo:
    """Extract resolution, method, and dates from a CIF block (AF3-aligned).

    Always returns a ``StructureInfo`` (never ``None``) — fields that can't
    be parsed are left as ``None`` so callers can rely on the dataclass
    being present even when the underlying tags are missing or malformed.
    """
    try:
        return StructureInfo(
            resolution=_get_resolution(block),
            method=_get_method(block),
            deposited=_get_deposition_date(block),
            released=_get_release_date(block),
        )
    except Exception:
        logger.debug("Failed to extract some metadata fields", exc_info=True)
        return StructureInfo()


def extract_mmcif_metadata(cif_path: Path) -> StructureInfo | None:
    """Read just the metadata block of an mmCIF (or .cif.gz / .cif.zst).

    Returns ``None`` if the file can't be parsed. Extracts resolution,
    method, deposited, and released using the same convention as AlphaFold
    3 (references/alphafold3/.../structure/mmcif.py:212-249) so the values
    are directly comparable to AF3's template-cutoff logic.
    """
    try:
        block = _read_cif_lenient(cif_path)[0]
    except Exception:
        logger.debug("Failed to read metadata from %s", cif_path, exc_info=True)
        return None
    return _extract_metadata(block)


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
    """Build canonical sequence string from residue list for entity/MSA dedup.

    Modified residues are normalized to their parent standard residue
    (AF3 §2.5.3) so that e.g. MSE chains cluster with MET chains.
    """
    return "|".join(const._NONSTANDARD_TO_STANDARD.get(res_name.strip(), res_name) for res_name, _, _ in residues)


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


def _estimate_assembly_chains(st: gemmi.Structure, assembly_id: str = "1") -> int | None:
    """Estimate chain count after assembly expansion without building it.

    Returns None if no assemblies are defined (falls back to asymmetric unit).
    """
    if not st.assemblies:
        return None
    assembly = None
    for a in st.assemblies:
        if a.name == assembly_id:
            assembly = a
            break
    if assembly is None:
        assembly = st.assemblies[0]
    return sum(len(g.operators) * len(g.subchains) for g in assembly.generators)


def process_mmcif(
    path: Path,
    use_assembly: bool = True,
    assembly_id: str = "1",
    ccd_atom_names: dict[str, set[str]] | None = None,
    ccd_leaving_atoms: dict[str, set[str]] | None = None,
    max_chains: int = 0,
) -> tuple[Structure, Record]:
    """Process a PDB mmCIF file into a DeepFold3 Structure and Record.

    Args:
        path: Path to .cif, .cif.gz, or .cif.zst file.
        use_assembly: If True, expand biological assembly (default: assembly 1).
        assembly_id: Which assembly to expand (default "1").
        ccd_atom_names: Optional CCD atom name cache for validation. If provided,
            atoms whose names are not in the CCD dictionary for their residue type
            are excluded (AF3 §2.5.4).
        ccd_leaving_atoms: Optional CCD leaving atom cache. If provided, leaving
            atoms are removed from residues involved in covalent bonds that have
            ``pdbx_leaving_atom_flag`` set (AF3 §2.5.4).
        max_chains: Skip assembly expansion if estimated chain count exceeds this.
            0 means no limit.

    Returns:
        Tuple of (Structure, Record).

    Raises:
        ValueError: If estimated assembly chain count exceeds *max_chains*.
    """
    # Some CIF files (e.g. PATCHR Atlas ModelCIF) have duplicate _chem_comp
    # loops; _read_cif_lenient strips them as a fallback.
    doc = _read_cif_lenient(path)
    st = gemmi.make_structure_from_block(doc[0])
    # Populate entity.full_sequence from _entity_poly_seq / _entity_poly so we can
    # compute canonical-sequence-based msa_id below (AF3 §2 spec). Idempotent.
    st.setup_entities()
    file_stem = Path(path).stem.split(".")[0].lower()
    # Use entry ID from CIF, but fall back to filename for generic IDs
    # (e.g. ModelCIF files use "_entry.id model")
    pdb_id = st.name.lower() if st.name and st.name.lower() != "model" else file_stem

    metadata = _extract_metadata(doc[0])

    # Pre-filter: estimate assembly chain count before expensive expansion
    if use_assembly and max_chains > 0:
        est = _estimate_assembly_chains(st, assembly_id)
        if est is not None and est > max_chains:
            raise ValueError(f"{est} chains > max {max_chains}")

    # Build entity → mol_type mapping
    entity_mol_type: dict[str, int] = {}
    # Build entity → canonical 3-letter / 1-letter sequence mappings.
    # Canonical sequence = _entity_poly.pdbx_seq_one_letter_code_can equivalent
    # (gemmi's entity.full_sequence after setup_entities()), with non-standard
    # residues mapped to standard parents. Used for ``msa_id`` hashing and the
    # MSA query string. Polymer entities only — ligands keep observed-derived
    # behaviour.
    entity_canonical_three: dict[str, list[str]] = {}
    entity_canonical_one: dict[str, str] = {}
    for entity in st.entities:
        mt = mol_type_from_gemmi(entity)
        if mt is not None:
            entity_mol_type[entity.name] = mt
        if mt is not None and mt != const.MOL_NONPOLYMER:
            three = get_canonical_three_letter(entity)
            if three is not None:
                entity_canonical_three[entity.name] = three
                one = get_canonical_one_letter(entity)
                if one is not None:
                    entity_canonical_one[entity.name] = one

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

    # AF3 §2.5.4: collect leaving atoms to exclude from covalent bond partners.
    # Uses auth_asym_id + auth_seq_id because gemmi exposes residues via
    # chain.name (= auth_asym_id) and res.seqid.num (= auth_seq_id).
    _leaving_atoms: set[tuple[str, int, str]] = set()  # (auth_asym_id, auth_seq_id, atom_name)
    if ccd_leaving_atoms:
        _conn_table = doc[0].find(
            "_struct_conn.",
            [
                "conn_type_id",
                "pdbx_leaving_atom_flag",
                "ptnr1_auth_asym_id",
                "ptnr1_label_comp_id",
                "ptnr1_auth_seq_id",
                "ptnr2_auth_asym_id",
                "ptnr2_label_comp_id",
                "ptnr2_auth_seq_id",
            ],
        )
        for row in _conn_table:
            flag = row[1].strip().strip('"').lower()
            if flag not in ("one", "both"):
                continue
            p1_chain, p1_comp, p1_seq = row[2].strip(), row[3].strip(), row[4].strip()
            p2_chain, p2_comp, p2_seq = row[5].strip(), row[6].strip(), row[7].strip()
            try:
                p1_seq_int = int(p1_seq) if p1_seq != "." else 0
                p2_seq_int = int(p2_seq) if p2_seq != "." else 0
            except ValueError:
                continue
            # "one" = partner 2 has leaving atoms; "both" = both partners
            if flag == "both" or flag == "one":
                for atom_name in ccd_leaving_atoms.get(p2_comp, ()):
                    _leaving_atoms.add((p2_chain, p2_seq_int, atom_name))
            if flag == "both":
                for atom_name in ccd_leaving_atoms.get(p1_comp, ()):
                    _leaving_atoms.add((p1_chain, p1_seq_int, atom_name))
        if _leaving_atoms:
            logger.debug(
                "%s: excluding %d leaving atoms from %d covalent bonds", pdb_id, len(_leaving_atoms), len(_conn_table)
            )

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
    chain_canonical_seqs: dict[int, str] = {}  # asym_id → canonical one-letter seq (polymer chains only)
    # Atom lookup for cross-entity bond extraction: (chain_label, res_id, atom_name) → global atom index
    # Uses parent chain.name as key because gemmi connections reference label_asym_id
    # which corresponds to the parent chain (not the subchain) after assembly expansion.
    atom_lookup: dict[tuple[str, int, str], int] = {}

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

                # Standard (single-token) polymer residues are the canonical
                # amino acids (0-19) and nucleotides (22-30, incl. unknown N/DN).
                # Only res_type 20 (UNK protein / modified residue) is atomized.
                # NB: ``res_type < 20`` wrongly atomised every nucleotide (22-29).
                is_standard = mol_type != const.MOL_NONPOLYMER and res_type != 20
                res_atom_start = atom_offset + len(chain_atom_data)
                res_atom_count = 0

                # CCD valid atom names for this residue (if cache provided)
                _ccd_valid = ccd_atom_names.get(res_name.strip()) if ccd_atom_names is not None else None

                for atom in res:
                    if atom.element in (_ELEMENT_H, _ELEMENT_D):
                        continue

                    # AF3 §2.5.4: filter atoms not in CCD dictionary
                    if _ccd_valid is not None and atom.name.strip() not in _ccd_valid:
                        continue

                    # AF3 §2.5.4: remove leaving atoms from covalent bond partners
                    # Key uses (chain.name, res_id) = (auth_asym_id, auth_seq_id)
                    if _leaving_atoms and (chain.name, res_id, atom.name.strip()) in _leaving_atoms:
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
                    global_idx = atom_offset + len(chain_atom_data) - 1
                    atom_lookup[(chain.name, res_id, atom.name.strip())] = global_idx
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

            # Entity/sym assignment.
            # Polymer chains: hash the canonical (entity.full_sequence) 3-letter list
            # so msa_id is stable under disorder and matches AF3 §2 / external MSA stashes.
            # Ligands and pathological inputs (no full_sequence): fall back to observed.
            canonical_three = entity_canonical_three.get(entity_name)
            if canonical_three is not None:
                seq_key = "|".join(canonical_three)
                # Stash canonical one-letter for the MSA-query path (Structure NPZ).
                canonical_one = entity_canonical_one.get(entity_name)
                if canonical_one is not None:
                    chain_canonical_seqs[asym_id] = canonical_one
            else:
                if mol_type != const.MOL_NONPOLYMER:
                    logger.warning(
                        "%s/%s: no canonical sequence for entity %s, falling back to observed",
                        pdb_id,
                        chain.name,
                        entity_name,
                    )
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
            # For nonpolymer (ligand) chains, store the CCD code for clustering
            ligand_res_name = chain_res_data[0][0].strip() if mol_type == const.MOL_NONPOLYMER else None

            chain_infos.append(
                ChainInfo(
                    chain_id=asym_id,
                    chain_name=chain_label[:4],
                    mol_type=mol_type,
                    cluster_id=seq_hash,  # placeholder until clustering
                    msa_id=seq_hash,
                    num_residues=chain_res_count,
                    entity_id=entity_id,
                    auth_chain_name=orig_label,
                    res_name=ligand_res_name,
                )
            )

            atom_offset += chain_atom_count
            res_offset += chain_res_count
            asym_id += 1

    # Build numpy arrays
    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    # Extract cross-entity bonds from mmCIF _struct_conn (disulfide, covalent, metal)
    conn_bonds: list[tuple[int, int, int]] = []
    _BOND_TYPES = {gemmi.ConnectionType.Covale, gemmi.ConnectionType.Disulf, gemmi.ConnectionType.MetalC}
    for conn in st.connections:
        if conn.type not in _BOND_TYPES:
            continue
        p1, p2 = conn.partner1, conn.partner2
        idx1 = atom_lookup.get((p1.chain_name, p1.res_id.seqid.num, p1.atom_name))
        idx2 = atom_lookup.get((p2.chain_name, p2.res_id.seqid.num, p2.atom_name))
        if idx1 is not None and idx2 is not None:
            conn_bonds.append((idx1, idx2, 1))  # bond order 1 (single)
    bonds_arr = np.array(conn_bonds, dtype=BOND_DTYPE) if conn_bonds else np.empty(0, dtype=BOND_DTYPE)

    structure = Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
        name=pdb_id,
        chain_canonical_seqs=chain_canonical_seqs if chain_canonical_seqs else None,
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
