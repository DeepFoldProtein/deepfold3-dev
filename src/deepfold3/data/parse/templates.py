"""Template structure parsing and feature extraction.

Downloads template structures from RCSB PDB, aligns them to query tokens,
and extracts dense atom positions/masks for the template embedding module.

Supports three alignment modes:
  - Residue-index matching (mmCIF directory input)
  - HHR alignment mapping (HHsearch output + PDB download)
  - JSON template hits (pre-computed query-to-template index mappings + local CIF files)

Template features (per template, aligned to query tokens):
    aatype         [N]         int64   residue type indices
    atom_positions [N, 24, 3]  float32 dense atom positions
    atom_mask      [N, 24]     bool    dense atom validity mask
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import gemmi
except ImportError:
    gemmi = None  # type: ignore[assignment]

from deepfold3.data import const
from deepfold3.data.parse.hhr import TemplateHit

if TYPE_CHECKING:
    from deepfold3.data.types import Structure

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TemplateFeatures:
    """Features extracted from a single structural template.

    All arrays are aligned to the query token sequence of length N.

    Attributes:
        aatype: [N] residue type indices (matching const.RES_TYPE_ORDER).
        atom_positions: [N, 24, 3] dense atom positions in Angstroms.
        atom_mask: [N, 24] mask indicating valid (resolved) atoms.
    """

    aatype: np.ndarray  # [N] int64
    atom_positions: np.ndarray  # [N, 24, 3] float32
    atom_mask: np.ndarray  # [N, 24] bool


def empty_template_features(num_tokens: int, max_atoms: int = const.MAX_ATOMS_PER_TOKEN) -> TemplateFeatures:
    """Create zero-filled template features for a given number of tokens.

    Args:
        num_tokens: Number of query tokens.
        max_atoms: Maximum atoms per token.

    Returns:
        TemplateFeatures with all-zero arrays and UNK aatype.
    """
    return TemplateFeatures(
        aatype=np.full(num_tokens, 20, dtype=np.int64),  # UNK
        atom_positions=np.zeros((num_tokens, max_atoms, 3), dtype=np.float32),
        atom_mask=np.zeros((num_tokens, max_atoms), dtype=bool),
    )


@dataclasses.dataclass
class TemplateHitInput:
    """A parsed template hit ready for feature extraction (no further I/O needed).

    Three alignment modes:
    1. JSON hits: query_indices + template_indices populated
    2. HHR hits: hhr_hit populated (carries query_to_hit_mapping)
    3. Structure: structure populated with query_to_template mapping
    """

    mmcif_text: str = ""
    query_indices: list[int] | None = None  # 0-based query positions (JSON mode)
    template_indices: list[int] | None = None  # label_seq_id numbers (JSON mode)
    hhr_hit: TemplateHit | None = None  # full HHR hit (HHR mode)
    chain_id: str | None = None
    structure: Structure | None = None  # Structure object (Structure mode)
    query_to_template: dict[int, int] | None = None  # query pos → template residue index


def template_hit_input_to_features(
    hit_input: TemplateHitInput,
    query_length: int,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> TemplateFeatures:
    """Convert a TemplateHitInput to TemplateFeatures (pure computation, no I/O).

    Dispatches to Structure-based, HHR-based, or index-pair-based extraction
    depending on which fields are populated.

    Args:
        hit_input: A TemplateHitInput with alignment info.
        query_length: Number of tokens in the query sequence.
        max_atoms: Maximum atoms per residue.

    Returns:
        TemplateFeatures aligned to query tokens.
    """
    # Structure mode
    if hit_input.structure is not None and hit_input.query_to_template is not None:
        return extract_template_features_from_structure(
            hit_input.structure,
            hit_input.query_to_template,
            query_length,
            max_atoms=max_atoms,
        )

    if hit_input.hhr_hit is not None:
        return extract_template_features_from_hhr(
            hit_input.mmcif_text,
            hit_input.hhr_hit,
            query_length,
            chain_id=hit_input.chain_id,
            max_atoms=max_atoms,
        )

    # JSON mode: query_indices + template_indices
    if hit_input.query_indices is None or hit_input.template_indices is None:
        logger.warning("TemplateHitInput has neither hhr_hit nor index pairs nor structure")
        return empty_template_features(query_length, max_atoms)

    return _extract_from_index_pairs(
        hit_input.mmcif_text,
        hit_input.query_indices,
        hit_input.template_indices,
        query_length,
        chain_id=hit_input.chain_id,
        max_atoms=max_atoms,
    )


def _extract_from_index_pairs(
    mmcif_text: str,
    query_indices: list[int],
    template_indices: list[int],
    query_length: int,
    chain_id: str | None = None,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> TemplateFeatures:
    """Extract template features using pre-computed query/template index pairs.

    Args:
        mmcif_text: mmCIF file contents.
        query_indices: 0-based query token positions.
        template_indices: Residue label_seq_id numbers in the template CIF.
        query_length: Total number of tokens in the query.
        chain_id: Chain to use from the template (None = first polymer).
        max_atoms: Maximum atoms per residue.

    Returns:
        TemplateFeatures aligned to query tokens.
    """
    result = _parse_mmcif_chain(mmcif_text, chain_id=chain_id)
    if result is None:
        return empty_template_features(query_length, max_atoms)
    _, target_chain = result

    # Build label_seq -> residue features lookup
    residue_lookup: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for residue in target_chain:
        residue_lookup[residue.label_seq] = _extract_residue_features(residue, max_atoms)

    aatype_arr = np.full(query_length, 20, dtype=np.int64)
    pos_arr = np.zeros((query_length, max_atoms, 3), dtype=np.float32)
    mask_arr = np.zeros((query_length, max_atoms), dtype=bool)

    matched = 0
    for qi, ti in zip(query_indices, template_indices, strict=False):
        if qi < 0 or qi >= query_length:
            continue
        if ti in residue_lookup:
            aa, pos, msk = residue_lookup[ti]
            aatype_arr[qi] = aa
            pos_arr[qi] = pos
            mask_arr[qi] = msk
            matched += 1

    logger.info("Index-pair template: %d/%d matched", matched, len(query_indices))
    return TemplateFeatures(aatype=aatype_arr, atom_positions=pos_arr, atom_mask=mask_arr)


def find_pdb_mmcif_path(pdb_id: str, pdb_mmcif_dir: Path) -> Path | None:
    """Find the file path of a PDB mmCIF in a local PDB mirror.

    Same search order as ``_find_pdb_mmcif_local`` but returns the ``Path``
    instead of reading the file content.  Useful for creating symbolic links.
    """
    pdb_lower = pdb_id.lower()
    mid2 = pdb_lower[1:3]
    candidates = [
        pdb_mmcif_dir / mid2 / f"{pdb_lower}.cif.gz",
        pdb_mmcif_dir / mid2 / f"{pdb_lower}.cif",
        pdb_mmcif_dir / f"{pdb_lower}.cif.gz",
        pdb_mmcif_dir / f"{pdb_lower}.cif",
        pdb_mmcif_dir / mid2 / f"{pdb_id}.cif.gz",
        pdb_mmcif_dir / mid2 / f"{pdb_id}.cif",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    # Glob fallback: flat directories with prefixed names (e.g. 102l_A_model_0.cif)
    matches = sorted(pdb_mmcif_dir.glob(f"{pdb_lower}*.cif"))
    if matches:
        return matches[0].resolve()
    matches = sorted(pdb_mmcif_dir.glob(f"{pdb_lower}*.cif.gz"))
    if matches:
        return matches[0].resolve()
    return None


def _find_pdb_mmcif_local(pdb_id: str, pdb_mmcif_dir: Path) -> str | None:
    """Look up a PDB mmCIF file in a local PDB mirror directory.

    Searches the standard PDB mirror layout::

        {pdb_mmcif_dir}/{mid2}/{pdb_id}.cif.gz
        {pdb_mmcif_dir}/{mid2}/{pdb_id}.cif

    where ``mid2`` is the middle two characters of the PDB ID (e.g. ``"3v"``
    for ``"a3vb"``).  Also tries the flat layout ``{pdb_mmcif_dir}/{pdb_id}.cif[.gz]``
    as a fallback.

    Args:
        pdb_id: 4-character PDB ID (case-insensitive; lowercased for path).
        pdb_mmcif_dir: Root directory of the local PDB mirror.

    Returns:
        mmCIF text on success, None if not found.
    """
    pdb_lower = pdb_id.lower()
    mid2 = pdb_lower[1:3]

    candidates = [
        pdb_mmcif_dir / mid2 / f"{pdb_lower}.cif.gz",
        pdb_mmcif_dir / mid2 / f"{pdb_lower}.cif",
        pdb_mmcif_dir / f"{pdb_lower}.cif.gz",
        pdb_mmcif_dir / f"{pdb_lower}.cif",
        # Also try uppercase variants
        pdb_mmcif_dir / mid2 / f"{pdb_id}.cif.gz",
        pdb_mmcif_dir / mid2 / f"{pdb_id}.cif",
    ]

    for path in candidates:
        if path.is_file():
            logger.info("Found local template %s at %s", pdb_id, path)
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return f.read()
            else:
                return path.read_text()

    return None


def fetch_pdb_mmcif(
    pdb_id: str,
    cache_dir: Path | None = None,
    pdb_mmcif_dir: Path | None = None,
) -> str:
    """Fetch an mmCIF file from a local PDB mirror or RCSB PDB.

    Resolution order:
    1. ``pdb_mmcif_dir`` — local PDB mirror (``{dir}/{mid2}/{id}.cif.gz``)
    2. ``cache_dir`` — previously downloaded file
    3. RCSB download (result cached if ``cache_dir`` is set)

    Args:
        pdb_id: 4-character PDB ID (e.g., "1ABC").
        cache_dir: Optional directory to cache downloaded files.
        pdb_mmcif_dir: Optional local PDB mmCIF mirror directory.
            Standard layout: ``{dir}/{mid2}/{id}.cif.gz`` where ``mid2``
            is the middle two characters of the lowercased PDB ID.

    Returns:
        mmCIF file contents as a string.

    Raises:
        ValueError: If pdb_id is not a valid 4-character code.
        urllib.error.URLError: If the download fails and no local source found.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB ID must be 4 characters, got: {pdb_id!r}")

    # 1. Local PDB mirror
    if pdb_mmcif_dir is not None:
        content = _find_pdb_mmcif_local(pdb_id, Path(pdb_mmcif_dir))
        if content is not None:
            return content
        logger.debug("PDB ID %s not found in local mirror %s", pdb_id, pdb_mmcif_dir)

    # 2. Cache
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{pdb_id}.cif"
        if cached.exists():
            return cached.read_text()

    # 3. RCSB download
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    logger.info("Downloading template %s from %s", pdb_id, url)

    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    if cache_dir is not None:
        cached.write_text(content)

    return content


def _parse_mmcif_chain(
    mmcif_text: str,
    chain_id: str | None = None,
    label: str = "template",
) -> tuple | None:
    """Parse an mmCIF string and select a target chain.

    Args:
        mmcif_text: mmCIF file contents.
        chain_id: Specific chain to use. If None, the first polymer chain is used.
        label: Label for log messages (e.g. PDB ID).

    Returns:
        (gemmi_structure, target_chain) or None if the structure is empty or
        the requested chain was not found.
    """
    if gemmi is None:
        raise ImportError("gemmi is required for template parsing: pip install gemmi")

    doc = gemmi.cif.read_string(mmcif_text)
    st = gemmi.make_structure_from_block(doc[0])
    st.setup_entities()

    if len(st) == 0 or len(st[0]) == 0:
        logger.warning("Template %s has no models/chains", label)
        return None

    model = st[0]

    # Select target chain
    target_chain = None
    if chain_id is not None:
        for ch in model:
            if ch.name == chain_id:
                target_chain = ch
                break
        if target_chain is None:
            logger.warning("Chain %s not found in template %s", chain_id, label)
            return None
    else:
        # First polymer chain
        for ch in model:
            entity = st.get_entity_of(ch.get_polymer())
            if entity is not None and entity.entity_type == gemmi.EntityType.Polymer:
                target_chain = ch
                break
        if target_chain is None:
            target_chain = model[0]

    return st, target_chain


def extract_template_features(
    mmcif_text: str,
    query_residue_index: np.ndarray,
    query_asym_id: np.ndarray | None = None,
    chain_id: str | None = None,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> TemplateFeatures:
    """Extract template features from an mmCIF structure, aligned to query tokens.

    Aligns template residues to query tokens by matching residue_index values.
    Unmatched query positions are filled with UNK type and zero coordinates.

    Args:
        mmcif_text: mmCIF file contents as a string.
        query_residue_index: [N] residue indices of the query sequence.
        query_asym_id: [N] chain (asym) IDs of the query. If provided, only
            the first chain's residues are aligned.
        chain_id: Specific chain to use from the template. If None, the first
            polymer chain is used.
        max_atoms: Maximum atoms per residue in dense representation.

    Returns:
        TemplateFeatures aligned to query tokens.
    """
    N = len(query_residue_index)
    result = _parse_mmcif_chain(mmcif_text, chain_id=chain_id)
    if result is None:
        return empty_template_features(N, max_atoms)
    _, target_chain = result

    # Build lookup: residue_index -> (aatype, positions, mask)
    residue_lookup: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for residue in target_chain:
        residue_lookup[residue.seqid.num] = _extract_residue_features(residue, max_atoms)

    # Align to query tokens
    aatype_arr = np.full(N, 20, dtype=np.int64)
    pos_arr = np.zeros((N, max_atoms, 3), dtype=np.float32)
    mask_arr = np.zeros((N, max_atoms), dtype=bool)

    for i, res_idx in enumerate(query_residue_index):
        res_idx_int = int(res_idx)
        if res_idx_int in residue_lookup:
            aa, pos, msk = residue_lookup[res_idx_int]
            aatype_arr[i] = aa
            pos_arr[i] = pos
            mask_arr[i] = msk

    return TemplateFeatures(
        aatype=aatype_arr,
        atom_positions=pos_arr,
        atom_mask=mask_arr,
    )


def parse_template_mmcifs(
    template_dir: Path,
    query_residue_index: np.ndarray,
    max_templates: int = 4,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> list[TemplateFeatures]:
    """Parse all mmCIF template files from a directory.

    Args:
        template_dir: Directory containing .cif files.
        query_residue_index: [N] residue indices of the query sequence.
        max_templates: Maximum number of templates to process.
        max_atoms: Maximum atoms per residue.

    Returns:
        List of TemplateFeatures, one per template (up to max_templates).
    """
    cif_files = sorted(template_dir.glob("*.cif"))[:max_templates]

    templates = []
    for cif_path in cif_files:
        try:
            text = cif_path.read_text()
            feats = extract_template_features(
                text,
                query_residue_index,
                max_atoms=max_atoms,
            )
            templates.append(feats)
            logger.info("Loaded template from %s", cif_path.name)
        except Exception:
            logger.warning("Failed to parse template %s", cif_path.name, exc_info=True)

    return templates


def _get_canonical_atoms(res_name: str) -> list[str]:
    """Get canonical atom order for a residue name, with nonstandard fallback."""
    canonical = const.REF_ATOMS.get(res_name)
    if canonical is None:
        std_name = const._NONSTANDARD_TO_STANDARD.get(res_name)
        if std_name is not None:
            canonical = const.REF_ATOMS.get(std_name)
    if canonical is None:
        canonical = const.REF_ATOMS.get("UNK", ["N", "CA", "C", "O", "CB"])
    return canonical


def _extract_residue_features(
    residue,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Extract aatype, atom positions and mask from a gemmi Residue.

    Returns:
        (aatype, positions [max_atoms, 3], mask [max_atoms]).
    """
    res_name = residue.name.strip()
    aatype = const.res_name_to_type(res_name)
    canonical_atoms = _get_canonical_atoms(res_name)

    positions = np.zeros((max_atoms, 3), dtype=np.float32)
    mask = np.zeros(max_atoms, dtype=bool)

    atom_name_to_idx = {name: i for i, name in enumerate(canonical_atoms[:max_atoms])}
    for atom in residue:
        idx = atom_name_to_idx.get(atom.name.strip())
        if idx is not None and idx < max_atoms:
            positions[idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
            mask[idx] = True

    return aatype, positions, mask


def extract_template_features_from_hhr(
    mmcif_text: str,
    hit: TemplateHit,
    query_length: int,
    chain_id: str | None = None,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> TemplateFeatures:
    """Extract template features using HHR alignment mapping.

    Unlike ``extract_template_features`` which matches by residue_index,
    this function uses the query-to-hit mapping from an HHR TemplateHit
    to place template atom features at the correct query positions.

    Args:
        mmcif_text: mmCIF file contents.
        hit: A TemplateHit from ``parse_hhr()``.
        query_length: Number of tokens in the query sequence.
        chain_id: Chain to use from the template. If None, uses the
            chain from ``hit.chain_id``, falling back to first polymer chain.
        max_atoms: Maximum atoms per residue.

    Returns:
        TemplateFeatures aligned to query tokens.
    """
    N = query_length
    result = _parse_mmcif_chain(mmcif_text, chain_id=chain_id or hit.chain_id, label=hit.pdb_id)
    if result is None:
        return empty_template_features(N, max_atoms)
    _, target_chain = result

    # Build sequential 0-based index -> residue features
    # (HHR hit indices are 0-based sequential positions in the chain)
    residue_by_seq_idx: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for seq_idx, residue in enumerate(target_chain):
        residue_by_seq_idx[seq_idx] = _extract_residue_features(residue, max_atoms)

    # Map query positions using HHR alignment
    mapping = hit.query_to_hit_mapping

    aatype_arr = np.full(N, 20, dtype=np.int64)
    pos_arr = np.zeros((N, max_atoms, 3), dtype=np.float32)
    mask_arr = np.zeros((N, max_atoms), dtype=bool)

    for query_idx, hit_idx in mapping.items():
        if query_idx >= N:
            continue
        if hit_idx in residue_by_seq_idx:
            aa, pos, msk = residue_by_seq_idx[hit_idx]
            aatype_arr[query_idx] = aa
            pos_arr[query_idx] = pos
            mask_arr[query_idx] = msk

    return TemplateFeatures(
        aatype=aatype_arr,
        atom_positions=pos_arr,
        atom_mask=mask_arr,
    )


def parse_hhr_templates(
    hhr_path: str | Path,
    query_length: int,
    max_templates: int = 4,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
    cache_dir: str | Path | None = None,
    pdb_mmcif_dir: str | Path | None = None,
) -> list[TemplateFeatures]:
    """Parse templates from an HHR file by downloading structures from RCSB.

    Reads the HHR file, takes the top hits in file order (already ranked by
    HHsearch), downloads each PDB structure, and extracts features via the
    HHR alignment.

    Args:
        hhr_path: Path to the .hhr file.
        query_length: Number of tokens in the query sequence.
        max_templates: Maximum number of templates to extract.
        max_atoms: Maximum atoms per residue.
        cache_dir: Optional directory to cache downloaded mmCIF files.
        pdb_mmcif_dir: Optional local PDB mmCIF mirror directory
            (layout: ``{dir}/{mid2}/{id}.cif.gz``).

    Returns:
        List of TemplateFeatures, one per successfully processed template.
    """
    from deepfold3.data.parse.hhr import parse_hhr

    hhr_text = Path(hhr_path).read_text()
    hits = parse_hhr(hhr_text)

    if not hits:
        logger.warning("No hits found in HHR file %s", hhr_path)
        return []

    pdb_dir = Path(pdb_mmcif_dir) if pdb_mmcif_dir is not None else None

    # HHR hits are already ranked by Probab/E-value; take top K in file order,
    # skipping failures until we have enough.
    templates: list[TemplateFeatures] = []
    for hit in hits:
        if len(templates) >= max_templates:
            break
        try:
            mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir, pdb_mmcif_dir=pdb_dir)
            feats = extract_template_features_from_hhr(
                mmcif_text,
                hit,
                query_length,
                max_atoms=max_atoms,
            )
            templates.append(feats)
            logger.info(
                "Loaded template %s_%s (e_value=%.1e, aligned_cols=%d)",
                hit.pdb_id,
                hit.chain_id,
                hit.e_value,
                hit.aligned_cols,
            )
        except Exception:
            logger.warning("Failed to process template %s_%s", hit.pdb_id, hit.chain_id, exc_info=True)

    return templates


def parse_json_templates(
    json_path: str | Path,
    query_length: int,
    max_templates: int = 4,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> list[TemplateFeatures]:
    """Parse templates from a JSON template hits file with pre-computed alignments.

    The JSON file contains an array of hits, each with a CIF filename and
    paired queryIndices/templateIndices arrays that map query token positions
    to template residue seqid numbers.

    Expected JSON format::

        [
            {
                "name": "template_hit_0.cif",
                "queryIndices": [0, 1, 5, ...],
                "templateIndices": [10, 11, 15, ...]
            },
            ...
        ]

    - ``name``: CIF filename in the same directory as the JSON file.
    - ``queryIndices``: 0-based token positions in the full query complex.
    - ``templateIndices``: Residue label_seq_id numbers (mmCIF canonical numbering) in the template CIF.

    Args:
        json_path: Path to the JSON template hits file.
        query_length: Total number of tokens in the query complex.
        max_templates: Maximum number of templates to process.
        max_atoms: Maximum atoms per residue.

    Returns:
        List of TemplateFeatures, one per successfully processed hit.
    """
    json_path = Path(json_path)
    json_dir = json_path.parent

    with open(json_path) as f:
        hits = json.load(f)

    if not isinstance(hits, list):
        logger.warning("JSON template hits file %s is not an array", json_path)
        return []

    templates: list[TemplateFeatures] = []
    for hit in hits[:max_templates]:
        try:
            cif_name = hit["name"]
            query_indices = hit["queryIndices"]
            template_indices = hit["templateIndices"]

            if len(query_indices) != len(template_indices):
                logger.warning("Mismatched index lengths in template %s, skipping", cif_name)
                continue

            cif_path = json_dir / cif_name
            if not cif_path.exists():
                logger.warning("Template CIF not found: %s", cif_path)
                continue

            mmcif_text = cif_path.read_text()
            feats = _extract_from_index_pairs(
                mmcif_text,
                query_indices,
                template_indices,
                query_length,
                max_atoms=max_atoms,
            )
            templates.append(feats)
            logger.info("Loaded JSON template %s", cif_name)

        except Exception:
            logger.warning("Failed to process JSON template hit", exc_info=True)

    return templates


def extract_template_features_from_structure(
    structure: Structure,
    query_to_template: dict[int, int],
    query_length: int,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
) -> TemplateFeatures:
    """Extract template features from a Structure object.

    Uses pre-computed query-to-template residue mapping to align template
    features to query token positions.

    Args:
        structure: A Structure object with atoms, residues, chains arrays.
        query_to_template: Maps 0-based query position → 0-based template
            residue index in the Structure's residues array.
        query_length: Number of tokens in the query sequence.
        max_atoms: Maximum atoms per residue.

    Returns:
        TemplateFeatures aligned to query tokens.
    """
    atoms = structure.atoms
    residues = structure.residues

    # Build residue index → features lookup
    residue_lookup: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}

    for ri in range(len(residues)):
        res = residues[ri]
        res_name = res["name"].decode("ascii").strip()
        aatype = const.res_name_to_type(res_name)
        canonical_atoms = _get_canonical_atoms(res_name)

        positions = np.zeros((max_atoms, 3), dtype=np.float32)
        mask = np.zeros(max_atoms, dtype=bool)

        atom_name_to_idx = {name: i for i, name in enumerate(canonical_atoms[:max_atoms])}

        atom_start = int(res["atom_idx"])
        atom_end = atom_start + int(res["atom_num"])
        for ai in range(atom_start, atom_end):
            a = atoms[ai]
            if not bool(a["is_present"]):
                continue
            a_name = a["name"].decode("ascii").strip()
            idx = atom_name_to_idx.get(a_name)
            if idx is not None and idx < max_atoms:
                positions[idx] = [float(a["x"]), float(a["y"]), float(a["z"])]
                mask[idx] = True

        residue_lookup[ri] = (aatype, positions, mask)

    # Align to query tokens
    aatype_arr = np.full(query_length, 20, dtype=np.int64)
    pos_arr = np.zeros((query_length, max_atoms, 3), dtype=np.float32)
    mask_arr = np.zeros((query_length, max_atoms), dtype=bool)

    matched = 0
    for qi, ti in query_to_template.items():
        if qi < 0 or qi >= query_length:
            continue
        if ti in residue_lookup:
            aa, pos, msk = residue_lookup[ti]
            aatype_arr[qi] = aa
            pos_arr[qi] = pos
            mask_arr[qi] = msk
            matched += 1

    logger.info("Structure template: %d/%d matched", matched, len(query_to_template))
    return TemplateFeatures(aatype=aatype_arr, atom_positions=pos_arr, atom_mask=mask_arr)
