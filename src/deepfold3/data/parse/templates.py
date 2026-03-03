"""Template structure parsing and feature extraction.

Downloads template structures from RCSB PDB, aligns them to query tokens,
and extracts dense atom positions/masks for the template embedding module.

Supports two alignment modes:
  - Residue-index matching (mmCIF directory input)
  - HHR alignment mapping (HHsearch output + PDB download)

Template features (per template, aligned to query tokens):
    aatype         [N]         int64   residue type indices
    atom_positions [N, 24, 3]  float32 dense atom positions
    atom_mask      [N, 24]     bool    dense atom validity mask
"""

from __future__ import annotations

import dataclasses
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

if TYPE_CHECKING:
    from deepfold3.data.parse.hhr import TemplateHit

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


def fetch_pdb_mmcif(pdb_id: str, cache_dir: Path | None = None) -> str:
    """Download an mmCIF file from RCSB PDB.

    Args:
        pdb_id: 4-character PDB ID (e.g., "1ABC").
        cache_dir: Optional directory to cache downloaded files. If the file
            already exists in the cache, it is read from disk.

    Returns:
        mmCIF file contents as a string.

    Raises:
        ValueError: If pdb_id is not a valid 4-character code.
        urllib.error.URLError: If the download fails.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB ID must be 4 characters, got: {pdb_id!r}")

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{pdb_id}.cif"
        if cached.exists():
            return cached.read_text()

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

    canonical_atoms = const.REF_ATOMS.get(res_name)
    if canonical_atoms is None:
        std_name = const._NONSTANDARD_TO_STANDARD.get(res_name)
        if std_name is not None:
            canonical_atoms = const.REF_ATOMS.get(std_name)
    if canonical_atoms is None:
        canonical_atoms = const.REF_ATOMS.get("UNK", ["N", "CA", "C", "O", "CB"])

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

    Returns:
        List of TemplateFeatures, one per successfully processed template.
    """
    from deepfold3.data.parse.hhr import parse_hhr

    hhr_text = Path(hhr_path).read_text()
    hits = parse_hhr(hhr_text)

    if not hits:
        logger.warning("No hits found in HHR file %s", hhr_path)
        return []

    # HHR hits are already ranked by Probab/E-value; take top K in file order,
    # skipping failures until we have enough.
    templates: list[TemplateFeatures] = []
    for hit in hits:
        if len(templates) >= max_templates:
            break
        try:
            mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir)
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
