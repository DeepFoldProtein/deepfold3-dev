"""Shared sequence extraction and FASTA utilities.

Used by ``deepfold3 cluster`` and ``deepfold3 msa-batch`` to extract
protein sequences from processed structures and cache them as FASTA.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from deepfold3.data.const import MOL_PROTEIN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------


def load_fasta(path: Path) -> dict[str, str]:
    """Load a single-line-per-sequence FASTA file into a dict.

    Returns:
        Dict mapping header (without ``>``) to sequence string.
    """
    sequences: dict[str, str] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:]
            elif name is not None:
                sequences[name] = line
                name = None
    return sequences


def save_fasta(sequences: dict[str, str], path: Path) -> None:
    """Save a dict of sequences to a single-line-per-sequence FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")


# ---------------------------------------------------------------------------
# Chain residue lookup
# ---------------------------------------------------------------------------


def get_chain_residue_types(structure, chain_id: int) -> np.ndarray:
    """Get residue type array for a chain identified by asym_id.

    Args:
        structure: A ``Structure`` object with ``.chains`` and ``.residues``.
        chain_id: The ``asym_id`` value to look up.

    Returns:
        1-D int array of residue types, or empty array if chain not found.
    """
    chain_matches = structure.chains[structure.chains["asym_id"] == chain_id]
    if len(chain_matches) == 0:
        return np.empty(0, dtype=np.int16)
    c = chain_matches[0]
    return structure.residues["res_type"][c["res_idx"] : c["res_idx"] + c["res_num"]]


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

# Lazily built lookup table: res_type index → one-letter amino acid code
_RT_TO_LETTER: dict[int, str] | None = None


def _build_rt_to_letter() -> dict[int, str]:
    global _RT_TO_LETTER
    if _RT_TO_LETTER is None:
        from deepfold3.data.const import RES_TYPE_INDEX_TO_NAME, THREE_LETTER_TO_ONE

        _RT_TO_LETTER = {i: THREE_LETTER_TO_ONE.get(name, "X") for i, name in RES_TYPE_INDEX_TO_NAME.items()}
    return _RT_TO_LETTER


def residue_types_to_sequence(res_types: np.ndarray) -> str:
    """Convert an array of residue type indices to a one-letter sequence string."""
    rt_map = _build_rt_to_letter()
    return "".join(rt_map.get(int(rt), "X") for rt in res_types)


def extract_sequences_by_msa_id(
    records: list,
    structures_dir: Path,
    protein_only: bool = True,
) -> dict[str, str]:
    """Extract unique protein sequences keyed by ``msa_id`` (SHA256).

    Used by ``msa-batch`` for MSA search — one sequence per unique hash.
    """
    from deepfold3.data.types import Structure

    sequences: dict[str, str] = {}
    seen: set[str] = set()

    for record in records:
        npz_path = structures_dir / f"{record.id}.npz"
        if not npz_path.exists():
            continue
        try:
            structure = Structure.load(npz_path)
        except Exception:
            logger.debug("Failed to load %s, skipping", npz_path)
            continue

        for chain_info in record.chains:
            if not chain_info.valid:
                continue
            if protein_only and chain_info.mol_type != MOL_PROTEIN:
                continue
            msa_id = str(chain_info.msa_id)
            if msa_id in seen:
                continue
            seen.add(msa_id)

            res_types = get_chain_residue_types(structure, chain_info.chain_id)
            if len(res_types) == 0:
                continue
            sequences[msa_id] = residue_types_to_sequence(res_types)

    return sequences


def build_complexes_json(
    records: list,
    protein_only: bool = True,
) -> dict[str, list[str]]:
    """Build a complexes mapping from manifest records.

    Returns a dict mapping complex ID to a list of ``msa_id`` values
    (matching ``msa_sequences.fasta`` headers), suitable for paired MSA
    pipelines. Only multimer complexes (>1 qualifying chain) are included.

    Args:
        records: List of Record objects (from manifest).
        protein_only: If True, only include protein chains.

    Returns:
        Dict mapping complex ID (PDB ID) to list of msa_id strings.
    """
    complexes: dict[str, list[str]] = {}

    for record in records:
        chain_ids: list[str] = []
        for chain_info in record.chains:
            if not chain_info.valid:
                continue
            if protein_only and chain_info.mol_type != MOL_PROTEIN:
                continue
            chain_ids.append(str(chain_info.msa_id))

        if len(chain_ids) > 1:
            complexes[record.id] = chain_ids

    return complexes


def save_complexes_json(complexes: dict[str, list[str]], path: Path) -> None:
    """Write complexes mapping to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(complexes, f, indent=2)


def extract_sequences_by_cluster_key(
    records: list,
    structures_dir: Path,
    min_length: int = 10,
) -> dict[str, str]:
    """Extract unique protein sequences keyed by ``{pdb_id}_{chain_name}``.

    Used by ``cluster`` for MMseqs2 clustering — one sequence per unique entity.
    """
    from deepfold3.data.prepare.clustering import sequence_to_cluster_key
    from deepfold3.data.types import Structure

    sequences: dict[str, str] = {}
    seen_entities: set[tuple[str, str | int | None]] = set()

    for record in records:
        npz_path = structures_dir / f"{record.id}.npz"
        if not npz_path.exists():
            continue
        try:
            structure = Structure.load(npz_path)
        except Exception:
            logger.debug("Failed to load %s, skipping", npz_path)
            continue

        for chain_info in record.chains:
            if not chain_info.valid or chain_info.mol_type != MOL_PROTEIN:
                continue
            entity_key = (record.id, chain_info.entity_id)
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            res_types = get_chain_residue_types(structure, chain_info.chain_id)
            if len(res_types) == 0:
                continue

            seq = residue_types_to_sequence(res_types)
            if len(seq) < min_length:
                continue

            key = sequence_to_cluster_key(record.id, chain_info.chain_name)
            sequences[key] = seq

    return sequences
