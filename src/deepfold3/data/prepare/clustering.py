"""Sequence clustering support for training data.

Loads pre-computed PDB sequence clusters (wwPDB 40% identity clusters)
and maps chain sequences to cluster IDs in the manifest.

The cluster file format is ``clusters-by-entity-40.txt`` from wwPDB:
each line contains space-separated ``PDB_CHAIN`` identifiers belonging
to the same cluster.  We assign the first entry as the cluster
representative and map all entries to the same integer cluster ID.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pdb_clusters(path: str | Path) -> dict[str, int]:
    """Load PDB 40% identity clusters from wwPDB format.

    Args:
        path: Path to ``clusters-by-entity-40.txt``.

    Returns:
        Mapping from ``pdb_chain`` string (e.g. ``"1abc_A"``) to integer
        cluster ID.  All members of the same cluster share the same ID.
    """
    path = Path(path)
    mapping: dict[str, int] = {}

    with open(path) as f:
        for cluster_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            members = line.split()
            for member in members:
                mapping[member.lower()] = cluster_id

    logger.info(f"Loaded {len(mapping)} chain-to-cluster mappings from {path}")
    return mapping


def sequence_to_cluster_key(pdb_id: str, chain_name: str) -> str:
    """Build the lookup key for the PDB cluster file.

    Args:
        pdb_id: 4-character PDB identifier (e.g. ``"1abc"``).
        chain_name: Chain name (e.g. ``"A"``).

    Returns:
        Lowercase key like ``"1abc_a"``.
    """
    return f"{pdb_id.lower()}_{chain_name.lower()}"


def sequence_hash_cluster(sequence: str) -> str:
    """Fallback cluster ID: SHA256 hash of the sequence.

    Used for chains not found in the PDB cluster file (e.g. novel
    sequences or non-protein chains).
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]
