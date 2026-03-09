"""Manifest: JSON index of all processed training records.

Layout::

    data_dir/
    ├── manifest.json
    ├── structures/
    │   └── {pdb_id}.npz
    └── msa/
        └── {msa_id}.npz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deepfold3.data.types import Record

logger = logging.getLogger(__name__)


def save_manifest(records: list[Record], path: Path) -> None:
    """Save a list of Records to a JSON manifest file."""
    data = [r.to_dict() for r in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved manifest with %d records to %s", len(records), path)


def load_manifest(path: Path) -> list[Record]:
    """Load a JSON manifest file into a list of Records."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    records = [Record.from_dict(d) for d in data]
    logger.info("Loaded manifest with %d records from %s", len(records), path)
    return records


def discover_records(data_dir: Path) -> list[Record]:
    """Discover records from a data directory.

    Looks for ``manifest.json`` first. If not found, scans the
    ``structures/`` subdirectory for NPZ files and builds minimal records.

    Args:
        data_dir: Root data directory.

    Returns:
        List of Record objects.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"

    if manifest_path.exists():
        return load_manifest(manifest_path)

    # Fallback: scan structures/ directory
    structures_dir = data_dir / "structures"
    if not structures_dir.exists():
        logger.warning("No manifest.json or structures/ directory found in %s", data_dir)
        return []

    from deepfold3.data.types import ChainInfo, Structure

    records = []
    for npz_path in sorted(structures_dir.glob("*.npz")):
        pdb_id = npz_path.stem
        try:
            structure = Structure.load(npz_path)
        except Exception:
            logger.warning("Failed to load %s, skipping", npz_path)
            continue

        chains = []
        for i, chain in enumerate(structure.chains):
            mol_type = int(chain["mol_type"])
            chain_name = chain["name"].decode("ascii", errors="replace").strip()
            num_residues = int(chain["res_num"])
            chains.append(
                ChainInfo(
                    chain_id=i,
                    chain_name=chain_name,
                    mol_type=mol_type,
                    cluster_id=pdb_id,
                    msa_id=f"{pdb_id}_{chain_name}",
                    num_residues=num_residues,
                    entity_id=int(chain["entity_id"]),
                )
            )
        records.append(Record(id=pdb_id, chains=chains, interfaces=[]))

    logger.info("Discovered %d records from %s", len(records), structures_dir)
    return records
