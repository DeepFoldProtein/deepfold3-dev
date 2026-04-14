"""Filesystem layout helpers for the training data directory.

All persistent files in ``{data_dir}/`` are sharded by the first two
characters of the file id (256 buckets) so that listing, backups, and
``rsync`` stay fast even with hundreds of thousands of files::

    {data_dir}/
    ├── structures/
    │   ├── 1a/
    │   │   └── 1a02.npz
    │   └── 8q/
    │       └── 8qy0.npz
    ├── masks/
    │   └── 1a/
    │       └── 1a02.npz
    └── msa/
        ├── unpaired/
        │   └── 00/
        │       └── 00019cc8…861bfd.npz
        └── paired/
            └── 1a/
                └── 1a02_paired_3aed842c…12b.npz

Use these helpers everywhere instead of constructing paths inline so the
layout stays consistent across the dataloader, prepare commands, import
commands, and search batch commands.

MSA-specific helpers live in :mod:`deepfold3.data.msa.paths` and re-use
the same shard rule defined here.
"""

from __future__ import annotations

from pathlib import Path


def shard(name: str) -> str:
    """Two-character shard key derived from a file id.

    Lowercased. Names shorter than two characters fall back to ``"__"``
    so the result is always a valid directory name.
    """
    return name[:2].lower() if len(name) >= 2 else "__"


def structure_path(structures_dir: Path, pdb_id: str) -> Path:
    """Path to a structure NPZ under the sharded layout."""
    return Path(structures_dir) / shard(pdb_id) / f"{pdb_id}.npz"


def mask_path(masks_dir: Path, pdb_id: str) -> Path:
    """Path to a token-mask NPZ under the sharded layout."""
    return Path(masks_dir) / shard(pdb_id) / f"{pdb_id}.npz"


def template_path(templates_dir: Path, pdb_id: str) -> Path:
    """Path to a multimer-template index JSON under the sharded layout (per record)."""
    return Path(templates_dir) / shard(pdb_id) / f"{pdb_id}.json"


def record_path(records_dir: Path, pdb_id: str) -> Path:
    """Path to a record JSON under the sharded layout."""
    return Path(records_dir) / shard(pdb_id) / f"{pdb_id}.json"


def monomer_template_path(monomer_templates_dir: Path, msa_id: str) -> Path:
    """Path to a monomer-template index JSON under the sharded layout (per sequence).

    Monomer templates are deduplicated by chain ``msa_id`` (sequence hash),
    so the same JSON serves every record that contains this sequence. The
    release-date cutoff is applied at training time by the dataloader, not
    at prepare time.
    """
    return Path(monomer_templates_dir) / shard(msa_id) / f"{msa_id}.json"
