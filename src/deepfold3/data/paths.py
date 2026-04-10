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
    """Path to a template-index JSON under the sharded layout."""
    return Path(templates_dir) / shard(pdb_id) / f"{pdb_id}.json"
