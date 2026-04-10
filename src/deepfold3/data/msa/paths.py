"""Filesystem layout helpers for MSA NPZ storage.

The training data ``msa/`` directory is split by MSA type and sharded by
the first two characters of the file id (256 buckets), matching the rule
in :mod:`deepfold3.data.paths`::

    {data_dir}/msa/
    ├── unpaired/
    │   ├── 00/
    │   │   └── 00019cc8…861bfd.npz
    │   └── ff/
    │       └── …
    └── paired/
        ├── 1a/
        │   └── 1a02_paired_3aed842c…12b.npz
        └── …

Use these helpers everywhere instead of constructing the path inline so
the layout stays consistent across the dataloader, the import commands,
and the search batch commands.
"""

from __future__ import annotations

from pathlib import Path

from deepfold3.data.paths import shard

UNPAIRED_DIR = "unpaired"
PAIRED_DIR = "paired"


def unpaired_msa_root(msa_dir: Path) -> Path:
    """Top-level directory for unpaired MSA NPZs."""
    return Path(msa_dir) / UNPAIRED_DIR


def paired_msa_root(msa_dir: Path) -> Path:
    """Top-level directory for paired MSA NPZs."""
    return Path(msa_dir) / PAIRED_DIR


def unpaired_msa_path(msa_dir: Path, msa_id: str) -> Path:
    """Path to ``{msa_id}.npz`` under the sharded unpaired layout."""
    return unpaired_msa_root(msa_dir) / shard(msa_id) / f"{msa_id}.npz"


def paired_msa_path(msa_dir: Path, paired_msa_id: str) -> Path:
    """Path to ``{paired_msa_id}.npz`` under the sharded paired layout."""
    return paired_msa_root(msa_dir) / shard(paired_msa_id) / f"{paired_msa_id}.npz"
