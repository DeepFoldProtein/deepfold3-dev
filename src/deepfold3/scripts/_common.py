"""Shared CLI scaffolding for the deepfold3 click commands.

Hosts the boilerplate that ~10 commands in ``scripts/prepare.py``,
``scripts/scan_mmcif.py``, and friends were repeating verbatim:

* ``setup_cli_logging`` — install a stdout INFO handler and return the logger.
* ``require_manifest`` — verify that ``{data_dir}/manifest.json`` exists or
  exit with a clear error.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


def setup_cli_logging(name: str | None = None) -> logging.Logger:
    """Install a stdout INFO handler and return the named logger.

    Idempotent: ``logging.basicConfig`` is a no-op if the root logger
    already has handlers, so calling this from multiple commands in the
    same process is safe.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


def require_manifest(data_dir: Path) -> Path:
    """Return ``{data_dir}/manifest.json`` or exit with a CLI error."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        click.echo(f"Error: {manifest_path} not found.", err=True)
        sys.exit(1)
    return manifest_path
