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


def pin_cuda_device(device: str | None):
    """Resolve ``device`` to a ``torch.device`` and pin it as the current CUDA device.

    Triton launches kernels on the *current* CUDA device's context, not the
    input tensor's device. A single-process entry point running on a non-default
    device (e.g. ``cuda:1``) while the current device is still 0 makes the Triton
    launcher reject the on-device pointers with "Pointer argument cannot be
    accessed from Triton (cpu tensor?)". Calling this once — after resolving the
    device, before any kernel launch — keeps the launch context aligned with the
    model/feature device. DDP training pins via its own ``set_device``; this is
    for the single-process commands (warmup-triton, inference).

    ``device`` may be a device string, or ``None`` to auto-select (``cuda`` if
    available, else ``cpu``). Returns the resolved ``torch.device``.
    """
    import torch  # local import keeps _common light for non-torch CLI commands

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev)
    return dev
