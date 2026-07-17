"""Shared helpers for the deepfold3 click commands.

Hosts the boilerplate that multiple commands were repeating verbatim:

* ``setup_cli_logging`` — install a stdout INFO handler and return the logger.
* ``require_manifest`` — verify that ``{data_dir}/manifest.json`` exists or
  exit with a clear error.
* ``select_snapshot`` / ``load_pt_snapshot`` — pick a weight snapshot out of a
  ``.pt`` training checkpoint (shared by ``inference``, ``inference-batch``,
  and ``export``).
* ``pin_cuda_device`` — resolve + pin the current CUDA device.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from deepfold3.model.config import ModelConfig


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


def select_snapshot(ckpt: dict, choice: str) -> dict:
    """Return the chosen weight snapshot from a ``.pt`` checkpoint as a state_dict.

    Shared by ``deepfold3 inference`` (load-time selection) and
    ``deepfold3 export`` (materialize a snapshot as ``.safetensors``).

    Vocabulary (per arxiv 2405.15682 Defazio 2024 + arxiv 2507.09846):
        - 'average' = SF's averaged iterate ``x`` (paper-recommended inference
          target). Stored in ``ckpt['model']`` because :class:`CheckpointManager`
          flips into eval mode before the save. For plain AdamW, this is the
          live params at save time.
        - 'shadow'  = External EMA shadow over ``x`` (or over live params for
          AdamW). An independent smoothing layer with its own time constant
          (~1/(1-decay) step), decoupled from SF's own averaging. Stored under
          ``ckpt['ema']['shadow']`` and overlaid onto ``ckpt['model']``.
        - 'fast'    = SF's fast iterate ``z`` (the SGD-like point that takes
          the raw step before averaging). Reconstructed from
          ``ckpt['optimizer']['state']``. Rarely the right choice for
          inference — exposed for ablation only.
        - 'auto'    = smart default: shadow if present (overlaid on
          ``ckpt['model']``), else ``ckpt['model']``. Inspects the
          ``ckpt['snapshot_kind']`` metadata for logging.
    """
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    shadow = (ckpt.get("ema") or {}).get("shadow") if isinstance(ckpt, dict) else None
    optim_state = ckpt.get("optimizer", {}).get("state", {}) if isinstance(ckpt, dict) else {}
    has_optim_z = any("z" in s for s in optim_state.values())

    if choice == "auto":
        kind = ckpt.get("snapshot_kind") if isinstance(ckpt, dict) else None
        if shadow:
            tag = f"shadow (auto: shadow present, kind={kind})" if kind else "shadow (auto: shadow present)"
            click.echo(f"  snapshot={tag} — overlay {len(shadow)} shadow params")
            return {**model, **shadow}
        click.echo(f"  snapshot=average (auto{f', kind={kind}' if kind else ''})")
        return model

    if choice == "shadow":
        if not shadow:
            raise click.ClickException("--snapshot shadow: no 'ema'/'shadow' entry in checkpoint")
        click.echo(f"  snapshot=shadow ({len(shadow)} shadow params)")
        return {**model, **shadow}

    if choice == "average":
        click.echo("  snapshot=average (= ckpt['model'])")
        return model

    if choice == "fast":
        if not has_optim_z:
            raise click.ClickException("--snapshot fast: no SF 'z' state in checkpoint")
        # Map (param_id_in_optim_state) → param_name by ordering. SF's z
        # always equals model.parameters() in shape and (insertion) order
        # because :class:`AdamWScheduleFree` adds entries one param at a
        # time during step(). Brittle if the model architecture differs
        # from save time, so this path errors loudly on length mismatch
        # and is documented as ablation-only.
        z_by_id = {pid: s["z"] for pid, s in optim_state.items() if "z" in s}
        model_items = list(model.items())
        if len(z_by_id) != len(model_items):
            raise click.ClickException(
                f"--snapshot fast: z count ({len(z_by_id)}) != model param count "
                f"({len(model_items)}); ablation path requires matched ordering."
            )
        z_state = {name: z for (name, _), z in zip(model_items, z_by_id.values(), strict=True)}
        click.echo(f"  snapshot=fast (= SF z; {len(z_state)} params)")
        return z_state

    raise click.ClickException(f"--snapshot: unknown choice {choice!r}")


def load_pt_snapshot(path: Path, snapshot: str) -> tuple[dict, ModelConfig | None]:
    """Load a ``.pt`` checkpoint and return ``(state_dict, model_cfg)``.

    Shared by ``inference``, ``inference-batch``, and ``export``: reads the
    checkpoint (``mmap`` so the unused optimizer / EMA storages are never paged
    in for the common ``average`` snapshot), selects the requested weight
    snapshot via :func:`select_snapshot`, strips any DDP ``module.`` prefix, and
    returns the embedded architectural :class:`ModelConfig` — ``None`` for a
    bare state_dict or a pre-metadata checkpoint. Callers need the config to
    rebuild the model with the flags used at training time (residual scaling,
    layer counts, etc.), whose dataclass defaults may differ from the trained
    values.

    The returned config aliases the checkpoint dict, but the checkpoint's other
    branches (optimizer, EMA) are dropped when the local ``ckpt`` reference goes
    out of scope, so no defensive copy is needed.
    """
    import torch

    # mmap requires torch's default new-zipfile format, which every checkpoint
    # this codebase writes uses.
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    state_dict = select_snapshot(ckpt, snapshot)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model_cfg = ckpt.get("model_cfg") if isinstance(ckpt, dict) else None
    return state_dict, model_cfg


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
