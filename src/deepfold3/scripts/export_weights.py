"""Export a training checkpoint (.pt) to inference-ready .safetensors.

Usage
-----
deepfold3 export \\
    --checkpoint runs/exp1/checkpoint_final.pt \\
    --output     weights.safetensors \\
    --snapshot   average

Selects one weight *snapshot* from the checkpoint (default ``average`` =
Schedule-Free's averaged iterate ``x`` / live params for plain AdamW; see
:func:`deepfold3.scripts._common.select_snapshot` for the full vocabulary),
strips any DDP ``module.`` prefix, and writes a single ``.safetensors`` file
with the architectural :class:`ModelConfig` embedded as JSON in the header
metadata. ``deepfold3 inference`` reads that metadata back automatically, so
the exported file is self-contained — no sidecar config needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from deepfold3.scripts import cli
from deepfold3.scripts._common import load_pt_snapshot


@cli.command("export")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training checkpoint (.pt) to export.",
)
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output .safetensors file path.")
@click.option(
    "--snapshot",
    type=click.Choice(["average", "auto", "shadow", "fast"]),
    default="average",
    show_default=True,
    help=(
        "Which weight snapshot to extract. 'average' = ckpt['model'] "
        "(SF averaged iterate x / live AdamW params). 'shadow' = external EMA "
        "shadow (errors if absent). 'auto' = shadow when present else average. "
        "'fast' = SF z buffer (ablation only)."
    ),
)
def export(checkpoint: Path, output: Path, snapshot: str) -> None:
    """Extract a weight snapshot from a .pt checkpoint into .safetensors."""
    if output.suffix != ".safetensors":
        raise click.ClickException(f"--output must end in .safetensors, got {output.name!r}")

    from safetensors.torch import save_file

    from deepfold3.model.config import model_cfg_to_metadata

    click.echo(f"[export] loading checkpoint {checkpoint} …")
    t0 = time.perf_counter()
    state_dict, model_cfg = load_pt_snapshot(checkpoint, snapshot)

    # Embed the architectural ModelConfig as JSON so the exported file is
    # self-contained (inference reconstructs the build from it). Absent only
    # for pre-metadata checkpoints — warn but still write the weights.
    metadata = None
    if model_cfg is not None:
        metadata = model_cfg_to_metadata(model_cfg)
    else:
        click.echo("[export] WARNING: checkpoint has no embedded model_cfg — writing weights without config metadata.")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output), metadata=metadata)
    click.echo(
        f"[export] wrote {len(state_dict)} tensors"
        f"{' + ModelConfig metadata' if metadata else ''} to {output.resolve()} "
        f"({time.perf_counter() - t0:.1f}s)"
    )


if __name__ == "__main__":
    export()
