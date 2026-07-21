"""Convert AF3 parameters to DeepFold3 safetensors format.

Usage
-----
uv run python -m deepfold3.scripts.convert_weights \\
    --params resources/af3.bin.zst \\
    --af3-ref references/alphafold3 \\
    --output resources/deepfold3.safetensors

This reads the AF3 binary parameters, converts Haiku weights to PyTorch
layout, extracts the Fourier embedding constants from the AF3 reference
source code, and saves everything as a single ``.safetensors`` file.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from deepfold3.scripts import cli


@cli.command("convert")
@click.option(
    "--params",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AF3 parameter file (af3.bin or af3.bin.zst)",
)
@click.option(
    "--af3-ref",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the AF3 reference repository",
)
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output .safetensors file path")
def convert(params: Path, af3_ref: Path, output: Path) -> None:
    """Convert AF3 parameters to safetensors format."""
    # Step 1: Read AF3 binary params
    click.echo(f"[convert] reading AF3 params from {params} …")
    t0 = time.perf_counter()
    from deepfold3.params import read_af3_params

    raw_params = read_af3_params(params)
    click.echo(f"  read {len(raw_params)} scopes in {time.perf_counter() - t0:.1f}s")

    # Step 2: Convert Haiku → PyTorch state dict
    click.echo("[convert] converting Haiku → PyTorch …")
    t0 = time.perf_counter()
    from deepfold3.convert.af3 import haiku_to_torch

    state_dict = haiku_to_torch(raw_params)
    click.echo(f"  {len(state_dict)} tensors in {time.perf_counter() - t0:.1f}s")

    # Step 3: Extract Fourier weights from AF3 reference source
    click.echo(f"[convert] extracting Fourier weights from {af3_ref} …")
    from deepfold3.convert.af3 import extract_fourier_weights

    fourier = extract_fourier_weights(af3_ref)
    for k, v in fourier.items():
        click.echo(f"  {k}: {tuple(v.shape)}")
        state_dict[k] = v

    # Step 4: Save as safetensors with embedded ModelConfig metadata.
    # Converted AF3 weights map onto the default ``ModelConfig`` as-is. In
    # particular, ``ref_element_zero_index`` must stay at its default (False):
    # AF3 one-hots the element at the raw atomic number Z (128 classes,
    # H=1, C=6, ...) — identical to DeepFold3's default — and the converter
    # copies ``embed_ref_element`` rows verbatim (DF3 input index i ↔ AF3
    # input index i ↔ atomic number i), so the model must one-hot at Z to
    # match. Do NOT set it True: Z-1 is the *Protenix* convention (118
    # classes), not AF3, and shifts every element down by one against the
    # AF3-trained weights (empirically: 8ey4 collapses, distogram corr
    # 0.35→0.44 and interfaces separate once corrected).
    click.echo(f"[convert] saving {output} …")
    output.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    from deepfold3.model.config import ModelConfig, model_cfg_to_metadata

    af3_cfg = ModelConfig()
    metadata = model_cfg_to_metadata(af3_cfg)
    save_file(state_dict, str(output), metadata=metadata)
    click.echo(f"  saved {len(state_dict)} tensors + ModelConfig metadata to {output.resolve()}")


if __name__ == "__main__":
    convert()
