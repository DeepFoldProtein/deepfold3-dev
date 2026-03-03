"""Inference script: run DeepFold3 from an A3M MSA file.

Usage
-----
uv run python -m deepfold3.scripts.inference \\
    --msa      path/to/query.a3m \\
    --output   output/prediction.cif \\
    --params   resources/deepfold3.safetensors \\
    --num-recycles 3 \\
    --num-samples  5 \\
    --seed     42

Supported parameter formats:
  - ``.safetensors`` — native format produced by ``convert_weights.py``
  - ``.bin`` / ``.bin.zst`` — raw AF3 params (on-the-fly conversion;
    Fourier weights will be missing — run ``convert_weights.py`` first)

The script produces one mmCIF file whose MODEL blocks are the requested
diffusion samples.  Per-atom pLDDT is stored in the B-factor column.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch


@click.command("inference")
@click.option(
    "--msa",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input A3M file (first sequence is query)",
)
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output mmCIF file path")
@click.option(
    "--params",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model parameters (.safetensors or .bin/.bin.zst)",
)
@click.option(
    "--mol-type",
    default="protein",
    type=click.Choice(["protein", "rna", "dna"]),
    show_default=True,
    help="Molecule type",
)
@click.option("--num-recycles", default=3, show_default=True, help="Evoformer recycling iterations")
@click.option("--num-samples", default=1, show_default=True, help="Number of diffusion samples")
@click.option("--seed", default=42, show_default=True, help="Random seed for diffusion sampling")
@click.option("--device", default=None, help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.")
def inference(
    msa: Path,
    output: Path,
    params: Path,
    mol_type: str,
    num_recycles: int,
    num_samples: int,
    seed: int,
    device: str | None,
) -> None:
    """Run structure prediction from an A3M MSA file."""
    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    click.echo(f"[inference] device: {dev}")

    # ------------------------------------------------------------------ #
    # Load model weights
    # ------------------------------------------------------------------ #
    click.echo(f"[inference] loading parameters from {params} …")
    t0 = time.perf_counter()
    from deepfold3.model.model import DeepFold3

    if params.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(params))
    else:
        import warnings

        from deepfold3.convert.af3 import haiku_to_torch
        from deepfold3.params import read_af3_params

        warnings.warn(
            "Loading raw AF3 params — Fourier weights will be missing. "
            "Run convert_weights.py to produce a .safetensors file.",
            stacklevel=1,
        )
        raw_params = read_af3_params(params)
        state_dict = haiku_to_torch(raw_params)

    model = DeepFold3()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        click.echo(f"  [WARN] missing keys   : {len(missing)}")
    if unexpected:
        click.echo(f"  [WARN] unexpected keys: {len(unexpected)}")
    model.to(dev).eval()
    click.echo(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # Featurize
    # ------------------------------------------------------------------ #
    click.echo(f"[inference] featurizing {msa} …")
    t0 = time.perf_counter()
    from deepfold3.data.module.dataset import InferenceInput

    a3m_text = msa.read_text()
    feats = InferenceInput.from_a3m(a3m_text, mol_type=mol_type)
    N = feats["res_type"].shape[0]
    S = feats["msa"].shape[0]
    click.echo(f"  {N} tokens, {S} MSA sequences  ({time.perf_counter() - t0:.1f}s)")

    # Move to device
    feats = {k: v.to(dev) for k, v in feats.items()}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    click.echo(f"[inference] running model  (recycles={num_recycles}, samples={num_samples}, seed={seed}) …")
    t0 = time.perf_counter()
    with torch.no_grad():
        model_output = model(
            feats,
            seed=seed,
            num_recycles=num_recycles,
            num_samples=num_samples,
        )
    elapsed = time.perf_counter() - t0
    click.echo(f"  done in {elapsed:.1f}s  —  positions: {tuple(model_output.atom_positions.shape)}")

    # ------------------------------------------------------------------ #
    # Confidence metrics + save
    # ------------------------------------------------------------------ #
    from deepfold3.io.confidences import (
        compute_iptm,
        compute_pde,
        compute_plddt,
        compute_ptm,
        compute_ranking_score,
        save_confidences_json,
    )

    atom_mask = model_output.atom_mask[0].float()  # [N, M] — same for all samples
    seq_mask = feats["seq_mask"].float().to(dev)
    pair_mask = feats["pair_mask"].float().to(dev)
    asym_id = feats["asym_id"].to(dev)

    for s in range(num_samples):
        metrics = []

        if model_output.plddt is not None:
            _, per_token = compute_plddt(model_output.plddt, atom_mask)
            mean_plddt = float(
                (per_token[s].cpu() * feats["seq_mask"].float().cpu()).sum()
                / feats["seq_mask"].float().cpu().sum().clamp(min=1)
            )
            metrics.append(f"pLDDT={mean_plddt:.1f}")

        if model_output.pae is not None:
            ptm = compute_ptm(model_output.pae[s], seq_mask, pair_mask)
            iptm = compute_iptm(model_output.pae[s], seq_mask, pair_mask, asym_id)
            ranking = compute_ranking_score(ptm, iptm)
            metrics.append(f"pTM={ptm:.3f}")
            if iptm == iptm:  # not NaN
                metrics.append(f"ipTM={iptm:.3f}")
            metrics.append(f"ranking={ranking:.3f}")

        if model_output.pde is not None:
            _, avg_pde = compute_pde(model_output.pde, pair_mask)
            metrics.append(f"gPDE={float(avg_pde[s]):.2f}")

        click.echo(f"  sample {s}: {', '.join(metrics)}")

    # Save mmCIF
    click.echo(f"[inference] writing {output} …")
    from deepfold3.io.mmcif import save_mmcif

    save_mmcif(output, model_output, feats)
    click.echo(f"  saved  {output.resolve()}")

    # Save confidences JSON (per sample)
    for s in range(num_samples):
        if num_samples == 1:
            json_path = output.with_name(output.stem + "_confidences.json")
        else:
            json_path = output.with_name(output.stem + f"_sample{s}_confidences.json")
        save_confidences_json(json_path, model_output, feats, sample_idx=s)
        click.echo(f"  saved  {json_path.resolve()}")


if __name__ == "__main__":
    inference()
