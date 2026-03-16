"""Inference script: run DeepFold3 from an A3M MSA file or JSON input.

Usage
-----
uv run python -m deepfold3.scripts.inference \\
    --msa      path/to/query.a3m \\
    --output   output/prediction.cif \\
    --params   resources/deepfold3.safetensors \\
    --num-samples  5 \\
    --seed     42

# Or from a Protenix-format JSON input:
uv run python -m deepfold3.scripts.inference \\
    --json     path/to/input.json \\
    --output   output/prediction.cif \\
    --params   resources/deepfold3.safetensors \\
    --template-cache /tmp/template_cache

Supported parameter formats:
  - ``.safetensors`` — native format produced by ``convert_weights.py``
  - ``.bin`` / ``.bin.zst`` — raw AF3 params (on-the-fly conversion;
    Fourier weights will be missing — run ``convert_weights.py`` first)

The script produces one mmCIF file whose MODEL blocks are the requested
diffusion samples.  Per-atom pLDDT is stored in the B-factor column.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import torch


@click.command("inference")
@click.option(
    "--msa",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input A3M file (first sequence is query)",
)
@click.option(
    "--json",
    "json_input",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Protenix-format JSON input file (alternative to --msa)",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path: directory for --json (filenames from job name), or .cif file for --msa",
)
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
@click.option("--num-recycles", default=10, show_default=True, help="Evoformer recycling iterations")
@click.option("--num-samples", default=1, show_default=True, help="Number of diffusion samples")
@click.option("--seed", default=42, show_default=True, help="Random seed for diffusion sampling")
@click.option("--device", default=None, help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.")
@click.option(
    "--templates",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Directory of template mmCIF files (optional, used with --msa)",
)
@click.option(
    "--template-cache",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to cache downloaded template mmCIF files (used with --json)",
)
@click.option(
    "--pdb-mmcif-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Local PDB mmCIF mirror directory (layout: {dir}/{mid2}/{id}.cif.gz)",
)
@click.option(
    "--optim",
    default=None,
    type=click.Choice(["none", "recommended", "all", "triton"]),
    help="Optimization backend: none (default), recommended (evo attn + triangle mul + compile), all, triton (fused kernels only)",
)
def inference(
    msa: Path | None,
    json_input: Path | None,
    output: Path,
    params: Path,
    mol_type: str,
    num_recycles: int,
    num_samples: int,
    seed: int,
    device: str | None,
    templates: Path | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    optim: str | None,
) -> None:
    """Run structure prediction from an A3M MSA file or JSON input."""
    # Validate input: exactly one of --msa or --json
    if msa is None and json_input is None:
        raise click.UsageError("Provide either --msa or --json")
    if msa is not None and json_input is not None:
        raise click.UsageError("Provide only one of --msa or --json, not both")
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    click.echo(f"[inference] device: {dev}")

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
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

    # Build model config with optimization settings
    from deepfold3.model.config import ModelConfig, OptimConfig

    model_cfg = ModelConfig()
    if optim == "all":
        model_cfg.optim = OptimConfig.all_enabled()
    elif optim == "recommended":
        model_cfg.optim = OptimConfig.recommended()
    elif optim == "triton":
        model_cfg.optim = OptimConfig.triton_only()
    # else: default OptimConfig (all False)

    if optim and optim != "none":
        click.echo(f"[inference] optimizations: {optim}")

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Fused kernel parameters are derived from base weights via _sync_to_fused — not real misses
    real_missing = [
        k
        for k in missing
        if not any(p in k for p in OptimConfig.FUSED_STATE_DICT_MARKERS)
    ]
    if real_missing:
        click.echo(f"  [WARN] missing keys   : {len(real_missing)}")
    if unexpected:
        click.echo(f"  [WARN] unexpected keys: {len(unexpected)}")
    model.to(dev).eval()
    click.echo(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Featurize
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from deepfold3.data.module.dataset import InferenceInput

    job_name: str | None = None
    entity_descriptions: dict[str, str] | None = None
    job_title: str | None = None
    if json_input is not None:
        click.echo(f"[inference] featurizing {json_input} …")
        from deepfold3.data.parse.json_input import parse_json_input

        jobs = parse_json_input(json_input, template_cache_dir=template_cache, pdb_mmcif_dir=pdb_mmcif_dir)
        if not jobs:
            raise click.UsageError("No jobs found in JSON input")
        job = jobs[0]
        job_name = job.name
        job_title = job.description or job.name

        # Build entity descriptions from chain descriptions
        chain_idx = 0
        desc_map: dict[str, str] = {}
        for chain_list in [job.protein_chains, job.dna_chains, job.rna_chains, job.ligand_chains]:
            for chain in chain_list:
                chain_name = chr(ord("A") + chain_idx)
                if chain.description:
                    desc_map[chain_name] = chain.description
                chain_idx += 1
        if desc_map:
            entity_descriptions = desc_map

        feats = InferenceInput.from_job(job)
    else:
        click.echo(f"[inference] featurizing {msa} …")
        a3m_text = msa.read_text()

        # Parse templates if directory provided
        template_feats = None
        if templates is not None:
            import numpy as np

            from deepfold3.data import const as data_const
            from deepfold3.data.parse.a3m import parse_a3m
            from deepfold3.data.parse.templates import parse_template_mmcifs

            mt = data_const.MOL_TYPE_IDS.get(mol_type.upper(), data_const.MOL_PROTEIN)
            msa_obj = parse_a3m(a3m_text, mt)
            n_tokens = msa_obj.sequences.shape[1]
            query_residue_index = np.arange(n_tokens, dtype=np.int64) + 1  # 1-based
            template_feats = parse_template_mmcifs(templates, query_residue_index)
            if template_feats:
                click.echo(f"  loaded {len(template_feats)} template(s) from {templates}")
            else:
                click.echo("  no valid templates found")
                template_feats = None

        feats = InferenceInput.from_a3m(a3m_text, mol_type=mol_type, templates=template_feats)

    N = feats["res_type"].shape[0]
    S = feats["msa"].shape[0]
    T = feats.get("template_aatype", torch.zeros(0)).shape[0]
    click.echo(f"  {N} tokens, {S} MSA sequences, {T} templates  ({time.perf_counter() - t0:.1f}s)")

    # Bucket padding for torch.compile efficiency (avoids per-shape recompilation)
    original_n = N
    use_bucket = model_cfg.optim.use_torch_compile
    if use_bucket:
        from deepfold3.model.bucket import next_bucket, pad_features

        bucket_n = next_bucket(N)
        if bucket_n > N:
            feats, original_n = pad_features(feats, bucket_n)
            click.echo(f"  bucket padding: {N} → {bucket_n} tokens")

    # Move to device
    feats = {k: v.to(dev) for k, v in feats.items()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    click.echo(f"[inference] running model  (recycles={num_recycles}, samples={num_samples}, seed={seed}) …")
    t0 = time.perf_counter()
    with torch.inference_mode():
        model_output = model(
            feats,
            seed=seed,
            num_recycles=num_recycles,
            num_samples=num_samples,
        )

    # Remove bucket padding from output and feats
    if use_bucket and original_n < feats["res_type"].shape[0]:
        from deepfold3.model.bucket import unpad_features, unpad_output

        model_output = unpad_output(model_output, original_n)
        feats = unpad_features(feats, original_n)

    elapsed = time.perf_counter() - t0
    click.echo(f"  done in {elapsed:.1f}s  —  positions: {tuple(model_output.atom_positions.shape)}")

    # ------------------------------------------------------------------
    # Confidence metrics + save
    # ------------------------------------------------------------------
    from deepfold3.io.confidences import (
        compute_iptm,
        compute_pde,
        compute_plddt,
        compute_ptm,
        compute_ranking_score,
        fraction_disordered,
        has_clash,
        save_confidences_json,
    )

    atom_mask = model_output.atom_mask[0].float()  # [N, M] — same for all samples
    seq_mask = feats["seq_mask"].float().to(dev)
    pair_mask = feats["pair_mask"].float().to(dev)
    asym_id = feats["asym_id"].to(dev)
    seq_mask_cpu = feats["seq_mask"].float().cpu()
    seq_mask_sum = seq_mask_cpu.sum().clamp(min=1)

    # Use frames_mask for pTM/ipTM (excludes frameless tokens like ions/ligands)
    if "frames_mask" in feats:
        tm_mask = feats["frames_mask"].float().to(dev)
        tm_pair_mask = (tm_mask.unsqueeze(-1) * tm_mask.unsqueeze(-2)) * pair_mask
    else:
        tm_mask = seq_mask
        tm_pair_mask = pair_mask

    # Compute pLDDT once for all samples
    per_token_plddt = None
    if model_output.plddt is not None:
        _, per_token_plddt = compute_plddt(model_output.plddt, atom_mask)

    # Polymer mask for clash/disorder
    polymer_mask = feats["frames_mask"].to(dev) if "frames_mask" in feats else seq_mask.bool()

    sample_rankings: list[tuple[float, int]] = []  # (ranking_score, sample_idx)

    for s in range(num_samples):
        metrics = []

        if per_token_plddt is not None:
            mean_plddt = float((per_token_plddt[s].cpu() * seq_mask_cpu).sum() / seq_mask_sum)
            metrics.append(f"pLDDT={mean_plddt:.1f}")

        ptm = 0.0
        iptm = float("nan")
        if model_output.pae is not None:
            ptm = compute_ptm(model_output.pae[s], tm_mask, tm_pair_mask)
            iptm = compute_iptm(model_output.pae[s], tm_mask, tm_pair_mask, asym_id)
            metrics.append(f"pTM={ptm:.3f}")
            if not math.isnan(iptm):
                metrics.append(f"ipTM={iptm:.3f}")

        if model_output.pde is not None:
            _, avg_pde = compute_pde(model_output.pde, pair_mask)
            metrics.append(f"gPDE={float(avg_pde[s]):.2f}")

        frac_disorder = 0.0
        if per_token_plddt is not None:
            frac_disorder = fraction_disordered(per_token_plddt[s].to(dev), polymer_mask)
        sample_clash = has_clash(
            model_output.atom_positions[s],
            atom_mask,
            feats["residue_index"].to(dev),
            asym_id,
            polymer_mask,
        )
        ranking = compute_ranking_score(ptm, iptm, frac_disorder, sample_clash)
        metrics.append(f"disorder={frac_disorder:.3f}")
        if sample_clash:
            metrics.append("CLASH")
        metrics.append(f"ranking={ranking:.3f}")

        sample_rankings.append((ranking, s))
        click.echo(f"  sample {s}: {', '.join(metrics)}")

    # Print ranked summary
    if num_samples > 1:
        sample_rankings.sort(reverse=True)
        click.echo("[inference] ranking:")
        for rank, (score, s_idx) in enumerate(sample_rankings, 1):
            name = f"{job_name}_sample{s_idx}" if job_name else f"sample_{s_idx}"
            click.echo(f"  #{rank}  {name}  ranking_score={score:.4f}")

    # Determine output paths
    if job_name is not None:
        # JSON input: output is a directory, filenames from job name
        output_dir = output
        output_dir.mkdir(parents=True, exist_ok=True)
        cif_path = output_dir / f"{job_name}.cif"
    else:
        # MSA input: output is a .cif file path
        cif_path = output
        output_dir = cif_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save mmCIF
    click.echo(f"[inference] writing {cif_path} …")
    from deepfold3.io.mmcif import save_mmcif

    save_mmcif(cif_path, model_output, feats, entity_descriptions=entity_descriptions, title=job_title)
    click.echo(f"  saved  {cif_path.resolve()}")

    # Save confidences JSON (per sample)
    for s in range(num_samples):
        stem = job_name if job_name is not None else cif_path.stem
        if num_samples == 1:
            json_path = output_dir / f"{stem}_confidences.json"
        else:
            json_path = output_dir / f"{stem}_sample{s}_confidences.json"
        save_confidences_json(json_path, model_output, feats, sample_idx=s, seed=seed)
        click.echo(f"  saved  {json_path.resolve()}")


if __name__ == "__main__":
    inference()
