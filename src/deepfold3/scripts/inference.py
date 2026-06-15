"""Inference script: run DeepFold3 from an A3M MSA file or JSON input.

Usage
-----
uv run python -m deepfold3.scripts.inference \\
    --msa      path/to/query.a3m \\
    --output   output/prediction.cif \\
    --params   resources/deepfold3.safetensors \\
    --num-samples  5 \\
    --seed     42

# Or from an AF3/Protenix-format JSON input:
uv run python -m deepfold3.scripts.inference \\
    --json     path/to/input.json \\
    --output   output_dir/ \\
    --params   resources/deepfold3.safetensors \\
    --template-cache /tmp/template_cache

Supported parameter formats:
  - ``.safetensors`` — native format produced by ``convert_weights.py``
  - ``.bin`` / ``.bin.zst`` — raw AF3 params (on-the-fly conversion;
    Fourier weights will be missing — run ``convert_weights.py`` first)

For ``--json`` input, the output directory is populated with one
sub-directory per (seed, sample) pair named ``seed-{seed}_sample-{idx}/``
containing ``model.cif`` and ``confidences.json``, plus a top-level
``ranking_scores.csv`` and ``{job_name}_summary_confidences.json``
mirroring the AlphaFold 3 output layout.

For ``--msa`` / ``--sequence`` input, a single mmCIF file with all
samples stacked as MODEL blocks is written to ``--output``.
Per-atom pLDDT is stored in the B-factor column.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch

from deepfold3.scripts import cli
from deepfold3.scripts._common import pin_cuda_device


def _select_snapshot(ckpt: dict, choice: str) -> dict:
    """Return the chosen weight snapshot as a state_dict.

    Vocabulary (per arxiv 2405.15682 Defazio 2024 + arxiv 2507.09846):
        - 'average' = SF's averaged iterate ``x`` (paper-recommended inference
          target). Stored in ``ckpt['model']`` because :class:`CheckpointManager`
          flips into eval mode before the save. For plain AdamW, this is the
          live params at save time.
        - 'shadow'  = External EMA shadow over ``x`` (or over live params for
          AdamW). An independent smoothing layer with its own time constant
          (~1/(1-decay) step), decoupled from SF's own averaging. Stored under
          ``ckpt['ema']['shadow']`` and overlaid onto ``ckpt['model']`` at
          load time.
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


@cli.command("inference")
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
    help="Model parameters (.safetensors converted weights, .pt training checkpoint, or .bin/.bin.zst raw AF3 params)",
)
@click.option(
    "--mol-type",
    default="protein",
    type=click.Choice(["protein", "rna", "dna"]),
    show_default=True,
    help="Molecule type",
)
@click.option("--num-recycles", default=10, show_default=True, help="Evoformer recycling iterations")
@click.option("--num-samples", default=5, show_default=True, help="Number of diffusion samples per seed")
@click.option(
    "--snapshot",
    type=click.Choice(["auto", "shadow", "average", "fast"]),
    default="auto",
    show_default=True,
    help=(
        "Which weight snapshot to load from a .pt training checkpoint. "
        "'auto' = shadow when present (overlaid on ckpt['model']), else ckpt['model']. "
        "'shadow' forces the external EMA shadow (errors if absent). "
        "'average' forces ckpt['model'] (= SF's averaged iterate x post eval-mode flip; "
        "= live params for AdamW). 'fast' forces the SF z buffer reconstructed from "
        "ckpt['optimizer']['state'] (SF only; ablation use). For .safetensors / raw "
        "AF3 params this flag is ignored."
    ),
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help=(
        "Random seed for diffusion sampling. For --json input, this is "
        "prepended to the JSON's modelSeeds list (all seeds are run). "
        "Defaults to 42 for --msa/--sequence input, and to the JSON's "
        "modelSeeds otherwise."
    ),
)
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
    default="inference",
    type=click.Choice(["none", "recommended", "inference", "all", "triton"]),
    show_default=True,
    help="Optimization preset: inference (default, Triton+tiling), recommended (training), triton (no tiling), none, all (testing)",
)
@click.option(
    "--plot-confidences",
    is_flag=True,
    default=False,
    help="Also write a PAE/PDE/contact-probability PNG next to each confidences JSON",
)
def inference(
    msa: Path | None,
    json_input: Path | None,
    output: Path,
    params: Path,
    mol_type: str,
    num_recycles: int,
    num_samples: int,
    snapshot: str,
    seed: int | None,
    device: str | None,
    templates: Path | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    optim: str | None,
    plot_confidences: bool,
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
    # pin_cuda_device also sets the current CUDA device so the fused Triton
    # kernels launch on the same device as the model/features — a non-default
    # --device (e.g. cuda:1) otherwise trips "cannot be accessed from Triton".
    dev = pin_cuda_device(device)
    click.echo(f"[inference] device: {dev}")

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
    click.echo(f"[inference] loading parameters from {params} …")
    t0 = time.perf_counter()
    from deepfold3.model.model import DeepFold3

    # Architectural ModelConfig embedded in training checkpoints — preserved
    # so the inference build matches whatever flags (e.g.
    # ``pairformer.residual_output_scaling``) were active at training time.
    # Stays ``None`` only for raw-AF3 ``.bin`` / ``.bin.zst`` paths and for
    # legacy ``.safetensors`` files exported before metadata embedding was
    # introduced — both fall back to ``ModelConfig()`` defaults.
    ckpt_model_cfg = None

    if params.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        from deepfold3.model.config import model_cfg_from_metadata

        # safetensors metadata read is cheap (header-only); pull ModelConfig
        # JSON before loading tensors so we can build the model with the
        # correct architectural flags.
        with safe_open(str(params), framework="pt") as f:
            ckpt_model_cfg = model_cfg_from_metadata(f.metadata())
        state_dict = load_file(str(params))
    elif params.suffix == ".pt":
        ckpt = torch.load(str(params), map_location="cpu", weights_only=False)
        state_dict = _select_snapshot(ckpt, snapshot)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        # Capture the embedded model_cfg before discarding the ckpt blob —
        # critical for runs that override architectural flags (residual
        # scaling, layer counts, etc.) whose dataclass defaults differ from
        # the values used during training.
        if isinstance(ckpt, dict) and "model_cfg" in ckpt:
            import copy

            ckpt_model_cfg = copy.deepcopy(ckpt["model_cfg"])
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

    if ckpt_model_cfg is not None:
        model_cfg = ckpt_model_cfg
        click.echo("[inference] using model_cfg embedded in checkpoint")
    else:
        model_cfg = ModelConfig()
        if params.suffix == ".pt":
            click.echo(
                "[inference] checkpoint has no embedded model_cfg — falling "
                "back to ModelConfig() defaults. If the checkpoint was trained "
                "with non-default architectural flags (e.g. residual_output_scaling), "
                "the inference build will silently disagree."
            )
    # ``--optim`` always overrides the OptimConfig sub-config so inference
    # picks fused / SDPA / tiled kernels regardless of how the checkpoint
    # was trained (training and inference often differ here).
    if optim == "all":
        model_cfg.optim = OptimConfig.all_enabled()
    elif optim == "recommended":
        model_cfg.optim = OptimConfig.recommended()
    elif optim == "inference":
        model_cfg.optim = OptimConfig.inference()
    elif optim == "triton":
        model_cfg.optim = OptimConfig.triton_only()
    # else: keep checkpoint's optim (or default OptimConfig if no checkpoint cfg)

    if optim and optim != "none":
        torch.set_float32_matmul_precision("high")  # TF32 for better perf
        click.echo(f"[inference] optimizations: {optim}")

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Filter legacy fused-kernel key names (no-op for current checkpoints).
    real_missing = [k for k in missing if not any(p in k for p in OptimConfig.FUSED_STATE_DICT_MARKERS)]
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
    # Determine seed list. JSON inputs honour the JSON's modelSeeds
    # (with --seed prepended if explicitly passed); MSA/sequence inputs
    # use a single seed (default 42).
    # ------------------------------------------------------------------
    if json_input is not None:
        json_seeds = list(job.model_seeds) if job.model_seeds else [42]
        if seed is not None:
            seed_list = [seed] + [s for s in json_seeds if s != seed]
        else:
            seed_list = json_seeds
    else:
        seed_list = [seed if seed is not None else 42]

    from deepfold3.io.confidences import (
        save_confidences_json,
        save_ranking_scores_csv,
        save_summary_confidences_json,
    )
    from deepfold3.io.mmcif import save_mmcif

    if json_input is not None:
        output_dir = output
    else:
        cif_path = output
        output_dir = cif_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []  # one entry per (seed, sample), augmented with seed/sample

    for current_seed in seed_list:
        click.echo(
            f"[inference] running model  (seed={current_seed}, recycles={num_recycles}, samples={num_samples}) …"
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            model_output = model(
                feats,
                seed=current_seed,
                num_recycles=num_recycles,
                num_samples=num_samples,
            )

        iter_feats = feats
        if use_bucket and original_n < feats["res_type"].shape[0]:
            from deepfold3.model.bucket import unpad_features, unpad_output

            model_output = unpad_output(model_output, original_n)
            iter_feats = unpad_features(feats, original_n)

        elapsed = time.perf_counter() - t0
        click.echo(f"  done in {elapsed:.1f}s  —  positions: {tuple(model_output.atom_positions.shape)}")

        # Non-JSON path writes one multi-MODEL mmCIF; JSON path writes
        # per-sample directories.  Either way, save_confidences_json
        # returns the per-sample summary used for ranking.
        if json_input is None:
            click.echo(f"[inference] writing {cif_path} …")
            save_mmcif(
                cif_path,
                model_output,
                iter_feats,
                entity_descriptions=entity_descriptions,
                title=job_title,
            )
            click.echo(f"  saved  {cif_path.resolve()}")

        for s in range(num_samples):
            if json_input is not None:
                sample_dir = output_dir / f"seed-{int(current_seed)}_sample-{int(s)}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_mmcif(
                    sample_dir / "model.cif",
                    model_output,
                    iter_feats,
                    entity_descriptions=entity_descriptions,
                    title=job_title,
                    sample_idx=s,
                )
                conf_path = sample_dir / "confidences.json"
            else:
                stem = cif_path.stem
                conf_path = output_dir / (
                    f"{stem}_confidences.json" if num_samples == 1 else f"{stem}_sample{s}_confidences.json"
                )

            summary = save_confidences_json(conf_path, model_output, iter_feats, sample_idx=s, seed=int(current_seed))
            summary = {**summary, "seed": int(current_seed), "sample": int(s)}
            summaries.append(summary)

            iptm_str = f", ipTM={summary['iptm']:.3f}" if summary.get("iptm") is not None else ""
            clash_str = ", CLASH" if summary["has_clash"] else ""
            click.echo(
                f"  seed {current_seed} sample {s}: "
                f"pLDDT={summary['mean_plddt']:.1f}, pTM={summary['ptm']:.3f}{iptm_str}, "
                f"gPDE={summary['average_pde']:.2f}, disorder={summary['fraction_disordered']:.3f}"
                f"{clash_str}, ranking={summary['ranking_score']:.3f}"
            )
            click.echo(f"  saved  {conf_path.resolve()}")

            if plot_confidences:
                from deepfold3.scripts.plot_confidences import plot_confidences_file

                plot_path = conf_path.with_suffix(".png")
                if plot_confidences_file(conf_path, plot_path):
                    click.echo(f"  saved  {plot_path.resolve()}")

    # ------------------------------------------------------------------
    # Aggregate ranking outputs (JSON input only)
    # ------------------------------------------------------------------
    if json_input is not None and summaries:
        ranking_csv = output_dir / "ranking_scores.csv"
        save_ranking_scores_csv(ranking_csv, summaries)
        click.echo(f"  saved  {ranking_csv.resolve()}")

        summary_name = f"{job_name}_summary_confidences.json" if job_name else "summary_confidences.json"
        summary_path = output_dir / summary_name
        save_summary_confidences_json(summary_path, summaries, job_name=job_name)
        click.echo(f"  saved  {summary_path.resolve()}")

        if len(summaries) > 1:
            ranked = sorted(summaries, key=lambda r: r["ranking_score"], reverse=True)
            click.echo("[inference] ranking:")
            for rank, row in enumerate(ranked, 1):
                click.echo(
                    f"  #{rank}  seed-{row['seed']}_sample-{row['sample']}  ranking_score={row['ranking_score']:.4f}"
                )


if __name__ == "__main__":
    inference()
