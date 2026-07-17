"""Batch inference — load the model once, run many JSON inputs back-to-back.

Use case: trajectory / benchmark sweeps that would otherwise re-load the
5.8 GB checkpoint for every query. Model load is ~12-15 s on B200; for a
sweep of 40 queries that's ~10 minutes purely in setup. This command
amortises it across the full input list.

Input format — text file, one query per line:

    {json_path} [TAB {output_dir}]

If only ``json_path`` is given on a line, the output directory is
``{output_template}`` rendered with ``{json_dir}`` (parent of the JSON
file), ``{json_stem}`` (file stem) and ``{params_stem}`` (parameter file
stem; useful for putting all sweep outputs under a per-checkpoint dir).
Blank lines and lines starting with ``#`` are ignored.

Usage::

    uv run deepfold3 inference-batch \\
        --inputs-file inputs.txt \\
        --params runs/df3-fapeS3/checkpoint_10000.pt \\
        --num-samples 30 \\
        --seed 42 --seed 43 --seed 44 --seed 45 \\
        --snapshot average \\
        --output-template '{json_dir}/inference/{params_stem}'

``--seed`` is repeatable: each input is folded once per seed into its own
``seed-{seed}_sample-{s}`` subdir (S seeds x num_samples structures). With
``--skip-existing`` a job is skipped only when every requested (seed, sample)
is already on disk, and any individually-complete seed is skipped, so an
interrupted multi-seed sweep resumes seed-by-seed.

Per-job options (num-recycles, num-samples, seed, plot-confidences,
template-cache, pdb-mmcif-dir) apply uniformly to every line.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch

from deepfold3.scripts import cli


def _load_model_once(params: Path, snapshot: str, dev: torch.device, optim: str):
    """Load weights + build model. Returns ``(model, model_cfg)``.

    Same as the top half of ``inference``. Extracted so batch loops can
    amortise the ~12 s checkpoint-read cost.
    """
    from deepfold3.model.config import ModelConfig, OptimConfig
    from deepfold3.model.model import DeepFold3
    from deepfold3.scripts._common import load_pt_snapshot

    click.echo(f"[batch] loading parameters from {params} …")
    t0 = time.perf_counter()
    ckpt_model_cfg = None
    if params.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        from deepfold3.model.config import model_cfg_from_metadata

        with safe_open(str(params), framework="pt") as f:
            ckpt_model_cfg = model_cfg_from_metadata(f.metadata())
        state_dict = load_file(str(params))
    elif params.suffix == ".pt":
        state_dict, ckpt_model_cfg = load_pt_snapshot(params, snapshot)
    else:
        import warnings

        from deepfold3.convert.af3 import haiku_to_torch
        from deepfold3.params import read_af3_params

        warnings.warn(
            "Loading raw AF3 params — Fourier weights will be missing.",
            stacklevel=1,
        )
        state_dict = haiku_to_torch(read_af3_params(params))

    model_cfg = ckpt_model_cfg if ckpt_model_cfg is not None else ModelConfig()
    if ckpt_model_cfg is None and params.suffix == ".pt":
        click.echo("[batch] checkpoint has no embedded model_cfg — using ModelConfig() defaults")

    if optim == "all":
        model_cfg.optim = OptimConfig.all_enabled()
    elif optim == "recommended":
        model_cfg.optim = OptimConfig.recommended()
    elif optim == "inference":
        model_cfg.optim = OptimConfig.inference()
    elif optim == "triton":
        model_cfg.optim = OptimConfig.triton_only()

    if optim and optim != "none":
        torch.set_float32_matmul_precision("high")
        click.echo(f"[batch] optimizations: {optim}")

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if not any(p in k for p in OptimConfig.FUSED_STATE_DICT_MARKERS)]
    if real_missing:
        click.echo(f"  [WARN] missing keys   : {len(real_missing)}")
    if unexpected:
        click.echo(f"  [WARN] unexpected keys: {len(unexpected)}")
    model.to(dev).eval()
    click.echo(f"  loaded in {time.perf_counter() - t0:.1f}s")
    return model, model_cfg


def _run_one_job(
    model,
    model_cfg,
    dev: torch.device,
    json_input: Path,
    output_dir: Path,
    *,
    num_recycles: int,
    num_samples: int,
    seeds: list[int],
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    plot_confidences: bool,
    save_distogram: bool = False,
    logits_scheme: str = "int8_softmax",
    skip_existing: bool,
    bucket: bool = False,
) -> None:
    """Featurize + forward + save for a single JSON job. Mirrors the JSON
    branch of ``inference``.

    ``seeds`` is the CLI-provided seed list; when empty each JSON's own
    ``modelSeeds`` are used. Every seed is folded independently and written to
    its own ``seed-{seed}_sample-{s}`` subdir, so a job with S seeds and
    ``num_samples`` samples yields S x num_samples structures. When
    ``skip_existing`` is set, any seed whose full set of sample CIFs already
    exists is skipped (per-seed resume)."""
    from deepfold3.data.module.dataset import InferenceInput
    from deepfold3.data.parse.json_input import parse_json_input
    from deepfold3.io.confidences import (
        save_confidences_json,
        save_distogram_logits,
        save_ranking_scores_csv,
        save_summary_confidences_json,
    )
    from deepfold3.io.mmcif import save_mmcif

    click.echo(f"[batch] featurizing {json_input} …")
    t0 = time.perf_counter()
    jobs = parse_json_input(json_input, template_cache_dir=template_cache, pdb_mmcif_dir=pdb_mmcif_dir)
    if not jobs:
        click.echo(f"[batch] skipping {json_input}: no jobs in JSON")
        return
    job = jobs[0]
    job_name = job.name
    job_title = job.description or job.name

    chain_idx = 0
    desc_map: dict[str, str] = {}
    for chain_list in [job.protein_chains, job.dna_chains, job.rna_chains, job.ligand_chains]:
        for chain in chain_list:
            chain_name = chr(ord("A") + chain_idx)
            if chain.description:
                desc_map[chain_name] = chain.description
            chain_idx += 1
    entity_descriptions = desc_map or None

    feats = InferenceInput.from_job(job)
    N = feats["res_type"].shape[0]
    S = feats["msa"].shape[0]
    T = feats.get("template_aatype", torch.zeros(0)).shape[0]
    click.echo(f"  {N} tokens, {S} MSA sequences, {T} templates  ({time.perf_counter() - t0:.1f}s)")

    original_n = N
    # Bucket-pad N up to the next BUCKET_SIZES entry. torch.compile *requires*
    # this (static shapes), but the fused Triton kernels benefit just as much:
    # they autotune on M = batch·N², so every distinct token count is a fresh
    # ~60 s autotune. Quantizing N to ~13 buckets collapses hundreds of shapes
    # to a handful, so a sweep autotunes each bucket once (and warmup-triton can
    # pre-warm them) instead of recompiling per target. Enabled by --bucket or
    # whenever torch.compile is on.
    use_bucket = bucket or model_cfg.optim.use_torch_compile
    if use_bucket:
        from deepfold3.model.bucket import next_bucket, pad_features

        bucket_n = next_bucket(N)
        if bucket_n > N:
            feats, original_n = pad_features(feats, bucket_n)
            click.echo(f"  bucket padding: {N} → {bucket_n} tokens")

    feats = {k: v.to(dev) for k, v in feats.items()}

    json_seeds = list(job.model_seeds) if job.model_seeds else [42]
    # CLI seeds (when provided) are authoritative — replace the JSON's own
    # seeds so a benchmark sweep folds exactly the requested set. Dedup while
    # preserving order.
    seed_list = list(dict.fromkeys(seeds)) if seeds else json_seeds

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for current_seed in seed_list:
        if skip_existing and all(
            (output_dir / f"seed-{int(current_seed)}_sample-{int(s)}" / "model.cif").exists()
            for s in range(num_samples)
        ):
            click.echo(f"[batch]   seed {current_seed}: all {num_samples} samples present — skip")
            continue
        click.echo(f"[batch] {json_input.name}  seed={current_seed} samples={num_samples} recycles={num_recycles}")
        t0 = time.perf_counter()
        with torch.inference_mode():
            model_output = model(feats, seed=current_seed, num_recycles=num_recycles, num_samples=num_samples)

        iter_feats = feats
        if use_bucket and original_n < feats["res_type"].shape[0]:
            from deepfold3.model.bucket import unpad_features, unpad_output

            model_output = unpad_output(model_output, original_n)
            iter_feats = unpad_features(feats, original_n)
        click.echo(f"  done in {time.perf_counter() - t0:.1f}s")

        # Trunk distogram is shared across this seed's samples (no sample axis),
        # so write it once per seed rather than into every sample dir.
        if save_distogram:
            distogram_npz = output_dir / f"seed-{int(current_seed)}_distogram.npz"
            if save_distogram_logits(distogram_npz, model_output, scheme=logits_scheme, seed=int(current_seed)):
                click.echo(f"  saved  {distogram_npz.name}")

        for s in range(num_samples):
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
            summary = save_confidences_json(conf_path, model_output, iter_feats, sample_idx=s, seed=int(current_seed))
            summary = {**summary, "seed": int(current_seed), "sample": int(s)}
            summaries.append(summary)
            if plot_confidences:
                from deepfold3.scripts.plot_confidences import plot_confidences_file

                plot_confidences_file(conf_path, conf_path.with_suffix(".png"))

    if summaries:
        save_ranking_scores_csv(output_dir / "ranking_scores.csv", summaries)
        summary_name = f"{job_name}_summary_confidences.json" if job_name else "summary_confidences.json"
        save_summary_confidences_json(output_dir / summary_name, summaries, job_name=job_name)


def _parse_inputs_file(path: Path, output_template: str, params: Path) -> list[tuple[Path, Path]]:
    """Read a TSV of ``json_path [TAB output_dir]`` lines. Returns
    ``[(json_path, output_dir), ...]``."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        json_path = Path(parts[0])
        if len(parts) == 2:
            output_dir = Path(parts[1])
        else:
            output_dir = Path(
                output_template.format(
                    json_dir=str(json_path.parent),
                    json_stem=json_path.stem,
                    json_name=json_path.name,
                    params_stem=params.stem,
                )
            )
        out.append((json_path, output_dir))
    return out


@cli.command("inference-batch")
@click.option(
    "--inputs-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file: 'json_path [TAB output_dir]' per line, '#' comments allowed",
)
@click.option(
    "--json",
    "json_paths",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Repeatable JSON input path. Output dir from --output-template.",
)
@click.option(
    "--output-template",
    default="{json_dir}/inference",
    help="Output dir template when only a JSON path is provided. "
    "Variables: {json_dir}, {json_stem}, {json_name}, {params_stem}.",
)
@click.option(
    "--params",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model parameters (.safetensors / .pt / .bin / .bin.zst)",
)
@click.option("--num-recycles", default=10, show_default=True)
@click.option("--num-samples", default=5, show_default=True)
@click.option("--snapshot", type=click.Choice(["auto", "shadow", "average", "fast"]), default="auto", show_default=True)
@click.option(
    "--seed",
    "seeds",
    multiple=True,
    type=int,
    help="Repeatable model seed (e.g. --seed 42 --seed 43). Each input is folded "
    "once per seed into its own seed-*_sample-* subdir. When given, replaces "
    "each JSON's modelSeeds; defaults to the JSON's own seeds.",
)
@click.option("--device", default=None)
@click.option("--template-cache", default=None, type=click.Path(path_type=Path))
@click.option("--pdb-mmcif-dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--optim",
    default="inference",
    type=click.Choice(["none", "recommended", "inference", "all", "triton"]),
    show_default=True,
)
@click.option("--plot-confidences", is_flag=True, default=False)
@click.option(
    "--save-distogram/--no-save-distogram",
    default=False,
    show_default=True,
    help="Persist the per-seed trunk distogram (seed-*_distogram.npz, shared "
    "across that seed's samples). Cheap ([N,N,64], one per seed); needed for "
    "distogram post-analysis (e.g. cross-chain interface-contact prediction).",
)
@click.option(
    "--logits-scheme",
    default="int8_softmax",
    type=click.Choice(["float32", "float16", "int8", "int8_softmax", "int4_softmax"]),
    show_default=True,
    help="Compression scheme for --save-distogram npz. int8_softmax preserves "
    "softmax(logits) within ~1e-4 at ~1 byte/value.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip jobs whose output dir already has any seed-* subdir.",
)
@click.option(
    "--bucket/--no-bucket",
    default=True,
    show_default=True,
    help="Pad token count N up to the next bucket size before the forward. The "
    "fused Triton kernels autotune on M=batch*N^2, so bucketing collapses "
    "hundreds of distinct N to ~13 shapes — each autotuned once (and "
    "pre-warmable via warmup-triton) instead of a ~60s re-autotune per target. "
    "Recommended for multi-target sweeps; auto-on when torch.compile is used.",
)
def inference_batch(
    inputs_file: Path | None,
    json_paths: tuple[Path, ...],
    output_template: str,
    params: Path,
    num_recycles: int,
    num_samples: int,
    snapshot: str,
    seeds: tuple[int, ...],
    device: str | None,
    template_cache: Path | None,
    pdb_mmcif_dir: Path | None,
    optim: str,
    plot_confidences: bool,
    save_distogram: bool,
    logits_scheme: str,
    skip_existing: bool,
    bucket: bool,
) -> None:
    """Run inference on many JSON inputs without reloading the model."""
    if not inputs_file and not json_paths:
        raise click.UsageError("Provide --inputs-file or one or more --json")

    jobs: list[tuple[Path, Path]] = []
    if inputs_file:
        jobs += _parse_inputs_file(inputs_file, output_template, params)
    for jp in json_paths:
        out_dir = Path(
            output_template.format(
                json_dir=str(jp.parent),
                json_stem=jp.stem,
                json_name=jp.name,
                params_stem=params.stem,
            )
        )
        jobs.append((jp, out_dir))

    click.echo(f"[batch] {len(jobs)} jobs queued")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    click.echo(f"[batch] device: {dev}")

    model, model_cfg = _load_model_once(params, snapshot, dev, optim)

    seed_list = list(dict.fromkeys(seeds))

    def _job_complete(od: Path) -> bool:
        """A job is complete when every requested (seed, sample) CIF exists.

        With CLI seeds we know the exact set; without them we can't cheaply
        know the JSON's own seeds, so fall back to the any-seed-dir heuristic.
        """
        if not od.exists():
            return False
        if seed_list:
            return all(
                (od / f"seed-{int(s)}_sample-{int(k)}" / "model.cif").exists()
                for s in seed_list
                for k in range(num_samples)
            )
        return any(p.name.startswith("seed-") for p in od.iterdir() if p.is_dir())

    n_done = 0
    n_skipped = 0
    t_total = time.perf_counter()
    for idx, (jp, od) in enumerate(jobs, 1):
        if skip_existing and _job_complete(od):
            click.echo(f"[batch] [{idx}/{len(jobs)}] SKIP {jp.name} → {od} (all seeds/samples present)")
            n_skipped += 1
            continue
        if not jp.exists():
            click.echo(f"[batch] [{idx}/{len(jobs)}] MISS {jp} — not found")
            continue
        click.echo(f"[batch] [{idx}/{len(jobs)}] {jp.name} → {od}")
        _run_one_job(
            model,
            model_cfg,
            dev,
            jp,
            od,
            num_recycles=num_recycles,
            num_samples=num_samples,
            seeds=seed_list,
            template_cache=template_cache,
            pdb_mmcif_dir=pdb_mmcif_dir,
            plot_confidences=plot_confidences,
            save_distogram=save_distogram,
            logits_scheme=logits_scheme,
            skip_existing=skip_existing,
            bucket=bucket,
        )
        n_done += 1
        # Release this job's cached blocks back to the driver before the next
        # target. Bucketing folds varied N (=> varied allocation sizes), and
        # per-shape Triton autotune allocates scratch, so cached blocks
        # fragment across a long shard and intermittently OOM the next (large)
        # target. The job's tensors are already freed by refcounting on return
        # (inference_mode => no autograd graph / cycles), so empty_cache alone
        # suffices — no gc.collect needed.
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_total
    click.echo(f"[batch] finished: {n_done} done, {n_skipped} skipped  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    inference_batch()
