"""``deepfold3 train`` -- training entry point.

Examples:
    Single-GPU (data_dir from config)::

        deepfold3 train --config configs/initial.yaml --output output/run1

    Single-GPU (explicit data-dir overrides config)::

        deepfold3 train --data-dir /data/training --output output/run1

    Multi-GPU (DDP via torchrun)::

        torchrun --nproc_per_node=4 -m deepfold3.scripts.train \\
            --config configs/initial.yaml --output output/run1

    Resume::

        deepfold3 train --output output/run1 \\
            --resume output/run1/checkpoint_1000.pt
"""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

from deepfold3.scripts import cli

logger = logging.getLogger(__name__)

# Dump Python tracebacks for all threads on SIGUSR1 — invaluable for
# diagnosing DDP deadlocks where one rank blocks in an NCCL collective.
faulthandler.enable()
if hasattr(faulthandler, "register"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=True)


def _git_info() -> dict:
    """Return ``{commit, branch, dirty}`` from the current git checkout, best-effort."""
    repo_dir = Path(__file__).resolve().parent

    def _run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=2,
                check=False,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    info: dict = {}
    commit = _run(["rev-parse", "HEAD"])
    if commit:
        info["commit"] = commit
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        info["branch"] = branch
    status = _run(["status", "--porcelain"])
    if status is not None:
        info["dirty"] = bool(status)
    return info


def _build_provenance(
    *,
    config_path: str | None,
    cli_argv: list[str],
    overrides: list[str],
    cli_datasets: list[str],
) -> dict:
    """Best-effort run-provenance dict: argv, config path, git, host, env."""
    import torch

    prov: dict = {
        "start_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "cli_argv": list(cli_argv),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "overrides": list(overrides),
        "cli_datasets": list(cli_datasets),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": {"version": str(torch.__version__), "cuda": str(torch.version.cuda) if torch.version.cuda else None},
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    git = _git_info()
    if git:
        prov["git"] = git
    return prov


def _apply_overrides(cfg: object, overrides: dict) -> None:
    """Recursively apply dict overrides to a nested dataclass config.

    Handles arbitrarily nested dataclass hierarchies: if a value is a dict
    and the corresponding attribute is a dataclass, recurse into it.
    """
    for key, val in overrides.items():
        if key.startswith("_"):
            continue  # skip metadata keys like _description, _usage
        if not hasattr(cfg, key):
            continue
        cur = getattr(cfg, key)
        if isinstance(val, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply_overrides(cur, val)
        else:
            setattr(cfg, key, val)


@cli.command()
@click.option(
    "--data-dir", default=None, type=click.Path(exists=True), help="Primary training data directory (overrides config)."
)
@click.option("--output", required=True, type=click.Path(), help="Output directory for checkpoints and logs.")
@click.option("--params", default="", help="Pretrained weights (.safetensors or .pt).")
@click.option("--resume", default=None, type=click.Path(exists=True), help="Resume from checkpoint.")
@click.option(
    "--config", "config_path", default=None, type=click.Path(exists=True), help="Training config (JSON or YAML)."
)
@click.option("--lr", default=None, type=float, help="Override learning rate.")
@click.option("--max-steps", default=None, type=int, help="Override max training steps.")
@click.option("--max-tokens", default=None, type=int, help="Override max tokens per crop.")
@click.option("--grad-accum", default=None, type=int, help="Override gradient accumulation steps.")
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option(
    "--deterministic",
    is_flag=True,
    default=False,
    help="Force deterministic algorithms (torch.use_deterministic_algorithms + cudnn.deterministic). "
    "Reproducible but slower; sets CUBLAS_WORKSPACE_CONFIG if unset.",
)
@click.option("--dtype", default=None, type=click.Choice(["bf16", "fp32", "fp16"]), help="Override precision.")
@click.option("--wandb", "wandb_project", default=None, help="WandB project name (enables tracking).")
@click.option("--wandb-entity", default=None, help="WandB team/org entity (default: personal workspace).")
@click.option("--override", multiple=True, help="Override config: key=value (e.g. --override alpha_pae=1.0).")
@click.option(
    "--ccd",
    default=None,
    type=click.Path(exists=True),
    help="Path to components.cif(.gz) for bulk CCD lookup.",
)
@click.option(
    "--include-list",
    default=None,
    type=click.Path(exists=True),
    help="Text file of PDB IDs to include (one per line). Only these records are used.",
)
@click.option(
    "--exclude-list",
    default=None,
    type=click.Path(exists=True),
    help="Text file of PDB IDs to exclude (one per line). These records are skipped.",
)
@click.option(
    "--dataset",
    "extra_datasets",
    multiple=True,
    help=(
        "Add an extra training dataset.  Format: name:path[:weight[:strategy[:distill[:format]]]].  "
        "format is 'df3' (default) or 'of3' (OpenFold 3 preprocessed, loaded on-the-fly).  "
        "Examples: --dataset afdb:/data/afdb:0.5:uniform:true  "
        "--dataset of3_mono:/data/of3/preprocessed:0.5:uniform:true:of3  "
        "--dataset pdb2:/data/pdb2:1.0:weighted.  Can be repeated."
    ),
)
@click.option(
    "--eval-cutoff-date",
    default=None,
    help="ISO date (YYYY-MM-DD): records released after this go to eval set (AF3-style).",
)
@click.option(
    "--mmcif-dir",
    default=None,
    type=click.Path(exists=True),
    help="PDB mmCIF mirror directory. Enables template loading during training.",
)
@click.option(
    "--ccd-cache",
    "ccd_cache_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to unified CCD cache JSON (atoms + ref_pos + automorphisms). "
    "Enables ligand atom-permutation loss. Build with `deepfold3 build-ccd-cache --automorphisms`.",
)
def train(
    data_dir: str | None,
    output: str,
    params: str,
    resume: str | None,
    config_path: str | None,
    lr: float | None,
    max_steps: int | None,
    max_tokens: int | None,
    grad_accum: int | None,
    seed: int | None,
    deterministic: bool,
    dtype: str | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    override: tuple[str, ...],
    ccd: str | None,
    include_list: str | None,
    exclude_list: str | None,
    extra_datasets: tuple[str, ...],
    eval_cutoff_date: str | None,
    mmcif_dir: str | None,
    ccd_cache_path: str | None,
) -> None:
    """Train DeepFold3 model."""
    from deepfold3.config_loader import load_config
    from deepfold3.model.config import DatasetConfig, ModelConfig, TrainingConfig
    from deepfold3.training.trainer import Trainer

    rank = int(os.environ.get("RANK", "0"))
    run_dir = Path(output)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    output = str(run_dir)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        log_file = run_dir / "train.log"
        handlers.append(logging.FileHandler(log_file))

    # INFO on every rank: deadlock diagnosis requires seeing where each rank
    # was last. Per-rank file separation is done by torchrun via
    # ``--redirects 3 --tee 3`` (writes ``<run_dir>/attempt_*/<rank>/stdout.log``);
    # no Python-side file-handler juggling needed.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [rank{rank}] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # Load bulk CCD if provided. Force the gemmi parse here (not on first
    # lookup) so the ~110 MB document is populated BEFORE DataLoader workers
    # and/or per-rank processes fork. Otherwise each worker re-parses the
    # file independently (32+ copies on a typical 8-GPU node, ~6 s each +
    # ~110 MB RAM per copy).
    if ccd:
        from deepfold3.data.ccd import fetch_ccd_block, set_ccd_components_path

        set_ccd_components_path(ccd)
        fetch_ccd_block("HOH")  # eager parse; any code triggers the lazy load

    # Build config from defaults + JSON override + CLI overrides
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if config_path:
        overrides = load_config(config_path)

        # Apply training config overrides
        for key, val in overrides.get("training", {}).items():
            if key == "extra_datasets":
                train_cfg.extra_datasets = [DatasetConfig(**d) for d in val]
            elif key == "eval_datasets":
                train_cfg.eval_datasets = [DatasetConfig(**d) for d in val]
            elif hasattr(train_cfg, key):
                setattr(train_cfg, key, val)

        # Apply model config overrides (recursively for nested dataclasses)
        _apply_overrides(model_cfg, overrides.get("model", {}))

    # Apply CLI overrides — data_dir from CLI takes precedence over config
    if data_dir is not None:
        train_cfg.data_dir = data_dir
    train_cfg.output_dir = output
    train_cfg.params_path = params
    if lr is not None:
        train_cfg.lr = lr
    if max_steps is not None:
        train_cfg.max_steps = max_steps
    if max_tokens is not None:
        train_cfg.max_tokens = max_tokens
    if grad_accum is not None:
        train_cfg.accumulate_grad_batches = grad_accum
    if seed is not None:
        if resume is not None and seed != train_cfg.seed:
            logger.warning(
                "--seed=%s overrides config seed (%s) on a --resume run; sampler RNG "
                "will diverge from the original run's data ordering at the resume point.",
                seed,
                train_cfg.seed,
            )
        train_cfg.seed = seed
    if deterministic:
        train_cfg.deterministic = True
    if dtype is not None:
        train_cfg.dtype = dtype
    if wandb_project is not None:
        train_cfg.wandb_project = wandb_project
        train_cfg.wandb_enabled = True
    if wandb_entity is not None:
        train_cfg.wandb_entity = wandb_entity
    if mmcif_dir is not None:
        train_cfg.mmcif_dir = mmcif_dir
    if ccd_cache_path is not None:
        train_cfg.ccd_cache_path = ccd_cache_path
    if eval_cutoff_date is not None:
        train_cfg.eval_cutoff_date = eval_cutoff_date

    # Apply --override key=value pairs (supports int, float, bool, str)
    for kv in override:
        if "=" not in kv:
            raise click.BadParameter(f"Override must be key=value, got: {kv}")
        key, val = kv.split("=", 1)
        if not hasattr(train_cfg, key):
            raise click.BadParameter(f"Unknown config key: {key}")
        cur = getattr(train_cfg, key)
        if isinstance(cur, bool):
            setattr(train_cfg, key, val.lower() in ("true", "1", "yes"))
        elif isinstance(cur, int):
            setattr(train_cfg, key, int(val))
        elif isinstance(cur, float):
            setattr(train_cfg, key, float(val))
        else:
            setattr(train_cfg, key, val)

    # Load records from manifest or by scanning data_dir
    from deepfold3.data.prepare.manifest import discover_records

    log = logging.getLogger(__name__)
    effective_data_dir = train_cfg.data_dir
    all_records = []
    if effective_data_dir:
        all_records = discover_records(Path(effective_data_dir))
        if not all_records:
            log.warning("No records found in %s", effective_data_dir)
    else:
        log.info("No primary data_dir specified; training will use extra_datasets only")

    # Filter by include/exclude lists
    if include_list or exclude_list:
        from deepfold3.data.prepare.manifest import load_id_list

        if include_list:
            allowed = load_id_list(Path(include_list))
            before = len(all_records)
            all_records = [r for r in all_records if r.id in allowed]
            log.info("Include list: %d → %d records", before, len(all_records))
        if exclude_list:
            denied = load_id_list(Path(exclude_list))
            before = len(all_records)
            all_records = [r for r in all_records if r.id not in denied]
            log.info("Exclude list: %d → %d records", before, len(all_records))

    # Drop giant assemblies by total chain count — BEFORE the dataset is built,
    # so they are never sampled or loaded (bounds per-sample featurisation cost
    # and the periodic dataloader stalls they cause). Operates on Record
    # metadata only; no structure/MSA I/O. ``model_cfg.data.max_total_chains``
    # of 0 disables the filter.
    max_total_chains = model_cfg.data.max_total_chains
    if max_total_chains and all_records:
        before = len(all_records)
        all_records = [r for r in all_records if len(r.chains) <= max_total_chains]
        log.info(
            "max_total_chains=%d: %d → %d records (dropped %d giant assemblies)",
            max_total_chains, before, len(all_records), before - len(all_records),
        )

    # Train / eval split — two modes by priority:
    #   1. eval_cutoff_date: records released AFTER cutoff → eval set (AF3-style)
    #   2. eval_datasets from config (separate dataset dirs)
    # If neither is configured, no eval is performed on the primary dataset.
    eval_records: list = []

    if train_cfg.eval_cutoff_date:
        # Mode 1: date-based split on primary records (AF3 §5.8)
        cutoff = train_cfg.eval_cutoff_date
        max_eval = train_cfg.eval_max_date
        records, eval_records = [], []
        for r in all_records:
            released = r.structure_info.released if r.structure_info else None
            if released and released > cutoff:
                if not max_eval or released <= max_eval:
                    eval_records.append(r)
                # Records after max_eval are excluded entirely
            else:
                records.append(r)
        log.info(
            "Date-based split (cutoff=%s, max=%s): %d train, %d eval",
            cutoff,
            max_eval or "none",
            len(records),
            len(eval_records),
        )
    else:
        records = all_records

    # Load distillation records if configured (legacy)
    distillation_records = []
    if train_cfg.distillation_data_dir:
        distillation_records = discover_records(Path(train_cfg.distillation_data_dir))
        log.info("Loaded %d distillation records from %s", len(distillation_records), train_cfg.distillation_data_dir)

    # Parse --dataset CLI flags into extra_datasets config
    existing_ds = {d.name: i for i, d in enumerate(train_cfg.extra_datasets)}
    for ds_spec in extra_datasets:
        parts = ds_spec.split(":")
        if len(parts) < 2:
            raise click.BadParameter(
                "--dataset format: name:path[:weight[:strategy[:distill[:format[:tpl_struct_space]]]]], got: " + ds_spec
            )
        entry = DatasetConfig(
            name=parts[0],
            data_dir=parts[1],
            weight=float(parts[2]) if len(parts) > 2 else 1.0,
            strategy=parts[3] if len(parts) > 3 else "uniform",
            is_distillation=parts[4].lower() in ("true", "1", "yes") if len(parts) > 4 else False,
            format=parts[5] if len(parts) > 5 else "df3",
            monomer_template_struct_space=parts[6].lower() in ("true", "1", "yes") if len(parts) > 6 else False,
        )
        # CLI dataset with same name replaces config-defined one
        if entry.name in existing_ds:
            train_cfg.extra_datasets[existing_ds[entry.name]] = entry
        else:
            existing_ds[entry.name] = len(train_cfg.extra_datasets)
            train_cfg.extra_datasets.append(entry)
        log.info(
            "CLI dataset: %s (dir=%s, weight=%.2f, strategy=%s, distill=%s)",
            entry.name,
            entry.data_dir,
            entry.weight,
            entry.strategy,
            entry.is_distillation,
        )

    provenance = _build_provenance(
        config_path=config_path,
        cli_argv=sys.argv,
        overrides=list(override),
        cli_datasets=list(extra_datasets),
    )

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        records=records,
        eval_records=eval_records,
        distillation_records=distillation_records,
        provenance=provenance,
    )
    try:
        trainer.run(resume_from=resume)
    except BaseException:
        # Log with rank tag before re-raising: peers will hang in NCCL until
        # torchrun reaps them, so the first-failing rank's traceback is the
        # only diagnostic signal pointing at root cause.
        log.exception("rank %d: trainer.run raised; peers may hang until torchrun reaps them", rank)
        raise


# Allow running as module for torchrun: python -m deepfold3.scripts.train
if __name__ == "__main__":
    train()
