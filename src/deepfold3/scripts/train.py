"""``deepfold3 train`` -- training entry point.

Examples:
    Single-GPU::

        deepfold3 train --data-dir /data/training --output output/run1

    Multi-GPU (DDP via torchrun)::

        torchrun --nproc_per_node=4 -m deepfold3.scripts.train \\
            --data-dir /data/training --output output/run1

    Resume::

        deepfold3 train --data-dir /data/training --output output/run1 \\
            --resume output/run1/checkpoint_1000.pt
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import click

from deepfold3.scripts import cli

# Dump Python tracebacks for all threads on SIGUSR1 — invaluable for
# diagnosing DDP deadlocks where one rank blocks in an NCCL collective.
faulthandler.enable()
if hasattr(faulthandler, "register"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=True)


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
@click.option("--data-dir", required=True, type=click.Path(exists=True), help="Training data directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for checkpoints and logs.")
@click.option("--params", default="", help="Pretrained weights (.safetensors or .pt).")
@click.option("--resume", default=None, type=click.Path(exists=True), help="Resume from checkpoint.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="Training config (JSON or YAML).")
@click.option("--lr", default=None, type=float, help="Override learning rate.")
@click.option("--max-steps", default=None, type=int, help="Override max training steps.")
@click.option("--max-tokens", default=None, type=int, help="Override max tokens per crop.")
@click.option("--grad-accum", default=None, type=int, help="Override gradient accumulation steps.")
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option("--dtype", default=None, type=click.Choice(["bf16", "fp32", "fp16"]), help="Override precision.")
@click.option("--wandb", "wandb_project", default=None, help="WandB project name (enables tracking).")
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
    "--eval-data-dir",
    default=None,
    type=click.Path(exists=True),
    help="Separate evaluation data directory (overrides date-based or random split).",
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
def train(
    data_dir: str,
    output: str,
    params: str,
    resume: str | None,
    config_path: str | None,
    lr: float | None,
    max_steps: int | None,
    max_tokens: int | None,
    grad_accum: int | None,
    seed: int | None,
    dtype: str | None,
    wandb_project: str | None,
    override: tuple[str, ...],
    ccd: str | None,
    include_list: str | None,
    exclude_list: str | None,
    extra_datasets: tuple[str, ...],
    eval_data_dir: str | None,
    eval_cutoff_date: str | None,
    mmcif_dir: str | None,
) -> None:
    """Train DeepFold3 model."""
    from deepfold3.config_loader import load_config
    from deepfold3.model.config import DatasetConfig, ModelConfig, TrainingConfig
    from deepfold3.training.trainer import Trainer

    # Auto-create a per-run subdirectory under the user-supplied output path.
    # Layout:  runs/<name>/<YYYYMMDD_HHMMSS>/{train.log, checkpoint_*.pt, ...}
    # - Rank 0 creates the subdir and writes INFO to stdout + train.log.
    # - Other ranks only emit WARNING/ERROR to stdout (prevents 8× duplication on DDP).
    # - When resuming, we respect the existing --output if it already looks like a
    #   timestamped run dir (contains a checkpoint or train.log).
    rank = int(os.environ.get("RANK", "0"))
    requested_output = Path(output)
    timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")

    is_existing_run = requested_output.is_dir() and any(
        p.name == "train.log" or p.name.startswith("checkpoint_") for p in requested_output.iterdir()
    )
    if resume is not None or is_existing_run:
        run_dir = requested_output
    else:
        run_dir = requested_output / timestamp

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Best-effort broadcast via filesystem: other ranks read the same path.
    # Update ``output`` so the Trainer checkpoints into the run dir.
    output = str(run_dir)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        log_file = run_dir / "train.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [%(levelname)s] [rank{rank}] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # Load bulk CCD if provided
    if ccd:
        from deepfold3.data.ccd import set_ccd_components_path

        set_ccd_components_path(ccd)

    # Build config from defaults + JSON override + CLI overrides
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if config_path:
        overrides = load_config(config_path)

        # Apply training config overrides
        for key, val in overrides.get("training", {}).items():
            if key == "extra_datasets":
                # Convert dicts to DatasetConfig objects
                train_cfg.extra_datasets = [DatasetConfig(**d) for d in val]
            elif hasattr(train_cfg, key):
                setattr(train_cfg, key, val)

        # Apply model config overrides (recursively for nested dataclasses)
        _apply_overrides(model_cfg, overrides.get("model", {}))

    # Apply CLI overrides
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
        train_cfg.seed = seed
    if dtype is not None:
        train_cfg.dtype = dtype
    if wandb_project is not None:
        train_cfg.wandb_project = wandb_project
        train_cfg.wandb_enabled = True
    if mmcif_dir is not None:
        train_cfg.mmcif_dir = mmcif_dir
    if eval_data_dir is not None:
        train_cfg.eval_data_dir = eval_data_dir
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

    all_records = discover_records(Path(data_dir))
    log = logging.getLogger(__name__)
    if not all_records:
        log.warning("No records found in %s", data_dir)

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

    # Train / eval split — three modes by priority:
    #   1. --eval-data-dir: separate dataset directory for eval
    #   2. eval_cutoff_date: records released AFTER cutoff → eval set (AF3-style)
    #   3. fallback: random split with fixed seed
    import random

    eval_records: list = []

    if train_cfg.eval_data_dir:
        # Mode 1: dedicated eval dataset
        eval_records = discover_records(Path(train_cfg.eval_data_dir))
        records = all_records
        log.info("Eval from separate dir: %d eval records from %s", len(eval_records), train_cfg.eval_data_dir)
    elif train_cfg.eval_cutoff_date:
        # Mode 2: date-based split (AF3 §5.8)
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
        # Mode 3: random split
        split_seed = train_cfg.seed
        rng = random.Random(split_seed)
        rng.shuffle(all_records)
        n_eval = max(1, int(len(all_records) * train_cfg.eval_fraction))
        records = all_records[:-n_eval] if len(all_records) > n_eval else all_records
        eval_records = all_records[-n_eval:] if len(all_records) > n_eval else []
        log.info(
            "Random split (seed=%d, frac=%.2f): %d train, %d eval",
            split_seed,
            train_cfg.eval_fraction,
            len(records),
            len(eval_records),
        )

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
                "--dataset format: name:path[:weight[:strategy[:distill[:format]]]], got: " + ds_spec
            )
        entry = DatasetConfig(
            name=parts[0],
            data_dir=parts[1],
            weight=float(parts[2]) if len(parts) > 2 else 1.0,
            strategy=parts[3] if len(parts) > 3 else "uniform",
            is_distillation=parts[4].lower() in ("true", "1", "yes") if len(parts) > 4 else False,
            format=parts[5] if len(parts) > 5 else "df3",
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

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        records=records,
        eval_records=eval_records,
        distillation_records=distillation_records,
    )
    trainer.run(resume_from=resume)


# Allow running as module for torchrun: python -m deepfold3.scripts.train
if __name__ == "__main__":
    train()
