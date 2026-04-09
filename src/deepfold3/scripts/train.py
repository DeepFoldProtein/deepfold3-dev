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

import json
import logging
import sys
from pathlib import Path

import click

from deepfold3.scripts import cli


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
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="Training config JSON.")
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
        "Add an extra training dataset.  Format: name:path[:weight[:strategy[:distill]]].  "
        "Examples: --dataset afdb:/data/afdb:0.5:uniform:true  "
        "--dataset pdb2:/data/pdb2:1.0:weighted.  Can be repeated."
    ),
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
) -> None:
    """Train DeepFold3 model."""
    from deepfold3.model.config import ModelConfig, TrainingConfig
    from deepfold3.training.trainer import Trainer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load bulk CCD if provided
    if ccd:
        from deepfold3.data.ccd import set_ccd_components_path

        set_ccd_components_path(ccd)

    # Build config from defaults + JSON override + CLI overrides
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)

        # Apply training config overrides
        for key, val in overrides.get("training", {}).items():
            if hasattr(train_cfg, key):
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

    # Shuffle with fixed seed for reproducible train/eval split
    import random

    rng = random.Random(42)
    rng.shuffle(all_records)

    # Split: last 5% for eval (at least 1 if available)
    n_eval = max(1, len(all_records) // 20)
    records = all_records[:-n_eval] if len(all_records) > n_eval else all_records
    eval_records = all_records[-n_eval:] if len(all_records) > n_eval else []

    # Load distillation records if configured (legacy)
    distillation_records = []
    if train_cfg.distillation_data_dir:
        distillation_records = discover_records(Path(train_cfg.distillation_data_dir))
        log.info("Loaded %d distillation records from %s", len(distillation_records), train_cfg.distillation_data_dir)

    # Parse --dataset CLI flags into extra_datasets config
    for ds_spec in extra_datasets:
        parts = ds_spec.split(":")
        if len(parts) < 2:
            raise click.BadParameter(f"--dataset format: name:path[:weight[:strategy[:distill]]], got: {ds_spec}")
        entry = {
            "name": parts[0],
            "data_dir": parts[1],
            "weight": float(parts[2]) if len(parts) > 2 else 1.0,
            "strategy": parts[3] if len(parts) > 3 else "uniform",
            "is_distillation": parts[4].lower() in ("true", "1", "yes") if len(parts) > 4 else False,
        }
        train_cfg.extra_datasets.append(entry)
        log.info(
            "CLI dataset: %s (dir=%s, weight=%.2f, strategy=%s, distill=%s)",
            entry["name"],
            entry["data_dir"],
            entry["weight"],
            entry["strategy"],
            entry["is_distillation"],
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
