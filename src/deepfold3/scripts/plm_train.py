"""``deepfold3 plm-train`` — train the complex language model.

Single node::

    torchrun --nproc-per-node=8 -m deepfold3.scripts.plm_train \\
        --config configs/plm-base.yaml --output runs/plm-r1

Overrides use dotted paths and take the config's own types::

    --override model.layer_pattern=ccg --override data.max_chain_len=2048
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from deepfold3.scripts import cli
from deepfold3.scripts._common import setup_cli_logging

logger = logging.getLogger(__name__)


@cli.command("plm-train")
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Experiment YAML/JSON.")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--resume", type=click.Path(exists=True, path_type=Path))
@click.option("--override", "overrides", multiple=True, metavar="KEY=VALUE", help="Dotted config override.")
@click.option("--esmc-weights", type=click.Path(exists=True), help="Warm start from an ESMC release.")
@click.option("--wandb", "wandb_project", help="WandB project (shorthand for train.wandb_project).")
@click.option("--wandb-entity")
@click.option("--dry-run", is_flag=True, help="Build everything, run a few steps, and stop.")
def plm_train(
    config: Path | None,
    output: Path,
    resume: Path | None,
    overrides: tuple[str, ...],
    esmc_weights: str | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    dry_run: bool,
) -> None:
    """Train the chain-RoPE / complex-NoPE masked language model."""
    setup_cli_logging(__name__)
    from deepfold3.complexlm.config import load_experiment
    from deepfold3.complexlm.trainer import ComplexLMTrainer

    parsed = dict(item.split("=", 1) for item in overrides)
    if esmc_weights:
        parsed["train.esmc_weights"] = esmc_weights
    if wandb_project:
        parsed["train.wandb_project"] = wandb_project
    if wandb_entity:
        parsed["train.wandb_entity"] = wandb_entity
    if dry_run:
        parsed.setdefault("train.max_steps", "20")
        parsed.setdefault("train.log_every", "5")
        parsed.setdefault("train.eval_every", "10")
        parsed.setdefault("train.eval_batches", "2")
        parsed.setdefault("train.save_every", "0")
    cfg = load_experiment(config, parsed)
    ComplexLMTrainer(cfg, output, resume).train()


if __name__ == "__main__":
    plm_train()
