"""Checkpoint save/load/rotate + pretrained-weight loading.

On-disk format is **unchanged** from the pre-refactor trainer — the
dict keys (``model``/``optimizer``/``scheduler``/``global_step``/
``effective_step``/``best_eval_loss``/``model_cfg``/``train_cfg``/
``ema``) match exactly so existing checkpoints round-trip through
this module.

RNG state + sampler epoch capture (accelerate-style) are intentionally
deferred to a later phase — they'd bump the format and deserve their
own round-trip tests.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn

from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.training.ema import EMA
from deepfold3.training.rank_zero import rank_zero_info
from deepfold3.training.scheduler import AF3LRScheduler
from deepfold3.training.state import TrainState

logger = logging.getLogger(__name__)


def load_pretrained_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool,
    device: torch.device,
) -> None:
    """Load pretrained weights from ``.safetensors`` or ``.pt``.

    Mirrors the previous ``Trainer._load_pretrained`` behavior: a
    ``.pt`` file may contain either a bare state_dict or a checkpoint
    dict with a ``"model"`` key.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(p), device=str(device))
        model.load_state_dict(state, strict=strict)
        rank_zero_info(f"Loaded pretrained weights from {p}")
    elif p.suffix == ".pt":
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=strict)
        rank_zero_info(f"Loaded pretrained weights from {p}")


class CheckpointManager:
    """Save + load + rotate + find-latest for the training checkpoint blob.

    A thin value-free utility — every public method takes the pieces
    it needs explicitly. Holds no state so the trainer can instantiate
    one and call methods directly.
    """

    def __init__(self, *, max_checkpoints: int, strict: bool, device: torch.device) -> None:
        self.max_checkpoints = max_checkpoints
        self.strict = strict
        self.device = device

    # ----- save path -----

    def save(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: AF3LRScheduler,
        ema: EMA | None,
        state: TrainState,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
    ) -> None:
        """Save the checkpoint blob + rotate old files.

        Keys match the pre-refactor format exactly so existing
        checkpoints remain loadable by this module.
        """
        blob: dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": state.global_step,
            "effective_step": state.effective_step,
            "best_eval_loss": state.best_eval_loss,
            "model_cfg": model_cfg,
            "train_cfg": train_cfg,
        }
        if ema is not None:
            blob["ema"] = ema.state_dict()
        torch.save(blob, path)
        logger.info(f"Saved checkpoint to {path}")

        if self.max_checkpoints > 0:
            self._rotate(path.parent)

    def _rotate(self, output_dir: Path) -> None:
        """Keep only the N most recent ``checkpoint_*.pt`` files.

        ``checkpoint_final.pt`` and ``checkpoint_best.pt`` are never
        rotated out.
        """
        ckpts = sorted(output_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
        ckpts = [c for c in ckpts if "final" not in c.stem and "best" not in c.stem]
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            old.unlink()
            logger.info(f"Removed old checkpoint: {old}")

    # ----- load path -----

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: AF3LRScheduler,
        ema: EMA | None,
        state: TrainState,
        train_cfg: TrainingConfig,
    ) -> None:
        """Resume training state from ``path``, mutating the live objects in place.

        Handles DDP ``module.`` prefix stripping automatically.
        Partial-load gates (``skip_load_optimizer`` / ``skip_load_scheduler`` /
        ``skip_load_step``) follow the training config.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = ckpt["model"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=self.strict)

        if "optimizer" in ckpt and not train_cfg.skip_load_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and not train_cfg.skip_load_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])

        if not train_cfg.skip_load_step:
            state.global_step = ckpt.get("global_step", 0)
            state.effective_step = ckpt.get("effective_step", 0)
        state.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))

        rank_zero_info(f"Resumed from {path} at step {state.effective_step}")

    # ----- discovery -----

    @staticmethod
    def find_latest(output_dir: Path) -> Path | None:
        """Find the ``checkpoint_*.pt`` file with the highest step number."""
        best_step, best_path = -1, None
        for p in output_dir.glob("checkpoint_*.pt"):
            m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step, best_path = step, p
        return best_path
