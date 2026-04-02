"""Learning rate schedulers for DeepFold3.

Two schedulers:

1. ``AF3LRScheduler``: linear warmup → plateau → step decay (initial training).
2. ``CosineAnnealingScheduler``: linear warmup → cosine decay (fine-tuning).

All parameters are configurable via ``TrainingConfig``.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from deepfold3.model.config import TrainingConfig


class AF3LRScheduler(LambdaLR):
    """AF3-style learning rate scheduler.

    Three-phase schedule:

    1. **Warmup** (steps ``[0, warmup_steps)``): linear ramp ``0 -> max_lr``.
    2. **Plateau** (steps ``[warmup_steps, start_decay_after)``): constant ``max_lr``.
    3. **Decay** (steps ``>= start_decay_after``):
       ``max_lr * decay_factor ** (steps_since // decay_every)``.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config with LR schedule parameters.
    """

    def __init__(self, optimizer: Optimizer, cfg: TrainingConfig):
        self.max_lr = cfg.lr
        self.warmup_steps = cfg.warmup_steps
        self.start_decay_after = cfg.start_decay_after
        self.decay_every = cfg.decay_every
        self.decay_factor = cfg.decay_factor

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        """Compute LR multiplier for the given step.

        Args:
            step: Current optimizer step.

        Returns:
            Multiplier in ``[0, 1]`` to apply to the base LR.
        """
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        if step < self.start_decay_after:
            return 1.0

        steps_since = step - self.start_decay_after
        n_decays = steps_since // max(1, self.decay_every)
        return self.decay_factor**n_decays


class CosineAnnealingScheduler(LambdaLR):
    """Cosine annealing with linear warmup for fine-tuning.

    1. **Warmup** (steps ``[0, warmup_steps)``): linear ramp ``0 -> max_lr``.
    2. **Cosine decay** (steps ``[warmup_steps, max_steps]``):
       ``max_lr * (min_ratio + (1 - min_ratio) * (1 + cos(pi * progress)) / 2)``.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config with cosine schedule parameters.
    """

    def __init__(self, optimizer: Optimizer, cfg: TrainingConfig):
        self.warmup_steps = cfg.warmup_steps
        self.max_steps = cfg.cosine_max_steps
        self.min_ratio = cfg.cosine_min_lr_ratio

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        progress = min(1.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
        return self.min_ratio + (1.0 - self.min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def build_scheduler(optimizer: Optimizer, cfg: TrainingConfig) -> LambdaLR:
    """Build the appropriate scheduler based on config.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training config.

    Returns:
        AF3LRScheduler or CosineAnnealingScheduler.
    """
    if cfg.finetune_scheduler == "cosine":
        return CosineAnnealingScheduler(optimizer, cfg)
    return AF3LRScheduler(optimizer, cfg)
