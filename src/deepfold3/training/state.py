"""Per-rank training state carried through the outer loop.

A single :class:`TrainState` dataclass replaces the scatter of
``self.global_step`` / ``self.effective_step`` / ``self._best_eval_loss``
attributes that used to live directly on :class:`Trainer`. Separating
state from the orchestrator lets pure-function helpers in
``training/step.py``, ``training/optim.py``, etc. mutate it explicitly
without reaching into a god-object.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrainState:
    """Run-local state that advances across optimizer steps.

    Attributes are mutated in place by the trainer and its helpers;
    this is intentional — the state is a one-of-one per run.

    Attributes:
        global_step: Microstep counter (includes accumulation steps).
        effective_step: Optimizer step counter (advances once per
            ``accumulate_grad_batches`` microsteps when no NaN skip).
        epoch: Current epoch index.
        best_eval_loss: Lowest ``eval/total`` seen so far (for best-ckpt
            tracking).
        consecutive_nan_skips: How many opt steps in a row were skipped
            due to non-finite loss or grads. Reset to 0 on a successful
            step. ``cfg.max_consecutive_nan_skips`` bounds this.
        last_grad_norm: Most recent global grad norm (for logging).
        last_num_recycles: Number of recycles sampled for the last
            microstep (for microstep logging).
    """

    global_step: int = 0
    effective_step: int = 0
    epoch: int = 0
    best_eval_loss: float = field(default_factory=lambda: float("inf"))
    consecutive_nan_skips: int = 0
    last_grad_norm: torch.Tensor | None = None
    last_num_recycles: int = 0


def seed_everything(seed: int, rank: int) -> None:
    """Set seeds for reproducibility, offset by rank.

    Offsetting by rank gives each rank an independent RNG stream while
    keeping the run as a whole deterministic with respect to the
    user-supplied seed.
    """
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
