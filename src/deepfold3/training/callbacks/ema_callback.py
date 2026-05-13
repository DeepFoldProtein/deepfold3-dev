"""EMA update callback — shadow-weight refresh on successful opt steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.optimizer import Optimizer

from deepfold3.training.callbacks import Callback
from deepfold3.training.ema import EMA
from deepfold3.training.schedule_free import AdamWScheduleFree, optimizer_eval_mode

if TYPE_CHECKING:
    from deepfold3.training.optim import StepResult
    from deepfold3.training.trainer import Trainer


class EMACallback(Callback):
    """Fire ``EMA.update`` after every successful optimizer step.

    Reached from ``CallbackHub.on_step_end``, which the trainer only
    invokes when ``result.advanced is True`` — NaN-skipped steps go to
    ``on_skip_step`` instead and never update the shadow weights.

    For Schedule-Free optimizers, ``ema.update(model)`` is wrapped in
    ``optimizer_eval_mode(optimizer)`` so the shadow tracks the SF
    averaged iterate ``x`` rather than the live ``y`` (= the gradient
    point). Per-step cost: 2 extra param-sized lerps (y→x flip and
    restore). For plain AdamW the wrap is a no-op and shadow tracks
    the live params directly.
    """

    def __init__(self, ema: EMA, optimizer: Optimizer | None = None) -> None:
        self._ema = ema
        # Only the SF case needs the eval-mode wrap; for AdamW the
        # context manager is a no-op so we keep optimizer=None to
        # short-circuit and avoid the conditional at every step.
        self._optimizer = optimizer if isinstance(optimizer, AdamWScheduleFree) else None

    def on_step_end(self, trainer: Trainer, result: StepResult) -> None:
        if self._optimizer is not None:
            with optimizer_eval_mode(self._optimizer):
                self._ema.update(trainer.model)
        else:
            self._ema.update(trainer.model)
