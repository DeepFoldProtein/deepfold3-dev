"""EMA update callback — shadow-weight refresh on successful opt steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepfold3.training.callbacks import Callback
from deepfold3.training.ema import EMA

if TYPE_CHECKING:
    from deepfold3.training.optim import StepResult
    from deepfold3.training.trainer import Trainer


class EMACallback(Callback):
    """Fire ``EMA.update`` after every successful optimizer step.

    Reached from ``CallbackHub.on_step_end``, which the trainer only
    invokes when ``result.advanced is True`` — NaN-skipped steps go to
    ``on_skip_step`` instead and never update the shadow weights.
    """

    def __init__(self, ema: EMA) -> None:
        self._ema = ema

    def on_step_end(self, trainer: Trainer, result: StepResult) -> None:
        self._ema.update(trainer.model)
