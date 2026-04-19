"""Experiment tracker abstraction — rank-0-only sinks behind a common protocol.

Keeps the main training loop free of per-backend branches. Add a new
tracker by subclassing :class:`Tracker` and appending it to
:class:`TrackerHub`. Plugin auto-discovery is intentionally absent —
the hub is a fixed list.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracker(Protocol):
    """Minimal experiment-tracker interface.

    Implementations are called only on the main rank by :class:`TrackerHub`.
    ``log`` may be called many times per second, so implementations should
    buffer or deduplicate as needed.
    """

    def start(self, config: dict[str, Any]) -> None: ...
    def log(self, values: dict[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...


class StdoutTracker:
    """No-op sink.

    Text-level progress is already emitted via ``logger.info`` in
    ``trainer.py``; this tracker exists so :class:`TrackerHub` always has
    at least one registered implementation and so ``log`` calls never
    branch on "is wandb configured?" at the call site.
    """

    def start(self, config: dict[str, Any]) -> None:  # noqa: D401
        return None

    def log(self, values: dict[str, Any], step: int) -> None:  # noqa: D401
        return None

    def finish(self) -> None:  # noqa: D401
        return None


class WandBTracker:
    """Weights & Biases tracker.

    Lazy-imports :mod:`wandb` in :meth:`start` so the module can be
    imported without the dependency installed. If import fails, the
    tracker silently becomes a no-op (matches the existing behavior in
    ``Trainer._init_wandb``).
    """

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        run_name: str,
    ) -> None:
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self._wandb = None

    def start(self, config: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")
            return
        self._wandb = wandb
        wandb.init(
            project=self.project,
            entity=self.entity or None,
            name=self.run_name,
            config=config,
        )
        # Declare custom step axis so train/eval/system metrics align on
        # ``train/step`` regardless of the step passed to ``wandb.log``.
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="train/step")
        wandb.define_metric("system/*", step_metric="train/step")
        logger.info(f"WandB initialized: project={self.project}, run={self.run_name}")

    def log(self, values: dict[str, Any], step: int) -> None:
        if self._wandb is None:
            return
        self._wandb.log(values, step=step)

    def finish(self) -> None:
        if self._wandb is None:
            return
        self._wandb.finish()


class TrackerHub:
    """Fan-out over a list of :class:`Tracker` instances.

    All methods are no-ops on non-main ranks so callers can invoke
    ``hub.log(...)`` unconditionally without guarding on rank. Rank
    membership is read from ``rank_zero_only.rank`` — the single source
    of rank truth across the trainer.
    """

    def __init__(self, trackers: list[Tracker]) -> None:
        from deepfold3.training.rank_zero import rank_zero_only

        self._trackers = list(trackers) if rank_zero_only.rank == 0 else []

    def start(self, config: dict[str, Any]) -> None:
        for t in self._trackers:
            t.start(config)

    def log(self, values: dict[str, Any], step: int) -> None:
        for t in self._trackers:
            t.log(values, step)

    def finish(self) -> None:
        for t in self._trackers:
            t.finish()
