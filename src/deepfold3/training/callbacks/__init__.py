"""Minimal callback/lifecycle-hook registry.

Trims a few responsibilities (EMA update, future WandB log fan-out,
future NaN-tracker bookkeeping) out of the trainer body into pluggable
objects. Intentionally tiny — no stage/setup/teardown machinery, no
automatic hook discovery, no exception swallowing. Compared with
``pytorch_lightning.Callback`` the surface is ~20 lines: the dispatch
is a for-loop, and hooks receive the live ``Trainer`` (not a
``LightningModule``) so callbacks can reach through to ``trainer.state``,
``trainer.model``, ``trainer.dist`` directly.

**Collective invariant:** callback methods run on every rank. A
callback MUST NOT issue a collective (``all_reduce`` / ``all_gather``
/ ``barrier``) inside a hook unless every rank is guaranteed to reach
the same call — the class of desync bug documented in
``memory/project_try_except_continue_deadlock.md`` applies equally
here. If you need cross-rank aggregation, factor it into the trainer
loop (where the call pattern is uniform) and let callbacks consume
the already-reduced values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from deepfold3.training.optim import StepResult
    from deepfold3.training.trainer import Trainer


class Callback:
    """Lifecycle-hook base class. Subclass and override what you need.

    All methods are no-ops by default. The ``Trainer`` argument gives
    access to the full trainer state for callbacks that need to read
    config / model / dist info.
    """

    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer, features: dict[str, Any]) -> None:
        pass

    def on_step_end(self, trainer: Trainer, result: StepResult) -> None:
        """Fires after a successful optimizer step (``result.advanced is True``)."""

    def on_skip_step(self, trainer: Trainer, result: StepResult) -> None:
        """Fires after a NaN-skipped optimizer step (``result.advanced is False``)."""

    def on_eval_start(self, trainer: Trainer) -> None:
        pass

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_checkpoint_save(self, trainer: Trainer, path: Path) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


_HOOKS = (
    "on_train_start",
    "on_step_start",
    "on_step_end",
    "on_skip_step",
    "on_eval_start",
    "on_eval_end",
    "on_checkpoint_save",
    "on_train_end",
)


class CallbackHub:
    """Ordered registry that dispatches lifecycle hooks to each callback.

    Ordering matches construction order. Exceptions propagate — no
    silent swallow — so a misbehaving callback kills the run instead
    of corrupting training state. Dispatch methods are generated
    per-hook from ``_HOOKS`` so adding a hook requires touching only
    the base class + that tuple.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = list(callbacks) if callbacks else []

    def __len__(self) -> int:
        return len(self._callbacks)

    def append(self, callback: Callback) -> None:
        self._callbacks.append(callback)


def _make_dispatch(hook: str):
    def dispatch(self: CallbackHub, *args, **kwargs) -> None:
        for cb in self._callbacks:
            getattr(cb, hook)(*args, **kwargs)

    dispatch.__name__ = hook
    dispatch.__qualname__ = f"CallbackHub.{hook}"
    return dispatch


for _hook in _HOOKS:
    setattr(CallbackHub, _hook, _make_dispatch(_hook))


__all__ = ["Callback", "CallbackHub"]
