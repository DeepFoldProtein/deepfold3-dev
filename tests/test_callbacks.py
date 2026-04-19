"""Unit tests for the callback hub + EMACallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deepfold3.training.callbacks import Callback, CallbackHub
from deepfold3.training.callbacks.ema_callback import EMACallback
from deepfold3.training.ema import EMA


class Recorder(Callback):
    """Records which hooks were called and with what arguments."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[tuple[str, tuple]] = []

    def on_train_start(self, trainer):
        self.calls.append(("train_start", ()))

    def on_step_start(self, trainer, features):
        self.calls.append(("step_start", (tuple(features.keys()),)))

    def on_step_end(self, trainer, result):
        self.calls.append(("step_end", (result,)))

    def on_skip_step(self, trainer, result):
        self.calls.append(("skip_step", (result,)))

    def on_eval_start(self, trainer):
        self.calls.append(("eval_start", ()))

    def on_eval_end(self, trainer, metrics):
        self.calls.append(("eval_end", (dict(metrics),)))

    def on_checkpoint_save(self, trainer, path):
        self.calls.append(("ckpt_save", (path,)))

    def on_train_end(self, trainer):
        self.calls.append(("train_end", ()))


def test_hub_dispatches_in_order():
    a = Recorder("a")
    b = Recorder("b")
    hub = CallbackHub([a, b])
    trainer = object()

    hub.on_train_start(trainer)
    hub.on_step_start(trainer, {"x": None})
    hub.on_eval_end(trainer, {"loss": 0.5})
    hub.on_train_end(trainer)

    assert [h for h, _ in a.calls] == ["train_start", "step_start", "eval_end", "train_end"]
    assert a.calls == b.calls  # same order for all callbacks


def test_hub_append_adds_to_tail():
    a = Recorder("a")
    hub = CallbackHub([a])
    b = Recorder("b")
    hub.append(b)
    assert len(hub) == 2

    hub.on_train_end(object())
    assert a.calls == [("train_end", ())]
    assert b.calls == [("train_end", ())]


def test_hub_empty_is_safe():
    hub = CallbackHub()
    # No callbacks → every hook is a no-op
    hub.on_train_start(None)
    hub.on_step_end(None, None)
    hub.on_train_end(None)


def test_callback_exception_propagates():
    class BadCallback(Callback):
        def on_train_start(self, trainer):
            raise ValueError("cb failure")

    hub = CallbackHub([Recorder("a"), BadCallback(), Recorder("b")])
    with pytest.raises(ValueError, match="cb failure"):
        hub.on_train_start(object())


def _tiny_ema_model() -> nn.Module:
    m = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.eye(4))
    return m


def test_ema_callback_updates_on_step_end():
    """EMACallback.on_step_end forwards to EMA.update(model)."""
    model = _tiny_ema_model()
    ema = EMA(model, decay=0.5)
    # Mutate the live model so EMA update has something to blend.
    with torch.no_grad():
        model.weight.mul_(2.0)

    callback = EMACallback(ema)
    trainer = SimpleNamespace(model=model)
    result = SimpleNamespace(advanced=True, was_nan=False, reason=None, grad_norm=None, nan_ranks=[])

    pre = ema.shadow["weight"].clone()
    callback.on_step_end(trainer, result)
    post = ema.shadow["weight"]

    # Warm-start decay: ``min(0.5, (1+num_updates)/(10+num_updates))`` — on
    # the first update ``num_updates=1`` so decay = min(0.5, 2/11) = 2/11.
    decay = min(0.5, 2 / 11)
    expected = decay * pre + (1 - decay) * model.weight
    torch.testing.assert_close(post, expected)


def test_ema_callback_does_not_fire_on_skip_step():
    """on_skip_step has no-op EMA behaviour (trainer gates by result.advanced)."""
    model = _tiny_ema_model()
    ema = EMA(model, decay=0.5)
    callback = EMACallback(ema)
    trainer = SimpleNamespace(model=model)
    result = SimpleNamespace(advanced=False, was_nan=True, reason="loss", grad_norm=None, nan_ranks=[0])

    pre = ema.shadow["weight"].clone()
    # on_skip_step inherits the base no-op; EMA untouched.
    callback.on_skip_step(trainer, result)
    torch.testing.assert_close(ema.shadow["weight"], pre)
