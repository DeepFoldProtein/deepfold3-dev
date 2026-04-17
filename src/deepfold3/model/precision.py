"""Region-level fp32 pinning for selective precision control.

Provides :func:`fp32_guard` — a decorator factory that, when enabled,
disables ``torch.autocast`` and upcasts floating inputs of the wrapped
callable to fp32 — and :func:`fp32_scope`, a context manager form for
modules that handle casting themselves. Used to harden numerically
sensitive regions (loss computation, final logit projections) against
bf16 overflow while the rest of the network remains in the training dtype.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


def upcast_fp32(x):
    """Recursively upcast float tensors to fp32; non-floats and None pass through.

    Walks dicts / lists / tuples so nested feature containers are handled
    transparently (critical for the loss path, which receives ``pred`` and
    ``features`` dicts).
    """
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: upcast_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(upcast_fp32(v) for v in x)
    return x


def fp32_guard(enabled: bool) -> Callable[[F], F]:
    """Return a decorator that runs the wrapped call in fp32 when ``enabled``.

    When ``enabled`` is True, the decorator enters
    ``torch.autocast("cuda", enabled=False)`` and recursively upcasts every
    floating Tensor in args / kwargs (including tensors nested inside dicts,
    lists, or tuples). When ``enabled`` is False, the decorator is a no-op.
    """

    def wrap(fn: F) -> F:
        if not enabled:
            return fn

        def inner(*args, **kwargs):
            with torch.autocast(device_type="cuda", enabled=False):
                return fn(
                    *(upcast_fp32(a) for a in args),
                    **{k: upcast_fp32(v) for k, v in kwargs.items()},
                )

        return inner  # type: ignore[return-value]

    return wrap


def fp32_scope(enabled: bool):
    """Context manager that disables CUDA autocast when ``enabled`` is True.

    Use when the wrapped block handles input casting explicitly (e.g. a
    module ``forward`` that knows which tensors to upcast). Returns a
    :class:`~contextlib.nullcontext` when ``enabled`` is False.
    """
    return torch.autocast(device_type="cuda", enabled=False) if enabled else nullcontext()
