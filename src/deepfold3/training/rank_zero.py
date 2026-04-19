"""Rank-zero-only logging utilities.

Replaces scattered ``if is_main: logger.info(...)`` idioms with a single
decorator + logger helpers. Mirrors the Lightning / lightning_utilities
pattern without the external dependency.

The module-level ``rank_zero_only.rank`` attribute is the single source
of truth; it is initialized from env vars (``RANK`` / ``LOCAL_RANK`` /
``SLURM_PROCID``) at import time and re-synced by
``init_process_group`` once the real rank is known. Patching this
attribute retroactively affects every already-decorated callable.

Use ``rank_zero_info`` / ``rank_zero_warn`` / ``rank_zero_debug`` /
``rank_zero_error`` instead of gating ``logger.info`` calls manually.
The helpers attribute the log record to the caller's module via the
``stacklevel`` mechanism, so log lines point at the caller's file/line
just as they would with a direct ``logger.info`` call.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _detect_rank() -> int:
    # Order matches Lightning: ``RANK`` (torchrun) wins over ``LOCAL_RANK`` /
    # SLURM so a cross-node launcher's global rank is authoritative.
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def rank_zero_only(fn: F) -> F:  # noqa: UP047 — PEP 695 syntax requires Python 3.12+, repo is 3.11
    """Run ``fn`` only on rank 0; return ``None`` on other ranks.

    Reads the module-level ``rank_zero_only.rank`` at call time (not
    decoration time), so patching the attribute after import works.
    """

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped  # type: ignore[return-value]


rank_zero_only.rank = _detect_rank()  # type: ignore[attr-defined]


def _caller_logger() -> logging.Logger:
    """Return the ``logging.Logger`` whose name matches the caller's module.

    Preserves the ``%(name)s`` field so rank-zero log lines are
    indistinguishable from direct ``logger.info`` calls.
    """
    # Frame 0 is this helper; frame 1 is ``_log``; frame 2 is the
    # ``rank_zero_*`` wrapper; frame 3 is the user's call site.
    frame = sys._getframe(3)
    name = frame.f_globals.get("__name__", "deepfold3.training.rank_zero")
    return logging.getLogger(name)


def _log(level: int, msg: str, *args: Any, **kwargs: Any) -> None:
    if rank_zero_only.rank != 0:
        return
    # ``stacklevel=3`` skips ``_log`` + the ``rank_zero_*`` wrapper so the
    # LogRecord's filename / lineno point to the user's call site.
    _caller_logger().log(level, msg, *args, stacklevel=3, **kwargs)


def _make_level_helper(level: int) -> Callable[..., None]:
    def helper(msg: str, *args: Any, **kwargs: Any) -> None:
        _log(level, msg, *args, **kwargs)

    return helper


rank_zero_debug = _make_level_helper(logging.DEBUG)
rank_zero_info = _make_level_helper(logging.INFO)
rank_zero_warn = _make_level_helper(logging.WARNING)
rank_zero_error = _make_level_helper(logging.ERROR)
rank_zero_critical = _make_level_helper(logging.CRITICAL)
