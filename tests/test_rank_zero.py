"""Unit tests for the rank-zero gating helpers."""

from __future__ import annotations

import logging

import pytest

from deepfold3.training.rank_zero import (
    rank_zero_debug,
    rank_zero_error,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)


@pytest.fixture
def restore_rank():
    """Restore the module-level rank after each test."""
    original = rank_zero_only.rank
    try:
        yield
    finally:
        rank_zero_only.rank = original


def test_decorator_runs_on_rank_zero(restore_rank):
    rank_zero_only.rank = 0
    calls = []

    @rank_zero_only
    def side_effect() -> str:
        calls.append(1)
        return "hello"

    assert side_effect() == "hello"
    assert calls == [1]


def test_decorator_noop_on_other_rank(restore_rank):
    rank_zero_only.rank = 3
    calls = []

    @rank_zero_only
    def side_effect() -> str:
        calls.append(1)
        return "hello"

    assert side_effect() is None
    assert calls == []


def test_rank_attribute_is_read_at_call_time(restore_rank):
    """Patching ``rank_zero_only.rank`` after decoration takes effect immediately."""
    rank_zero_only.rank = 0

    @rank_zero_only
    def fn() -> int:
        return 42

    assert fn() == 42
    rank_zero_only.rank = 1
    assert fn() is None
    rank_zero_only.rank = 0
    assert fn() == 42


def test_log_helpers_emit_on_rank_zero(restore_rank, caplog):
    rank_zero_only.rank = 0
    # caplog installs a handler at the root level; propagation carries
    # records up from the caller's module logger.
    with caplog.at_level(logging.DEBUG):
        rank_zero_debug("debug msg %d", 1)
        rank_zero_info("info msg %d", 2)
        rank_zero_warn("warn msg %d", 3)
        rank_zero_error("error msg %d", 4)

    assert [r.getMessage() for r in caplog.records] == [
        "debug msg 1",
        "info msg 2",
        "warn msg 3",
        "error msg 4",
    ]


def test_log_helpers_silent_on_non_zero_rank(restore_rank, caplog):
    rank_zero_only.rank = 2
    with caplog.at_level(logging.DEBUG):
        rank_zero_info("should not appear")
        rank_zero_warn("should not appear")

    assert caplog.records == []


def test_log_record_attributes_caller_module(restore_rank, caplog):
    """Log record ``name`` should reflect the caller's module, not rank_zero."""
    rank_zero_only.rank = 0
    with caplog.at_level(logging.INFO):
        rank_zero_info("hello")

    assert len(caplog.records) == 1
    # Caller module is this test module.
    assert caplog.records[0].name == __name__
    # stacklevel=3 should make pathname point at the test file.
    assert caplog.records[0].filename == "test_rank_zero.py"
