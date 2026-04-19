"""Unit tests for ``collective_guard`` — barrier semantics without NCCL."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from deepfold3.training.collective_guard import collective_guard
from deepfold3.training.distributed import DistInfo


def _dist(world_size: int) -> DistInfo:
    return DistInfo(rank=0, world_size=world_size, local_rank=0, device=torch.device("cpu"))


def test_noop_in_single_process():
    """No barrier calls when ``is_distributed`` is False."""
    with patch("deepfold3.training.collective_guard.dist.barrier") as barrier:
        with collective_guard("step_or_skip", _dist(world_size=1)):
            pass
    barrier.assert_not_called()


def test_issues_entry_and_exit_barrier():
    calls = []
    with patch("deepfold3.training.collective_guard.dist.barrier", side_effect=lambda **kw: calls.append(kw)):
        with collective_guard("step_or_skip", _dist(world_size=2)):
            calls.append("body")
    assert calls == [{"device_ids": [0]}, "body", {"device_ids": [0]}]


def test_body_exception_skips_exit_barrier():
    """If the body raises, the exit barrier is skipped (failing rank exits;
    other ranks hit the NCCL watchdog on their next collective)."""
    calls = []
    with patch("deepfold3.training.collective_guard.dist.barrier", side_effect=lambda **kw: calls.append("barrier")):
        with pytest.raises(ValueError, match="boom"):
            with collective_guard("step_or_skip", _dist(world_size=2)):
                calls.append("body")
                raise ValueError("boom")
    assert calls == ["barrier", "body"]


def test_entry_barrier_failure_is_renamed():
    """A RuntimeError from ``dist.barrier`` on entry is re-raised with the guard name."""
    with patch(
        "deepfold3.training.collective_guard.dist.barrier",
        side_effect=RuntimeError("NCCL watchdog timeout"),
    ):
        with pytest.raises(RuntimeError, match=r"collective_guard\('step_or_skip'\) entry barrier"):
            with collective_guard("step_or_skip", _dist(world_size=2)):
                pytest.fail("body should not have run after entry barrier failed")


def test_exit_barrier_failure_is_renamed():
    calls = {"n": 0}

    def flaky_barrier(**_kw):
        calls["n"] += 1
        if calls["n"] == 2:  # entry ok, exit fails
            raise RuntimeError("peer rank missing")

    with patch("deepfold3.training.collective_guard.dist.barrier", side_effect=flaky_barrier):
        with pytest.raises(RuntimeError, match=r"collective_guard\('step_or_skip'\) exit barrier"):
            with collective_guard("step_or_skip", _dist(world_size=2)):
                pass


def test_return_inside_guard_runs_exit_barrier():
    """Returns inside the ``with`` block must still trigger the exit barrier."""
    calls = []

    def run() -> int:
        with patch(
            "deepfold3.training.collective_guard.dist.barrier", side_effect=lambda **kw: calls.append("barrier")
        ):
            with collective_guard("step_or_skip", _dist(world_size=2)):
                calls.append("body")
                return 42
        return -1  # pragma: no cover

    assert run() == 42
    assert calls == ["barrier", "body", "barrier"]
