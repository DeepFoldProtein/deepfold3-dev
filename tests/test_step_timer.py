"""Tests for the CUDA-event step timer (``training/step_timer.py``)."""

from __future__ import annotations

import math

import torch

from deepfold3.training.step_timer import StepTimeMeter, _reduce


def test_reduce_groups_microsteps_into_effective_steps():
    # accumulate=2: three effective steps, the third carries a 91s stall.
    durs = [8.0, 11.0, 9.0, 10.0, 7.0, 91.0]
    bounds = [False, True, False, True, False, True]
    total, mn, mx = _reduce(durs, bounds)
    assert total == 136.0
    assert mn == 19.0  # min effective-step total (8+11)
    assert mx == 98.0  # max effective-step total (7+91) — the burst


def test_reduce_no_boundary_is_nan_minmax_but_real_sum():
    total, mn, mx = _reduce([3.0, 4.0], [False, False])
    assert total == 7.0
    assert math.isnan(mn) and math.isnan(mx)


def test_reduce_empty_window():
    total, mn, mx = _reduce([], [])
    assert total == 0.0
    assert math.isnan(mn) and math.isnan(mx)


def test_reduce_accumulate_one():
    # accumulate=1: every microstep is its own effective step.
    total, mn, mx = _reduce([5.0, 20.0, 6.0], [True, True, True])
    assert total == 31.0
    assert mn == 5.0
    assert mx == 20.0


def _run_window(meter, per_estep_microsteps):
    """Drive the meter through effective steps and flush once."""
    for microsteps in per_estep_microsteps:
        for j in range(microsteps):
            meter.start()
            # A little real work so the CUDA events bracket a nonzero span.
            if torch.cuda.is_available():
                x = torch.randn(256, 256, device="cuda")
                (x @ x).sum()
            meter.stop(is_boundary=(j == microsteps - 1))
    return meter.flush()


def test_step_time_meter_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meter = StepTimeMeter(device)

    # Window 1: three effective steps of 2 microsteps each.
    total, mn, mx = _run_window(meter, [2, 2, 2])
    assert total > 0.0
    assert 0.0 < mn <= mx
    mean = total / 3
    assert mn <= mean <= mx + 1e-9

    # Flush must reset: a bare flush now sees an empty window.
    total2, mn2, mx2 = meter.flush()
    assert total2 == 0.0
    assert math.isnan(mn2) and math.isnan(mx2)

    # Window 2 reuses the event pool and works identically.
    total3, mn3, mx3 = _run_window(meter, [2, 2])
    assert total3 > 0.0
    assert 0.0 < mn3 <= mx3


def test_step_time_meter_skipped_step_extends_bucket():
    # A skipped effective step (no boundary) folds its microsteps into the
    # next finalized bucket rather than crashing.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meter = StepTimeMeter(device)
    # microstep boundaries: [F, F, T] → one effective step spanning 3 microsteps
    for j in range(3):
        meter.start()
        if torch.cuda.is_available():
            x = torch.randn(128, 128, device="cuda")
            (x @ x).sum()
        meter.stop(is_boundary=(j == 2))
    total, mn, mx = meter.flush()
    assert total > 0.0
    assert mn == mx  # a single finalized effective step → min == max
