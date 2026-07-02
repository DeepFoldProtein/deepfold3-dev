"""Per-effective-step GPU time (mean / min / max) via CUDA events.

CPU ``time.monotonic()`` around a training microstep is distorted by CUDA
async execution: it measures when kernels are *launched*, not when the GPU
finishes them. A microstep that races ahead of the GPU reads short, and its
real compute leaks into whatever operation next forces a device sync — so the
per-step min/max (the burst signal) are only as accurate as the incidental
syncs that happen to bracket each step (here, ``meters.update``'s
``bool(isfinite)`` and the optimizer's NaN ``.item()``). That coupling is
fragile: a refactor that defers those checks would silently skew the timings.

:class:`StepTimeMeter` records a start/end CUDA event pair per microstep — a
near-free stream marker, no CPU stall — buffers the pairs across the log
window, and reads their elapsed GPU time *once* at the window boundary, where
the metrics all-reduce already synchronizes the device. Consecutive
microsteps accumulate into the current effective step (delimited by the
optimizer-step boundary); each effective step's total feeds the window
min/max, and the grand total feeds the mean / throughput denominator.

The end event is recorded after the optimizer step, i.e. after the trainer has
already CPU-synced on the gradient all-reduce (the NaN ``.item()`` consensus),
so the collective wait is inside the bracket and shows up in the timing.

Falls back to CPU ``time.monotonic()`` when CUDA is unavailable (CPU tests).
Bracketing is per microstep (start recorded after the dataloader fetch), so
inter-microstep dataloader wait is excluded — matching the CPU ``dt`` it
replaces.
"""

from __future__ import annotations

import time

import torch


def _reduce(durations: list[float], boundaries: list[bool]) -> tuple[float, float, float]:
    """Fold per-microstep durations into ``(window_sum, estep_min, estep_max)``.

    Microsteps are grouped into effective steps delimited by a ``True`` in
    ``boundaries`` (the optimizer step). A window with no boundary (all
    microsteps skipped, or a partial tail) yields ``NaN`` for min/max but a
    real sum. Durations are in seconds.
    """
    total = 0.0
    estep = 0.0
    mn = float("inf")
    mx = 0.0
    finalized = 0
    for d, is_boundary in zip(durations, boundaries, strict=True):
        total += d
        estep += d
        if is_boundary:
            mn = min(mn, estep)
            mx = max(mx, estep)
            estep = 0.0
            finalized += 1
    if finalized == 0:
        return total, float("nan"), float("nan")
    return total, mn, mx


class StepTimeMeter:
    """CUDA-event step timer with a CPU-monotonic fallback.

    Args:
        device: CUDA device the events record on. Ignored on the CPU path.
    """

    def __init__(self, device: torch.device | None = None):
        self._cuda = torch.cuda.is_available()
        self._device = device
        # CUDA path: a flat pool of events reused every window (2 per
        # microstep: [start_0, end_0, start_1, end_1, ...]). Growing it lazily
        # avoids ``cudaEventCreate`` on the steady-state hot path.
        self._pool: list[torch.cuda.Event] = []
        self._boundary: list[bool] = []
        self._n = 0  # microsteps recorded in the current window
        # CPU fallback: buffer durations + boundary flags, reduced at flush.
        self._cpu_t0 = 0.0
        self._cpu_durs: list[float] = []
        self._cpu_bounds: list[bool] = []

    def start(self) -> None:
        """Mark the start of a microstep (record on the current CUDA stream)."""
        if self._cuda:
            i = 2 * self._n
            while len(self._pool) <= i + 1:
                self._pool.append(torch.cuda.Event(enable_timing=True))
            self._pool[i].record()
        else:
            self._cpu_t0 = time.monotonic()

    def stop(self, is_boundary: bool) -> None:
        """Mark the end of a microstep.

        Args:
            is_boundary: True on the optimizer-step (sync) microstep that
                closes an effective step — the unit ``dt`` min/max report on.
        """
        if self._cuda:
            self._pool[2 * self._n + 1].record()
            self._boundary.append(is_boundary)
            self._n += 1
        else:
            self._cpu_durs.append(time.monotonic() - self._cpu_t0)
            self._cpu_bounds.append(is_boundary)

    def flush(self) -> tuple[float, float, float]:
        """Return ``(window_sum_s, estep_min_s, estep_max_s)`` and reset.

        On the CUDA path this issues exactly one device sync (on the last end
        event, which drains all earlier ones) before reading elapsed times.
        Call only at a point where a sync is acceptable — e.g. the metrics
        log boundary, which already all-reduces.
        """
        if self._cuda:
            durations: list[float] = []
            if self._n > 0:
                # One sync drains the whole window (events are stream-ordered).
                self._pool[2 * self._n - 1].synchronize()
                durations = [self._pool[2 * i].elapsed_time(self._pool[2 * i + 1]) / 1000.0 for i in range(self._n)]
            out = _reduce(durations, self._boundary)
            self._boundary = []
            self._n = 0
            return out
        out = _reduce(self._cpu_durs, self._cpu_bounds)
        self._cpu_durs = []
        self._cpu_bounds = []
        return out
