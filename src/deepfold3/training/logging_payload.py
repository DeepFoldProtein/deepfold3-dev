"""Build the per-step metrics payload consumed by trackers.

Pure function: takes the already-reduced averages, a few scalars, and
returns the dict that gets handed to ``TrackerHub.log``.
"""

from __future__ import annotations

import torch


def build_step_payload(
    *,
    avg: dict[str, float],
    effective_step: int,
    lr: float,
    epoch: int,
    grad_norm: float,
    dt: float,
    tok_per_sec: float,
    peak_mem_gb: float,
    data_wait: float = 0.0,
    grad_norm_max: float = float("nan"),
    dt_min: float = float("nan"),
    dt_max: float = float("nan"),
) -> dict[str, float]:
    """Assemble the training-step payload.

    Mirrors the shape expected by ``wandb.define_metric("train/*", ...)`` —
    ``train/*`` keys for model/optimizer state, ``system/*`` for resource
    usage, and a top-level ``train/step`` axis.
    """
    payload: dict[str, float] = {f"train/{k}": v for k, v in avg.items()}
    payload["train/step"] = effective_step
    payload["train/lr"] = lr
    payload["train/epoch"] = epoch
    payload["train/grad_norm"] = grad_norm
    payload["train/grad_norm_max"] = grad_norm_max
    payload["train/step_time_s"] = dt
    payload["train/step_time_min_s"] = dt_min
    payload["train/step_time_max_s"] = dt_max
    payload["train/data_wait_s"] = data_wait
    payload["train/tokens_per_sec"] = tok_per_sec
    cuda_avail = torch.cuda.is_available()
    payload["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1 << 30) if cuda_avail else 0.0
    payload["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1 << 30) if cuda_avail else 0.0
    payload["system/gpu_peak_memory_gb"] = peak_mem_gb
    return payload
