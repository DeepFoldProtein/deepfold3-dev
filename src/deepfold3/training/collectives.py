"""Pytree helpers used inside the DDP training step.

Kept in a dedicated module so cross-cutting tensor utilities (batch-dim
slicing today; potentially ``reduce``/``gather`` helpers later) live next
to the distributed primitives they wrap, rather than inside
``trainer.py``.
"""

from __future__ import annotations

import torch


def slice_batch_dict(d: dict, b: int, batch_size: int) -> dict:
    """Slice each tensor value in ``d`` along dim 0 at index ``b``.

    Only tensors whose leading dim equals ``batch_size`` are sliced.
    Shared weight tensors (e.g. ``pae_weight: [num_bins, c_pair]``),
    pre-aggregated scalars (``__diffusion_total__``), and other shapes
    that don't carry the batch dim pass through unchanged. Sub-dicts
    are recursed into.

    Used by the per-batch loss loop in ``_training_step`` when
    ``batch_size_per_rank > 1``.
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.dim() >= 1 and v.shape[0] == batch_size:
                out[k] = v[b]
            else:
                out[k] = v
        elif isinstance(v, dict):
            out[k] = slice_batch_dict(v, b, batch_size)
        else:
            out[k] = v
    return out
