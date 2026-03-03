"""Batch collation: pads each feature to the max shape, then stacks."""

from __future__ import annotations

import torch


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad and stack a list of feature dicts into a batched dict.

    Each tensor is padded along all dimensions (after the batch dim) to
    the maximum size seen in the batch.
    """
    if not batch:
        return {}

    keys = list(batch[0].keys())
    out: dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [sample[key] for sample in batch if key in sample]
        if not tensors:
            continue

        # Find max shape
        shapes = [t.shape for t in tensors]
        if len(shapes[0]) == 0:
            # Scalar
            out[key] = torch.stack(tensors)
            continue

        max_shape = [max(s[d] for s in shapes) for d in range(len(shapes[0]))]

        padded_list: list[torch.Tensor] = []
        for t in tensors:
            pad_config: list[int] = []
            # torch.nn.functional.pad expects padding in reverse order (last dim first)
            for d in reversed(range(len(t.shape))):
                pad_config.extend([0, max_shape[d] - t.shape[d]])
            if any(p > 0 for p in pad_config):
                padded = torch.nn.functional.pad(t, pad_config, value=0)
            else:
                padded = t
            padded_list.append(padded)

        out[key] = torch.stack(padded_list, dim=0)

    return out
