"""Batch collation: pads each feature to the max shape, then stacks.

When ``bucket_pad=True`` (the production training default), each sample is
first passed through :func:`deepfold3.model.bucket.pad_features` so its N
dimension lands on the next entry in ``BUCKET_SIZES``. This keeps the
Triton kernel input shapes (and the persistent autotune cache keys)
deterministic across microsteps; without it the cache thrashes on every
distinct sample length.

Padding writes ``0`` into both data and mask tensors. The seq/pair/atom/msa
masks become 0 in padded slots so the forward pass and loss computation
ignore them; data tensors hold whatever 0 means in their dtype but are
protected by the corresponding mask.
"""

from __future__ import annotations

import torch

from deepfold3.model.bucket import next_bucket, pad_features


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    *,
    bucket_pad: bool = False,
) -> dict[str, torch.Tensor]:
    """Pad and stack a list of feature dicts into a batched dict.

    Args:
        batch: List of per-sample feature dicts from the dataset.
        bucket_pad: If True, each sample's N dim is rounded up to the next
            bucket size from :data:`deepfold3.model.bucket.BUCKET_SIZES` via
            :func:`deepfold3.model.bucket.pad_features`. Required for the
            Triton autotune cache to stay warm in production training.
    """
    if not batch:
        return {}

    if bucket_pad:
        n_max = max(sample["res_type"].shape[0] for sample in batch if "res_type" in sample)
        bucket_n = next_bucket(n_max)
        batch = [pad_features(sample, bucket_n=bucket_n)[0] for sample in batch]

    keys = list(batch[0].keys())
    out: dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [sample[key] for sample in batch if key in sample]
        if not tensors:
            continue

        shapes = [t.shape for t in tensors]
        if len(shapes[0]) == 0:
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
