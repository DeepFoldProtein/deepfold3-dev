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
    fixed_bucket_n: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad and stack a list of feature dicts into a batched dict.

    Args:
        batch: List of per-sample feature dicts from the dataset.
        bucket_pad: If True, each sample's N dim is rounded up to a bucket
            size via :func:`deepfold3.model.bucket.pad_features`. Required
            for the Triton autotune cache to stay warm in production.
        fixed_bucket_n: If given (and ``bucket_pad`` is True), pad every
            sample to **this exact N** regardless of its local size. Set to
            ``cfg.max_tokens`` in distributed training so every rank
            produces identical-shape batches — without it each rank picks
            a different local bucket (via ``next_bucket(batch_n_max)``) and
            the rank with the longest crop stalls the collective on every
            step. When ``None``, falls back to per-batch ``next_bucket``.
    """
    if not batch:
        return {}

    if bucket_pad:
        # Strip non-tensor diagnostics (e.g. ``_record_id``) before bucket
        # padding — ``pad_features`` only knows tensor shapes and would
        # warn "unknown key" on every microstep otherwise. Reattach after.
        sidecars: list[dict] = []
        tensor_batch: list[dict] = []
        for sample in batch:
            sidecar = {k: v for k, v in sample.items() if not isinstance(v, torch.Tensor)}
            sidecars.append(sidecar)
            tensor_batch.append({k: v for k, v in sample.items() if isinstance(v, torch.Tensor)})
        if fixed_bucket_n is not None:
            bucket_n = fixed_bucket_n
        else:
            n_max = max(sample["res_type"].shape[0] for sample in tensor_batch if "res_type" in sample)
            bucket_n = next_bucket(n_max)
        padded = [pad_features(sample, bucket_n=bucket_n)[0] for sample in tensor_batch]
        batch = [{**p, **s} for p, s in zip(padded, sidecars, strict=True)]

    out: dict[str, torch.Tensor] = {}
    all_keys = {k for sample in batch for k in sample}

    for key in all_keys:
        tensors_present = [(i, sample[key]) for i, sample in enumerate(batch) if key in sample]
        if not tensors_present:
            continue

        # Non-tensor fields (e.g. ``_record_id`` strings for diagnostics) are
        # passed through as a plain Python list aligned with the batch dim.
        # Missing samples get None so positional indexing stays consistent.
        if not isinstance(tensors_present[0][1], torch.Tensor):
            values: list = [None] * len(batch)
            for i, v in tensors_present:
                values[i] = v
            out[key] = values
            continue

        # When a key appears in only some samples (e.g. mask channels from
        # mixed datasets), fill missing entries with zeros so the batch dim
        # is consistent and _slice_batch_dict can slice correctly.
        # Skip scalars (0-dim) that aren't in all samples — zero-filling
        # them changes semantics (e.g. resolution=0.0 vs absent).
        if len(tensors_present) < len(batch):
            ref = tensors_present[0][1]
            if ref.dim() == 0:
                continue
            full: list[torch.Tensor] = [torch.zeros_like(ref) for _ in range(len(batch))]
            for i, t in tensors_present:
                full[i] = t
            tensors = full
        else:
            tensors = [sample[key] for sample in batch]

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
