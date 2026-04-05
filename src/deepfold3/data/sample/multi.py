"""Multi-dataset sampler with per-dataset weights and strategies.

Combines multiple training datasets (e.g. PDB + distillation) into a single
sampler that respects per-dataset sampling weights and strategies.

Used with ``torch.utils.data.ConcatDataset`` — indices are offset by each
dataset's cumulative length so the DataLoader can route to the correct
underlying dataset.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

from deepfold3.data.sample.weighted import compute_cluster_weights


class MultiDatasetSampler(Sampler[int]):
    """Sampler that draws from multiple datasets with configurable weights and strategies.

    Each dataset entry specifies:
      - ``records``: the Record list (for weighted strategy)
      - ``dataset_len``: number of items in the dataset
      - ``weight``: relative sampling probability for this dataset
      - ``strategy``: ``"weighted"`` (cluster-balanced) or ``"uniform"``

    Args:
        dataset_configs: List of dicts, each with keys:
            ``records``, ``dataset_len``, ``weight``, ``strategy``.
        epoch_size: Total samples per epoch across all datasets.
        num_replicas: DDP world size.
        rank: DDP rank.
        seed: Random seed.
        weighted_kwargs: Extra kwargs passed to ``WeightedClusterSampler``
            (e.g. ``alpha_prot``, ``beta_chain``).
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        epoch_size: int = 10000,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        **weighted_kwargs,
    ):
        self.epoch_size = epoch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.configs = dataset_configs
        self.weighted_kwargs = weighted_kwargs

        # Compute cumulative offsets for ConcatDataset indexing
        self.offsets: list[int] = []
        offset = 0
        for cfg in dataset_configs:
            self.offsets.append(offset)
            offset += cfg["dataset_len"]

        # Normalize dataset weights to probabilities
        raw_weights = np.array([cfg["weight"] for cfg in dataset_configs], dtype=np.float64)
        self.dataset_probs = raw_weights / raw_weights.sum()

        # Pre-build per-dataset sampling weights
        self._weighted_weights: list[np.ndarray | None] = []
        for cfg in dataset_configs:
            if cfg["strategy"] == "weighted":
                self._weighted_weights.append(compute_cluster_weights(cfg["records"], **weighted_kwargs))
            else:
                self._weighted_weights.append(None)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Draw epoch_size samples
        indices = np.empty(self.epoch_size, dtype=np.int64)
        # Choose which dataset for each sample
        ds_choices = rng.choice(len(self.configs), size=self.epoch_size, p=self.dataset_probs)

        for ds_idx in range(len(self.configs)):
            mask = ds_choices == ds_idx
            count = mask.sum()
            if count == 0:
                continue

            w = self._weighted_weights[ds_idx]
            if w is not None:
                # Cluster-weighted sampling
                local_indices = rng.choice(len(w), size=count, replace=True, p=w)
            else:
                # Uniform sampling
                local_indices = rng.integers(0, self.configs[ds_idx]["dataset_len"], size=count)

            # Offset to ConcatDataset global index
            indices[mask] = local_indices + self.offsets[ds_idx]

        # Shuffle to interleave datasets
        rng.shuffle(indices)

        # Shard by rank
        shard = indices[self.rank :: self.num_replicas]
        yield from shard.tolist()

    def __len__(self) -> int:
        return self.epoch_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
