"""Cluster-balanced weighted sampler for training data.

Implements AF3-style cluster-balanced sampling (§2.5.6):
  w_chain     = beta_chain / cluster_size * (alpha_prot * is_prot + alpha_nucl * is_nucl + alpha_lig * is_lig)
  w_interface = beta_interface / cluster_size * (alpha_prot * n_prot + alpha_nucl * n_nucl + alpha_lig * n_lig)

Each record's weight is the sum of its chain and interface weights, normalized
to a probability distribution. Samples are drawn with replacement.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from torch.utils.data import Sampler

from deepfold3.data import const
from deepfold3.data.types import Record


class WeightedClusterSampler(Sampler[int]):
    """Cluster-balanced sampler for training datasets.

    Compatible with ``DistributedDataParallel``: each rank draws its own
    shard of ``epoch_size // num_replicas`` samples per epoch.

    Args:
        records: List of Record objects with chain/interface metadata.
        epoch_size: Number of samples per synthetic epoch.
        alpha_prot: Weight multiplier for protein chains.
        alpha_nucl: Weight multiplier for nucleic acid chains.
        alpha_lig: Weight multiplier for ligand chains.
        beta_chain: Weight multiplier for chain-level contributions.
        beta_interface: Weight multiplier for interface-level contributions.
        num_replicas: Total number of DDP replicas.
        rank: Rank of this replica.
        seed: Random seed.
    """

    def __init__(
        self,
        records: list[Record],
        epoch_size: int = 10000,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_lig: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.records = records
        self.epoch_size = epoch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Compute per-record weights
        cluster_sizes = self._compute_cluster_sizes(records)
        weights = np.zeros(len(records), dtype=np.float64)

        for i, rec in enumerate(records):
            w = 0.0
            for chain in rec.chains:
                if not chain.valid:
                    continue
                cs = max(cluster_sizes.get(chain.cluster_id, 1), 1)
                alpha = self._mol_type_alpha(chain.mol_type, alpha_prot, alpha_nucl, alpha_lig)
                w += beta_chain * alpha / cs

            for iface in rec.interfaces:
                if not iface.valid:
                    continue
                # Interface weight: sum of mol_type alphas for both chains
                c1 = self._find_chain(rec, iface.chain_1)
                c2 = self._find_chain(rec, iface.chain_2)
                if c1 is None or c2 is None:
                    continue
                cs1 = max(cluster_sizes.get(c1.cluster_id, 1), 1)
                cs2 = max(cluster_sizes.get(c2.cluster_id, 1), 1)
                alpha_sum = self._mol_type_alpha(c1.mol_type, alpha_prot, alpha_nucl, alpha_lig) + self._mol_type_alpha(
                    c2.mol_type, alpha_prot, alpha_nucl, alpha_lig
                )
                # Use geometric mean of cluster sizes for interface
                cs = max((cs1 * cs2) ** 0.5, 1)
                w += beta_interface * alpha_sum / cs

            weights[i] = w

        # Normalize to probability distribution
        total = weights.sum()
        self.weights = weights / total if total > 0 else np.ones(len(records)) / len(records)

    @staticmethod
    def _compute_cluster_sizes(records: list[Record]) -> dict:
        """Count records per cluster_id."""
        counter: Counter = Counter()
        for rec in records:
            for chain in rec.chains:
                if chain.valid:
                    counter[chain.cluster_id] += 1
        return dict(counter)

    @staticmethod
    def _mol_type_alpha(mol_type: int, alpha_prot: float, alpha_nucl: float, alpha_lig: float) -> float:
        if mol_type == const.MOL_PROTEIN:
            return alpha_prot
        elif mol_type in (const.MOL_RNA, const.MOL_DNA):
            return alpha_nucl
        elif mol_type == const.MOL_NONPOLYMER:
            return alpha_lig
        return 1.0

    @staticmethod
    def _find_chain(rec: Record, chain_id: int):
        for c in rec.chains:
            if c.chain_id == chain_id:
                return c
        return None

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Draw epoch_size samples, then shard by rank
        indices = rng.choice(len(self.records), size=self.epoch_size, replace=True, p=self.weights)
        # Each rank gets its slice
        shard = indices[self.rank :: self.num_replicas]
        yield from shard.tolist()

    def __len__(self) -> int:
        return self.epoch_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
