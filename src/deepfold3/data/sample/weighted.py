"""Cluster-balanced weighted sampler for training data.

Implements AlphaFold 3 §2.5.1 cluster-balanced sampling. The sampling
table is **flattened** so each ``(record, anchor)`` candidate is its own row:

* one row per **valid chain** of every record
  ``w_chain     = beta_chain / N_clust * alpha(mol_type)``
* one row per **valid interface** of every record
  ``w_interface = beta_interface / N_clust * (alpha(c1.mol_type) + alpha(c2.mol_type))``

For chains, ``N_clust`` is the number of valid chains sharing the same
chain cluster ID.

For interfaces, ``N_clust`` is the number of valid interfaces sharing
the same **interface cluster ID**, defined per AF3 §2.5.3 as the
(unordered) pair of constituent chain cluster IDs:
``C_interface = frozenset({C_chain_1, C_chain_2})``.

Sampling a row tells the dataloader BOTH which record to load AND which
chain/interface should anchor the crop. This is critical: a 20-chain
multimer should generate ~20 chain-anchor draws and many interface
draws so each chain/interface gets training time, instead of being
collapsed into a single per-record weight that the cropper then ignores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Sampler

from deepfold3.data import const
from deepfold3.data.types import Record


@dataclass(frozen=True)
class SamplingRow:
    """One row in the flattened ``(record, anchor)`` sampling table.

    ``anchor_kind`` is ``"chain"`` (anchor_a is the chain id, anchor_b is
    -1) or ``"interface"`` (anchor_a, anchor_b are the two chain ids).
    The dataloader uses ``anchor_kind`` + ``anchor_a`` (+ ``anchor_b``)
    to seed the cropper.
    """

    record_idx: int
    anchor_kind: str  # "chain" | "interface"
    anchor_a: int
    anchor_b: int = -1


def _mol_type_alpha(mol_type: int, alpha_prot: float, alpha_nucl: float, alpha_lig: float) -> float:
    if mol_type == const.MOL_PROTEIN:
        return alpha_prot
    elif mol_type in (const.MOL_RNA, const.MOL_DNA):
        return alpha_nucl
    elif mol_type == const.MOL_NONPOLYMER:
        return alpha_lig
    return 1.0


def _compute_chain_cluster_sizes(records: list[Record]) -> dict:
    """Count valid chains per chain cluster ID."""
    counter: Counter = Counter()
    for rec in records:
        for chain in rec.chains:
            if chain.valid:
                counter[chain.cluster_id] += 1
    return dict(counter)


def _interface_cluster_id(c1_cluster, c2_cluster):
    """Build an interface cluster ID from two chain cluster IDs (AF3 §2.5.3).

    Interface clustering is a join on chain cluster pairs: interfaces share
    a cluster iff their constituent chain pairs map to the same pair of chain
    clusters (unordered).
    """
    return frozenset((c1_cluster, c2_cluster))


def _compute_interface_cluster_sizes(records: list[Record]) -> dict:
    """Count valid interfaces per interface cluster ID.

    The interface cluster ID is the unordered pair of constituent chain
    cluster IDs, per AF3 §2.5.3.
    """
    counter: Counter = Counter()
    for rec in records:
        chain_map = {c.chain_id: c for c in rec.chains}
        for iface in rec.interfaces:
            if not iface.valid:
                continue
            c1 = chain_map.get(iface.chain_1)
            c2 = chain_map.get(iface.chain_2)
            if c1 is None or c2 is None or not c1.valid or not c2.valid:
                continue
            icid = _interface_cluster_id(c1.cluster_id, c2.cluster_id)
            counter[icid] += 1
    return dict(counter)


def build_sampling_rows(
    records: list[Record],
    alpha_prot: float = 3.0,
    alpha_nucl: float = 3.0,
    alpha_lig: float = 1.0,
    beta_chain: float = 0.5,
    beta_interface: float = 1.0,
) -> tuple[list[SamplingRow], np.ndarray]:
    """Flatten records into one sampling row per chain + interface.

    Returns ``(rows, weights)`` where ``weights`` is a normalised
    probability distribution of length ``len(rows)``. Records with no
    valid chains contribute no rows.

    Chain cluster sizes are counted across all valid chains. Interface
    cluster sizes are counted across all valid interfaces, using the
    AF3 §2.5.3 interface cluster ID = unordered pair of chain cluster IDs.
    """
    chain_cluster_sizes = _compute_chain_cluster_sizes(records)
    iface_cluster_sizes = _compute_interface_cluster_sizes(records)
    rows: list[SamplingRow] = []
    weight_list: list[float] = []

    for record_idx, rec in enumerate(records):
        chain_map = {c.chain_id: c for c in rec.chains}

        for chain in rec.chains:
            if not chain.valid:
                continue
            cs = max(chain_cluster_sizes.get(chain.cluster_id, 1), 1)
            alpha = _mol_type_alpha(chain.mol_type, alpha_prot, alpha_nucl, alpha_lig)
            w = beta_chain * alpha / cs
            if w > 0:
                rows.append(SamplingRow(record_idx, "chain", chain.chain_id))
                weight_list.append(w)

        for iface in rec.interfaces:
            if not iface.valid:
                continue
            c1 = chain_map.get(iface.chain_1)
            c2 = chain_map.get(iface.chain_2)
            if c1 is None or c2 is None or not c1.valid or not c2.valid:
                continue
            alpha_sum = _mol_type_alpha(c1.mol_type, alpha_prot, alpha_nucl, alpha_lig) + _mol_type_alpha(
                c2.mol_type, alpha_prot, alpha_nucl, alpha_lig
            )
            # AF3 §2.5.3: N_clust = size of interface cluster (pair of chain clusters)
            icid = _interface_cluster_id(c1.cluster_id, c2.cluster_id)
            cs = max(iface_cluster_sizes.get(icid, 1), 1)
            w = beta_interface * alpha_sum / cs
            if w > 0:
                rows.append(SamplingRow(record_idx, "interface", c1.chain_id, c2.chain_id))
                weight_list.append(w)

    if not rows:
        # Degenerate fallback: one uniform row per record so the dataloader
        # can still iterate (e.g. tests with no valid chains).
        rows = [SamplingRow(i, "chain", -1) for i in range(len(records))]
        weight_list = [1.0] * len(records)

    weights = np.asarray(weight_list, dtype=np.float64)
    total = weights.sum()
    return rows, weights / total if total > 0 else np.ones(len(rows)) / len(rows)


def compute_cluster_weights(
    records: list[Record],
    alpha_prot: float = 3.0,
    alpha_nucl: float = 3.0,
    alpha_lig: float = 1.0,
    beta_chain: float = 0.5,
    beta_interface: float = 1.0,
) -> np.ndarray:
    """Per-record weights (legacy helper).

    Aggregates the chain + interface weights of every record into a single
    record-level draw probability. Kept for backward compatibility with
    callers that don't need the per-anchor flat layout. New code should
    use :func:`build_sampling_rows` instead.
    """
    chain_cluster_sizes = _compute_chain_cluster_sizes(records)
    iface_cluster_sizes = _compute_interface_cluster_sizes(records)
    weights = np.zeros(len(records), dtype=np.float64)

    for i, rec in enumerate(records):
        w = 0.0
        chain_map = {c.chain_id: c for c in rec.chains}
        for chain in rec.chains:
            if not chain.valid:
                continue
            cs = max(chain_cluster_sizes.get(chain.cluster_id, 1), 1)
            alpha = _mol_type_alpha(chain.mol_type, alpha_prot, alpha_nucl, alpha_lig)
            w += beta_chain * alpha / cs
        for iface in rec.interfaces:
            if not iface.valid:
                continue
            c1 = chain_map.get(iface.chain_1)
            c2 = chain_map.get(iface.chain_2)
            if c1 is None or c2 is None or not c1.valid or not c2.valid:
                continue
            alpha_sum = _mol_type_alpha(c1.mol_type, alpha_prot, alpha_nucl, alpha_lig) + _mol_type_alpha(
                c2.mol_type, alpha_prot, alpha_nucl, alpha_lig
            )
            icid = _interface_cluster_id(c1.cluster_id, c2.cluster_id)
            cs = max(iface_cluster_sizes.get(icid, 1), 1)
            w += beta_interface * alpha_sum / cs
        weights[i] = w
    total = weights.sum()
    return weights / total if total > 0 else np.ones(len(records)) / len(records)


class WeightedClusterSampler(Sampler[int]):
    """Cluster-balanced sampler that yields **flat sampling row indices**.

    The yielded ``int`` is an index into ``self.rows``; the dataset is
    expected to look up ``rows[idx]`` and dispatch the cropper using the
    row's ``anchor_kind`` / ``anchor_a`` / ``anchor_b``. Compatible with
    DDP — each rank takes ``indices[rank::num_replicas]``.

    Args:
        records: List of Record objects with chain/interface metadata.
        epoch_size: Number of samples per synthetic epoch.
        alpha_prot, alpha_nucl, alpha_lig: Molecule-type weight multipliers.
        beta_chain, beta_interface: Chain vs interface weight balance.
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

        self.rows, self.weights = build_sampling_rows(
            records,
            alpha_prot=alpha_prot,
            alpha_nucl=alpha_nucl,
            alpha_lig=alpha_lig,
            beta_chain=beta_chain,
            beta_interface=beta_interface,
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(len(self.rows), size=self.epoch_size, replace=True, p=self.weights)
        shard = indices[self.rank :: self.num_replicas]
        yield from shard.tolist()

    def __len__(self) -> int:
        return self.epoch_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class WeightedEvalSampler(Sampler[int]):
    """Cluster-weighted fixed-size eval sampler.

    Each ``set_epoch`` / ``__iter__`` draws ``n_samples`` record indices
    **without replacement** using per-record cluster weights from
    :func:`compute_cluster_weights`. This avoids evaluating the entire
    (potentially huge) eval set while still covering sequence families
    proportionally.

    Yielded indices are plain record indices (no ``sampling_rows``
    indirection) since eval doesn't need per-anchor crop control.

    Compatible with DDP — each rank takes ``indices[rank::num_replicas]``.
    """

    def __init__(
        self,
        records: list[Record],
        n_samples: int = 50,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.n_records = len(records)
        self.n_samples = min(n_samples, self.n_records)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.weights = compute_cluster_weights(records)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(self.n_records, size=self.n_samples, replace=False, p=self.weights)
        shard = indices[self.rank :: self.num_replicas]
        yield from shard.tolist()

    def __len__(self) -> int:
        return -(-self.n_samples // self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class UniformEvalSampler(Sampler[int]):
    """Uniform random fixed-size eval sampler (no cluster weights).

    Like :class:`WeightedEvalSampler` but draws indices uniformly — used
    when per-record cluster weights are not available (e.g. YAML
    ``eval_datasets`` or ``ConcatDataset``).
    """

    def __init__(
        self,
        n_records: int,
        n_samples: int = 50,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.n_records = n_records
        self.n_samples = min(n_samples, n_records)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(self.n_records, size=self.n_samples, replace=False)
        shard = indices[self.rank :: self.num_replicas]
        yield from shard.tolist()

    def __len__(self) -> int:
        return -(-self.n_samples // self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
