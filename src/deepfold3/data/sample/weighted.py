"""Cluster-balanced weighted sampling for training data.

Implements AlphaFold 3 §2.5.1 cluster-balanced sampling as a **two-stage
decomposition** of the per-anchor joint distribution ``p(record, anchor) ∝ w_a``:

1. **Record marginal** (the sampler): draw a record with probability
   ``p(record) = Σ_{a∈record} w_a / W`` — exactly :func:`compute_cluster_weights`.
2. **Anchor conditional** (the dataset, at load time): given the record, draw a
   chain/interface anchor with probability ``p(a | record) = w_a / Σ_{a∈record} w_a``
   — from :func:`build_record_anchor_tables`.

By the chain rule ``p(record)·p(a|record) = w_a / W``, so this reproduces the
AF3 §2.5.1 flat per-anchor table's joint **exactly**, while keeping the sampler
per-record (DDP-simple) and moving the anchor choice next to the cropper that
consumes it. Per-anchor weights:

* per **valid chain**   ``w_chain     = beta_chain / N_clust * alpha(mol_type)``
* per **valid interface** ``w_interface = beta_interface / N_clust *
  (alpha(c1.mol_type) + alpha(c2.mol_type))``

For chains, ``N_clust`` is the number of valid chains sharing the same chain
cluster ID. For interfaces, ``N_clust`` is the number of valid interfaces
sharing the same **interface cluster ID**, defined per AF3 §2.5.3 as the
(unordered) pair of constituent chain cluster IDs
``C_interface = frozenset({C_chain_1, C_chain_2})``.

The anchor a record draws seeds the crop (chain → contiguous/spatial seed pool,
interface → interface-seeded spatial crop). A 20-chain multimer still earns
~Σ-of-anchors draw mass and, across draws, sees each chain/interface in
proportion to its cluster-balanced weight — instead of collapsing to a single
per-record weight the cropper then self-seeds with a size bias.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from torch.utils.data import Sampler

from deepfold3.data import const
from deepfold3.data.types import Record

# A crop anchor threaded from the record's anchor draw into the cropper:
# ``("chain", chain_id, -1)`` or ``("interface", chain_1_id, chain_2_id)``.
CropAnchor = tuple[str, int, int]


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


def _record_anchor_rows(
    rec: Record,
    chain_cluster_sizes: dict,
    iface_cluster_sizes: dict,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_lig: float,
    beta_chain: float,
    beta_interface: float,
) -> tuple[list[CropAnchor], list[float]]:
    """One record's ``(anchors, weights)`` — the shared AF3 §2.5.1 weight core.

    ``anchors`` is a list of :data:`CropAnchor` tuples (one per valid chain and
    valid interface); ``weights`` are the matching per-anchor ``w_a`` (chain →
    ``beta_chain·alpha/N_clust``, interface → ``beta_interface·(alpha1+alpha2)/N_clust``).
    Used both to build the per-record anchor conditional
    (:func:`build_record_anchor_tables`) and to aggregate the record marginal
    (:func:`compute_cluster_weights`).
    """
    # chain_map only resolves interface endpoints — skip it for monomers.
    chain_map = {c.chain_id: c for c in rec.chains} if rec.interfaces else {}
    anchors: list[CropAnchor] = []
    weights: list[float] = []

    for chain in rec.chains:
        if not chain.valid:
            continue
        cs = max(chain_cluster_sizes.get(chain.cluster_id, 1), 1)
        alpha = _mol_type_alpha(chain.mol_type, alpha_prot, alpha_nucl, alpha_lig)
        w = beta_chain * alpha / cs
        if w > 0:
            anchors.append(("chain", int(chain.chain_id), -1))
            weights.append(w)

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
            anchors.append(("interface", int(c1.chain_id), int(c2.chain_id)))
            weights.append(w)

    return anchors, weights


def build_record_anchor_tables(
    records: list[Record],
    alpha_prot: float = 3.0,
    alpha_nucl: float = 3.0,
    alpha_lig: float = 1.0,
    beta_chain: float = 0.5,
    beta_interface: float = 1.0,
) -> dict[int, tuple[list[CropAnchor], list[float]]]:
    """Per-record anchor conditional for load-time (two-stage) anchor sampling.

    Returns ``{record_idx: (anchors, weights)}`` where ``anchors`` is a list of
    :data:`CropAnchor` tuples and ``weights`` the matching per-anchor ``w_a``
    (unnormalised — ``random.choices`` renormalises). Records with no valid
    anchor are **omitted**, so the caller leaves ``crop_anchor=None`` and the
    cropper self-seeds.

    The weights are the same ``w_a`` used by the flat AF3 §2.5.1 table, so
    drawing a record by :func:`compute_cluster_weights` (marginal) and then an
    anchor from this table (conditional) reproduces the flat table's joint
    ``p(record, anchor) ∝ w_a`` exactly. Must use the **same** ``alpha_*`` /
    ``beta_*`` as the record-marginal sampler for the identity to hold.
    """
    chain_cluster_sizes = _compute_chain_cluster_sizes(records)
    iface_cluster_sizes = _compute_interface_cluster_sizes(records)
    tables: dict[int, tuple[list[CropAnchor], list[float]]] = {}

    for record_idx, rec in enumerate(records):
        anchors, weights = _record_anchor_rows(
            rec,
            chain_cluster_sizes,
            iface_cluster_sizes,
            alpha_prot,
            alpha_nucl,
            alpha_lig,
            beta_chain,
            beta_interface,
        )
        if anchors:
            tables[record_idx] = (anchors, weights)

    return tables


def compute_cluster_weights(
    records: list[Record],
    alpha_prot: float = 3.0,
    alpha_nucl: float = 3.0,
    alpha_lig: float = 1.0,
    beta_chain: float = 0.5,
    beta_interface: float = 1.0,
) -> np.ndarray:
    """Per-record draw distribution — the AF3 §2.5.1 record marginal.

    Each record's weight is ``Σ_{a∈record} w_a`` (the aggregate of its chain +
    interface anchor weights), so drawing a record here and then an anchor from
    :func:`build_record_anchor_tables` reproduces the flat per-anchor joint
    exactly (chain rule). Returns a normalised distribution of length
    ``len(records)``; records with no valid anchor get weight 0.
    """
    chain_cluster_sizes = _compute_chain_cluster_sizes(records)
    iface_cluster_sizes = _compute_interface_cluster_sizes(records)
    weights = np.zeros(len(records), dtype=np.float64)

    for i, rec in enumerate(records):
        _, w_list = _record_anchor_rows(
            rec,
            chain_cluster_sizes,
            iface_cluster_sizes,
            alpha_prot,
            alpha_nucl,
            alpha_lig,
            beta_chain,
            beta_interface,
        )
        weights[i] = float(sum(w_list))
    total = weights.sum()
    return weights / total if total > 0 else np.ones(len(records)) / len(records)


class WeightedClusterSampler(Sampler[int]):
    """Cluster-balanced sampler that yields **record indices** (AF3 §2.5.1 marginal).

    Draws records with probability ``∝`` :func:`compute_cluster_weights` (the
    per-record aggregate of chain + interface anchor weights). The chain /
    interface anchor that seeds the crop is drawn **at load time** inside the
    dataset via :func:`build_record_anchor_tables` — the two-stage decomposition
    that reproduces the flat per-anchor joint exactly (see module docstring),
    so the sampler no longer carries a flat row table. Compatible with DDP —
    each rank takes ``indices[rank::num_replicas]``.

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
        epoch_size: int = 100_000,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_lig: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        infinite: bool = False,
    ):
        self.records = records
        self.epoch_size = epoch_size
        self.infinite = infinite
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        # Within-epoch position. Advanced once per yielded index. Reset
        # by ``set_epoch`` only when the epoch number actually changes,
        # so a resumed sampler whose state was loaded after construction
        # is not clobbered by the trainer's ``set_epoch(state.epoch)``
        # at the top of the loop.
        self.sample_idx = 0

        # Per-record marginal p(record) ∝ Σ_{a∈record} w_a; ``len(self.weights)``
        # is the record count (the draw population).
        self.weights = compute_cluster_weights(
            records,
            alpha_prot=alpha_prot,
            alpha_nucl=alpha_nucl,
            alpha_lig=alpha_lig,
            beta_chain=beta_chain,
            beta_interface=beta_interface,
        )

    def __iter__(self):
        if not self.infinite:
            # Finite epoch (legacy): one epoch_size draw, sharded, resumable.
            # The trainer recreates this iterator each epoch — at the boundary
            # the prefetch buffer drains, so the first batch of the new epoch
            # waits (a ``data_wait`` spike). Use ``infinite=True`` to avoid it.
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.choice(len(self.weights), size=self.epoch_size, replace=True, p=self.weights)
            shard = indices[self.rank :: self.num_replicas]
            for i in range(self.sample_idx, len(shard)):
                self.sample_idx = i + 1
                yield int(shard[i])
            # Iterator exhausted — leave ``sample_idx`` at len(shard) so a
            # state_dict captured after the iterator drained reads as
            # "epoch fully consumed". ``set_epoch`` resets it on the next
            # outer-loop tick.
            return

        # Infinite: never raises StopIteration. The trainer creates a single
        # long-lived iterator and pulls forever, so workers keep prefetching
        # across "epoch" reseeds with no buffer drain — no epoch-boundary
        # data_wait spike. Because sampling is with-replacement, an "epoch" is
        # just an arbitrary ``epoch_size`` chunk; we reseed per chunk for draw
        # variety and advance ``self.epoch`` purely as a logging / resume
        # counter. Resume picks up from the seeded ``self.epoch``/``sample_idx``.
        epoch = self.epoch
        pos = self.sample_idx
        while True:
            rng = np.random.default_rng(self.seed + epoch)
            indices = rng.choice(len(self.weights), size=self.epoch_size, replace=True, p=self.weights)
            shard = indices[self.rank :: self.num_replicas]
            while pos < len(shard):
                self.epoch = epoch
                self.sample_idx = pos + 1
                yield int(shard[pos])
                pos += 1
            epoch += 1
            pos = 0

    def __len__(self) -> int:
        return self.epoch_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.sample_idx = 0
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Snapshot for checkpoint resume.

        Captures epoch + within-epoch position so the iterator can
        fast-forward past indices that were already trained on. Saving
        ``seed`` makes resume catch the same-seed mismatch loud.
        """
        return {"epoch": self.epoch, "sample_idx": self.sample_idx, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        if "seed" in state and state["seed"] != self.seed:
            # Seed change is allowed (different runs, different shuffles)
            # but signal it so a silent divergence isn't surprising later.
            import warnings

            warnings.warn(
                f"WeightedClusterSampler resume with seed={self.seed} but ckpt sampler had "
                f"seed={state['seed']}; sample sequence will diverge.",
                stacklevel=2,
            )
        self.epoch = state["epoch"]
        self.sample_idx = state.get("sample_idx", 0)


class WeightedEvalSampler(Sampler[int]):
    """Cluster-weighted fixed-size eval sampler.

    Each ``set_epoch`` / ``__iter__`` draws ``n_samples`` record indices
    **without replacement** using per-record cluster weights from
    :func:`compute_cluster_weights`. This avoids evaluating the entire
    (potentially huge) eval set while still covering sequence families
    proportionally.

    Yielded indices are plain record indices and eval datasets skip the
    load-time anchor draw — eval doesn't need per-anchor crop control.

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
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.weights = compute_cluster_weights(records)
        # rng.choice(replace=False, p=weights) raises when fewer records have
        # nonzero weight than the requested count (records with no valid chain
        # get weight 0 from compute_cluster_weights). Cap at the drawable count.
        n_nonzero = int((self.weights > 0).sum())
        self.n_samples = min(n_samples, self.n_records, n_nonzero)

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
