"""Hierarchical mixture sampling over the sources.

Sampling in proportion to raw dataset size would hand the model to
AFDB-multimer, which is 1.59M complexes of which ~96% are homodimers. A model
trained on that is a homodimer model. So the draw is factored, exactly as the
design note has it::

    D ~ p(D)            which source
    T ~ p(T | D)        homomeric or heteromeric, within that source
    c ~ p(c | D, T)     which cluster, uniformly
    x ~ p(x | c)        which complex in the cluster, uniformly

The last two steps are the cluster-balanced draw AF3 uses; the first two are
what keep the mixture from being decided by whichever source happens to be
largest. Every level is configurable, and ``p(T | D) = None`` keeps a source's
natural composition, which is itself an ablation arm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepfold3.complexlm.config import SourceConfig
from deepfold3.complexlm.store import ComplexStore

logger = logging.getLogger(__name__)


@dataclass
class Bucket:
    """Complex indices, optionally grouped by cluster.

    ``members`` is sorted by cluster and ``cluster_off`` indexes into it, so a
    cluster-balanced draw is two integer draws and no allocation.
    """

    members: np.ndarray
    cluster_off: np.ndarray | None

    @property
    def n_clusters(self) -> int:
        return 0 if self.cluster_off is None else int(self.cluster_off.size - 1)

    def draw(self, rng: np.random.Generator, cluster_balanced: bool) -> int:
        if cluster_balanced and self.cluster_off is not None:
            c = int(rng.integers(self.n_clusters))
            lo, hi = int(self.cluster_off[c]), int(self.cluster_off[c + 1])
            return int(self.members[rng.integers(lo, hi)])
        return int(self.members[rng.integers(self.members.size)])


def _group_by_cluster(members: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Sort ``members`` by cluster and return the CSR offsets, or ``None``.

    Returns ``None`` when the store has no usable clusters (all ``-1``), which
    is the signal to fall back to a uniform draw rather than to invent one
    cluster per complex — those are not the same distribution.
    """
    keys = clusters[members]
    if members.size == 0 or (keys < 0).all():
        return None
    order = np.argsort(keys, kind="stable")
    ordered = members[order]
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    offsets = np.concatenate(([0], boundaries, [ordered.size]))
    return ordered, offsets


def eligible_indices(
    store: ComplexStore,
    cfg: SourceConfig,
    min_chains: int,
    max_chains: int,
    exclude_clusters: set[int] | None = None,
) -> np.ndarray:
    """Complexes surviving this source's filters."""
    n = len(store)
    keep = np.ones(n, dtype=bool)
    keep &= store.cx_nchain >= min_chains
    # max_chains is enforced by the cropper too; dropping here as well keeps a
    # 200-chain assembly from being drawn only to lose 95% of itself.
    keep &= store.cx_nchain <= max_chains

    if cfg.date_max is not None or cfg.date_min is not None:
        dates = np.asarray(store.cx_date) if store.cx_date is not None else np.zeros(n, dtype=np.int32)
        dated = dates > 0
        if cfg.date_max is not None:
            keep &= dated & (dates <= cfg.date_max) | (~dated & cfg.allow_undated)
        if cfg.date_min is not None:
            keep &= dated & (dates > cfg.date_min) | (~dated & cfg.allow_undated)
        if not cfg.allow_undated and not dated.all():
            logger.info("%s: %d/%d complexes undated, dropped by the date filter", cfg.name, int((~dated).sum()), n)

    if cfg.exclude_ids:
        excluded = {line.strip() for line in Path(cfg.exclude_ids).read_text().splitlines() if line.strip()}
        if excluded:
            drop = np.array([cid in excluded for cid in store.ids], dtype=bool)
            keep &= ~drop
            logger.info("%s: excluded %d complexes listed in %s", cfg.name, int(drop.sum()), cfg.exclude_ids)

    if cfg.holdout_every:
        held = np.arange(n) % cfg.holdout_every == 0
        keep &= held if cfg.holdout_side == "eval" else ~held

    if exclude_clusters:
        clusters = np.asarray(store.cx_cluster)
        if (clusters < 0).all():
            logger.warning("%s: cluster-disjoint filter skipped, store has no cluster ids", cfg.name)
        else:
            drop = np.isin(clusters, np.fromiter(exclude_clusters, dtype=np.int64)) & (clusters >= 0)
            logger.info("%s: %d complexes share a training cluster, dropped", cfg.name, int((drop & keep).sum()))
            keep &= ~drop

    # max_complexes subsamples last, so the cap counts complexes that survived
    # every filter rather than being spent on ones about to be dropped.
    idx = np.flatnonzero(keep)
    if cfg.max_complexes is not None and idx.size > cfg.max_complexes:
        idx = idx[:: max(1, idx.size // cfg.max_complexes)][: cfg.max_complexes]
    return idx


class SourceView:
    """One source with its filters applied and its type buckets built.

    Args:
        exclude_clusters: Complex-cluster ids to drop. Used to make an
            evaluation view disjoint from training at the *cluster* level —
            a date cutoff alone does not do that (on PDB, 51% of post-cutoff
            complexes belong to a family that also appears before it, so a
            date-only holdout partly measures memorisation).
    """

    def __init__(
        self,
        store: ComplexStore,
        cfg: SourceConfig,
        min_chains: int,
        max_chains: int,
        exclude_clusters: set[int] | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.interface_masking = cfg.interface_masking and store.has_iface
        if cfg.interface_masking and not store.has_iface:
            logger.warning("%s: interface masking requested but the store has no interface flags", cfg.name)

        idx = eligible_indices(store, cfg, min_chains, max_chains)
        if exclude_clusters:
            clusters = np.asarray(store.cx_cluster)[idx]
            if (clusters < 0).all():
                logger.warning("%s: cluster-disjoint filter skipped, store has no cluster ids", cfg.name)
            else:
                drop = np.isin(clusters, np.fromiter(exclude_clusters, dtype=np.int64)) & (clusters >= 0)
                logger.info(
                    "%s: dropped %d/%d complexes sharing a training cluster", cfg.name, int(drop.sum()), idx.size
                )
                idx = idx[~drop]
        if idx.size == 0:
            raise ValueError(f"{cfg.name}: no complexes survive the filters")
        clusters = np.asarray(store.cx_cluster)

        if cfg.homo_weight is None and cfg.hetero_weight is None:
            keys, groups = ["all"], [idx]
            weights = [1.0]
        else:
            n_ent = store.n_entities()[idx]
            homo, hetero = idx[n_ent == 1], idx[n_ent > 1]
            keys, groups, weights = [], [], []
            for key, group, weight in (
                ("homo", homo, cfg.homo_weight or 0.0),
                ("hetero", hetero, cfg.hetero_weight or 0.0),
            ):
                if group.size and weight > 0:
                    keys.append(key)
                    groups.append(group)
                    weights.append(weight)
            if not keys:  # both weights zero, or one bucket empty and the other unweighted
                keys, groups, weights = ["all"], [idx], [1.0]

        self.keys = keys
        self.buckets = []
        for group in groups:
            grouped = _group_by_cluster(group, clusters)
            self.buckets.append(Bucket(group, None) if grouped is None else Bucket(*grouped))
        self.type_p = np.asarray(weights, dtype=np.float64) / float(sum(weights))
        self.n_eligible = int(idx.size)

    def draw(self, rng: np.random.Generator) -> int:
        b = int(rng.choice(len(self.buckets), p=self.type_p)) if len(self.buckets) > 1 else 0
        return self.buckets[b].draw(rng, self.cfg.cluster_balanced)

    def describe(self) -> dict:
        return {
            "source": self.cfg.name,
            "n_complexes": len(self.store),
            "n_eligible": self.n_eligible,
            "weight": self.cfg.weight,
            "interface_masking": self.interface_masking,
            "cluster_balanced": self.cfg.cluster_balanced,
            "buckets": {
                key: {"n": int(b.members.size), "n_clusters": b.n_clusters, "p": float(p)}
                for key, b, p in zip(self.keys, self.buckets, self.type_p, strict=True)
            },
        }


class MixtureSampler:
    """Draws ``(source, complex)`` pairs from the configured mixture."""

    def __init__(self, views: list[SourceView]) -> None:
        if not views:
            raise ValueError("mixture has no sources")
        self.views = views
        weights = np.array([v.cfg.weight for v in views], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("source weights must be positive")
        self.source_p = weights / weights.sum()

    def draw(self, rng: np.random.Generator) -> tuple[int, int]:
        s = int(rng.choice(len(self.views), p=self.source_p))
        return s, self.views[s].draw(rng)

    def describe(self) -> list[dict]:
        return [{**v.describe(), "p_source": float(p)} for v, p in zip(self.views, self.source_p, strict=True)]


def build_views(
    data_root: str | Path,
    sources: list[SourceConfig],
    min_chains: int,
    max_chains: int,
    exclude_clusters: dict[str, set[int]] | None = None,
) -> list[SourceView]:
    views = []
    for cfg in sources:
        store = ComplexStore(Path(data_root) / cfg.name)
        excluded = (exclude_clusters or {}).get(cluster_scope(store))
        views.append(SourceView(store, cfg, min_chains, max_chains, excluded))
        logger.info("mixture source %s", views[-1].describe())
    return views


def cluster_scope(store: ComplexStore) -> str:
    """Key under which this store's cluster ids may be compared with others.

    Complex-cluster ids are positions in one table built across every store of
    a clustering run, so they are comparable exactly within a run. A store that
    has never been clustered gets a key of its own, which makes every
    cross-store comparison a miss rather than a false match.
    """
    return store.cluster_run or f"unclustered:{store.name}"


def clusters_in_use(views: list[SourceView]) -> dict[str, set[int]]:
    """``{cluster scope: complex-cluster ids reachable in these views}``.

    Keyed by clustering run rather than by store, so an evaluation store can be
    made disjoint from a *different* training store — which is the case that
    matters: ``string_valid`` shares 18.5% of its complex clusters with
    ``string``, and ``pplm_pdb_valid`` shares 39.9% with ``pdb``.
    """
    out: dict[str, set[int]] = {}
    for view in views:
        clusters = np.asarray(view.store.cx_cluster)
        bucket = out.setdefault(cluster_scope(view.store), set())
        for b in view.buckets:
            bucket.update(clusters[b.members].tolist())
    return out
