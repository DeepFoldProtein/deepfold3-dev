"""From stores to packed batches.

One example is built by: draw a complex -> permute its chains -> crop to the
length budgets -> mask. Examples are then concatenated into a flat token axis
until the token budget is reached, and that is the batch. There is no padding
anywhere, so a batch of one 4000-residue assembly and a batch of forty
100-residue dimers cost the same.

Chain order is permuted on every draw. The order chains appear in a PDB entry
is an artefact of deposition, and the model is supposed to represent a *set*;
permuting is both the augmentation that discourages order-dependence and the
thing that makes the permutation-sensitivity diagnostic meaningful — a model
trained on one canonical order would have no reason to be invariant.

Evaluation is deterministic in a specific way: the mask of complex *i* is
seeded from *i* alone, so two checkpoints, two architectures, or two runs are
scored on exactly the same masked positions. Without that, the difference
between two models on a 20-batch eval is dominated by which residues got
masked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from deepfold3.complexlm.batch import IGNORE_INDEX, PackedBatch
from deepfold3.complexlm.config import DataConfig
from deepfold3.complexlm.crop import crop_complex
from deepfold3.complexlm.masking import apply_masking
from deepfold3.complexlm.sampler import MixtureSampler, SourceView

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """One cropped, masked complex, still as a list of chains."""

    tokens: list[np.ndarray]
    labels: list[np.ndarray]
    iface: list[np.ndarray]
    entities: np.ndarray
    source: int
    index: int
    #: Whether the source carries interface labels at all. A sequence-only
    #: source gets all-zero flags, which is not the same statement as "none of
    #: these residues is at an interface".
    iface_known: bool = False

    @property
    def n_res(self) -> int:
        return int(sum(t.size for t in self.tokens))


def build_example(
    view: SourceView,
    index: int,
    rng: np.random.Generator,
    cfg: DataConfig,
    permute: bool | None = None,
) -> Example | None:
    """Draw complex ``index`` from ``view`` and turn it into an example."""
    stored = view.store.example(index)
    order = np.arange(stored.n_chains)
    if cfg.permute_chains if permute is None else permute:
        rng.shuffle(order)

    seqs = [stored.seqs[i] for i in order]
    iface = [stored.iface[i] for i in order] if stored.iface is not None else None
    entities = stored.chain_rows[order]

    spec = crop_complex(
        rng,
        [s.size for s in seqs],
        entities,
        iface,
        max_chain_len=cfg.max_chain_len,
        max_complex_len=cfg.max_complex_len,
        max_chains=cfg.max_chains,
    )
    if not spec.slots:
        return None
    seqs = spec.apply(seqs)
    iface_cropped = spec.apply(iface) if iface is not None else None
    entities = entities[spec.slots]

    tokens, labels = apply_masking(rng, seqs, iface_cropped, entities, cfg.masking, view.interface_masking)
    flags = iface_cropped or [np.zeros(s.size, dtype=np.uint8) for s in seqs]
    return Example(tokens, labels, flags, entities, view_index(view), index, iface_cropped is not None)


def view_index(view: SourceView) -> int:
    """Source id stamped into the batch; assigned by :func:`assign_source_ids`."""
    return getattr(view, "_source_id", 0)


def assign_source_ids(views: list[SourceView]) -> None:
    for i, view in enumerate(views):
        view._source_id = i


def pack(examples: list[Example]) -> PackedBatch:
    """Concatenate examples into one token axis with both cu_seqlens vectors."""
    tokens, labels, positions, global_positions = [], [], [], []
    chain_lens: list[int] = []
    complex_lens: list[int] = []
    chain_index, complex_index, entity_index, iface, iface_known = [], [], [], [], []
    sources: list[int] = []
    n_chain = 0
    n_entity = 0

    for ci, ex in enumerate(examples):
        entity_ids, _ = np.unique(ex.entities, return_inverse=True)
        remap = {int(e): n_entity + i for i, e in enumerate(entity_ids)}
        offset = 0
        for slot, (tok, lab) in enumerate(zip(ex.tokens, ex.labels, strict=True)):
            length = tok.size
            tokens.append(tok)
            labels.append(lab)
            positions.append(np.arange(length, dtype=np.int32))
            global_positions.append(np.arange(offset, offset + length, dtype=np.int32))
            chain_index.append(np.full(length, n_chain, dtype=np.int32))
            complex_index.append(np.full(length, ci, dtype=np.int32))
            entity_index.append(np.full(length, remap[int(ex.entities[slot])], dtype=np.int32))
            iface.append(ex.iface[slot])
            iface_known.append(np.full(length, ex.iface_known, dtype=bool))
            chain_lens.append(length)
            n_chain += 1
            offset += length
        n_entity += len(entity_ids)
        complex_lens.append(offset)
        sources.append(ex.source)

    def cumulative(lengths: list[int]) -> torch.Tensor:
        out = np.zeros(len(lengths) + 1, dtype=np.int32)
        np.cumsum(lengths, out=out[1:])
        return torch.from_numpy(out)

    label = torch.from_numpy(np.concatenate(labels))
    return PackedBatch(
        tokens=torch.from_numpy(np.concatenate(tokens)),
        labels=torch.where(label < 0, torch.full_like(label, IGNORE_INDEX), label),
        positions=torch.from_numpy(np.concatenate(positions)),
        global_positions=torch.from_numpy(np.concatenate(global_positions)),
        chain_cu=cumulative(chain_lens),
        complex_cu=cumulative(complex_lens),
        chain_index=torch.from_numpy(np.concatenate(chain_index)),
        complex_index=torch.from_numpy(np.concatenate(complex_index)),
        entity_index=torch.from_numpy(np.concatenate(entity_index)),
        iface=torch.from_numpy(np.concatenate(iface)).bool(),
        iface_known=torch.from_numpy(np.concatenate(iface_known)),
        source=torch.tensor(sources, dtype=torch.int32),
        max_chain_len=max(chain_lens),
        max_complex_len=max(complex_lens),
    )


class PackedComplexStream(IterableDataset):
    """Yields :class:`~deepfold3.complexlm.batch.PackedBatch` objects.

    Args:
        views: Sources, already filtered.
        cfg: Data configuration; ``token_budget`` sets the batch size.
        sampler: Mixture sampler for an infinite training stream. Mutually
            exclusive with ``plan``.
        plan: Explicit ``(view, complex)`` list for a finite, reproducible
            evaluation stream.
        rank / world_size: Used only to decorrelate the per-worker seeds. The
            stream is i.i.d., so sharding by rank is unnecessary — every rank
            drawing independently from the same mixture *is* the shard.
    """

    def __init__(
        self,
        views: list[SourceView],
        cfg: DataConfig,
        sampler: MixtureSampler | None = None,
        plan: list[tuple[int, int]] | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        if (sampler is None) == (plan is None):
            raise ValueError("pass exactly one of sampler= or plan=")
        if cfg.max_complex_len > cfg.token_budget:
            raise ValueError(f"token_budget={cfg.token_budget} < max_complex_len={cfg.max_complex_len}")
        assign_source_ids(views)
        self.views = views
        self.cfg = cfg
        self.sampler = sampler
        self.plan = plan
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def _iter_examples(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        n_workers = info.num_workers if info else 1
        if self.sampler is not None:
            rng = np.random.default_rng([self.seed, self.rank, worker_id])
            while True:
                view_idx, index = self.sampler.draw(rng)
                example = build_example(self.views[view_idx], index, rng, self.cfg)
                if example is not None:
                    yield example
        else:
            for step, (view_idx, index) in enumerate(self.plan):
                if step % n_workers != worker_id:
                    continue
                # Seeded per complex, not per worker: the same complex gets the
                # same mask under every model and every worker count.
                rng = np.random.default_rng([self.seed, view_idx, index])
                example = build_example(self.views[view_idx], index, rng, self.cfg, permute=False)
                if example is not None:
                    yield example

    def __iter__(self):
        buffer: list[Example] = []
        used = 0
        for example in self._iter_examples():
            if buffer and used + example.n_res > self.cfg.token_budget:
                yield pack(buffer)
                buffer, used = [], 0
            buffer.append(example)
            used += example.n_res
        if buffer:
            yield pack(buffer)


def evaluation_plan(views: list[SourceView], n_per_source: int, seed: int = 0) -> list[tuple[int, int]]:
    """A fixed, evenly spaced sample of each source's eligible complexes.

    Evenly spaced rather than randomly drawn so that the plan is stable when a
    store is rebuilt with more complexes: it stays a deterministic function of
    the store's order, and the same complexes appear in every run.
    """
    plan: list[tuple[int, int]] = []
    for v, view in enumerate(views):
        members = np.concatenate([b.members for b in view.buckets])
        members = np.unique(members)
        if members.size > n_per_source:
            members = members[np.linspace(0, members.size - 1, n_per_source).astype(np.int64)]
        plan.extend((v, int(i)) for i in members)
    return plan


def make_dataloader(stream: PackedComplexStream, cfg: DataConfig) -> DataLoader:
    """A dataloader over already-batched objects (``batch_size=None``)."""
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
        persistent_workers=bool(cfg.num_workers),
    )
