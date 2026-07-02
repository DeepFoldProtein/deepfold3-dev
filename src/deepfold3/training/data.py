"""Train + eval dataloader construction.

Pure factory functions. The legacy ``_init_dataloaders`` mega-method
on ``Trainer`` conflated dataset discovery, sampler choice, multi-DS
assembly, and eval sampler logic. Splitting it makes each concern
testable in isolation.

Load-bearing side effect: when the primary dataset uses
:class:`WeightedClusterSampler`, the sampler's ``rows`` table is
attached to ``train_ds.sampling_rows`` so ``TrainingDataset.__getitem__``
can resolve each sampled index back to (record, chain/interface
anchor). This closes the AF3 §2.5.1 loop. See
``memory/feedback_chain_selection_training_time.md``.
"""

from __future__ import annotations

import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.training.distributed import DistInfo

logger = logging.getLogger(__name__)


# Bucket-padded collation keeps Triton kernel input shapes (and the
# persistent autotune cache keys) deterministic across microsteps. For
# distributed training we ALSO force ``fixed_bucket_n=cfg.max_tokens`` so
# every rank emits batches of identical N — otherwise each rank picks a
# local bucket via ``next_bucket(batch_n_max)`` and the rank with the
# longest crop stalls every per-step collective.
def _make_bucket_collate(max_tokens: int):
    return partial(collate_fn, bucket_pad=True, fixed_bucket_n=max_tokens)


def _seed_worker(worker_id: int) -> None:
    """Per-worker RNG init.

    ``torch.initial_seed()`` already derives a unique 64-bit seed per
    worker from the DataLoader's base ``generator`` seed; this just
    propagates the low 32 bits to ``random`` and ``numpy`` so all RNG
    sources a dataset/cropper might use are seeded consistently.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader_generator(base_seed: int, rank: int) -> torch.Generator:
    """Seed a ``torch.Generator`` that drives DataLoader worker seeds."""
    g = torch.Generator()
    g.manual_seed(base_seed + rank)
    return g


def _apply_cutoff(records: list, cutoff: str, *, keep_after: bool) -> list:
    """Filter records by release-date cutoff.

    ``keep_after=False`` — training pre-cutoff split: keep records
    released on or before ``cutoff``; also keep records with unknown
    release date.
    ``keep_after=True`` — eval post-cutoff split: keep records released
    strictly after ``cutoff``; drop records with unknown date.
    """
    if not cutoff:
        return records
    filtered = []
    for r in records:
        released = r.structure_info.released if r.structure_info else None
        if released is None:
            if not keep_after:
                filtered.append(r)
        elif keep_after:
            if released > cutoff:
                filtered.append(r)
        else:
            if released <= cutoff:
                filtered.append(r)
    return filtered


def build_train_sampler(records: list, train_ds: TrainingDataset, cfg: TrainingConfig, dist: DistInfo):
    """Build training sampler — cluster-weighted or plain distributed.

    For :class:`WeightedClusterSampler` also attach ``sampler.rows``
    to ``train_ds.sampling_rows`` so the dataset's ``__getitem__``
    can resolve sampled indices back to their chain / interface
    anchor — closes the AF3 §2.5.1 loop.
    """
    has_real_clusters = any(isinstance(c.cluster_id, int) for r in records for c in r.chains)

    if has_real_clusters:
        from deepfold3.data.sample.weighted import WeightedClusterSampler

        sampler = WeightedClusterSampler(
            records=records,
            epoch_size=cfg.epoch_size,
            num_replicas=dist.world_size,
            rank=dist.rank,
            seed=cfg.seed,
            infinite=True,  # single long-lived iterator (no epoch-boundary prefetch drain)
        )
        train_ds.sampling_rows = sampler.rows
        return sampler

    if dist.is_distributed:
        return DistributedSampler(
            train_ds,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=True,
            seed=cfg.seed,
        )
    return None


def build_eval_sampler(eval_ds, eval_records: list, cfg: TrainingConfig, dist: DistInfo):
    """Build eval sampler — fixed-size subset or full-coverage distributed.

    When ``eval_samples > 0`` and < full size, uses
    :class:`WeightedEvalSampler` (cluster-weighted) when
    ``eval_records`` covers the full eval dataset; otherwise falls
    back to :class:`UniformEvalSampler`. When ``eval_samples`` covers
    the full dataset (or is 0), uses a plain ``DistributedSampler``
    (or ``None`` on single GPU).
    """
    n_eval = cfg.eval_samples
    ds_len = len(eval_ds)
    if n_eval <= 0 or n_eval >= ds_len:
        if dist.is_distributed:
            return DistributedSampler(
                eval_ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, seed=cfg.seed
            )
        return None

    if eval_records and len(eval_records) >= ds_len:
        from deepfold3.data.sample.weighted import WeightedEvalSampler

        return WeightedEvalSampler(
            records=eval_records,
            n_samples=n_eval,
            num_replicas=dist.world_size,
            rank=dist.rank,
            seed=cfg.seed,
        )

    from deepfold3.data.sample.weighted import UniformEvalSampler

    return UniformEvalSampler(
        n_records=ds_len,
        n_samples=n_eval,
        num_replicas=dist.world_size,
        rank=dist.rank,
        seed=cfg.seed,
    )


def build_dataloaders(
    *,
    cfg: TrainingConfig,
    model_cfg: ModelConfig,
    dist: DistInfo,
    records: list,
    eval_records: list,
    distillation_records: list,
) -> tuple[DataLoader, DataLoader | None]:
    """Build ``(train_dl, eval_dl)``.

    Supports multi-dataset training via ``extra_datasets`` config or
    the legacy ``distillation_data_dir`` option. The primary dataset
    (``--data-dir``) always uses cluster-weighted sampling when
    cluster IDs are available; extra datasets specify their own
    strategy.
    """
    ds_kwargs = dict(
        max_tokens=cfg.max_tokens,
        max_atoms=cfg.max_atoms,
        max_msa_seqs=cfg.max_msa_seqs,
        mmcif_dir=Path(cfg.mmcif_dir) if cfg.mmcif_dir else None,
        max_templates=model_cfg.data.max_templates,
        template_cutoff_days=model_cfg.data.template_cutoff_days,
        max_template_date=model_cfg.data.max_template_date,
        prefer_monomer_templates=model_cfg.data.prefer_monomer_templates,
        max_neighborhood=model_cfg.data.max_neighborhood,
        contiguous_crop_weight=model_cfg.data.contiguous_crop_weight,
        spatial_crop_weight=model_cfg.data.spatial_crop_weight,
        interface_crop_weight=model_cfg.data.interface_crop_weight,
        interface_distance=model_cfg.data.interface_distance,
        max_chains=model_cfg.data.max_chains,
        ccd_cache_path=cfg.ccd_cache_path or None,
        pdb_chain_cache_dir=cfg.pdb_chain_cache_dir or None,
        pdb_chain_cache_mem=cfg.pdb_chain_cache_mem,
        slow_sample_log_threshold_s=cfg.slow_sample_log_threshold_s,
        # MSA pre-crop: featurizer emits a fixed [(max_recycles+1)*crop, N] MSA
        # stack so the GPU MSA shape is constant (kills allocator fragmentation).
        # crop size must match the model's; n_crops covers the max recycles used
        # by forward_training (training + eval).
        msa_crop_size=model_cfg.msa_crop_size,
        n_msa_crops=(cfg.max_recycles + 1) if cfg.msa_precrop else 0,
    )

    # Keys accepted by OF3Dataset (subset of ds_kwargs).
    of3_kwarg_names = {
        "max_tokens",
        "max_atoms",
        "max_msa_seqs",
        "max_neighborhood",
        "contiguous_crop_weight",
        "spatial_crop_weight",
        "interface_crop_weight",
        "interface_distance",
        "mmcif_dir",
        "max_templates",
        "max_template_date",
        "msa_crop_size",
        "n_msa_crops",
    }
    of3_kwargs = {k: v for k, v in ds_kwargs.items() if k in of3_kwarg_names}

    # Cache discovered records by (data_dir, format) to avoid re-scanning.
    record_cache: dict[tuple[str, str], list] = {}

    def discover(data_dir: str, fmt: str) -> list:
        key = (data_dir, fmt)
        if key not in record_cache:
            if fmt == "of3":
                from deepfold3.data.module.of3_dataset import discover_of3_records

                record_cache[key] = discover_of3_records(Path(data_dir))
            else:
                from deepfold3.data.prepare.manifest import discover_records

                record_cache[key] = discover_records(Path(data_dir))
        # Return a copy so cutoff filtering doesn't mutate the cache.
        return list(record_cache[key])

    def load_from_spec(spec, *, keep_after: bool, preloaded_records=None):
        recs = preloaded_records if preloaded_records is not None else discover(spec.data_dir, spec.format)
        if spec.cutoff_date:
            recs = _apply_cutoff(recs, spec.cutoff_date, keep_after=keep_after)
        if not recs:
            return None, []
        if spec.format == "of3":
            from deepfold3.data.module.of3_dataset import OF3Dataset

            ds = OF3Dataset(
                records=recs,
                of3_dir=Path(spec.data_dir),
                is_distillation=spec.is_distillation,
                name=spec.name,
                exclude_loss_mol_types=tuple(spec.exclude_loss_mol_types),
                **of3_kwargs,
            )
        else:
            ds = TrainingDataset(
                records=recs,
                data_dir=Path(spec.data_dir),
                is_distillation=spec.is_distillation,
                name=spec.name,
                exclude_loss_mol_types=tuple(spec.exclude_loss_mol_types),
                **ds_kwargs,
            )
        return ds, recs

    # Primary (PDB) dataset — may be empty when only extra_datasets are used.
    pdb_ds = None
    if records and cfg.data_dir:
        pdb_ds = TrainingDataset(
            records=records,
            data_dir=Path(cfg.data_dir),
            is_distillation=False,
            name="main",
            **ds_kwargs,
        )

    extra_specs = list(cfg.extra_datasets)

    # Legacy: convert distillation_data_dir → an extra_datasets entry.
    if cfg.distillation_data_dir and distillation_records and not any(s.name == "_distillation" for s in extra_specs):
        from deepfold3.model.config import DatasetConfig

        extra_specs.append(
            DatasetConfig(
                name="_distillation",
                data_dir=cfg.distillation_data_dir,
                weight=cfg.distillation_weight,
                strategy="uniform",
                is_distillation=True,
            )
        )

    if extra_specs:
        from deepfold3.data.sample.multi import MultiDatasetSampler

        datasets: list = []
        sampler_configs: list[dict] = []

        if pdb_ds is not None and len(pdb_ds) > 0:
            datasets.append(pdb_ds)
            sampler_configs.append(
                {
                    "records": records,
                    "dataset_len": len(pdb_ds),
                    "weight": 1.0,
                    "strategy": "weighted",
                }
            )

        for spec in extra_specs:
            preloaded = distillation_records if spec.name == "_distillation" else None
            ds, recs = load_from_spec(spec, keep_after=False, preloaded_records=preloaded)
            if ds is None:
                logger.warning("Extra dataset %r: no records in %s, skipping.", spec.name, spec.data_dir)
                continue
            datasets.append(ds)
            sampler_configs.append(
                {
                    "records": recs,
                    "dataset_len": len(ds),
                    "weight": spec.weight,
                    "strategy": spec.strategy,
                }
            )
            logger.info(
                "Extra dataset %r: %d records, weight=%.2f, strategy=%s, distillation=%s",
                spec.name,
                len(recs),
                spec.weight,
                spec.strategy,
                spec.is_distillation,
            )

        train_ds = ConcatDataset(datasets)
        train_sampler = MultiDatasetSampler(
            dataset_configs=sampler_configs,
            epoch_size=cfg.epoch_size,
            num_replicas=dist.world_size,
            rank=dist.rank,
            seed=cfg.seed,
            infinite=True,  # single long-lived iterator (no epoch-boundary prefetch drain)
        )
    else:
        if pdb_ds is None:
            raise RuntimeError(
                "No training data: --data-dir is empty and no extra_datasets configured. "
                "Provide --data-dir or define datasets in the config file."
            )
        train_ds = pdb_ds
        train_sampler = build_train_sampler(records, pdb_ds, cfg, dist)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_per_rank,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=_make_bucket_collate(cfg.max_tokens),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=_seed_worker,
        generator=_make_loader_generator(cfg.seed, dist.rank),
    )

    eval_dl: DataLoader | None = None
    eval_datasets: list = []

    # Primary eval records (from date-based split on primary data_dir).
    if eval_records and cfg.data_dir:
        eval_datasets.append(
            TrainingDataset(
                records=eval_records,
                data_dir=Path(cfg.data_dir),
                **ds_kwargs,
            )
        )

    for spec in cfg.eval_datasets:
        ds, recs = load_from_spec(spec, keep_after=True)
        if ds is None:
            logger.warning("Eval dataset %r: no records in %s, skipping.", spec.name, spec.data_dir)
            continue
        eval_datasets.append(ds)
        logger.info("Eval dataset %r: %d records from %s", spec.name, len(recs), spec.data_dir)

    if eval_datasets:
        eval_ds = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]
        eval_sampler = build_eval_sampler(eval_ds, eval_records, cfg, dist)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size_per_rank,
            sampler=eval_sampler,
            shuffle=False,
            collate_fn=_make_bucket_collate(cfg.max_tokens),
            num_workers=cfg.eval_num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.eval_num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.eval_num_workers > 0 else None,
            worker_init_fn=_seed_worker,
            generator=_make_loader_generator(cfg.seed, dist.rank),
        )

    return train_dl, eval_dl
