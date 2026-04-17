"""DDP trainer for DeepFold3.

Native PyTorch ``DistributedDataParallel`` training loop following
Protenix's procedure with Boltz-style code organization.  No Lightning
dependency.

Example::

    trainer = Trainer(model_cfg, train_cfg)
    trainer.run()
"""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import random
import re
import signal
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import fp32_guard
from deepfold3.training.ema import EMA
from deepfold3.training.loss import training_loss
from deepfold3.training.scheduler import AF3LRScheduler, build_scheduler

# Bucket-padded collation keeps Triton kernel input shapes (and the
# persistent autotune cache keys) deterministic across microsteps.
_collate_bucket_pad = partial(collate_fn, bucket_pad=True)

logger = logging.getLogger(__name__)


def _slice_batch_dict(d: dict, b: int, batch_size: int) -> dict:
    """Slice each tensor value in ``d`` along dim 0 at index ``b``.

    Only tensors whose leading dim equals ``batch_size`` are sliced.
    Shared weight tensors (e.g. ``pae_weight: [num_bins, c_pair]``),
    pre-aggregated scalars (``__diffusion_total__``), and other shapes
    that don't carry the batch dim pass through unchanged. Sub-dicts
    are recursed into.

    Used by the per-batch loss loop in ``_training_step`` when
    ``batch_size_per_rank > 1``.
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.dim() >= 1 and v.shape[0] == batch_size:
                out[k] = v[b]
            else:
                out[k] = v
        elif isinstance(v, dict):
            out[k] = _slice_batch_dict(v, b, batch_size)
        else:
            out[k] = v
    return out


def _yaml_safe(value):
    """Recursively convert tuples to lists so the result is `yaml.safe_dump` clean."""
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    return value


def serialize_run_config(train_cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Build a round-trippable resolved-config dict.

    Mirrors the input YAML layout accepted by :func:`deepfold3.config_loader.load_config`:
    top-level ``training``, ``model``, optional ``datasets`` and ``eval_datasets``.
    The result can be re-fed via ``--config`` to reproduce the run.
    """
    train_dict = _yaml_safe(dataclasses.asdict(train_cfg))
    extras = train_dict.pop("extra_datasets", []) or []
    evals = train_dict.pop("eval_datasets", []) or []
    out: dict = {
        "training": train_dict,
        "model": _yaml_safe(dataclasses.asdict(model_cfg)),
    }
    if extras:
        out["datasets"] = extras
    if evals:
        out["eval_datasets"] = evals
    return out


class DistributedAverageMeters:
    """Group of named metrics with deferred DDP all-reduce.

    Per-step ``update`` is collective-free — values accumulate locally as GPU
    tensors. ``sync`` packs all metric sums + a shared count into a single
    f64 tensor and does ONE ``all_reduce`` to compute the global weighted
    average across ranks. The "all-reduce only at log time" pattern avoids
    the per-step sync barrier that would otherwise serialize fast and slow
    ranks at every step.

    Important: ``sync`` MUST be called by every rank in lockstep (it issues a
    collective). Place the call right after the optimizer step, where DDP's
    own gradient all-reduce has already synchronized the ranks for free —
    this makes the meter's all-reduce essentially zero-cost.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}  # f64 device tensors
        self._count: int = 0

    def update(self, values: dict[str, torch.Tensor], n: int = 1) -> None:
        """Accumulate a step's metrics. Tensors are detached and stored on device."""
        for k, v in values.items():
            # Squeeze to scalar — with batch_per_rank > 1 the per-batch
            # loss loop may produce 0-d or 1-element tensors.
            v_f64 = v.detach().to(torch.float64).squeeze() * n
            if k in self._sums:
                self._sums[k] += v_f64
            else:
                self._sums[k] = v_f64.clone()
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    def sync(self) -> tuple[dict[str, float], dict[str, float]]:
        """All-reduce across ranks and return ``(averages, sums)``.

        Both dicts share the same keys. ``averages`` divides by global count
        (for losses). ``sums`` gives the raw global totals (for counters like
        token counts that should not be averaged).
        """
        if not self._sums:
            return {}, {}
        keys = sorted(self._sums.keys())
        # Pack [sum_k0, sum_k1, ..., sum_kN-1, count] into one f64 tensor.
        packed = torch.empty(len(keys) + 1, dtype=torch.float64, device=self.device)
        for i, k in enumerate(keys):
            packed[i] = self._sums[k]
        packed[-1] = float(self._count)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_count = packed[-1].item()
        if global_count == 0:
            return dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0.0)
        # Single GPU→CPU sync at the very end.
        raw = packed[:-1].tolist()
        sums = dict(zip(keys, raw, strict=True))
        averaged = {k: v / global_count for k, v in sums.items()}
        return averaged, sums

    def reset(self) -> None:
        for t in self._sums.values():
            t.zero_()
        self._count = 0


class Trainer:
    """DeepFold3 distributed trainer.

    Orchestrates DDP setup, training loop, checkpointing, EMA, and evaluation.

    Args:
        model_cfg: Model architecture config.
        train_cfg: Training hyperparameters.
        records: Training data records.
        eval_records: Optional evaluation data records.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        records: list | None = None,
        eval_records: list | None = None,
        distillation_records: list | None = None,
        provenance: dict | None = None,
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.records = records or []
        self.eval_records = eval_records or []
        self.distillation_records = distillation_records or []
        self.provenance = provenance

        # Will be set during setup
        self.model: DeepFold3 | None = None
        self.ddp_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: AF3LRScheduler | None = None
        self.ema: EMA | None = None
        self.scaler: GradScaler | None = None
        self._wandb = None

        self.global_step = 0
        self.effective_step = 0
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self._last_grad_norm: torch.Tensor | None = None
        self._best_eval_loss: float = float("inf")
        self._last_num_recycles: int = 0
        # GPU NaN/Inf accumulator — ORed across microsteps, all-reduced once
        # per opt step. Materialised lazily on first optimizer step (so we
        # pick up the final device after ``setup_distributed``).
        self._nan_accum: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Initialize process group if running in distributed mode."""
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0

        self.is_main = self.rank == 0
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

    def _seed_everything(self) -> None:
        """Set seeds for reproducibility, offset by rank."""
        seed = self.cfg.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Model and optimizer setup
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Create model, optimizer, scheduler, EMA, and wrap in DDP."""
        self.model = DeepFold3(self.model_cfg).to(self.device)

        # Load pretrained weights if specified
        if self.cfg.params_path:
            self._load_pretrained(self.cfg.params_path)

        # Freeze parameters for confidence-only training (stage 3)
        if self.cfg.train_confidence_only:
            self._freeze_for_confidence_only()

        # Wrap in DDP. ``static_graph=True`` is safe in the unchunked path:
        #   * Recycles 1..N-1 run inside ``torch.no_grad()`` (model.py:396-409),
        #     so only the last recycle is in the autograd graph and the graph
        #     topology is identical iteration-to-iteration regardless of
        #     ``num_recycles``.
        #   * Gradient accumulation uses ``no_sync()`` on non-final
        #     microsteps (see ``_optimizer_step``), so only the final
        #     backward per optimizer step triggers DDP sync.
        # Confidence-loss resolution gating is a tensor multiply (loss.py)
        # rather than a Python branch, so the set of grad-receiving parameters
        # is also identical every step.
        #
        # ``static_graph=True`` is INCOMPATIBLE with ``model.no_sync()``:
        # PyTorch raises if the two are combined.  ``no_sync`` is used in
        # two places: (a) chunked-diffusion to defer DDP all_reduce until
        # after the outer aux backward, and (b) gradient-accumulation
        # microsteps in ``_optimizer_step`` to defer sync until the final
        # microstep.  Auto-disable ``static_graph`` when either is active.
        chunked_diffusion = self._chunked_diffusion_active()
        needs_no_sync = chunked_diffusion or self.cfg.accumulate_grad_batches > 1
        use_static_graph = self.cfg.static_graph and not needs_no_sync
        if self.world_size > 1:
            if self.is_main and self.cfg.static_graph and not use_static_graph:
                reasons = []
                if chunked_diffusion:
                    reasons.append(f"diffusion_chunk_size={self.cfg.diffusion_chunk_size}")
                if self.cfg.accumulate_grad_batches > 1:
                    reasons.append(f"accumulate_grad_batches={self.cfg.accumulate_grad_batches}")
                logger.info(
                    "static_graph auto-disabled (%s): model.no_sync() is required, which conflicts with static_graph.",
                    ", ".join(reasons),
                )
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.find_unused_parameters,
                static_graph=use_static_graph,
                bucket_cap_mb=self.cfg.bucket_cap_mb,
                gradient_as_bucket_view=self.cfg.gradient_as_bucket_view,
            )
        else:
            self.ddp_model = self.model

        # AdamW with decoupled weight decay. Protenix / nanoGPT convention:
        # decay 2D+ params (matmul weights, embeddings); skip 1D params
        # (LayerNorm gamma/beta, biases) — applying L2 to scale/shift terms
        # collapses them and hurts training stability.
        #
        # Additional refinement: many AF3 projections are ``LinearNoBias(1, C)``
        # or ``LinearNoBias(C, 1)`` — scalar-input or scalar-output linears that
        # behave as per-channel scale factors. Their weight is technically 2D
        # (``[C, 1]`` or ``[1, C]``) but semantically equivalent to a 1D param.
        # We treat any 2D weight where ``min(shape) == 1`` as no-decay.
        decay_params = []
        no_decay_params = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or min(param.shape) == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use the fused AdamW kernel on CUDA — single kernel launch per param
        # group, ~1.3-1.5x faster than the foreach implementation at our scale.
        use_fused_adam = torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
            fused=use_fused_adam,
        )
        if self.is_main:
            n_decay = sum(p.numel() for p in decay_params) / 1e6
            n_nodecay = sum(p.numel() for p in no_decay_params) / 1e6
            logger.info(
                f"AdamW: fused={use_fused_adam}, weight_decay={self.cfg.weight_decay} "
                f"({n_decay:.1f}M decayed, {n_nodecay:.2f}M no-decay)"
            )

        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        # EMA
        if self.cfg.ema_decay > 0:
            self.ema = EMA(self.model, decay=self.cfg.ema_decay)

        # Mixed precision scaler (only for fp16; bf16 doesn't need scaling)
        self.scaler = GradScaler("cuda", enabled=(self.cfg.dtype == "fp16"))

        if self.is_main:
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
            logger.info(f"Model parameters: {n_params:.1f}M total, {n_trainable:.1f}M trainable")
            logger.info(
                f"DDP: world_size={self.world_size}, static_graph={use_static_graph}, "
                f"accumulate_grad_batches={self.cfg.accumulate_grad_batches}"
            )

    def _freeze_for_confidence_only(self) -> None:
        """Freeze all parameters except confidence head for stage 3 training."""
        for name, param in self.model.named_parameters():
            if "confidence" not in name:
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Confidence-only training: {trainable / 1e6:.1f}M trainable parameters")

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained weights from safetensors or checkpoint.

        Args:
            path: Path to ``.safetensors`` or ``.pt`` file.
        """
        p = Path(path)
        strict = self.cfg.load_strict
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(p), device=str(self.device))
            self.model.load_state_dict(state, strict=strict)
            if self.is_main:
                logger.info(f"Loaded pretrained weights from {p}")
        elif p.suffix == ".pt":
            ckpt = torch.load(str(p), map_location=self.device, weights_only=False)
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=strict)
            else:
                self.model.load_state_dict(ckpt, strict=strict)
            if self.is_main:
                logger.info(f"Loaded pretrained weights from {p}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if configured.

        Gated on ``wandb_enabled`` (explicit on/off) and ``wandb_project``
        (the W&B project name). The run name is either the user-provided
        ``wandb_run_name`` *as-is*, or — when unset — ``{output_dir}_{ts}``
        where ``ts = YYYYMMDD-HHMMSS`` so multiple runs into the same
        output directory don't collapse into one wandb run.
        """
        if not self.cfg.wandb_enabled or not self.is_main:
            return
        if not self.cfg.wandb_project:
            logger.warning("wandb_enabled=True but wandb_project is empty; skipping wandb init")
            return
        try:
            import dataclasses

            import wandb

            if self.cfg.wandb_run_name:
                run_name = self.cfg.wandb_run_name
            else:
                run_name = Path(self.cfg.output_dir).name

            self._wandb = wandb
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity or None,
                name=run_name,
                config=dataclasses.asdict(self.cfg),
            )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")
            wandb.define_metric("system/*", step_metric="train/step")
            logger.info(f"WandB initialized: project={self.cfg.wandb_project}, run={run_name}")
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _init_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Create training and optional evaluation dataloaders.

        Supports multi-dataset training via ``extra_datasets`` config or
        the legacy ``distillation_data_dir`` option.  The primary dataset
        (``--data-dir``) always uses cluster-weighted sampling when cluster
        IDs are available; extra datasets specify their own strategy.

        Returns:
            Tuple of ``(train_dl, eval_dl)``.  ``eval_dl`` is ``None``
            when no evaluation records are provided.
        """
        ds_kwargs = dict(
            max_tokens=self.cfg.max_tokens,
            max_atoms=self.cfg.max_atoms,
            max_msa_seqs=self.cfg.max_msa_seqs,
            mmcif_dir=Path(self.cfg.mmcif_dir) if self.cfg.mmcif_dir else None,
            max_templates=self.model_cfg.data.max_templates,
            template_cutoff_days=self.model_cfg.data.template_cutoff_days,
            max_template_date=self.model_cfg.data.max_template_date,
            prefer_monomer_templates=self.model_cfg.data.prefer_monomer_templates,
            max_neighborhood=self.model_cfg.data.max_neighborhood,
            contiguous_crop_weight=self.model_cfg.data.contiguous_crop_weight,
            spatial_crop_weight=self.model_cfg.data.spatial_crop_weight,
            interface_crop_weight=self.model_cfg.data.interface_crop_weight,
            interface_distance=self.model_cfg.data.interface_distance,
            max_chains=self.model_cfg.data.max_chains,
        )

        # Keys accepted by OF3Dataset (subset of ds_kwargs)
        _OF3_KWARGS = {
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
        }
        of3_kwargs = {k: v for k, v in ds_kwargs.items() if k in _OF3_KWARGS}

        # Cache discovered records by (data_dir, format) to avoid re-scanning
        _record_cache: dict[tuple[str, str], list] = {}

        def _discover(data_dir: str, fmt: str) -> list:
            key = (data_dir, fmt)
            if key not in _record_cache:
                if fmt == "of3":
                    from deepfold3.data.module.of3_dataset import discover_of3_records

                    _record_cache[key] = discover_of3_records(Path(data_dir))
                else:
                    from deepfold3.data.prepare.manifest import discover_records

                    _record_cache[key] = discover_records(Path(data_dir))
            # Return a copy so cutoff filtering doesn't mutate the cache
            return list(_record_cache[key])

        def _apply_cutoff(records, cutoff: str, *, keep_after: bool):
            """Filter records by release date cutoff."""
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

        def _load_dataset_from_spec(spec, *, keep_after: bool, preloaded_records=None):
            """Load a dataset from a DatasetConfig spec.

            Returns (dataset, records) or (None, []) if no records found.
            """
            if preloaded_records is not None:
                records = preloaded_records
            else:
                records = _discover(spec.data_dir, spec.format)
            if spec.cutoff_date:
                records = _apply_cutoff(records, spec.cutoff_date, keep_after=keep_after)
            if not records:
                return None, []
            if spec.format == "of3":
                from deepfold3.data.module.of3_dataset import OF3Dataset

                ds = OF3Dataset(
                    records=records,
                    of3_dir=Path(spec.data_dir),
                    is_distillation=spec.is_distillation,
                    **of3_kwargs,
                )
            else:
                ds = TrainingDataset(
                    records=records,
                    data_dir=Path(spec.data_dir),
                    is_distillation=spec.is_distillation,
                    **ds_kwargs,
                )
            return ds, records

        # Primary (PDB) dataset — may be empty when only extra_datasets are used
        pdb_ds = None
        if self.records and self.cfg.data_dir:
            pdb_ds = TrainingDataset(
                records=self.records,
                data_dir=Path(self.cfg.data_dir),
                is_distillation=False,
                **ds_kwargs,
            )

        # Collect extra datasets from config
        extra_specs = list(self.cfg.extra_datasets)

        # Legacy: convert distillation_data_dir to an extra_datasets entry
        if (
            self.cfg.distillation_data_dir
            and self.distillation_records
            and not any(s.name == "_distillation" for s in extra_specs)
        ):
            from deepfold3.model.config import DatasetConfig

            extra_specs.append(
                DatasetConfig(
                    name="_distillation",
                    data_dir=self.cfg.distillation_data_dir,
                    weight=self.cfg.distillation_weight,
                    strategy="uniform",
                    is_distillation=True,
                )
            )

        if extra_specs:
            from torch.utils.data import ConcatDataset

            from deepfold3.data.sample.multi import MultiDatasetSampler

            datasets = []
            sampler_configs = []

            # Only include primary dataset if it has records
            if pdb_ds is not None and len(pdb_ds) > 0:
                datasets.append(pdb_ds)
                sampler_configs.append(
                    {
                        "records": self.records,
                        "dataset_len": len(pdb_ds),
                        "weight": 1.0,
                        "strategy": "weighted",
                    },
                )

            for spec in extra_specs:
                preloaded = self.distillation_records if spec.name == "_distillation" else None
                ds, records = _load_dataset_from_spec(spec, keep_after=False, preloaded_records=preloaded)
                if ds is None:
                    logger.warning("Extra dataset %r: no records in %s, skipping.", spec.name, spec.data_dir)
                    continue
                datasets.append(ds)
                sampler_configs.append(
                    {
                        "records": records,
                        "dataset_len": len(ds),
                        "weight": spec.weight,
                        "strategy": spec.strategy,
                    }
                )
                logger.info(
                    "Extra dataset %r: %d records, weight=%.2f, strategy=%s, distillation=%s",
                    spec.name,
                    len(records),
                    spec.weight,
                    spec.strategy,
                    spec.is_distillation,
                )

            train_ds = ConcatDataset(datasets)
            train_sampler = MultiDatasetSampler(
                dataset_configs=sampler_configs,
                epoch_size=self.cfg.epoch_size,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.cfg.seed,
            )
        else:
            if pdb_ds is None:
                raise RuntimeError(
                    "No training data: --data-dir is empty and no extra_datasets configured. "
                    "Provide --data-dir or define datasets in the config file."
                )
            train_ds = pdb_ds
            train_sampler = self._build_train_sampler(pdb_ds)

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size_per_rank,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=_collate_bucket_pad,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )

        eval_dl = None
        eval_datasets = []

        # Primary eval records (from date-based split on primary data_dir)
        if self.eval_records and self.cfg.data_dir:
            eval_datasets.append(
                TrainingDataset(
                    records=self.eval_records,
                    data_dir=Path(self.cfg.data_dir),
                    **ds_kwargs,
                )
            )

        # Eval datasets from config (eval_datasets YAML section)
        for spec in self.cfg.eval_datasets:
            ds, recs = _load_dataset_from_spec(spec, keep_after=True)
            if ds is None:
                logger.warning("Eval dataset %r: no records in %s, skipping.", spec.name, spec.data_dir)
                continue
            eval_datasets.append(ds)
            logger.info("Eval dataset %r: %d records from %s", spec.name, len(recs), spec.data_dir)

        if eval_datasets:
            from torch.utils.data import ConcatDataset as ConcatDS

            eval_ds = ConcatDS(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]
            eval_sampler = self._build_eval_sampler(eval_ds)
            eval_dl = DataLoader(
                eval_ds,
                batch_size=self.cfg.batch_size_per_rank,
                sampler=eval_sampler,
                shuffle=False,
                collate_fn=_collate_bucket_pad,
                num_workers=self.cfg.eval_num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=self.cfg.eval_num_workers > 0,
                prefetch_factor=self.cfg.prefetch_factor if self.cfg.eval_num_workers > 0 else None,
            )

        return train_dl, eval_dl

    def _build_train_sampler(self, train_ds: TrainingDataset):
        """Build training sampler — weighted or distributed.

        For ``WeightedClusterSampler`` we also attach the sampler's flat
        ``rows`` table to ``train_ds.sampling_rows`` so the dataset's
        ``__getitem__`` can resolve each sampled index back to a record
        and the chain/interface anchor that earned the draw — closing
        the loop the AF3 §2.5.1 sampler depends on.
        """
        has_real_clusters = any(isinstance(c.cluster_id, int) for r in self.records for c in r.chains)

        if has_real_clusters:
            from deepfold3.data.sample.weighted import WeightedClusterSampler

            sampler = WeightedClusterSampler(
                records=self.records,
                epoch_size=self.cfg.epoch_size,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.cfg.seed,
            )
            train_ds.sampling_rows = sampler.rows
            return sampler

        if self.world_size > 1:
            return DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.cfg.seed,
            )

        return None

    def _build_eval_sampler(self, eval_ds):
        """Build eval sampler — subsample with cluster-weighted probabilities.

        When ``eval_samples > 0``, uses :class:`WeightedEvalSampler` to
        draw a fixed-size subset each eval pass using per-record cluster
        weights (so large families aren't over-represented). Each pass
        gets a different subset via ``set_epoch(effective_step)``.

        When ``eval_samples == 0`` (use all), falls back to a plain
        ``DistributedSampler`` (DDP) or ``None`` (single GPU).
        """
        n_eval = self.cfg.eval_samples
        ds_len = len(eval_ds)
        if n_eval <= 0 or n_eval >= ds_len:
            if self.world_size > 1:
                # shuffle=True so each eval call sees a different (still
                # full-coverage) ordering; set_epoch(effective_step) is
                # already called in the train loop to vary the seed.
                return DistributedSampler(
                    eval_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.cfg.seed
                )
            return None

        # WeightedEvalSampler needs per-record cluster weights from
        # self.eval_records.  When the eval set comes from YAML
        # eval_datasets (or a ConcatDataset), eval_records may not cover
        # the full dataset — fall back to a uniform random subset sampler.
        if self.eval_records and len(self.eval_records) >= ds_len:
            from deepfold3.data.sample.weighted import WeightedEvalSampler

            return WeightedEvalSampler(
                records=self.eval_records,
                n_samples=n_eval,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.cfg.seed,
            )

        from deepfold3.data.sample.weighted import UniformEvalSampler

        return UniformEvalSampler(
            n_records=ds_len,
            n_samples=n_eval,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.cfg.seed,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _prepare_features(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move features to device.

        Always keeps the leading ``[B, ...]`` batch dimension so every
        module consistently takes the batched code path (even when
        ``batch_size_per_rank == 1``).
        """
        return {k: v.to(self.device, non_blocking=True) for k, v in features.items()}

    def _training_step(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one forward pass and compute all losses.

        Delegates to :meth:`DeepFold3.forward_training` for the full
        training pipeline (evoformer, diffusion, confidence, distogram),
        then computes the combined loss.

        Args:
            features: Batched feature dict from the dataloader.

        Returns:
            Tuple of ``(total_loss, loss_dict)``.
        """
        cfg = self.cfg
        features = self._prepare_features(features)

        # Number of recycles per microstep. AF3 convention: total trunk
        # passes = num_recycles + 1, sampled uniformly from {0..max_recycles}.
        # max_recycles=3 → up to 4 trunk passes (matches AF3 paper).
        # Two modes:
        #   * uniform_recycle=True (default): all DDP ranks pick the SAME
        #     num_recycles via a fresh ``random.Random`` seeded from
        #     ``(cfg.seed, global_step)`` — both identical on every rank.
        #     Avoids the straggler effect where one rank does 3 no_grad
        #     warmup passes and another does 0, blocking DDP all_reduce
        #     on the slowest rank (~4× wasted compute at max_recycles=3).
        #   * uniform_recycle=False: legacy per-rank Python random — each
        #     rank picks independently from its own seeded RNG. Trains on
        #     more diverse recycle counts per global batch but pays the
        #     straggler cost.
        if cfg.uniform_recycle:
            num_recycles = random.Random(cfg.seed + self.global_step).randint(0, cfg.max_recycles)
        else:
            num_recycles = random.randint(0, cfg.max_recycles)
        # Stash for ``log_microsteps`` debug logging in the outer train loop.
        self._last_num_recycles = num_recycles

        # Chunked-diffusion path: ``forward_training`` back-props each chunk
        # internally and runs the trunk flow-back at the end of the chunked
        # block. Those backwards must run under ``model.no_sync()`` so DDP
        # doesn't fire an all_reduce per chunk. The aux backward (in
        # ``_optimizer_step``) runs OUTSIDE no_sync and fires the single
        # DDP all_reduce per microstep.
        chunked = self._chunked_diffusion_active()
        if chunked:
            sync_ctx = self.ddp_model.no_sync() if self.world_size > 1 else nullcontext()
            backward_fn = self._scaled_backward
            loss_cfg_arg = cfg
            chunk_size_arg = cfg.diffusion_chunk_size
        else:
            sync_ctx = nullcontext()
            backward_fn = None
            loss_cfg_arg = None
            chunk_size_arg = 0

        with sync_ctx:
            pred = self.model.forward_training(
                features,
                num_recycles=num_recycles,
                diffusion_batch_size=cfg.diffusion_batch_size,
                mini_rollout_steps=cfg.mini_rollout_steps,
                mini_rollout_samples=cfg.mini_rollout_samples,
                seed=self.global_step,
                fused_ce=cfg.use_fused_ce_loss,
                diffusion_chunk_size=chunk_size_arg,
                backward_fn=backward_fn,
                loss_cfg=loss_cfg_arg,
            )

        # ``training_loss`` assumes unbatched ``[N, ...]`` features — slice per
        # batch element and average (even for B=1, since the batch dim is kept).
        # ``fp32_loss`` runs the combined loss with autocast disabled so
        # softmax / CE always see fp32 logits.
        loss_fn = fp32_guard(self.model_cfg.precision.fp32_loss)(training_loss)
        B = cfg.batch_size_per_rank
        losses_list: list[torch.Tensor] = []
        loss_dict_acc: dict[str, torch.Tensor] = {}
        for b in range(B):
            pred_b = _slice_batch_dict(pred, b, B)
            features_b = _slice_batch_dict(features, b, B)
            loss_b, ld_b = loss_fn(pred_b, features_b, cfg)
            losses_list.append(loss_b)
            for k, v in ld_b.items():
                if k in loss_dict_acc:
                    loss_dict_acc[k] = loss_dict_acc[k] + v
                else:
                    loss_dict_acc[k] = v
        total_loss = torch.stack(losses_list).mean()
        loss_dict = {k: v / B for k, v in loss_dict_acc.items()}
        return total_loss, loss_dict

    def _chunked_diffusion_active(self) -> bool:
        cfg = self.cfg
        return cfg.diffusion_chunk_size > 0 and cfg.diffusion_batch_size % cfg.diffusion_chunk_size == 0

    def _scaled_backward(self, loss: torch.Tensor) -> None:
        """``scaler.scale(loss / accumulate).backward()`` — used by both the
        chunked-diffusion per-chunk callback and the outer ``_optimizer_step``.
        """
        scaled = loss / self.cfg.accumulate_grad_batches
        self.scaler.scale(scaled).backward()

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self, loss: torch.Tensor) -> bool:
        """Backward pass, gradient clipping, and optimizer step.

        When ``accumulate_grad_batches > 1``, non-final microsteps run
        under ``ddp_model.no_sync()`` so DDP defers the gradient
        all-reduce until the final microstep — otherwise every backward
        fires a redundant all-reduce per microstep.

        NaN/Inf detection runs once per optimizer step at the sync
        boundary (instead of once per microstep) via ``_nan_accum``: each
        microstep ORs its local NaN flag into the accumulator on-device
        with no collective. At the sync boundary a single ``all_reduce``
        checks whether any rank saw a bad loss across the whole window —
        if so, all grads are zeroed and the opt step is skipped. This
        replaces N per-window collectives (one per microstep) with 1.

        Args:
            loss: Scalar loss tensor from ``_training_step()``.

        Returns:
            True if an optimizer step was taken (effective_step advanced).
        """
        # Per-microstep local NaN/Inf flag (no collective, no CPU sync).
        if self._nan_accum is None:
            self._nan_accum = torch.zeros((), device=loss.device)
        self._nan_accum.add_((torch.isnan(loss) | torch.isinf(loss)).float())

        next_step = self.global_step + 1
        is_sync_step = next_step % self.cfg.accumulate_grad_batches == 0

        # Suppress DDP gradient all-reduce on non-final accumulation steps.
        sync_ctx = self.ddp_model.no_sync() if (not is_sync_step and self.world_size > 1) else nullcontext()
        with sync_ctx:
            self._scaled_backward(loss)

        self.global_step = next_step

        if not is_sync_step:
            return False

        # Sync boundary: single collective for NaN check across ranks +
        # all microsteps in this accumulation window.
        if self.world_size > 1:
            dist.all_reduce(self._nan_accum, op=dist.ReduceOp.MAX)
        nan_detected = self._nan_accum.item() > 0  # one CPU sync per opt step
        self._nan_accum.zero_()

        if nan_detected:
            # Any microstep on any rank produced NaN/Inf — discard the whole
            # accumulation window (NaN would have propagated via DDP all_reduce
            # on the sync-step backward above).
            self.optimizer.zero_grad(set_to_none=True)
            if self.is_main:
                logger.warning(f"Step {self.effective_step}: NaN/Inf loss in accumulation window, skipping")
            return False

        self.scaler.unscale_(self.optimizer)

        # Always compute the global pre-clip grad norm for logging,
        # even when grad_clip_norm <= 0 (use a huge cap as no-op).
        clip = self.cfg.grad_clip_norm if self.cfg.grad_clip_norm > 0 else float("inf")
        self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip).detach()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.effective_step += 1

        if self.ema is not None:
            self.ema.update(self.model)

        return True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, eval_dl: DataLoader) -> dict[str, float]:
        """Run evaluation on the eval set.

        Swaps in EMA weights if available, runs forward_training on each
        batch, computes losses and distogram lDDT, then restores weights.

        Args:
            eval_dl: Evaluation dataloader.

        Returns:
            Dict mapping metric names to averaged values.
        """
        if self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()
        metrics: dict[str, list[float]] = {}
        n_batches = len(eval_dl)

        for batch_idx, features in enumerate(eval_dl):
            features = self._prepare_features(features)

            try:
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type="cuda",
                        dtype=self.cfg.torch_dtype,
                        enabled=(self.cfg.dtype != "fp32"),
                    ),
                ):
                    pred = self.model.forward_training(
                        features,
                        num_recycles=min(self.cfg.max_recycles, 4),
                        diffusion_batch_size=1,
                        mini_rollout_steps=self.cfg.mini_rollout_steps,
                        mini_rollout_samples=1,
                        seed=0,
                    )
                    # Slice batch dim (always kept) before loss.
                    # Use actual batch size from the data (not cfg) — the last
                    # eval batch may be smaller than batch_size_per_rank when the
                    # dataset size isn't evenly divisible.
                    B_actual = features["res_type"].shape[0]
                    pred_0 = _slice_batch_dict(pred, 0, B_actual)
                    features_0 = _slice_batch_dict(features, 0, B_actual)
                    eval_loss_fn = fp32_guard(self.model_cfg.precision.fp32_loss)(training_loss)
                    _, loss_dict = eval_loss_fn(pred_0, features_0, self.cfg)

                for k, v in loss_dict.items():
                    metrics.setdefault(f"eval/{k}", []).append(v.item())

                if self.is_main and self.cfg.log_microsteps:
                    logger.info("eval: %d/%d loss=%.2f", batch_idx + 1, n_batches, loss_dict["total"])
            except Exception as e:
                logger.warning("Eval error at batch %d: %s", batch_idx, e, exc_info=True)
                continue

        self.model.train()

        if self.ema is not None:
            self.ema.restore(self.model)

        return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

    def _run_eval_and_track_best(self, eval_dl: DataLoader, output_dir: Path) -> None:
        """Reseed eval sampler, run one eval pass, log metrics, save best checkpoint."""
        if hasattr(eval_dl.sampler, "set_epoch"):
            eval_dl.sampler.set_epoch(self.effective_step)
        eval_metrics = self._evaluate(eval_dl)
        if self.is_main:
            logger.info(f"Eval at step {self.effective_step}: {eval_metrics}")
            if self._wandb is not None:
                self._wandb.log(eval_metrics, step=self.effective_step)
            eval_total = eval_metrics.get("eval/total", float("inf"))
            if eval_total < self._best_eval_loss:
                self._best_eval_loss = eval_total
                self._save_checkpoint(output_dir / "checkpoint_best.pt")
                logger.info(f"New best eval loss: {eval_total:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path) -> None:
        """Save model, optimizer, scheduler, EMA, and step counters.

        Args:
            path: File path for the checkpoint ``.pt`` file.
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "effective_step": self.effective_step,
            "best_eval_loss": self._best_eval_loss,
            "model_cfg": self.model_cfg,
            "train_cfg": self.cfg,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

        # Rotate old checkpoints
        if self.cfg.max_checkpoints > 0:
            self._rotate_checkpoints(path.parent)

    def _rotate_checkpoints(self, output_dir: Path) -> None:
        """Keep only the N most recent checkpoints."""
        pattern = "checkpoint_*.pt"
        ckpts = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        # Exclude checkpoint_final.pt from rotation
        ckpts = [c for c in ckpts if "final" not in c.stem and "best" not in c.stem]
        while len(ckpts) > self.cfg.max_checkpoints:
            old = ckpts.pop(0)
            old.unlink()
            logger.info(f"Removed old checkpoint: {old}")

    @staticmethod
    def _find_latest_checkpoint(output_dir: Path) -> Path | None:
        """Find the checkpoint with the highest step number in *output_dir*."""
        best_step, best_path = -1, None
        for p in output_dir.glob("checkpoint_*.pt"):
            m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step, best_path = step, p
        return best_path

    def _load_checkpoint(self, path: Path) -> None:
        """Resume training from a checkpoint.

        Handles DDP ``module.`` prefix stripping automatically.
        Supports partial loading via config flags.

        Args:
            path: Path to a checkpoint saved by ``_save_checkpoint()``.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Handle DDP module. prefix
        state_dict = ckpt["model"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=self.cfg.load_strict)

        if "optimizer" in ckpt and not self.cfg.skip_load_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and not self.cfg.skip_load_scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt and self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])

        if not self.cfg.skip_load_step:
            self.global_step = ckpt.get("global_step", 0)
            self.effective_step = ckpt.get("effective_step", 0)
        self._best_eval_loss = ckpt.get("best_eval_loss", float("inf"))

        if self.is_main:
            logger.info(f"Resumed from {path} at step {self.effective_step}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self, resume_from: str | Path | None = None) -> None:
        """Run the full training loop.

        Args:
            resume_from: Optional checkpoint path to resume from.
        """
        self._run_inner(resume_from)

    def _run_inner(self, resume_from: str | Path | None) -> None:
        self._init_distributed()
        self._seed_everything()

        self._init_model()
        self._init_wandb()

        train_dl, eval_dl = self._init_dataloaders()
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config (all CLI/YAML overrides applied) for reproducibility.
        # Layout mirrors configs/*.yaml so the dump can be re-fed via --config.
        if self.is_main:
            config_out = output_dir / "config.yaml"
            with open(config_out, "w") as f:
                yaml.safe_dump(
                    serialize_run_config(self.cfg, self.model_cfg),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"Resolved config saved to {config_out}")
            if self.provenance is not None:
                prov_out = output_dir / "provenance.yaml"
                with open(prov_out, "w") as f:
                    yaml.safe_dump(
                        _yaml_safe(self.provenance), f, default_flow_style=False, sort_keys=False
                    )
                logger.info(f"Run provenance saved to {prov_out}")

        # Auto-resume: find latest checkpoint in output_dir if none specified
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))
        elif self.cfg.auto_resume:
            latest = self._find_latest_checkpoint(output_dir)
            if latest is not None:
                self._load_checkpoint(latest)

        cfg = self.cfg
        if self.is_main:
            tokens_per_rank = cfg.max_tokens * cfg.batch_size_per_rank
            effective_batch_size = cfg.batch_size_per_rank * self.world_size * cfg.accumulate_grad_batches
            tokens_per_opt_step = tokens_per_rank * self.world_size * cfg.accumulate_grad_batches
            logger.info(f"Starting training: max_steps={cfg.max_steps}")
            logger.info(f"  confidence_only={cfg.train_confidence_only}")
            logger.info(f"  lr={cfg.lr}, warmup={cfg.warmup_steps}, scheduler={cfg.finetune_scheduler}")
            logger.info(
                f"  crop: max_tokens={cfg.max_tokens}, max_atoms={cfg.max_atoms}, "
                f"max_msa_seqs={cfg.max_msa_seqs}, max_recycles={cfg.max_recycles}"
            )
            logger.info(
                f"  batch: world_size={self.world_size}, batch_per_rank={cfg.batch_size_per_rank}, "
                f"accumulate={cfg.accumulate_grad_batches} → "
                f"effective_batch_size={effective_batch_size} (tokens_per_opt_step={tokens_per_opt_step})"
            )
            logger.info(
                f"  diffusion: batch_size={cfg.diffusion_batch_size}, chunk_size={cfg.diffusion_chunk_size}, "
                f"mini_rollout_steps={cfg.mini_rollout_steps}, mini_rollout_samples={cfg.mini_rollout_samples}"
            )
            logger.info(f"  dtype={cfg.dtype}, grad_clip={cfg.grad_clip_norm}, ema_decay={cfg.ema_decay}")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        meters = DistributedAverageMeters(self.device)
        t0 = time.monotonic()
        t_last_ckpt = t0
        # Pure microstep compute time accumulator (excludes eval / checkpoint /
        # summary-log wall-clock). Flushed at each summary log.
        microstep_compute_time = 0.0
        # _best_eval_loss is initialized in __init__ and may be overwritten
        # by _load_checkpoint; no need for getattr guard here.

        # SIGUSR2 → request immediate checkpoint (preemption safety)
        # SIGTERM/SIGINT → checkpoint + graceful exit (SLURM preemption, kill, Ctrl+C)
        self._checkpoint_requested = False
        self._shutdown_requested = False
        if platform.system() != "Windows":
            signal.signal(signal.SIGUSR2, lambda *_: setattr(self, "_checkpoint_requested", True))

            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_shutdown_requested", True))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "_shutdown_requested", True))

        shutdown_flag = torch.zeros(1, dtype=torch.int32, device=self.device)

        if cfg.eval_before and eval_dl is not None:
            self._run_eval_and_track_best(eval_dl, output_dir)

        while self.effective_step < self.cfg.max_steps:
            sampler = train_dl.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for features in train_dl:
                if self.effective_step >= self.cfg.max_steps:
                    break

                t_microstep_start = time.monotonic()
                try:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.cfg.torch_dtype,
                        enabled=(self.cfg.dtype != "fp32"),
                    ):
                        loss, loss_dict = self._training_step(features)
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(
                        f"Step {self.effective_step}: OOM, skipping batch: {e}",
                        exc_info=True,
                    )
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    continue
                except Exception as e:
                    logger.warning(
                        f"Step {self.effective_step}: skipping batch ({e})",
                        exc_info=True,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    continue

                # NaN/Inf detection is deferred to the opt-step boundary inside
                # ``_optimizer_step`` — a single all-reduce per window replaces
                # one-per-microstep. Backward runs unconditionally here; any
                # NaN grads are discarded at the boundary before the opt step.
                on_opt_step_boundary = self._optimizer_step(loss)

                # Accumulate logging metrics locally (no collective).
                # Inject token count so it rides the same all-reduce as losses.
                seq_mask = features.get("seq_mask")
                n_tokens = (
                    seq_mask.sum()
                    if seq_mask is not None
                    else torch.tensor(cfg.max_tokens * cfg.batch_size_per_rank, device=self.device)
                )
                loss_dict["_tokens"] = n_tokens
                meters.update(loss_dict, n=1)

                # Pure microstep compute time (forward + backward + step). Excludes
                # anything that happens between iterations (eval, checkpoint, summary
                # log), so rate / ETA stay stable across eval boundaries. Accumulated
                # into ``microstep_compute_time`` for the effective-step summary log.
                dt_micro = time.monotonic() - t_microstep_start
                microstep_compute_time += dt_micro

                if cfg.log_microsteps and self.is_main:
                    samples_per_microstep = self.world_size * cfg.batch_size_per_rank
                    samples_per_sec = samples_per_microstep / max(dt_micro, 1e-9)
                    # Position within the current accumulation window: 1..accumulate
                    # (the accumulate-th microstep is the sync step that fires
                    # optimizer.step and increments effective_step).  ``target`` is
                    # the effective_step this microstep is building toward — on the
                    # sync step it equals the just-incremented effective_step;
                    # otherwise it's effective_step+1.
                    in_step = ((self.global_step - 1) % cfg.accumulate_grad_batches) + 1
                    target = self.effective_step if on_opt_step_boundary else self.effective_step + 1
                    logger.info(
                        f"micro: target={target} "
                        f"global={self.global_step} "
                        f"in_step={in_step}/{cfg.accumulate_grad_batches} "
                        f"recycles={self._last_num_recycles} "
                        f"loss={loss.item():.4f} "
                        f"dt_micro={dt_micro:.2f}s samples/s={samples_per_sec:.3f}"
                    )

                # Release forward-pass outputs and input features immediately.
                # Without this, local vars hold references to large GPU tensors
                # (denoised [S,N,M,3], atom_gt_positions [N,M,3], etc.) until
                # Python rebinds them at the next iteration — wasting GPU memory
                # during the dataloader fetch and gradient accumulation micro-steps.
                del loss, loss_dict, features

                # --- Logging ---
                # Gate on ``on_opt_step_boundary`` (returned by
                # ``_optimizer_step``) so the log always fires AFTER
                # ``clip_grad_norm`` has populated ``_last_grad_norm``.
                if on_opt_step_boundary and self.effective_step % self.cfg.log_interval == 0 and meters.count > 0:
                    avg, sums = meters.sync()
                    meters.reset()

                    # Flush the compute-time accumulator regardless of is_main so
                    # all ranks stay in sync if the flag ever changes.
                    compute_time_window = microstep_compute_time
                    microstep_compute_time = 0.0

                    if self.is_main:
                        grad_norm = self._last_grad_norm.item() if self._last_grad_norm is not None else float("nan")
                        elapsed = time.monotonic() - t0
                        # ``dt`` is per-effective-step compute time (forward+backward+step).
                        # Eval / checkpoint / logging time is intentionally excluded so the
                        # displayed rate/ETA reflect pure training throughput.
                        dt = compute_time_window / max(self.cfg.log_interval, 1)
                        lr = self.optimizer.param_groups[0]["lr"]

                        # Throughput (global token sum from all ranks via all-reduce).
                        # Same compute-time denominator as ``dt`` for a consistent picture.
                        tok_per_sec = sums.pop("_tokens", 0) / max(compute_time_window, 1e-9)
                        avg.pop("_tokens", None)

                        # GPU memory
                        peak_mem_gb = torch.cuda.max_memory_allocated() / (1 << 30) if torch.cuda.is_available() else 0

                        # ETA
                        remaining = cfg.max_steps - self.effective_step
                        eta_sec = remaining * dt
                        eta_h, eta_rem = divmod(int(eta_sec), 3600)
                        eta_str = f"{eta_h}:{eta_rem // 60:02d}" if eta_h > 0 else f"{int(eta_sec // 60)}m"

                        parts = [f"step={self.effective_step}"]
                        parts.append(f"loss={avg.get('total', 0):.4f}")
                        for k, v in sorted(avg.items()):
                            if k != "total":
                                parts.append(f"{k}={v:.4f}")
                        parts.append(f"grad_norm={grad_norm:.3f}")
                        parts.append(f"lr={lr:.2e}")
                        if tok_per_sec >= 1000:
                            parts.append(f"tok/s={tok_per_sec / 1000:.1f}k")
                        else:
                            parts.append(f"tok/s={tok_per_sec:.0f}")
                        parts.append(f"mem={peak_mem_gb:.1f}G")
                        parts.append(f"eta={eta_str}")
                        parts.append(f"dt={dt:.1f}s")
                        parts.append(f"t={elapsed:.0f}s")
                        logger.info(" | ".join(parts))

                        if self._wandb is not None:
                            log_data = {f"train/{k}": v for k, v in avg.items()}
                            log_data["train/step"] = self.effective_step
                            log_data["train/lr"] = lr
                            log_data["train/epoch"] = epoch
                            log_data["train/grad_norm"] = grad_norm
                            log_data["train/step_time_s"] = dt
                            log_data["train/tokens_per_sec"] = tok_per_sec
                            log_data["system/gpu_memory_allocated_gb"] = (
                                torch.cuda.memory_allocated() / (1 << 30) if torch.cuda.is_available() else 0
                            )
                            log_data["system/gpu_memory_reserved_gb"] = (
                                torch.cuda.memory_reserved() / (1 << 30) if torch.cuda.is_available() else 0
                            )
                            log_data["system/gpu_peak_memory_gb"] = peak_mem_gb
                            self._wandb.log(log_data, step=self.effective_step)

                # --- Checkpoint (before eval so weights are saved even if eval crashes) ---
                need_ckpt = False
                if on_opt_step_boundary and self.is_main:
                    if self.effective_step % self.cfg.checkpoint_interval == 0:
                        need_ckpt = True
                    elif (
                        cfg.checkpoint_every_hours > 0
                        and (time.monotonic() - t_last_ckpt) / 3600 >= cfg.checkpoint_every_hours
                    ):
                        need_ckpt = True
                    elif self._checkpoint_requested:
                        self._checkpoint_requested = False
                        need_ckpt = True
                        logger.info("Signal-triggered checkpoint requested")

                if need_ckpt:
                    self._save_checkpoint(output_dir / f"checkpoint_{self.effective_step}.pt")
                    t_last_ckpt = time.monotonic()

                # --- Evaluation ---
                # Gate on ``on_opt_step_boundary`` so eval fires exactly once
                # per effective step — without the gate, ``effective_step``
                # stays constant across ``accumulate_grad_batches`` microsteps
                # and the check would trigger on every one of them.
                if on_opt_step_boundary and eval_dl is not None and self.effective_step % self.cfg.eval_interval == 0:
                    self._run_eval_and_track_best(eval_dl, output_dir)

                # --- Shutdown ---
                # Propagate on every microstep (cheap 1-elem all_reduce) so
                # all ranks bail together, not just at opt-step boundaries.
                # MAX so any rank's signal (SIGINT → all; SIGTERM may → rank 0
                # only) reaches everyone.
                if self.world_size > 1:
                    shutdown_flag.fill_(int(self._shutdown_requested))
                    dist.all_reduce(shutdown_flag, op=dist.ReduceOp.MAX)
                    self._shutdown_requested = bool(shutdown_flag.item())

                if self._shutdown_requested:
                    # Mid-window: drop partial grads and rewind global_step
                    # to the last completed boundary so resume starts a fresh
                    # accumulation window. scheduler / scaler / EMA stay as-is
                    # (they only advance on sync steps).
                    partial = 0
                    if not on_opt_step_boundary:
                        partial = self.global_step - self.effective_step * cfg.accumulate_grad_batches
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step = self.effective_step * cfg.accumulate_grad_batches
                    if self.is_main:
                        if partial:
                            logger.info(
                                f"Shutdown mid-window at step {self.effective_step}: discarded "
                                f"{partial}/{cfg.accumulate_grad_batches} partial microsteps"
                            )
                        else:
                            logger.info(f"Shutdown at step {self.effective_step}")
                        self._save_checkpoint(output_dir / f"checkpoint_{self.effective_step}.pt")
                    break

            if self._shutdown_requested:
                break
            epoch += 1

        # Final checkpoint
        if self.is_main:
            if not self._shutdown_requested:
                self._save_checkpoint(output_dir / "checkpoint_final.pt")
            logger.info(
                f"Training {'interrupted' if self._shutdown_requested else 'complete'} at step {self.effective_step}"
            )

        if self._wandb is not None:
            self._wandb.finish()

        # Per PyTorch reference: every trainer process must call
        # ``destroy_process_group`` exactly once at the end. The call is a
        # collective — all ranks must reach it, so it lives AFTER the
        # training loop (not in a ``try/finally`` which could fire at
        # different times on different ranks and cause a deadlock).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # ensure all ranks finished before teardown
            dist.destroy_process_group()
