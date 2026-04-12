"""DDP trainer for DeepFold3.

Native PyTorch ``DistributedDataParallel`` training loop following
Protenix's procedure with Boltz-style code organization.  No Lightning
dependency.

Example::

    trainer = Trainer(model_cfg, train_cfg)
    trainer.run()
"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
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

    def sync(self) -> dict[str, float]:
        """All-reduce across ranks and return the global weighted average dict."""
        if not self._sums:
            return {}
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
            return dict.fromkeys(keys, 0.0)
        # Single GPU→CPU sync at the very end.
        averaged = (packed[:-1] / global_count).tolist()
        return dict(zip(keys, averaged, strict=True))

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
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.records = records or []
        self.eval_records = eval_records or []
        self.distillation_records = distillation_records or []

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
        self._last_num_recycles: int = 0

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
        #   * Gradient accumulation is pure-scaling (no ``no_sync()``), so
        #     every backward triggers full DDP sync.
        # Confidence-loss resolution gating is a tensor multiply (loss.py)
        # rather than a Python branch, so the set of grad-receiving parameters
        # is also identical every step.
        #
        # Chunked-diffusion path is INCOMPATIBLE with ``static_graph=True``:
        # it uses ``model.no_sync()`` to defer DDP all_reduce until after the
        # outer aux backward (otherwise we'd issue an all_reduce per chunk,
        # swamping NCCL). PyTorch raises if ``no_sync`` is used under
        # ``static_graph=True`` — auto-disable in that case.
        chunked_diffusion = self._chunked_diffusion_active()
        use_static_graph = self.cfg.static_graph and not chunked_diffusion
        if self.world_size > 1:
            if self.is_main and self.cfg.static_graph and not use_static_graph:
                logger.info(
                    "static_graph auto-disabled (diffusion_chunk_size=%d > 0): "
                    "chunked diffusion uses model.no_sync(), which conflicts with static_graph.",
                    self.cfg.diffusion_chunk_size,
                )
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.find_unused_parameters,
                static_graph=use_static_graph,
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
        self.scaler = GradScaler(enabled=(self.cfg.dtype == "fp16"))

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
            import datetime

            import wandb

            if self.cfg.wandb_run_name:
                run_name = self.cfg.wandb_run_name
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                run_name = f"{Path(self.cfg.output_dir).name}_{timestamp}"

            self._wandb = wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=run_name,
                config=dataclasses.asdict(self.cfg),
            )
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
        )

        # Primary (PDB) dataset
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
            and not any(s.get("name") == "_distillation" for s in extra_specs)
        ):
            extra_specs.append(
                {
                    "name": "_distillation",
                    "data_dir": self.cfg.distillation_data_dir,
                    "weight": self.cfg.distillation_weight,
                    "strategy": "uniform",
                    "is_distillation": True,
                    "_records": self.distillation_records,
                }
            )

        if extra_specs:
            from torch.utils.data import ConcatDataset

            from deepfold3.data.prepare.manifest import discover_records
            from deepfold3.data.sample.multi import MultiDatasetSampler

            datasets = [pdb_ds]
            sampler_configs = [
                {
                    "records": self.records,
                    "dataset_len": len(pdb_ds),
                    "weight": 1.0,
                    "strategy": "weighted",
                },
            ]

            for spec in extra_specs:
                # Load records: from pre-loaded _records or by discovering from data_dir
                records = spec.get("_records")
                if records is None:
                    records = discover_records(Path(spec["data_dir"]))
                if not records:
                    logger.warning(
                        "Extra dataset %r: no records found in %s, skipping.", spec.get("name", "?"), spec["data_dir"]
                    )
                    continue

                ds = TrainingDataset(
                    records=records,
                    data_dir=Path(spec["data_dir"]),
                    is_distillation=spec.get("is_distillation", False),
                    **ds_kwargs,
                )
                datasets.append(ds)
                sampler_configs.append(
                    {
                        "records": records,
                        "dataset_len": len(ds),
                        "weight": spec.get("weight", 1.0),
                        "strategy": spec.get("strategy", "uniform"),
                    }
                )
                logger.info(
                    "Extra dataset %r: %d records, weight=%.2f, strategy=%s, distillation=%s",
                    spec.get("name", "?"),
                    len(records),
                    spec.get("weight", 1.0),
                    spec.get("strategy", "uniform"),
                    spec.get("is_distillation", False),
                )

            train_ds = ConcatDataset(datasets)
            train_sampler = MultiDatasetSampler(
                dataset_configs=sampler_configs,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.cfg.seed,
            )
        else:
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
        if self.eval_records:
            eval_ds = TrainingDataset(
                records=self.eval_records,
                data_dir=Path(self.cfg.data_dir),
                **ds_kwargs,
            )
            eval_dl = DataLoader(
                eval_ds,
                batch_size=self.cfg.batch_size_per_rank,
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
        the loop the AF3 §2.5.6 sampler depends on.
        """
        has_real_clusters = any(isinstance(c.cluster_id, int) for r in self.records for c in r.chains)

        if has_real_clusters:
            from deepfold3.data.sample.weighted import WeightedClusterSampler

            sampler = WeightedClusterSampler(
                records=self.records,
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

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _prepare_features(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move features to device. Squeeze the leading batch dim only when
        ``batch_size_per_rank == 1`` so the model uses its (slightly cheaper)
        unbatched code path. With ``batch_size_per_rank > 1`` we keep the
        ``[B, ...]`` layout so trunk + diffusion + heads run in batched mode.
        """
        features = {k: v.to(self.device, non_blocking=True) for k, v in features.items()}
        if self.cfg.batch_size_per_rank == 1:
            for k, v in features.items():
                if v.dim() > 0 and v.shape[0] == 1:
                    features[k] = v.squeeze(0)
        return features

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

        # Loss computation. ``training_loss`` and the underlying per-loss
        # helpers assume unbatched ``[N, ...]`` features. With
        # ``batch_size_per_rank > 1`` we slice ``pred`` and ``features``
        # per batch element, call ``training_loss`` per element, and
        # average. The expensive forward (trunk + diffusion) already ran
        # batched, so this loop only adds the (cheap) loss-computation
        # overhead per element.
        if cfg.batch_size_per_rank > 1:
            B = cfg.batch_size_per_rank
            losses_list: list[torch.Tensor] = []
            loss_dict_acc: dict[str, torch.Tensor] = {}
            for b in range(B):
                pred_b = _slice_batch_dict(pred, b, B)
                features_b = _slice_batch_dict(features, b, B)
                loss_b, ld_b = training_loss(pred_b, features_b, cfg)
                losses_list.append(loss_b)
                for k, v in ld_b.items():
                    if k in loss_dict_acc:
                        loss_dict_acc[k] = loss_dict_acc[k] + v
                    else:
                        loss_dict_acc[k] = v.clone()
            total_loss = torch.stack(losses_list).mean()
            loss_dict = {k: v / B for k, v in loss_dict_acc.items()}
        else:
            total_loss, loss_dict = training_loss(pred, features, cfg)
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

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Backward pass, gradient clipping, and optimizer step.

        Args:
            loss: Scalar loss tensor from ``_training_step()``.
        """
        self._scaled_backward(loss)

        self.global_step += 1

        if self.global_step % self.cfg.accumulate_grad_batches == 0:
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

    # ------------------------------------------------------------------
    # NaN detection
    # ------------------------------------------------------------------

    def _check_nan(self, loss: torch.Tensor) -> bool:
        """Check if loss is NaN/Inf across all ranks.

        In distributed mode, uses all-reduce so all ranks agree on
        whether to skip the batch (prevents hangs from rank divergence).

        Returns:
            True if loss is NaN/Inf on any rank.
        """
        # Use bitwise OR to avoid Python bool conversion triggering two GPU syncs
        flag = (torch.isnan(loss) | torch.isinf(loss)).float()
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0  # single GPU sync

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

        for features in eval_dl:
            features = self._prepare_features(features)

            try:
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.cfg.torch_dtype,
                    enabled=(self.cfg.dtype != "fp32"),
                ):
                    pred = self.model.forward_training(
                        features,
                        num_recycles=min(self.cfg.max_recycles, 4),
                        diffusion_batch_size=1,
                        mini_rollout_steps=self.cfg.mini_rollout_steps,
                        mini_rollout_samples=1,
                        seed=0,
                    )
                    _, loss_dict = training_loss(pred, features, self.cfg)

                for k, v in loss_dict.items():
                    metrics.setdefault(f"eval/{k}", []).append(v.item())
            except Exception as e:
                logger.warning(f"Eval error: {e}")
                continue

        self.model.train()

        if self.ema is not None:
            self.ema.restore(self.model)

        return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

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
        ckpts = [c for c in ckpts if "final" not in c.stem]
        while len(ckpts) > self.cfg.max_checkpoints:
            old = ckpts.pop(0)
            old.unlink()
            logger.info(f"Removed old checkpoint: {old}")

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

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

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
        # DDP-friendly metric accumulator: per-step ``update`` is local-only,
        # one ``all_reduce`` per log boundary. Placed AFTER backward so DDP
        # has already synced ranks → the meter's all-reduce is essentially
        # free (no straggler barrier).
        meters = DistributedAverageMeters(self.device)
        t0 = time.time()
        t_last_log = t0  # wall-clock at the previous log emission
        # Per-microstep wall-clock for ``log_microsteps`` debug logging.
        # Updated AFTER the optimizer-step so it captures forward+backward+
        # optimizer time of one microstep.
        t_last_microstep = t0

        while self.effective_step < self.cfg.max_steps:
            sampler = train_dl.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for features in train_dl:
                if self.effective_step >= self.cfg.max_steps:
                    break

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

                # Distributed NaN check
                if self._check_nan(loss):
                    if self.is_main:
                        logger.warning(f"Step {self.effective_step}: NaN/Inf loss, skipping")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    del loss, loss_dict
                    continue

                self._optimizer_step(loss)

                # Accumulate logging metrics locally (no collective).
                meters.update(loss_dict, n=1)

                # Per-microstep timing log (debug). Triggered every microstep
                # when ``cfg.log_microsteps`` is True. Reports per-microstep
                # wall-clock and per-example wall-clock so iteration cost can
                # be measured before reaching an effective optimizer step.
                if cfg.log_microsteps and self.is_main:
                    now = time.time()
                    dt_micro = now - t_last_microstep
                    t_last_microstep = now
                    # System-wide throughput: examples processed per second.
                    # Each microstep processes ``world_size × batch_size_per_rank``
                    # examples in ``dt_micro`` seconds.
                    samples_per_microstep = self.world_size * cfg.batch_size_per_rank
                    samples_per_sec = samples_per_microstep / max(dt_micro, 1e-9)
                    in_step = self.global_step % cfg.accumulate_grad_batches
                    logger.info(
                        f"micro: effective={self.effective_step} "
                        f"global={self.global_step} "
                        f"in_step={in_step}/{cfg.accumulate_grad_batches} "
                        f"recycles={self._last_num_recycles} "
                        f"loss={loss.item():.4f} "
                        f"dt_micro={dt_micro:.2f}s samples/s={samples_per_sec:.3f}"
                    )
                elif cfg.log_microsteps:
                    # Non-main ranks still need to keep their wall-clock fresh.
                    t_last_microstep = time.time()

                # Release forward-pass outputs and input features immediately.
                # Without this, local vars hold references to large GPU tensors
                # (denoised [S,N,M,3], atom_gt_positions [N,M,3], etc.) until
                # Python rebinds them at the next iteration — wasting GPU memory
                # during the dataloader fetch and gradient accumulation micro-steps.
                del loss, loss_dict, features

                # --- Logging ---
                # All ranks call ``meters.sync()`` (it issues an all_reduce),
                # but only rank 0 prints / logs to wandb. The all_reduce is
                # essentially free here because DDP's gradient all-reduce in
                # the just-completed backward already brought all ranks into
                # lockstep — no straggler barrier. Gate on optimizer-step
                # boundaries (``global_step % accumulate == 0``) so the log
                # always fires AFTER ``clip_grad_norm`` has populated
                # ``_last_grad_norm`` — otherwise mid-accumulation logs show
                # ``grad_norm=nan``.
                on_opt_step_boundary = self.global_step % cfg.accumulate_grad_batches == 0
                if on_opt_step_boundary and self.effective_step % self.cfg.log_interval == 0 and meters.count > 0:
                    avg = meters.sync()
                    meters.reset()

                    if self.is_main:
                        grad_norm = self._last_grad_norm.item() if self._last_grad_norm is not None else float("nan")
                        now = time.time()
                        elapsed = now - t0
                        dt = (now - t_last_log) / max(self.cfg.log_interval, 1)
                        t_last_log = now
                        lr = self.optimizer.param_groups[0]["lr"]
                        parts = [f"step={self.effective_step}"]
                        parts.append(f"loss={avg.get('total', 0):.4f}")
                        for k, v in sorted(avg.items()):
                            if k != "total":
                                parts.append(f"{k}={v:.4f}")
                        parts.append(f"grad_norm={grad_norm:.3f}")
                        parts.append(f"lr={lr:.2e}")
                        parts.append(f"dt={dt:.1f}s")
                        parts.append(f"t={elapsed:.0f}s")
                        logger.info(" | ".join(parts))

                        if self._wandb is not None:
                            log_data = {f"train/{k}": v for k, v in avg.items()}
                            log_data["train/lr"] = lr
                            log_data["train/epoch"] = epoch
                            log_data["train/grad_norm"] = grad_norm
                            log_data["train/step_time_s"] = dt
                            self._wandb.log(log_data, step=self.effective_step)

                # --- Evaluation ---
                if eval_dl is not None and self.effective_step % self.cfg.eval_interval == 0:
                    eval_metrics = self._evaluate(eval_dl)
                    if self.is_main:
                        logger.info(f"Eval at step {self.effective_step}: {eval_metrics}")
                        if self._wandb is not None:
                            self._wandb.log(eval_metrics, step=self.effective_step)

                # --- Checkpoint ---
                if self.is_main and self.effective_step % self.cfg.checkpoint_interval == 0:
                    ckpt_path = output_dir / f"checkpoint_{self.effective_step}.pt"
                    self._save_checkpoint(ckpt_path)

            epoch += 1

        # Final checkpoint
        if self.is_main:
            self._save_checkpoint(output_dir / "checkpoint_final.pt")
            logger.info(f"Training complete at step {self.effective_step}")

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
