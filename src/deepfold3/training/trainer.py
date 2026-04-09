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

logger = logging.getLogger(__name__)


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

        # Wrap in DDP
        if self.world_size > 1:
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.find_unused_parameters,
                static_graph=self.cfg.static_graph,
            )
        else:
            self.ddp_model = self.model

        # Optimizer: separate weight decay for >=2D params (weights) vs <2D (biases, norms)
        decay_params = []
        no_decay_params = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
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
            logger.info(f"DDP: world_size={self.world_size}, static_graph={self.cfg.static_graph}")

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
        """Initialize Weights & Biases logging if configured."""
        if not self.cfg.wandb_project or not self.is_main:
            return
        try:
            import dataclasses

            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name or None,
                config=dataclasses.asdict(self.cfg),
            )
            logger.info(f"WandB initialized: project={self.cfg.wandb_project}")
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
            batch_size=1,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
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
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.cfg.eval_num_workers,
                pin_memory=self.cfg.pin_memory,
            )

        return train_dl, eval_dl

    def _build_train_sampler(self, train_ds: TrainingDataset):
        """Build training sampler — weighted or distributed."""
        has_real_clusters = any(isinstance(c.cluster_id, int) for r in self.records for c in r.chains)

        if has_real_clusters:
            from deepfold3.data.sample.weighted import WeightedClusterSampler

            return WeightedClusterSampler(
                records=self.records,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.cfg.seed,
            )

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
        """Move features to device and squeeze batch dim if batch_size=1."""
        features = {k: v.to(self.device, non_blocking=True) for k, v in features.items()}
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

        # Random number of recycles per batch
        num_recycles = random.randint(1, cfg.max_recycles)

        pred = self.model.forward_training(
            features,
            num_recycles=num_recycles,
            diffusion_batch_size=cfg.diffusion_batch_size,
            mini_rollout_steps=cfg.mini_rollout_steps,
            mini_rollout_samples=cfg.mini_rollout_samples,
            seed=self.global_step,
            fused_ce=cfg.use_fused_ce_loss,
        )

        total_loss, loss_dict = training_loss(pred, features, cfg)
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Backward pass, gradient clipping, and optimizer step.

        Args:
            loss: Scalar loss tensor from ``_training_step()``.
        """
        scaled_loss = loss / self.cfg.accumulate_grad_batches
        self.scaler.scale(scaled_loss).backward()

        self.global_step += 1

        if self.global_step % self.cfg.accumulate_grad_batches == 0:
            self.scaler.unscale_(self.optimizer)

            if self.cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

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
        self._init_distributed()
        self._seed_everything()
        self._init_model()
        self._init_wandb()

        train_dl, eval_dl = self._init_dataloaders()
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

        if self.is_main:
            logger.info(f"Starting training: max_steps={self.cfg.max_steps}")
            logger.info(f"  confidence_only={self.cfg.train_confidence_only}")
            logger.info(f"  accumulate_grad_batches={self.cfg.accumulate_grad_batches}")
            logger.info(f"  lr={self.cfg.lr}, warmup={self.cfg.warmup_steps}")
            logger.info(f"  scheduler={self.cfg.finetune_scheduler}")
            logger.info(f"  diffusion_batch_size={self.cfg.diffusion_batch_size}")
            logger.info(f"  mini_rollout_steps={self.cfg.mini_rollout_steps}")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        running_loss: dict[str, float] = {}
        log_count = 0
        t0 = time.time()

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
                except Exception as e:
                    logger.warning(f"Step {self.effective_step}: skipping batch ({e})")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    continue

                # Distributed NaN check
                if self._check_nan(loss):
                    logger.warning(f"Step {self.effective_step}: NaN/Inf loss, skipping")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    del loss, loss_dict
                    continue

                self._optimizer_step(loss)

                # Accumulate logging metrics as detached tensors (no GPU sync here)
                for k, v in loss_dict.items():
                    if k in running_loss:
                        running_loss[k] = running_loss[k] + v.detach()
                    else:
                        running_loss[k] = v.detach().clone()
                log_count += 1

                # Release forward-pass outputs and input features immediately.
                # Without this, local vars hold references to large GPU tensors
                # (denoised [S,N,M,3], atom_gt_positions [N,M,3], etc.) until
                # Python rebinds them at the next iteration — wasting GPU memory
                # during the dataloader fetch and gradient accumulation micro-steps.
                del loss, loss_dict, features

                # --- Logging ---
                if self.is_main and self.effective_step % self.cfg.log_interval == 0 and log_count > 0:
                    # GPU sync happens here only, at log time
                    avg = {k: (v / log_count).item() for k, v in running_loss.items()}
                    elapsed = time.time() - t0
                    lr = self.optimizer.param_groups[0]["lr"]
                    parts = [f"step={self.effective_step}"]
                    parts.append(f"loss={avg.get('total', 0):.4f}")
                    for k, v in sorted(avg.items()):
                        if k != "total":
                            parts.append(f"{k}={v:.4f}")
                    parts.append(f"lr={lr:.2e}")
                    parts.append(f"t={elapsed:.0f}s")
                    logger.info(" | ".join(parts))

                    if self._wandb is not None:
                        log_data = {f"train/{k}": v for k, v in avg.items()}
                        log_data["train/lr"] = lr
                        log_data["train/epoch"] = epoch
                        self._wandb.log(log_data, step=self.effective_step)

                    running_loss.clear()
                    log_count = 0

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

        if self.world_size > 1:
            dist.destroy_process_group()
