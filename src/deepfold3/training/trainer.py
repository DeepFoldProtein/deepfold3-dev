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
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from deepfold3.data.module.collate import collate_fn
from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.training.accumulator import AccumulationLoop
from deepfold3.training.callbacks import CallbackHub
from deepfold3.training.callbacks.ema_callback import EMACallback
from deepfold3.training.checkpointing import CheckpointManager, load_pretrained_weights
from deepfold3.training.data import build_dataloaders
from deepfold3.training.distributed import DistInfo, destroy_process_group_safely, init_process_group
from deepfold3.training.ema import EMA
from deepfold3.training.eval import run_eval
from deepfold3.training.logging_payload import build_step_payload
from deepfold3.training.meters import DistributedAverageMeters
from deepfold3.training.optim import (
    WrappedOptimizer,
    build_optimizer,
    freeze_for_confidence,
    is_chunked_diffusion_active,
    register_reduce_hook,
    wrap_ddp,
)
from deepfold3.training.rank_zero import rank_zero_info, rank_zero_warn
from deepfold3.training.scheduler import AF3LRScheduler, build_scheduler
from deepfold3.training.signals import CheckpointRequestHandler, ShutdownCoordinator
from deepfold3.training.state import TrainState, seed_everything
from deepfold3.training.step import scaled_backward, training_step
from deepfold3.training.tracking import StdoutTracker, Tracker, TrackerHub, WandBTracker

# Bucket-padded collation keeps Triton kernel input shapes (and the
# persistent autotune cache keys) deterministic across microsteps.
_collate_bucket_pad = partial(collate_fn, bucket_pad=True)

logger = logging.getLogger(__name__)


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

        self.model: DeepFold3 | None = None
        self.ddp_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: AF3LRScheduler | None = None
        self.ema: EMA | None = None
        self.scaler: GradScaler | None = None
        self.trackers: TrackerHub | None = None
        self.accumulator: AccumulationLoop | None = None
        self.ckpt_manager: CheckpointManager | None = None
        self.wrapped_optimizer: WrappedOptimizer | None = None
        self.callbacks: CallbackHub = CallbackHub()
        self.dist: DistInfo | None = None
        self.state = TrainState()
        self._train_sampler: object | None = None

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Bring up the process group — delegates to :func:`init_process_group`."""
        self.dist = init_process_group(timeout_s=self.cfg.collective_timeout_s)

    def _seed_everything(self) -> None:
        """Seed all RNGs with a per-rank offset."""
        seed_everything(self.cfg.seed, self.dist.rank, deterministic=self.cfg.deterministic)

    # ------------------------------------------------------------------
    # Model and optimizer setup
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Create model, optimizer, scheduler, EMA, and wrap in DDP."""
        self.model = DeepFold3(self.model_cfg).to(self.dist.device)

        # Load pretrained weights if specified
        if self.cfg.params_path:
            self._load_pretrained(self.cfg.params_path)

        # Freeze parameters for confidence-only training (stage 3)
        if self.cfg.train_confidence_only:
            freeze_for_confidence(self.model)

        self.ddp_model, use_static_graph = wrap_ddp(self.model, self.dist, self.cfg)
        self._install_precision_policy()
        self.accumulator = AccumulationLoop(
            ddp_model=self.ddp_model,
            dist=self.dist,
            num_steps=self.cfg.accumulate_grad_batches,
        )
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        # EMA — owned by the trainer (for eval swap + checkpoint save/load),
        # driven by ``EMACallback`` on every successful optimizer step.
        if self.cfg.ema_decay > 0:
            self.ema = EMA(self.model, decay=self.cfg.ema_decay)
            self.callbacks.append(EMACallback(self.ema))

        # Mixed precision scaler (only for fp16; bf16 doesn't need scaling)
        self.scaler = GradScaler("cuda", enabled=(self.cfg.dtype == "fp16"))

        self.wrapped_optimizer = WrappedOptimizer(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model=self.model,
            dist_info=self.dist,
            cfg=self.cfg,
        )

        self.ckpt_manager = CheckpointManager(
            max_checkpoints=self.cfg.max_checkpoints,
            strict=self.cfg.load_strict,
            device=self.dist.device,
        )

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        rank_zero_info(f"Model parameters: {n_params:.1f}M total, {n_trainable:.1f}M trainable")
        rank_zero_info(
            f"DDP: world_size={self.dist.world_size}, static_graph={use_static_graph}, "
            f"accumulate_grad_batches={self.cfg.accumulate_grad_batches}"
        )

    def _install_precision_policy(self) -> None:
        """Validate cross-config precision invariants and install the comm hook.

        Runs after ``wrap_ddp`` (needs the DDP-wrapped model). Fails
        loud on combinations that would otherwise silently drift:
        bf16 grad-accum across microsteps drops small contributions;
        bf16 reduce on an fp32-compute run downcasts for no benefit.
        """
        precision = self.model_cfg.precision
        if precision.grad_acc_dtype == "bf16" and self.cfg.accumulate_grad_batches > 1:
            raise ValueError(
                f"grad_acc_dtype='bf16' is incompatible with accumulate_grad_batches="
                f"{self.cfg.accumulate_grad_batches} (>1): bf16 rounding silently drops "
                "small microstep contributions from the accumulator. Use 'fp32' for grad_acc."
            )
        if self.cfg.dtype == "fp32" and precision.reduce_dtype == "bf16":
            raise ValueError(
                "reduce_dtype='bf16' is incompatible with compute dtype='fp32' — the hook "
                "downcasts fp32 grads to bf16 before allreduce, losing precision without "
                "any bandwidth benefit in an fp32 run."
            )

        comm_hook_name = (
            register_reduce_hook(self.ddp_model, precision.reduce_dtype)
            if self.dist.is_distributed
            else "default (fp32 allreduce)"
        )
        rank_zero_info(
            "Precision policy: compute=%s, grad_acc=%s, reduce=%s, loss=%s; comm_hook=%s",
            self.cfg.dtype,
            precision.grad_acc_dtype,
            precision.reduce_dtype,
            precision.loss_dtype,
            comm_hook_name,
        )

    def _load_pretrained(self, path: str) -> None:
        """Thin wrapper around :func:`load_pretrained_weights`."""
        load_pretrained_weights(self.model, path, strict=self.cfg.load_strict, device=self.dist.device)

    def _init_trackers(self) -> None:
        """Build a :class:`TrackerHub` from the training config.

        Non-main ranks get an empty hub so ``log()`` is a safe no-op on
        every rank. On rank 0 the hub always contains a ``StdoutTracker``
        plus a ``WandBTracker`` when ``wandb_enabled`` + a non-empty
        ``wandb_project`` are set.
        """
        trackers: list[Tracker] = [StdoutTracker()]
        if self.dist.is_main and self.cfg.wandb_enabled:
            if not self.cfg.wandb_project:
                rank_zero_warn("wandb_enabled=True but wandb_project is empty; skipping wandb init")
            else:
                run_name = self.cfg.wandb_run_name or Path(self.cfg.output_dir).name
                trackers.append(
                    WandBTracker(
                        project=self.cfg.wandb_project,
                        entity=self.cfg.wandb_entity,
                        run_name=run_name,
                    )
                )
        self.trackers = TrackerHub(trackers)
        self.trackers.start(dataclasses.asdict(self.cfg))

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _init_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Thin wrapper around :func:`build_dataloaders`."""
        return build_dataloaders(
            cfg=self.cfg,
            model_cfg=self.model_cfg,
            dist=self.dist,
            records=self.records,
            eval_records=self.eval_records,
            distillation_records=self.distillation_records,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _training_step(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Thin wrapper around :func:`training_step` — keeps the call sites tidy."""
        # MUST pass the DDP-wrapped model so that DDP's ``_pre_forward`` runs
        # and primes the reducer for the upcoming backward — otherwise the
        # bucket AllReduce is never scheduled and ranks train independently.
        total_loss, loss_dict, num_recycles = training_step(
            self.ddp_model,
            features,
            cfg=self.cfg,
            model_cfg=self.model_cfg,
            global_step=self.state.global_step,
            accumulator=self.accumulator,
            scaler=self.scaler,
            device=self.dist.device,
        )
        self.state.last_num_recycles = num_recycles
        return total_loss, loss_dict

    def _chunked_diffusion_active(self) -> bool:
        return is_chunked_diffusion_active(self.cfg)

    def _scaled_backward(self, loss: torch.Tensor) -> None:
        scaled_backward(self.scaler, loss, self.cfg.accumulate_grad_batches)

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self, loss: torch.Tensor, record_ids: list[str | None] | None = None) -> bool:
        """Accumulate the microstep's NaN flag, backward, and — on sync — step.

        Delegates the NaN collective + unscale + clip + step contract
        to :class:`WrappedOptimizer`; this shell handles the
        no_sync/sync-step DDP gating and EMA update. ``record_ids`` are
        the dataset ids of the samples in this microstep's batch, used
        for NaN-skip diagnostics.
        """
        self.wrapped_optimizer.add_microstep_loss(loss, record_ids=record_ids)

        is_sync_step = self.accumulator.is_sync_step(self.state.global_step)
        with self.accumulator.step_context(is_sync_step=is_sync_step):
            self._scaled_backward(loss)
        self.state.global_step += 1

        if not is_sync_step:
            return False

        result = self.wrapped_optimizer.step_or_skip(self.state)
        if result.advanced:
            self.callbacks.on_step_end(self, result)
        else:
            self.callbacks.on_skip_step(self, result)
        return result.advanced

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _run_eval_and_track_best(self, eval_dl: DataLoader, output_dir: Path) -> None:
        """Reseed eval sampler, run one eval pass, log metrics, save best checkpoint."""
        if hasattr(eval_dl.sampler, "set_epoch"):
            eval_dl.sampler.set_epoch(self.state.effective_step)
        eval_metrics = run_eval(
            self.model,
            self.ema,
            eval_dl,
            self.cfg,
            self.model_cfg,
            optimizer=self.optimizer,
            device=self.dist.device,
            log_microsteps=self.cfg.log_microsteps,
        )
        eval_total = eval_metrics.get("eval/total", float("inf"))
        save_best = eval_total < self.state.best_eval_loss
        if self.dist.is_main:
            logger.info(f"Eval at step {self.state.effective_step}: {eval_metrics}")
            # Inject ``train/step`` so wandb's ``step_metric="train/step"``
            # places eval points on the correct x-axis. Without it, an eval
            # fired before any train-step log call (e.g. eval_before on
            # resume) is charted at the wandb internal step (=0 for a fresh
            # run) instead of ``effective_step``.
            self.trackers.log(
                {**eval_metrics, "train/step": self.state.effective_step},
                step=self.state.effective_step,
            )
        # Broadcast the save decision so every rank participates in the
        # RNG-gather inside ``_save_checkpoint``. ``eval_total`` may be
        # rank-0-only when meters aren't all-reduced, so don't trust it
        # on non-main ranks for the comparison.
        if self.dist.is_distributed:
            flag = torch.tensor(int(save_best), device=self.dist.device)
            torch_dist.broadcast(flag, src=0)
            save_best = bool(flag.item())
        if save_best:
            self.state.best_eval_loss = eval_total
            self._save_checkpoint(output_dir / "checkpoint_best.pt")
            if self.dist.is_main:
                logger.info(f"New best eval loss: {eval_total:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path) -> None:
        """Save checkpoint via :class:`CheckpointManager`.

        All ranks call this so the cross-rank RNG-state gather can run;
        only rank 0 actually writes the file. Callers that previously
        guarded with ``self.dist.is_main`` no longer need to (the gate
        moved into ``CheckpointManager.save``).
        """
        self.ckpt_manager.save(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            scaler=self.scaler,
            state=self.state,
            model_cfg=self.model_cfg,
            train_cfg=self.cfg,
            dist_info=self.dist,
            train_sampler=self._train_sampler,
        )

    @staticmethod
    def _find_latest_checkpoint(output_dir: Path) -> Path | None:
        return CheckpointManager.find_latest(output_dir)

    def _load_checkpoint(self, path: Path) -> None:
        """Load checkpoint via :class:`CheckpointManager`.

        Computes ``steps_per_epoch`` and ``samples_per_step_per_rank``
        so old checkpoints (no ``epoch`` / ``sampler`` keys) can derive
        a sampler-epoch + within-epoch position from ``effective_step``
        — the sampler RNG keys on epoch, so we want to land in the
        right basin and skip records already trained on, even when
        resuming from a pre-sampler-persistence ckpt.
        """
        cfg = self.cfg
        effective_batch_size = cfg.batch_size_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        steps_per_epoch = max(1, cfg.epoch_size // max(1, effective_batch_size)) if cfg.epoch_size > 0 else 0
        samples_per_step_per_rank = cfg.batch_size_per_rank * cfg.accumulate_grad_batches
        self.ckpt_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            scaler=self.scaler,
            state=self.state,
            train_cfg=self.cfg,
            dist_info=self.dist,
            steps_per_epoch=steps_per_epoch,
            train_sampler=self._train_sampler,
            samples_per_step_per_rank=samples_per_step_per_rank,
        )

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
        self._init_trackers()

        train_dl, eval_dl = self._init_dataloaders()
        # Stash the train sampler so the checkpoint manager can persist
        # its within-epoch position (only one sampler per run; capturing
        # it once here avoids re-deriving from ``train_dl.sampler`` at
        # every save site).
        self._train_sampler = train_dl.sampler
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config (all CLI/YAML overrides applied) for reproducibility.
        # Layout mirrors configs/*.yaml so the dump can be re-fed via --config.
        if self.dist.is_main:
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
                    yaml.safe_dump(_yaml_safe(self.provenance), f, default_flow_style=False, sort_keys=False)
                logger.info(f"Run provenance saved to {prov_out}")

        # Auto-resume: find latest checkpoint in output_dir if none specified
        resumed = False
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))
            resumed = True
        elif self.cfg.auto_resume:
            latest = self._find_latest_checkpoint(output_dir)
            if latest is not None:
                self._load_checkpoint(latest)
                resumed = True

        # Resume LR override: optimizer.load_state_dict and
        # scheduler.load_state_dict both restore the lr/initial_lr/base_lrs
        # baked into the checkpoint, which silently shadows any --config
        # change. When the configured lr differs from the saved peak, force
        # the new value into both so the schedule honors the override.
        # SF-AdamW's accumulator state (z buffer, lr_max, weight_sum) is
        # preserved — only the per-step `lr`/`initial_lr` is bumped.
        if resumed and self.optimizer.param_groups:
            saved_lr = float(self.optimizer.param_groups[0].get("initial_lr", self.optimizer.param_groups[0]["lr"]))
            target_lr = float(self.cfg.lr)
            if abs(saved_lr - target_lr) > 1e-12:
                rank_zero_info(
                    f"[resume] LR override: {saved_lr:.3e} → {target_lr:.3e} "
                    "(forcing optimizer.param_groups['lr','initial_lr'] + scheduler.base_lrs; "
                    "SF accumulator state preserved)"
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = target_lr
                    g["initial_lr"] = target_lr
                base_lrs = getattr(self.scheduler, "base_lrs", None)
                if base_lrs is not None:
                    self.scheduler.base_lrs = [target_lr] * len(base_lrs)

        cfg = self.cfg
        tokens_per_rank = cfg.max_tokens * cfg.batch_size_per_rank
        effective_batch_size = cfg.batch_size_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        tokens_per_opt_step = tokens_per_rank * self.dist.world_size * cfg.accumulate_grad_batches
        rank_zero_info(f"Starting training: max_steps={cfg.max_steps}")
        rank_zero_info(f"  confidence_only={cfg.train_confidence_only}")
        rank_zero_info(f"  lr={cfg.lr}, warmup={cfg.warmup_steps}, scheduler={cfg.finetune_scheduler}")
        rank_zero_info(
            f"  crop: max_tokens={cfg.max_tokens}, max_atoms={cfg.max_atoms}, "
            f"max_msa_seqs={cfg.max_msa_seqs}, max_recycles={cfg.max_recycles}"
        )
        rank_zero_info(
            f"  batch: world_size={self.dist.world_size}, batch_per_rank={cfg.batch_size_per_rank}, "
            f"accumulate={cfg.accumulate_grad_batches} → "
            f"effective_batch_size={effective_batch_size} (tokens_per_opt_step={tokens_per_opt_step})"
        )
        rank_zero_info(
            f"  diffusion: batch_size={cfg.diffusion_batch_size}, chunk_size={cfg.diffusion_chunk_size}, "
            f"mini_rollout_steps={cfg.mini_rollout_steps}, mini_rollout_samples={cfg.mini_rollout_samples}"
        )
        rank_zero_info(f"  dtype={cfg.dtype}, grad_clip={cfg.grad_clip_norm}, ema_decay={cfg.ema_decay}")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Advance through ``state.epoch`` (carries over a resumed value
        # when present in the checkpoint, or is derived from
        # ``effective_step // steps_per_epoch`` for old checkpoints).
        meters = DistributedAverageMeters(self.dist.device)
        t0 = time.monotonic()
        t_last_ckpt = t0
        # Pure microstep compute time accumulator (excludes eval / checkpoint /
        # summary-log wall-clock). Flushed at each summary log.
        microstep_compute_time = 0.0

        # SIGUSR2 → request immediate checkpoint (preemption safety)
        # SIGTERM/SIGINT → checkpoint + graceful exit (SLURM preemption, kill, Ctrl+C)
        shutdown = ShutdownCoordinator(self.dist)
        ckpt_request = CheckpointRequestHandler()

        # Single-track stability probe (file-only logging, no WandB).
        # Attached only on rank 0 — the per-block stats are local to each
        # rank's forward and we don't need a cross-rank reduction.
        self._single_track_probe = None
        if cfg.single_track_probe and self.dist.is_main:
            from deepfold3.training.single_track_probe import SingleTrackProbe

            probe_path = output_dir / "single_track_probe.jsonl"
            self._single_track_probe = SingleTrackProbe(self.model, probe_path)
            rank_zero_info(
                f"Single-track probe enabled → {probe_path} (every {cfg.single_track_probe_interval} steps)"
            )

        if cfg.eval_before and eval_dl is not None:
            self._run_eval_and_track_best(eval_dl, output_dir)

        while self.state.effective_step < self.cfg.max_steps:
            sampler = train_dl.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.state.epoch)

            for features in train_dl:
                if self.state.effective_step >= self.cfg.max_steps:
                    break

                t_microstep_start = time.monotonic()
                # Non-tensor diagnostic: pop before the forward pass so the
                # model / ``prepare_features`` never see a string. Threaded
                # into the optimizer so NaN-skip warnings can name the
                # offending record(s).
                record_ids = features.pop("_record_id", None)
                # Single-track probe gate. Fire on the LAST microstep of the
                # accumulation window that builds toward a target effective
                # step matching the probe interval. Captures per-block stats
                # during this microstep's forward pass.
                probe = self._single_track_probe
                if probe is not None:
                    # Clear any stale flag from a skipped opt step (NaN/Inf)
                    # so non-probe microsteps don't accidentally accumulate.
                    probe.discard_step()
                    last_microstep_in_window = (
                        (self.state.global_step + 1) % self.cfg.accumulate_grad_batches
                    ) == 0
                    target_step = self.state.effective_step + (1 if last_microstep_in_window else 0)
                    should_probe = (
                        last_microstep_in_window
                        and target_step > 0
                        and target_step % self.cfg.single_track_probe_interval == 0
                    )
                    if should_probe:
                        probe.begin_step(target_step)
                # No except-and-continue on OOM/exception: ``continue`` skips
                # the per-microstep shutdown_flag all_reduce + the sync-boundary
                # NaN all_gather, which desynchronizes collective participation
                # across ranks → peers hang on a 600s NCCL watchdog. Fail hard
                # so torchrun reaps the whole process group; the surviving
                # traceback (logged via the rank-tagged handler in train.py)
                # identifies the offending rank.
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.cfg.torch_dtype,
                    enabled=(self.cfg.dtype != "fp32"),
                ):
                    loss, loss_dict = self._training_step(features)

                # NaN/Inf detection is deferred to the opt-step boundary inside
                # ``_optimizer_step`` — a single all-reduce per window replaces
                # one-per-microstep. Backward runs unconditionally here; any
                # NaN grads are discarded at the boundary before the opt step.
                on_opt_step_boundary = self._optimizer_step(loss, record_ids=record_ids)

                # Flush probe row (rank-0 only); pass per-microstep loss
                # (total + components from ``loss_dict``) so the probe JSONL
                # is self-contained — plot scripts don't need to align with
                # the slower ``log_interval`` stdout cadence.
                if probe is not None and probe.enabled:
                    probe_extra = {"loss": float(loss.detach().item())}
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            probe_extra[k] = float(v.detach().item())
                    probe.end_step(extra=probe_extra)

                # Accumulate logging metrics locally (no collective).
                # Inject token count so it rides the same all-reduce as losses.
                seq_mask = features.get("seq_mask")
                n_tokens = (
                    seq_mask.sum()
                    if seq_mask is not None
                    else torch.tensor(cfg.max_tokens * cfg.batch_size_per_rank, device=self.dist.device)
                )
                loss_dict["_tokens"] = n_tokens
                meters.update(loss_dict, n=1)

                # Pure microstep compute time (forward + backward + step). Excludes
                # anything that happens between iterations (eval, checkpoint, summary
                # log), so rate / ETA stay stable across eval boundaries. Accumulated
                # into ``microstep_compute_time`` for the effective-step summary log.
                dt_micro = time.monotonic() - t_microstep_start
                microstep_compute_time += dt_micro

                if cfg.log_microsteps:
                    samples_per_microstep = self.dist.world_size * cfg.batch_size_per_rank
                    samples_per_sec = samples_per_microstep / max(dt_micro, 1e-9)
                    # Position within the current accumulation window: 1..accumulate
                    # (the accumulate-th microstep is the sync step that fires
                    # optimizer.step and increments effective_step).  ``target`` is
                    # the effective_step this microstep is building toward — on the
                    # sync step it equals the just-incremented effective_step;
                    # otherwise it's effective_step+1.
                    in_step = ((self.state.global_step - 1) % cfg.accumulate_grad_batches) + 1
                    target = self.state.effective_step if on_opt_step_boundary else self.state.effective_step + 1
                    rank_zero_info(
                        f"micro: target={target} "
                        f"global={self.state.global_step} "
                        f"in_step={in_step}/{cfg.accumulate_grad_batches} "
                        f"recycles={self.state.last_num_recycles} "
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
                # Gate fires after clip_grad_norm populates ``_last_grad_norm``.
                # Collective: ``meters.sync`` issues ``all_gather_object`` +
                # ``all_reduce``, so the gate MUST be identical on all ranks
                # (no per-rank conditions like ``meters.count > 0``).
                if on_opt_step_boundary and self.state.effective_step % self.cfg.log_interval == 0:
                    avg, sums = meters.sync()
                    meters.reset()

                    # Flush the compute-time accumulator regardless of is_main so
                    # all ranks stay in sync if the flag ever changes.
                    compute_time_window = microstep_compute_time
                    microstep_compute_time = 0.0

                    if self.dist.is_main:
                        grad_norm = (
                            self.state.last_grad_norm.item() if self.state.last_grad_norm is not None else float("nan")
                        )
                        elapsed = time.monotonic() - t0
                        # ``dt`` is per-effective-step compute time (forward+backward+step).
                        # Eval / checkpoint / logging time is intentionally excluded so the
                        # displayed rate/ETA reflect pure training throughput.
                        dt = compute_time_window / max(self.cfg.log_interval, 1)
                        # Schedule-Free folds warmup + bias-correction into ``scheduled_lr``;
                        # ``lr`` stays at the configured peak. Prefer the scheduled value so
                        # the log/WandB curve reflects what's actually applied.
                        g = self.optimizer.param_groups[0]
                        lr = g.get("scheduled_lr", g["lr"])

                        # Throughput (global token sum from all ranks via all-reduce).
                        # Same compute-time denominator as ``dt`` for a consistent picture.
                        tok_per_sec = sums.pop("_tokens", 0) / max(compute_time_window, 1e-9)
                        avg.pop("_tokens", None)

                        # GPU memory
                        peak_mem_gb = torch.cuda.max_memory_allocated() / (1 << 30) if torch.cuda.is_available() else 0

                        # ETA
                        remaining = cfg.max_steps - self.state.effective_step
                        eta_sec = remaining * dt
                        eta_h, eta_rem = divmod(int(eta_sec), 3600)
                        eta_str = f"{eta_h}:{eta_rem // 60:02d}" if eta_h > 0 else f"{int(eta_sec // 60)}m"

                        parts = [f"step={self.state.effective_step}"]
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

                        self.trackers.log(
                            build_step_payload(
                                avg=avg,
                                effective_step=self.state.effective_step,
                                lr=lr,
                                epoch=self.state.epoch,
                                grad_norm=grad_norm,
                                dt=dt,
                                tok_per_sec=tok_per_sec,
                                peak_mem_gb=peak_mem_gb,
                            ),
                            step=self.state.effective_step,
                        )

                # --- Checkpoint (before eval so weights are saved even if eval crashes) ---
                # ``effective_step`` is identical on all ranks, so the
                # interval check decides locally without a broadcast.
                # The time-based and signal-based triggers are rank-0
                # local; we broadcast a single bool so save fires
                # synchronously on every rank (RNG-gather requirement).
                need_ckpt = on_opt_step_boundary and (
                    self.state.effective_step % self.cfg.checkpoint_interval == 0
                )
                if on_opt_step_boundary and self.dist.is_main and not need_ckpt:
                    if (
                        cfg.checkpoint_every_hours > 0
                        and (time.monotonic() - t_last_ckpt) / 3600 >= cfg.checkpoint_every_hours
                    ):
                        need_ckpt = True
                    elif ckpt_request.consume():
                        need_ckpt = True
                        logger.info("Signal-triggered checkpoint requested")
                if self.dist.is_distributed and on_opt_step_boundary:
                    flag = torch.tensor(int(need_ckpt), device=self.dist.device)
                    torch_dist.broadcast(flag, src=0)
                    need_ckpt = bool(flag.item())

                if need_ckpt:
                    self._save_checkpoint(output_dir / f"checkpoint_{self.state.effective_step}.pt")
                    t_last_ckpt = time.monotonic()

                # --- Evaluation ---
                # Gate on ``on_opt_step_boundary`` so eval fires exactly once
                # per effective step — without the gate, ``effective_step``
                # stays constant across ``accumulate_grad_batches`` microsteps
                # and the check would trigger on every one of them.
                if (
                    on_opt_step_boundary
                    and eval_dl is not None
                    and self.state.effective_step % self.cfg.eval_interval == 0
                ):
                    self._run_eval_and_track_best(eval_dl, output_dir)

                # --- Shutdown ---
                # Propagate on every microstep (cheap 1-elem all_reduce) so
                # all ranks bail together, not just at opt-step boundaries.
                shutdown_now = shutdown.tick()

                if shutdown_now:
                    # Mid-window: drop partial grads and rewind global_step
                    # to the last completed boundary so resume starts a fresh
                    # accumulation window. scheduler / scaler / EMA stay as-is
                    # (they only advance on sync steps).
                    partial = 0
                    if not on_opt_step_boundary:
                        partial = self.state.global_step - self.state.effective_step * cfg.accumulate_grad_batches
                        self.optimizer.zero_grad(set_to_none=True)
                        self.state.global_step = self.state.effective_step * cfg.accumulate_grad_batches
                    if self.dist.is_main:
                        if partial:
                            logger.info(
                                f"Shutdown mid-window at step {self.state.effective_step}: discarded "
                                f"{partial}/{cfg.accumulate_grad_batches} partial microsteps"
                            )
                        else:
                            logger.info(f"Shutdown at step {self.state.effective_step}")
                    # All ranks must call save so the RNG-gather collective
                    # runs on every rank (rank-0 write gate is inside).
                    self._save_checkpoint(output_dir / f"checkpoint_{self.state.effective_step}.pt")
                    break

            if shutdown.requested:
                break
            self.state.epoch += 1

        # Final checkpoint — call on every rank so the cross-rank RNG
        # gather inside ``CheckpointManager.save`` runs on all ranks
        # (the rank-0 write gate is inside the manager).
        if not shutdown.requested:
            self._save_checkpoint(output_dir / "checkpoint_final.pt")
        if self.dist.is_main:
            logger.info(
                f"Training {'interrupted' if shutdown.requested else 'complete'} at step {self.state.effective_step}"
            )

        if self.trackers is not None:
            self.trackers.finish()

        # Collective teardown — see ``destroy_process_group_safely`` for the
        # ordering rationale (must live outside ``try/finally``).
        destroy_process_group_safely()
