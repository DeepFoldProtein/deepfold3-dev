"""Optimizer, scheduler, DDP wrapping, and the NaN-hardened step contract.

:class:`WrappedOptimizer` owns the full per-microstep NaN handling and
the sync-boundary collective. It is the one place where the
``all_gather_into_tensor`` across ranks runs; dropping the call on any
rank desynchronizes collective participation and hangs NCCL (see
``memory/project_try_except_continue_deadlock.md``), so the call MUST
run unconditionally on every rank when ``is_sync_step`` is true.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from deepfold3.model.config import TrainingConfig
from deepfold3.training.distributed import DistInfo
from deepfold3.training.state import TrainState

logger = logging.getLogger(__name__)


def is_chunked_diffusion_active(cfg: TrainingConfig) -> bool:
    """Return True when diffusion is run in fixed-size chunks inside each microstep.

    Kept as a pure function of the config so ``wrap_ddp`` can decide
    whether to force-disable ``static_graph`` without instantiating the
    trainer.
    """
    return cfg.diffusion_chunk_size > 0 and cfg.diffusion_batch_size % cfg.diffusion_chunk_size == 0


def freeze_for_confidence(model: nn.Module) -> None:
    """Freeze all parameters except the confidence head (stage-3 training)."""
    for name, param in model.named_parameters():
        if "confidence" not in name:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Confidence-only training: {trainable / 1e6:.1f}M trainable parameters")


def build_optimizer(model: nn.Module, cfg: TrainingConfig, *, is_main: bool) -> torch.optim.Optimizer:
    """Build AdamW with decoupled weight decay.

    Protenix / nanoGPT convention: decay 2D+ params (matmul weights,
    embeddings); skip 1D params (LayerNorm gamma/beta, biases) —
    applying L2 to scale/shift terms collapses them and hurts training
    stability.

    Additional AF3-specific refinement: many AF3 projections are
    ``LinearNoBias(1, C)`` or ``LinearNoBias(C, 1)`` — scalar-input or
    scalar-output linears that behave as per-channel scale factors.
    Their weight is technically 2D (``[C, 1]`` or ``[1, C]``) but
    semantically equivalent to a 1D param. Treat any 2D weight where
    ``min(shape) == 1`` as no-decay.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or min(param.shape) == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Fused AdamW on CUDA: single kernel launch per param group,
    # ~1.3-1.5x faster than the foreach implementation at our scale.
    use_fused_adam = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=use_fused_adam,
    )
    if is_main:
        n_decay = sum(p.numel() for p in decay_params) / 1e6
        n_nodecay = sum(p.numel() for p in no_decay_params) / 1e6
        logger.info(
            f"AdamW: fused={use_fused_adam}, weight_decay={cfg.weight_decay} "
            f"({n_decay:.1f}M decayed, {n_nodecay:.2f}M no-decay)"
        )
    return optimizer


def wrap_ddp(model: nn.Module, dist: DistInfo, cfg: TrainingConfig) -> tuple[nn.Module, bool]:
    """Wrap the model in :class:`DistributedDataParallel` (if distributed).

    Returns ``(wrapped_model, use_static_graph)``. The static-graph
    decision matrix:

    * ``static_graph=True`` is safe in the unchunked path — recycles
      1..N-1 run inside ``torch.no_grad()`` (model.py:396-409), so only
      the last recycle is in the autograd graph and the graph topology
      is identical iteration-to-iteration regardless of
      ``num_recycles``. Gradient accumulation uses ``no_sync()`` on
      non-final microsteps, so only the final backward per optimizer
      step triggers DDP sync. Confidence-loss resolution gating is a
      tensor multiply (loss.py) rather than a Python branch, so the set
      of grad-receiving parameters is identical every step.
    * ``static_graph=True`` is INCOMPATIBLE with ``model.no_sync()``.
      PyTorch raises if the two are combined. ``no_sync`` is used in
      two places: (a) chunked-diffusion to defer DDP all_reduce until
      after the outer aux backward, and (b) gradient-accumulation
      microsteps to defer sync until the final microstep. We
      auto-disable ``static_graph`` when either is active.
    """
    chunked_diffusion = is_chunked_diffusion_active(cfg)
    needs_no_sync = chunked_diffusion or cfg.accumulate_grad_batches > 1
    use_static_graph = cfg.static_graph and not needs_no_sync

    if not dist.is_distributed:
        return model, use_static_graph

    if dist.is_main and cfg.static_graph and not use_static_graph:
        reasons = []
        if chunked_diffusion:
            reasons.append(f"diffusion_chunk_size={cfg.diffusion_chunk_size}")
        if cfg.accumulate_grad_batches > 1:
            reasons.append(f"accumulate_grad_batches={cfg.accumulate_grad_batches}")
        logger.info(
            "static_graph auto-disabled (%s): model.no_sync() is required, which conflicts with static_graph.",
            ", ".join(reasons),
        )

    ddp_model = DDP(
        model,
        device_ids=[dist.local_rank],
        find_unused_parameters=cfg.find_unused_parameters,
        static_graph=use_static_graph,
        bucket_cap_mb=cfg.bucket_cap_mb,
        gradient_as_bucket_view=cfg.gradient_as_bucket_view,
    )
    return ddp_model, use_static_graph


@dataclass
class StepResult:
    """Outcome of one optimizer-step attempt."""

    advanced: bool
    was_nan: bool
    reason: Literal["loss", "grad"] | None
    grad_norm: torch.Tensor | None
    nan_ranks: list[int]


class WrappedOptimizer:
    """Optimizer wrapper that enforces the NaN/scaler/clip/step contract.

    The trainer calls :meth:`add_microstep_loss` after each microstep
    backward (to OR the local NaN flag into the on-device accumulator)
    and :meth:`step_or_skip` at the sync boundary (to run the single
    cross-rank collective, unscale + clip, and either step or skip).

    Invariants preserved verbatim from the pre-refactor trainer:

    * ``_nan_accum`` is materialised lazily on the first call, shape
      ``[1]`` so ``all_gather_into_tensor`` can concatenate across
      ranks into ``[world_size]``.
    * The NaN collective runs EXACTLY ONCE per opt step on every rank.
    * Two NaN checks: loss-side (from the accumulator) + grad-side
      (after ``clip_grad_norm_`` — catches finite-loss / NaN-grad
      paths like log/sqrt/div).
    * ``cfg.max_consecutive_nan_skips`` raises after N skips in a row.
    * The scheduler is stepped ONLY on a successful opt step (skips
      on NaN leave the LR unchanged).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: GradScaler,
        model: nn.Module,
        dist_info: DistInfo,
        cfg: TrainingConfig,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model = model
        self.dist = dist_info
        self.cfg = cfg
        self._nan_accum: torch.Tensor | None = None
        self._nan_gather_buf: torch.Tensor | None = None
        self._grad_norm_gather_buf: torch.Tensor | None = None

    # ----- per-microstep -----

    def add_microstep_loss(self, loss: torch.Tensor) -> None:
        """OR the local NaN/Inf flag into the on-device accumulator.

        No collective, no CPU sync — this is called on every microstep.
        """
        if self._nan_accum is None:
            self._nan_accum = torch.zeros(1, device=loss.device)
        self._nan_accum.add_((torch.isnan(loss) | torch.isinf(loss)).float().view(1))

    # ----- sync boundary -----

    def step_or_skip(self, state: TrainState) -> StepResult:
        """Run the collective NaN check, then either step or skip.

        MUST be called by EVERY rank when ``is_sync_step`` is true.
        Dropping the call on any rank desyncs the
        ``all_gather_into_tensor`` below and hangs NCCL.
        """
        assert self._nan_accum is not None, "add_microstep_loss must run at least once before step_or_skip"

        # Sync boundary: single collective for NaN check across ranks +
        # all microsteps in this accumulation window. ``all_gather`` so
        # the warning can name the offending rank(s).
        if self.dist.is_distributed:
            if self._nan_gather_buf is None:
                self._nan_gather_buf = torch.zeros(self.dist.world_size, device=self._nan_accum.device)
            dist.all_gather_into_tensor(self._nan_gather_buf, self._nan_accum)
            nan_flags = self._nan_gather_buf
        else:
            nan_flags = self._nan_accum
        nan_detected = bool(nan_flags.any().item())  # one CPU sync per opt step
        self._nan_accum.zero_()

        attempted_step = state.effective_step + 1

        if nan_detected:
            # Any microstep on any rank produced NaN/Inf — discard the
            # whole accumulation window. NaN would have propagated via
            # DDP all_reduce on the sync-step backward.
            self.optimizer.zero_grad(set_to_none=True)
            bad_ranks = (nan_flags > 0).nonzero(as_tuple=False).flatten().tolist()
            if self.dist.is_main:
                logger.warning(
                    f"Step {attempted_step}: NaN/Inf loss in accumulation window (ranks={bad_ranks}), skipping"
                )
            self._account_skip(state, attempted_step, reason="loss")
            return StepResult(advanced=False, was_nan=True, reason="loss", grad_norm=None, nan_ranks=bad_ranks)

        self.scaler.unscale_(self.optimizer)

        # Always compute the global pre-clip grad norm for logging,
        # even when grad_clip_norm <= 0 (use a huge cap as no-op).
        clip = self.cfg.grad_clip_norm if self.cfg.grad_clip_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip).detach()
        state.last_grad_norm = grad_norm

        # clip_grad_norm_ returns a NaN/Inf total when ANY param grad
        # is non-finite, so this single check covers the whole model.
        # Without it, a finite loss with non-finite grads (log/sqrt/div
        # paths) silently corrupts weights via optimizer.step.
        #
        # Collective consensus: even with DDP-synced grads, bf16 rounding
        # in ``clip_grad_norm_``'s sum-of-squared-norms can overflow on
        # one rank but stay finite on another. A per-rank ``isfinite``
        # check then returns different ``advanced`` values → ranks
        # disagree on ``on_opt_step_boundary`` → the log/ckpt/eval gates
        # in ``trainer._run_inner`` diverge → NCCL hang. All-gather the
        # scalar so every rank sees the same vector, then reduce with
        # ``any()`` for identical advance/skip decisions.
        if self.dist.is_distributed:
            if self._grad_norm_gather_buf is None:
                self._grad_norm_gather_buf = torch.zeros(self.dist.world_size, device=grad_norm.device)
            dist.all_gather_into_tensor(self._grad_norm_gather_buf, grad_norm.view(1))
            grad_norms_per_rank = self._grad_norm_gather_buf
        else:
            grad_norms_per_rank = grad_norm.view(1)
        grad_finite_per_rank = torch.isfinite(grad_norms_per_rank)
        if not bool(grad_finite_per_rank.all().item()):
            self.optimizer.zero_grad(set_to_none=True)
            bad_ranks = (~grad_finite_per_rank).nonzero(as_tuple=False).flatten().tolist()
            if self.dist.is_main:
                per_rank = ", ".join(f"rank{r}={grad_norms_per_rank[r].item():.3e}" for r in bad_ranks)
                logger.warning(
                    f"Step {attempted_step}: NaN/Inf in grads (ranks={bad_ranks}, {per_rank}), skipping"
                )
            self._account_skip(state, attempted_step, reason="grad")
            return StepResult(advanced=False, was_nan=True, reason="grad", grad_norm=grad_norm, nan_ranks=bad_ranks)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        state.effective_step += 1
        state.consecutive_nan_skips = 0
        return StepResult(advanced=True, was_nan=False, reason=None, grad_norm=grad_norm, nan_ranks=[])

    def _account_skip(self, state: TrainState, attempted_step: int, *, reason: Literal["loss", "grad"]) -> None:
        """Bump the consecutive-skip counter and raise if it exceeds the limit.

        Every rank raises the same exception when the limit is hit —
        ``RuntimeError`` propagates cleanly through torchrun so all
        ranks die together rather than deadlocking on the next
        collective.
        """
        state.consecutive_nan_skips += 1
        limit = self.cfg.max_consecutive_nan_skips
        if limit > 0 and state.consecutive_nan_skips >= limit:
            msg = (
                f"Aborting: {state.consecutive_nan_skips} consecutive non-finite "
                f"{reason} skips at step {attempted_step} (limit={limit})."
            )
            if self.dist.is_main:
                logger.critical(msg)
            raise RuntimeError(msg)
