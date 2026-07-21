"""Evaluation pass — EMA-swapped forward over the eval loader.

Exposes :func:`run_eval` as a pure-ish function: it takes the model +
loader + configs and returns the averaged eval metrics. Logging,
tracker fan-out, and best-checkpoint saving live in the trainer.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import fp32_guard
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.ema import EMA
from deepfold3.training.loss import training_loss
from deepfold3.training.rank_zero import rank_zero_info
from deepfold3.training.schedule_free import optimizer_eval_mode

logger = logging.getLogger(__name__)


def _prepare_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in features.items()}


def _reduce_eval_metrics(
    metrics: dict[str, list[float]],
    device: torch.device,
) -> dict[str, float]:
    """All-reduce per-rank eval metrics into a global mean.

    Eval uses ``DistributedSampler`` (each rank sees ``1/world_size`` of the
    eval set), so a per-rank ``mean(values)`` would bias the displayed
    metric — and the best-checkpoint gate — to whichever shard rank 0 drew.
    Pack ``(sum, count)`` per key into one f64 tensor, all_reduce SUM, then
    divide for the true global mean.

    Non-finite per-batch values are dropped (mirrors :class:`DistributedAverageMeters`):
    a single NaN batch on one rank shouldn't poison the eval-pass average.
    Per-rank key sets may diverge if some ranks happen to draw only
    distillation samples — keys are unioned via ``all_gather_object``.
    """
    multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    local_sums: dict[str, float] = {}
    local_counts: dict[str, int] = {}
    for k, vs in metrics.items():
        s = 0.0
        c = 0
        for v in vs:
            if math.isfinite(v):
                s += v
                c += 1
        if c > 0:
            local_sums[k] = s
            local_counts[k] = c

    if multi:
        gathered: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, sorted(local_sums.keys()))
        keys = sorted({k for ks in gathered if ks for k in ks})
    else:
        keys = sorted(local_sums.keys())

    if not keys:
        return {}

    packed = torch.zeros(2 * len(keys), dtype=torch.float64, device=device)
    for i, k in enumerate(keys):
        if k in local_sums:
            packed[2 * i] = local_sums[k]
            packed[2 * i + 1] = float(local_counts[k])
    if multi:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    raw = packed.tolist()
    out: dict[str, float] = {}
    for i, k in enumerate(keys):
        global_sum = raw[2 * i]
        global_count = raw[2 * i + 1]
        out[k] = global_sum / global_count if global_count > 0 else 0.0
    return out


@torch.no_grad()
def run_eval(
    model: DeepFold3,
    ema: EMA | None,
    eval_dl: DataLoader,
    cfg: TrainingConfig,
    model_cfg: ModelConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device,
    log_microsteps: bool,
) -> dict[str, float]:
    """Run one evaluation pass.

    Swaps in EMA weights (if present), calls ``forward_training`` under
    ``no_grad``, computes the same losses as training, then restores
    the live weights. The EMA apply/restore sandwich is also the place
    where fused-kernel re-sync fires — don't reorder.

    If ``optimizer`` is a Schedule-Free optimizer, ``optimizer_eval_mode``
    flips its live params from ``y`` (gradient point) to ``x`` (inference
    target) for the duration of the pass and restores ``y`` on exit.
    For plain AdamW this is a no-op — and because AdamW has no averaged
    iterate, ``eval_snapshot='average'`` on an AdamW run with EMA active is
    upgraded to the EMA shadow (the deployment weights); otherwise it would
    silently evaluate the raw live params, which are never deployed.
    """
    use_shadow = cfg.eval_snapshot == "shadow"
    # A plain (non-Schedule-Free) optimizer has NO averaged iterate: eval_snapshot
    # 'average' would fall through to the raw *live* params, which are not the
    # deployment target when EMA is enabled. AdamW's only meaningful smoothed /
    # inference snapshot is the EMA shadow, so upgrade 'average' -> shadow whenever
    # EMA is present on a non-SF optimizer (matches what `inference --snapshot auto`
    # picks for these checkpoints). SF is untouched: its 'average' is the genuine x.
    is_schedule_free = getattr(optimizer, "eval_mode", None) is not None
    if cfg.eval_snapshot == "average" and not is_schedule_free and ema is not None:
        use_shadow = True
        logger.info(
            "eval_snapshot='average' on a non-Schedule-Free optimizer with EMA active — "
            "AdamW has no averaged iterate; evaluating on the EMA shadow (the deployment weights)."
        )
    if use_shadow and ema is None:
        logger.warning(
            "eval_snapshot='shadow' requested but EMA is disabled "
            "(ema_decay <= 0); falling back to 'average' (or live params "
            "if optimizer is not Schedule-Free)."
        )
        use_shadow = False
    with optimizer_eval_mode(optimizer):
        if ema is not None and use_shadow:
            ema.apply(model)

        model.eval()
        metrics: dict[str, list[float]] = {}
        n_batches = len(eval_dl)

        for batch_idx, features in enumerate(eval_dl):
            features.pop("_record_id", None)
            features = _prepare_features(features, device)

            try:
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type="cuda",
                        dtype=cfg.torch_dtype,
                        enabled=(cfg.dtype != "fp32"),
                    ),
                ):
                    pred = model.forward_training(
                        features,
                        # ``min(.., max_recycles)`` so the recycle count never
                        # exceeds the featurizer's crop budget (n=max_recycles+1)
                        # when msa_precrop is on.
                        num_recycles=min(cfg.max_recycles, 4),
                        diffusion_batch_size=1,
                        mini_rollout_steps=cfg.mini_rollout_steps,
                        mini_rollout_samples=1,
                        seed=0,
                        msa_precropped=cfg.msa_precrop,
                        chain_perm_gate_k=(cfg.chain_perm_gate_k if cfg.chain_perm_stability_gate else 1),
                        permute_gt_before_noise=cfg.permute_gt_before_noise,
                    )
                    # Slice batch dim (always kept) before loss.
                    # Use actual batch size from the data (not cfg) — the
                    # last eval batch may be smaller than batch_size_per_rank
                    # when the dataset size isn't evenly divisible.
                    b_actual = features["res_type"].shape[0]
                    pred_0 = slice_batch_dict(pred, 0, b_actual)
                    features_0 = slice_batch_dict(features, 0, b_actual)
                    eval_loss_fn = fp32_guard(model_cfg.precision.fp32_loss)(training_loss)
                    _, loss_dict = eval_loss_fn(pred_0, features_0, cfg)

                for k, v in loss_dict.items():
                    metrics.setdefault(f"eval/{k}", []).append(v.item())

                if log_microsteps:
                    rank_zero_info("eval: %d/%d loss=%.2f", batch_idx + 1, n_batches, loss_dict["total"])
            except Exception as e:
                # NOTE: ``except: continue`` inside a loop that calls into
                # ``forward_training`` is the class of pattern that caused
                # the prior DDP deadlock in training. Eval runs under
                # ``no_grad`` so there's no DDP backward collective, but
                # this still deserves an audit before turning DDP collectives
                # on inside the model forward (e.g. sharded eval).
                logger.warning("Eval error at batch %d: %s", batch_idx, e, exc_info=True)
                continue

        model.train()

        if ema is not None and use_shadow:
            ema.restore(model)

    return _reduce_eval_metrics(metrics, device)
