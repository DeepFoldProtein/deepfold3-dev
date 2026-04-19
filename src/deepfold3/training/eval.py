"""Evaluation pass — EMA-swapped forward over the eval loader.

Exposes :func:`run_eval` as a pure-ish function: it takes the model +
loader + configs and returns the averaged eval metrics. Logging,
tracker fan-out, and best-checkpoint saving live in the trainer.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from deepfold3.model.config import ModelConfig, TrainingConfig
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import fp32_guard
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.ema import EMA
from deepfold3.training.loss import training_loss
from deepfold3.training.rank_zero import rank_zero_info

logger = logging.getLogger(__name__)


def _prepare_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in features.items()}


@torch.no_grad()
def run_eval(
    model: DeepFold3,
    ema: EMA | None,
    eval_dl: DataLoader,
    cfg: TrainingConfig,
    model_cfg: ModelConfig,
    *,
    device: torch.device,
    log_microsteps: bool,
) -> dict[str, float]:
    """Run one evaluation pass.

    Swaps in EMA weights (if present), calls ``forward_training`` under
    ``no_grad``, computes the same losses as training, then restores
    the live weights. The EMA apply/restore sandwich is also the place
    where fused-kernel re-sync fires — don't reorder.
    """
    if ema is not None:
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
                    num_recycles=min(cfg.max_recycles, 4),
                    diffusion_batch_size=1,
                    mini_rollout_steps=cfg.mini_rollout_steps,
                    mini_rollout_samples=1,
                    seed=0,
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

    if ema is not None:
        ema.restore(model)

    return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}
