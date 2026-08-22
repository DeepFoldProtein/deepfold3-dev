"""Visualize predicted vs ground-truth distogram for a trained checkpoint.

Runs the trunk + distogram head only (no diffusion) on a specific sample,
then plots side-by-side heatmaps:
  1. GT distogram bin indices from representative atom positions
  2. Predicted distogram via argmax of logits
  3. Predicted expected distance (probability-weighted bin centres)
  4. Residual: |GT_dist - predicted_dist|

Also plots the per-pair probability distribution for a few sampled pairs
to inspect confidence.

Usage:
    python scripts/check_distogram.py \
        --ckpt runs/round1_ln/1s_pair/checkpoint_3033.pt \
        --data-dir /data/distillation/patchr_training \
        --record 1xr1 \
        --out runs/round1_ln/distogram_check_1s_pair.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepfold3.config_loader import load_config
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.config import ModelConfig
from deepfold3.model.model import DeepFold3
from deepfold3.scripts.train import _apply_overrides


def build_model_from_run(run_cfg_path: Path) -> tuple[DeepFold3, ModelConfig, dict]:
    full_cfg = load_config(run_cfg_path)
    mcfg = ModelConfig()
    _apply_overrides(mcfg, full_cfg.get("model", {}))
    model = DeepFold3(mcfg)
    return model, mcfg, full_cfg


def load_sample(record_id: str, data_dir: Path, mcfg: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    records = discover_records(data_dir)
    record = next((r for r in records if r.id == record_id), None)
    if record is None:
        raise ValueError(f"record {record_id} not found under {data_dir}")
    ds_kwargs = {
        "records": [record],
        "data_dir": data_dir,
        "max_tokens": mcfg.data.max_tokens if hasattr(mcfg.data, "max_tokens") else 384,
        "max_atoms": 4096,
        "max_msa_seqs": 512,
        "is_distillation": False,
        "max_templates": mcfg.data.max_templates,
        "template_cutoff_days": mcfg.data.template_cutoff_days,
        "max_template_date": mcfg.data.max_template_date,
        "contiguous_crop_weight": 1.0,
        "spatial_crop_weight": 0.0,
        "interface_crop_weight": 0.0,
    }
    ds = TrainingDataset(**ds_kwargs)
    torch.manual_seed(seed)
    np.random.seed(seed)
    features = ds[0]
    features.pop("_record_id", None)
    batch = collate_fn([features])
    return batch


def gt_distogram(
    ref_pos: torch.Tensor,
    seq_mask: torch.Tensor,
    first_break: float,
    last_break: float,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(ref_pos, ref_pos)
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=dist.device)
    bins = torch.bucketize(dist, breaks)
    pair_mask = seq_mask.unsqueeze(-1) & seq_mask.unsqueeze(-2)
    return dist, bins, pair_mask


def expected_distance_from_logits(
    logits: torch.Tensor, first_break: float, last_break: float, num_bins: int
) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=logits.device)
    last_gap = breaks[-1] - breaks[-2]
    bin_lower = torch.cat([torch.zeros(1, device=logits.device), breaks])
    bin_upper = torch.cat([breaks, breaks[-1:] + last_gap])
    bin_centres = 0.5 * (bin_lower + bin_upper)
    return (probs * bin_centres).sum(dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--record", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-recycles", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = args.ckpt.parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found next to checkpoint: {cfg_path}")

    model, mcfg, _ = build_model_from_run(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    print(f"[check_distogram] loading ckpt {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

    print(f"[check_distogram] loading sample {args.record}")
    batch = load_sample(args.record, args.data_dir, mcfg, seed=args.seed)
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    with torch.no_grad():
        logits = model.compute_distogram(batch, num_recycles=args.num_recycles)
    logits = logits[0]  # [N, N, 64] (drop batch dim)

    seq_mask = batch["seq_mask"][0].bool()
    atom_pos = batch["atom_gt_positions"][0].float()  # [N, M_atom, 3]
    tok_rep = batch["token_to_rep_atom"][0].long()  # [N]
    N = atom_pos.shape[0]
    ref_pos = atom_pos[torch.arange(N, device=atom_pos.device), tok_rep]  # [N, 3]

    first_break = mcfg.distogram.first_break
    last_break = mcfg.distogram.last_break
    num_bins = mcfg.distogram.num_bins

    gt_dist, gt_bins, pair_mask = gt_distogram(ref_pos, seq_mask, first_break, last_break, num_bins)
    pred_bins = logits.argmax(dim=-1)
    pred_exp = expected_distance_from_logits(logits, first_break, last_break, num_bins)

    mask_np = pair_mask.cpu().numpy()
    gt_bins_np = gt_bins.cpu().numpy().astype(float)
    gt_dist_np = gt_dist.cpu().numpy()
    pred_bins_np = pred_bins.cpu().numpy().astype(float)
    pred_exp_np = pred_exp.cpu().numpy()

    # Mask invalid pairs to nan for display.
    for m in (gt_bins_np, gt_dist_np, pred_bins_np, pred_exp_np):
        m[~mask_np] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmax_bin = num_bins - 1
    im0 = axes[0, 0].imshow(gt_bins_np, cmap="viridis", vmin=0, vmax=vmax_bin)
    axes[0, 0].set_title(f"GT distogram bins (N={int(seq_mask.sum().item())})")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.045)

    im1 = axes[0, 1].imshow(pred_bins_np, cmap="viridis", vmin=0, vmax=vmax_bin)
    axes[0, 1].set_title("Predicted argmax bin")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.045)

    diff_bin = pred_bins_np - gt_bins_np
    im2 = axes[0, 2].imshow(diff_bin, cmap="RdBu_r", vmin=-10, vmax=10)
    axes[0, 2].set_title("Δ bins (pred − GT)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.045)

    vmax_d = float(np.nanmax(gt_dist_np))
    im3 = axes[1, 0].imshow(gt_dist_np, cmap="magma", vmin=0, vmax=vmax_d)
    axes[1, 0].set_title("GT distances [Å]")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.045)

    im4 = axes[1, 1].imshow(pred_exp_np, cmap="magma", vmin=0, vmax=vmax_d)
    axes[1, 1].set_title("Predicted E[d] [Å]")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.045)

    diff_d = pred_exp_np - gt_dist_np
    im5 = axes[1, 2].imshow(diff_d, cmap="RdBu_r", vmin=-5, vmax=5)
    axes[1, 2].set_title("Δ distance [Å]")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.045)

    valid = mask_np & ~np.isnan(diff_bin)
    mae_bin = np.abs(diff_bin[valid]).mean()
    mae_d = np.abs(diff_d[valid]).mean()
    rmse_d = np.sqrt(np.mean(diff_d[valid] ** 2))

    fig.suptitle(
        f"{args.record}  ckpt={args.ckpt.name}  MAE_bin={mae_bin:.2f}  MAE_d={mae_d:.2f}Å  RMSE_d={rmse_d:.2f}Å"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[check_distogram] saved {args.out}")
    print(f"[check_distogram] MAE_bin={mae_bin:.3f}  MAE_d={mae_d:.3f}  RMSE_d={rmse_d:.3f}")


if __name__ == "__main__":
    main()
