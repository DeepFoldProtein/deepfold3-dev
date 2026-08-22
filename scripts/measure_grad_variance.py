"""Measure per-sample gradient NORM and DIRECTION spread from a checkpoint.

Diagnostic for per_rank_grad_clip and for understanding WHY the gradient norm is
large. Runs one clean (unchunked) forward+backward PER training example.

Key trick — direction stats need almost no memory. With unit gradients ûᵢ:
    mean_{i≠j} cos(gᵢ,gⱼ) = (‖Σûᵢ‖² − n) / (n²−n)
    resultant length R      = ‖Σûᵢ‖ / n        ∈ [0,1]
Both come from the running sum Σûᵢ alone, so we can STREAM hundreds of samples.
Only the full pairwise cosine matrix needs per-sample storage, so we keep just a
small subset (plus the running norm-outlier's direction) for that picture.

Regime read-out:
  R≈1 / mean-cos≈1  → directions agree (coherent signal; big norm = real steep descent)
  R≈1/√n / mean-cos≈0 → directions orthogonal (noise; big norm = idiosyncratic sample)
  a norm outlier orthogonal to the bulk ⇒ garbage per-rank clip should bound.

Usage:
    uv run python scripts/measure_grad_variance.py [n_total] [diffusion_bs] [store_cap] [clip]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.loss import training_loss

CKPT = "/home/jovyan/work/deepfold3-dev/runs/df3-stage0-r5/checkpoint_6500.pt"
DATA_DIR = "/data/distillation/patchr_training"
DEVICE = "cuda:0"
SCRATCH = "/tmp/claude-1000/-home-jovyan-work-deepfold3-dev/cc837a17-556b-4492-bd56-b3517cb762ce/scratchpad"

N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 64
DIFF_BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
STORE_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CLIP = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
NUM_RECYCLES = 3


def flat_grad(params) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def main() -> None:
    torch.manual_seed(0)
    print(f"Loading {CKPT} ...", flush=True)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    model = DeepFold3(model_cfg).to(DEVICE)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    del blob
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    D = sum(p.numel() for p in params)
    print(
        f"model {D / 1e6:.0f}M params | missing={len(missing)} unexpected={len(unexpected)} | "
        f"store {STORE_CAP} grads = {STORE_CAP * D * 4 / 1e9:.1f} GB",
        flush=True,
    )

    ds_kwargs = dict(
        max_tokens=train_cfg.max_tokens,
        max_atoms=train_cfg.max_atoms,
        max_msa_seqs=train_cfg.max_msa_seqs,
        mmcif_dir=Path(train_cfg.mmcif_dir) if train_cfg.mmcif_dir else None,
        max_templates=model_cfg.data.max_templates,
        template_cutoff_days=model_cfg.data.template_cutoff_days,
        max_template_date=model_cfg.data.max_template_date,
        default_template_cutoff_date=model_cfg.data.default_template_cutoff_date,
        prefer_monomer_templates=model_cfg.data.prefer_monomer_templates,
        max_neighborhood=model_cfg.data.max_neighborhood,
        contiguous_crop_weight=model_cfg.data.contiguous_crop_weight,
        spatial_crop_weight=model_cfg.data.spatial_crop_weight,
        interface_crop_weight=model_cfg.data.interface_crop_weight,
        interface_distance=model_cfg.data.interface_distance,
        max_chains=model_cfg.data.max_chains,
        atomize_prob=model_cfg.data.atomize_prob,
        atomize_source=model_cfg.data.atomize_source,
        atomize_max_residues=model_cfg.data.atomize_max_residues,
        ccd_cache_path=train_cfg.ccd_cache_path or None,
        pdb_chain_cache_dir=train_cfg.pdb_chain_cache_dir or None,
        pdb_chain_cache_mem=train_cfg.pdb_chain_cache_mem,
        msa_crop_size=model_cfg.msa_crop_size,
        n_msa_crops=(train_cfg.max_recycles + 1) if train_cfg.msa_precrop else 0,
    )
    ds = TrainingDataset(
        records=discover_records(Path(DATA_DIR)),
        data_dir=Path(DATA_DIR),
        is_distillation=False,
        name="main",
        **ds_kwargs,
    )
    collate = lambda items: collate_fn(items, bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
    print(f"dataset: {len(ds)} records", flush=True)

    S_raw = torch.zeros(D, device=DEVICE)  # Σ gᵢ
    S_unit = torch.zeros(D, device=DEVICE)  # Σ ûᵢ
    S_clip = torch.zeros(D, device=DEVICE)  # Σ min(‖g‖,C)·ûᵢ
    u_outlier = torch.zeros(D, device=DEVICE)
    max_norm = -1.0
    U_store = torch.zeros(STORE_CAP, D, device=DEVICE)
    stored = 0
    norms: list[float] = []
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(N_TOTAL * 4, len(ds)), replace=False).tolist()

    n = 0
    for idx in idxs:
        if n >= N_TOTAL:
            break
        try:
            batch = collate([ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"  [skip idx={idx}] load {type(e).__name__}", flush=True)
            continue
        features = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.forward_training(
                    features,
                    num_recycles=NUM_RECYCLES,
                    diffusion_batch_size=DIFF_BS,
                    mini_rollout_steps=train_cfg.mini_rollout_steps,
                    mini_rollout_samples=train_cfg.mini_rollout_samples,
                    seed=n,
                    diffusion_chunk_size=0,
                    compute_confidence=(train_cfg.alpha_confidence > 0),
                    msa_precropped=train_cfg.msa_precrop,
                )
            pred_b, features_b = slice_batch_dict(pred, 0, 1), slice_batch_dict(features, 0, 1)
            loss, ld = training_loss(pred_b, features_b, train_cfg)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip idx={idx}] OOM", flush=True)
            continue

        g = flat_grad(params)
        gnorm = float(g.double().norm().item())
        if not np.isfinite(gnorm) or gnorm == 0:
            print(f"  [skip idx={idx}] non-finite/zero grad", flush=True)
            del g
            continue
        u = g / gnorm
        del g
        S_raw.add_(u, alpha=gnorm)
        S_unit.add_(u)
        S_clip.add_(u, alpha=min(gnorm, CLIP))
        if gnorm > max_norm:
            max_norm = gnorm
            u_outlier.copy_(u)
        if stored < STORE_CAP:
            U_store[stored].copy_(u)
            stored += 1
        norms.append(gnorm)
        n_tok = int(features_b["seq_mask"].sum().item()) if "seq_mask" in features_b else -1
        comps = {k: float(v) for k, v in ld.items() if torch.is_tensor(v) and v.numel() == 1}
        top = sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:4]
        print(
            f"  [{n + 1:>3}/{N_TOTAL}] tok={n_tok:>4} loss={float(loss):8.3f} ||g||={gnorm:12.3f}  "
            + " ".join(f"{k}={v:.2f}" for k, v in top),
            flush=True,
        )
        n += 1
        del u

    if n < 2:
        print("Need >=2 samples.")
        return

    arr = np.array(norms)
    # Exact direction stats from the running sum of unit vectors.
    su_sq = float(S_unit.double().pow(2).sum())
    R = float(np.sqrt(su_sq) / n)
    mean_cos = (su_sq - n) / (n * n - n)
    raw_mean_norm = float(S_raw.double().norm() / n)
    clip_mean_norm = float(S_clip.double().norm() / n)
    mean_dir = S_unit / (S_unit.double().norm().float() + 1e-12)
    outlier_cos_mean = float((u_outlier.double() * mean_dir.double()).sum())

    cos = (U_store[:stored] @ U_store[:stored].T).cpu().double().numpy()
    off = cos[~np.eye(stored, dtype=bool)]
    np.save(f"{SCRATCH}/cos.npy", cos)
    np.save(f"{SCRATCH}/norms.npy", arr)

    p = print
    p("\n" + "=" * 74)
    p(f"n={n} (stored {stored})  diffusion_bs={DIFF_BS}  num_recycles={NUM_RECYCLES}  clip={CLIP}")
    p("-" * 74)
    p("MAGNITUDE  ||g_i||:")
    p(
        f"  mean={arr.mean():.3f} median={np.median(arr):.3f} min={arr.min():.3f} max={arr.max():.3f} "
        f"max/min={arr.max() / max(arr.min(), 1e-9):.1f}x"
    )
    p(
        f"  percentiles p50={np.percentile(arr, 50):.2f} p90={np.percentile(arr, 90):.2f} "
        f"p99={np.percentile(arr, 99):.2f}   >clip({CLIP}): {(arr > CLIP).mean() * 100:.0f}%"
    )
    p(
        f"  ||mean g|| = {raw_mean_norm:.3f}   mean||g|| = {arr.mean():.3f}   "
        f"ratio = {raw_mean_norm / max(arr.mean(), 1e-9):.3f}"
    )
    p("-" * 74)
    p("DIRECTION  cos(g_i, g_j)   (magnitude-independent):")
    p(f"  mean pairwise cos = {mean_cos:+.4f}   [exact over all {n} samples]")
    p(f"  resultant length R = {R:.4f}   (orthogonal ref 1/sqrt(n)={1 / np.sqrt(n):.4f}, aligned=1)")
    p(f"  stored subset ({stored}): off-diag cos mean={off.mean():+.3f} min={off.min():+.3f} max={off.max():+.3f}")
    p(f"  norm outlier ||g||={max_norm:.1f}: cos with mean direction = {outlier_cos_mean:+.4f}")
    p("-" * 74)
    p("UPDATE the optimizer steps on:")
    p(f"  GLOBAL clip  = clip(||mean g||,{CLIP})   = {min(raw_mean_norm, CLIP):.3f}")
    p(
        f"  PER-RANK clip= ||mean clip(g_i,{CLIP})|| = {clip_mean_norm:.3f}   "
        f"(per-rank/global = {clip_mean_norm / max(min(raw_mean_norm, CLIP), 1e-9):.3f})"
    )
    p("=" * 74)


if __name__ == "__main__":
    main()
