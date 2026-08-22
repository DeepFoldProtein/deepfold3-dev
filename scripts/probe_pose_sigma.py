"""σ-resolved, per-sample inter-chain pose diagnostic (no gradient).

Answers three questions the reduced ``eval/fape_inter`` scalar cannot:

1. **Is 0.1835 a typical value or a mean over a bimodal mixture?**
   → per-example distribution (p10/p50/p90), not just the mean.
2. **At which σ does the one-shot denoiser actually decide the pose?**
   The training draw is σ = σ_data·exp(p_mean + p_std·z) (median ≈ 4.8 Å),
   but iid per-atom noise barely moves a chain centroid (‖shift‖ ~ σ/√n_atoms),
   so most samples get the pose handed to them. Sweeping σ on a *grid* —
   possible only because no gradient is needed — separates "recovering blur"
   from "deciding the docking".
3. **How far is the one-shot denoise from the multi-step rollout?**
   ``fape_inter`` is measured on ``denoised`` (1 step from noised GT) while
   ``tm_true`` is measured on ``mini_rollout_pos`` (20 steps from pure noise).
   Scoring both on the same metrics makes them comparable for the first time.

Per (example, σ) it records the normalized FAPE terms plus two shape probes
targeted at the documented collapse failure mode (``rg_decompose``:
D_pred/D_gt ∈ 0.33–0.68):

  d_ratio  — mean over chain pairs of ‖Δc_pred‖ / ‖Δc_gt‖   (1.0 = correct separation)
  rg_ratio — Rg(pred) / Rg(gt)                              (<1 = collapsed)

``d_ratio`` is also recorded for the *noised input* (``kind=noisy_input``),
which shows how much pose information survives the forward process at each σ
— the upper bound on what the denoiser could possibly use.

Usage (8 GPUs, one process per shard):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 uv run python scripts/probe_pose_sigma.py \
        --shard $i --num-shards 8 --n 32 --out /tmp/pose_probe &
    done; wait
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from deepfold3.config_loader import load_config
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import upcast_fp32
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.data import _apply_cutoff
from deepfold3.training.loss import fape_loss, permute_gt_for_diffusion

# σ grid. Training draws σ = 16·exp(-1.2 + 1.5·z): median 4.8, P(σ>20)=17%,
# P(σ>50)=6%. Inference spans σ_data·[smin, smax] = [0.0064, 2560]. The grid
# straddles both so the "pose handed over" → "pose must be decided" → "blob"
# transitions all land inside it.
SIGMAS = (0.5, 1.5, 4.8, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0)

# Loss-side config that produced the 0.1835 number (configs/stage2.yaml).
FAPE_KW = dict(
    clip_distance=10.0,
    clip_distance_inter=30.0,
    normalize=True,
    all_atom_points=True,
    mode="both",
    chunk_size=64,  # bound the [1, C, N, M, 3] peak; exact (plain sum over chunks)
)


def _ridge_r(x_tr, y_tr, x_te, y_te, lam: float = 10.0) -> float:
    """Test Pearson r of a ridge fit — the ``r_cc_z`` classifier from Result 5,
    recomputed here so the sigma sweep can be grouped by failure TYPE. It has to
    be in this same pass: the crop is redrawn on every ``__getitem__``."""
    if x_tr.shape[0] < 200 or x_te.shape[0] < 200:
        return float("nan")
    xt, xe = x_tr.double(), x_te.double()
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp(min=1e-6)
    xt, xe = (xt - mu) / sd, (xe - mu) / sd
    xt = torch.cat([xt, torch.ones(xt.shape[0], 1, device=xt.device, dtype=torch.float64)], 1)
    xe = torch.cat([xe, torch.ones(xe.shape[0], 1, device=xe.device, dtype=torch.float64)], 1)
    a = xt.T @ xt
    a.diagonal().add_(lam)
    try:
        w = torch.linalg.solve(a, xt.T @ y_tr.double())
    except Exception:  # noqa: BLE001
        return float("nan")
    pr = xe @ w
    if not torch.isfinite(pr).all():
        return float("nan")
    pr = pr - pr.mean()
    e = y_te.double() - y_te.double().mean()
    if float(pr.norm()) < 1e-6 * float(e.norm().clamp(min=1e-12)):
        return float("nan")
    return max(-1.0, min(1.0, float((pr @ e) / (pr.norm() * e.norm()))))


def _center(pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-centre [N, M, 3]. The sampler works in a centred frame, so the
    σ-sweep input must be centred too; rotation is skipped (every metric here
    is rigid-invariant) to keep the sweep deterministic."""
    m = mask.float().unsqueeze(-1)
    return (pos.float() - (pos.float() * m).sum((0, 1), keepdim=True) / m.sum().clamp(min=1e-6)) * m


def _chain_centroids(pos: torch.Tensor, mask: torch.Tensor, asym: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per-chain mask-weighted atom centroid. ``pos`` [N, M, 3], ``asym`` [N]."""
    out = {}
    for c in asym.unique().tolist():
        sel = asym == c
        m = mask[sel].float().unsqueeze(-1)
        tot = m.sum()
        if tot < 1:
            continue
        out[int(c)] = (pos[sel].float() * m).sum((0, 1)) / tot
    return out


def _shape_probes(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, asym: torch.Tensor):
    """(d_ratio, rg_ratio). d_ratio = mean over chain pairs of the predicted
    inter-centroid distance over the true one — the ``rg_decompose`` D_pred/D_gt.
    Both are invariant to rigid motion, so no superposition is needed."""
    cp, cg = _chain_centroids(pred, mask, asym), _chain_centroids(gt, mask, asym)
    keys = sorted(set(cp) & set(cg))
    dp, dg = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g = float((cg[keys[i]] - cg[keys[j]]).norm())
            if g > 1e-3:
                dg.append(g)
                dp.append(float((cp[keys[i]] - cp[keys[j]]).norm()))
    # Σ/Σ, not mean-of-ratios: a pair of interdigitated chains has a tiny GT
    # separation and its ratio would otherwise dominate the average.
    d_ratio = (sum(dp) / sum(dg)) if dg else float("nan")

    def rg(p):
        m = mask.float().unsqueeze(-1)
        tot = m.sum().clamp(min=1e-6)
        c = (p.float() * m).sum((0, 1), keepdim=True) / tot
        return float(((((p.float() - c) ** 2).sum(-1, keepdim=True) * m).sum() / tot).sqrt())

    rg_gt = rg(gt)
    return d_ratio, (rg(pred) / rg_gt if rg_gt > 1e-3 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--config", default="configs/stage2.yaml", help="for the FAPE/eval-split settings")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30", help="eval split = records released AFTER this")
    ap.add_argument("--out", default="/tmp/pose_probe")
    ap.add_argument("--n", type=int, default=32, help="examples per shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101, help="matches training seed for the record draw")
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(1000 + args.shard)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    # EMA shadow = the deployment weights, and what run_eval actually scores
    # (it upgrades eval_snapshot 'average' → 'shadow' for AdamW+EMA).
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
        print(f"[shard {args.shard}] applied EMA shadow ({len(blob['ema']['shadow'])} tensors)", flush=True)
    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[shard {args.shard}] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del blob, state
    model.to(dev).eval()

    # stage2's FAPE settings are what produced the 0.1835 number; the ckpt's own
    # train_cfg is stage1 (pooled / NCACO+CB), so override from the YAML.
    fape_kw = dict(FAPE_KW)
    ycfg = load_config(args.config).get("training", {})
    for k, y in (("clip_distance", "fape_clip_distance"), ("clip_distance_inter", "fape_clip_distance_inter")):
        if y in ycfg:
            fape_kw[k] = float(ycfg[y])

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
        atomize_prob=0.0,  # eval split is un-atomized (atomize_eval: false)
        atomize_source=model_cfg.data.atomize_source,
        atomize_max_residues=model_cfg.data.atomize_max_residues,
        ccd_cache_path=train_cfg.ccd_cache_path or None,
        pdb_chain_cache_dir=train_cfg.pdb_chain_cache_dir or None,
        pdb_chain_cache_mem=train_cfg.pdb_chain_cache_mem,
        msa_crop_size=model_cfg.msa_crop_size,
        n_msa_crops=(train_cfg.max_recycles + 1) if train_cfg.msa_precrop else 0,
        template_struct_space=True,  # patchr indices are structure-space
    )
    recs = _apply_cutoff(discover_records(Path(args.data_dir)), args.cutoff_date, keep_after=True)
    ds = TrainingDataset(records=recs, data_dir=Path(args.data_dir), is_distillation=False, name="patchr_eval", **ds_kwargs)
    print(f"[shard {args.shard}] eval split: {len(ds)} records (released after {args.cutoff_date})", flush=True)

    # Same uniform draw the in-training eval sampler uses at epoch 0, then sharded.
    order = np.random.default_rng(args.seed).permutation(len(ds)).tolist()[args.shard :: args.num_shards]

    R = train_cfg.max_recycles
    gate_k = train_cfg.chain_perm_gate_k
    dh = model.diffusion_head
    sig_t = torch.tensor(SIGMAS, device=dev, dtype=torch.float32)

    out_path = Path(args.out) / f"pose_probe_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["record_id", "idx", "n_chains", "n_tokens", "kind", "sigma",
                 "fape_intra", "fape_inter", "d_ratio", "rg_ratio", "r_cc_z"])
    rng_np = np.random.default_rng(6000 + args.shard)

    done = 0
    t0 = time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] load {type(e).__name__}: {e}", flush=True)
            continue
        rid = batch.get("record_id", [f"idx{idx}"])
        rid = rid[0] if isinstance(rid, (list, tuple)) else str(rid)
        features = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # --- trunk (R warm-up passes + 1 final), mirroring forward_training ---
                if R > 0:
                    emb_det, *_ = model.run_evoformer(features, R - 1, seed=0, msa_precropped=train_cfg.msa_precrop, crop_offset=0)
                    features["_prev_pair"], features["_prev_single"] = emb_det["pair"], emb_det["single"]
                embeddings, rel_feat, target_feat, seq_mask, _ = model.run_evoformer(
                    features, 0, seed=R, msa_precropped=train_cfg.msa_precrop, crop_offset=R
                )
                features.pop("_prev_pair", None)
                features.pop("_prev_single", None)
                embeddings["target_feat"] = target_feat
                ref_kw = model._ref_kwargs(features)

                # --- mini-rollout: 20 steps from pure noise, gate_k samples ---
                orig = dh.cfg.steps, dh.cfg.num_samples
                dh.cfg.steps, dh.cfg.num_samples = train_cfg.mini_rollout_steps, max(train_cfg.mini_rollout_samples, gate_k)
                try:
                    mini = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)["atom_positions"]
                finally:
                    dh.cfg.steps, dh.cfg.num_samples = orig

                # --- chain permutation, resolved once from the rollout (as in training) ---
                pred_b = {"mini_rollout_pos": mini[0, : train_cfg.mini_rollout_samples], "chain_perm_gate_pos": mini[0]}
                features_b = slice_batch_dict(features, 0, 1)
                gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(pred_b, features_b, stability_gate_k=gate_k)

                asym = features_b["asym_id"]
                seq_m = features_b["seq_mask"].bool()
                if c_keep is not None:
                    seq_m = seq_m & c_keep
                n_chains = int(asym[seq_m].unique().numel())
                n_tok = int(seq_m.sum())
                if n_chains < 2:
                    continue  # single-chain crop has no cross-chain pairs at all

                atom_m = gt_mask.bool()
                frm_m = features_b["frames_mask"].bool() & atom_m[
                    features_b["frame_token_index"], features_b["frame_atom_index"]
                ].all(dim=-1)
                pair_m = frm_m.unsqueeze(-1) & features_b["pair_mask"].bool()

                # r_cc_z: is the cross-chain distance linearly decodable from the
                # trunk pair rep? Token-disjoint split (a random PAIR split leaks
                # through the triangle inequality).
                rep_tok = features_b["token_to_rep_atom"]
                ar_t = torch.arange(rep_tok.shape[0], device=dev)
                gt_rep_t = gt_pos[ar_t, rep_tok].float()
                d_gt_t = torch.cdist(gt_rep_t, gt_rep_t)
                rep_ok_t = seq_m & atom_m[ar_t, rep_tok]
                vld = rep_ok_t.unsqueeze(1) & rep_ok_t.unsqueeze(0)
                vld.fill_diagonal_(False)
                cc_m = vld & (asym.unsqueeze(1) != asym.unsqueeze(0))
                n_all_t = rep_tok.shape[0]
                sd_t = torch.zeros(n_all_t, dtype=torch.bool, device=dev)
                sd_t[torch.as_tensor(rng_np.permutation(n_all_t)[: n_all_t // 2].copy(), device=dev)] = True
                zpair = embeddings["pair"][0].float()

                def _sub(m, cap=40000):
                    q = m.nonzero(as_tuple=False)
                    if q.shape[0] > cap:
                        q = q[torch.as_tensor(rng_np.permutation(q.shape[0])[:cap].copy(), device=dev)]
                    return q

                q_tr = _sub(cc_m & sd_t.unsqueeze(1) & sd_t.unsqueeze(0))
                q_te = _sub(cc_m & (~sd_t).unsqueeze(1) & (~sd_t).unsqueeze(0))
                r_cc_z = _ridge_r(zpair[q_tr[:, 0], q_tr[:, 1]], d_gt_t[q_tr[:, 0], q_tr[:, 1]],
                                  zpair[q_te[:, 0], q_te[:, 1]], d_gt_t[q_te[:, 0], q_te[:, 1]])

                gt_c = _center(gt_pos, atom_m)

                # Loop-scoped tensors are bound as defaults so the closure does
                # not depend on rebinding later in the iteration (ruff B023).
                def score(pred_pos, _gt=gt_c, _f=features_b, _fm=frm_m, _sm=seq_m,
                          _pm=pair_m, _am=atom_m, _az=asym):
                    a, b = fape_loss(
                        pred_pos.unsqueeze(0).float(), _gt, None, None, None,
                        _f["frame_atom_index"], _f["frame_token_index"],
                        _fm, _sm, _pm, atom_mask=_am, asym_id=_az, **fape_kw,
                    )
                    return float(a), float(b)
                rows = []
                # The training denoise runs with autocast OFF on fp32 inputs
                # (``DiffusionConfig.skip_amp_sample_training=True``, which this
                # run sets). Reproducing that here is not optional: under bf16
                # the same call scores ~2x worse, and the gap widens with σ
                # because the input's signal-to-noise shrinks into bf16's
                # ~3 significant digits. ``sample()`` upcasts internally
                # (``skip_amp_sample=True``), so the mini-rollout above is
                # already correct and only this block needs the context.
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    emb32 = upcast_fp32(embeddings)
                    rk32 = {k: (upcast_fp32(v) if k in ("ref_pos", "ref_charge", "ref_mask") else v) for k, v in ref_kw.items()}
                    static = dh.compute_static(
                        embeddings=emb32, seq_mask=upcast_fp32(seq_mask), rel_feat=upcast_fp32(rel_feat), **rk32
                    )

                    # --- σ grid: one-shot denoise of the noised GT ---
                    noisy = gt_c.unsqueeze(0) + torch.randn(len(SIGMAS), *gt_c.shape, device=dev) * sig_t.view(-1, 1, 1, 1)
                    den = dh._denoise_samples(noisy, sig_t, static=static).float()  # [S, N, M, 3]
                    for s, sg in enumerate(SIGMAS):
                        fi, fx = score(den[s])
                        d, rgr = _shape_probes(den[s], gt_c, atom_m, asym)
                        rows.append([rid, idx, n_chains, n_tok, "oneshot", sg, fi, fx, d, rgr, r_cc_z])
                        dn, _ = _shape_probes(noisy[s], gt_c, atom_m, asym)
                        rows.append([rid, idx, n_chains, n_tok, "noisy_input", sg, "", "", dn, "", r_cc_z])

                    # --- multi-step rollout, scored on the SAME metrics ---
                    fi, fx = score(mini[0, 0])
                    d, rgr = _shape_probes(mini[0, 0], gt_c, atom_m, asym)
                    rows.append([rid, idx, n_chains, n_tok, "mini_rollout", "", fi, fx, d, rgr, r_cc_z])

                    # --- the training draw itself: reproduces the logged eval/fape_inter.
                    # Includes the σ mixture (this run has sigma_mix_alpha=0.02,
                    # kind=uniform → 2% of draws are log-uniform over the FULL
                    # inference range) and the low-σ clamp, or the reproduction
                    # misses the heavy right tail those add.
                    dc = model_cfg.diffusion
                    n_td = 4
                    log_base = math.log(dc.sigma_data) + dc.p_mean + dc.p_std * torch.randn(n_td, device=dev)
                    if dc.sigma_mix_alpha > 0.0:
                        if dc.sigma_mix_kind == "uniform":
                            lo, hi = math.log(dc.sigma_data * dc.sigma_mix_smin), math.log(dc.sigma_data * dc.sigma_mix_smax)
                            log_mix = lo + torch.rand(n_td, device=dev) * (hi - lo)
                        else:
                            log_mix = math.log(dc.sigma_data) + dc.sigma_mix_p_mean + dc.sigma_mix_p_std * torch.randn(n_td, device=dev)
                        tsig = torch.where(torch.rand(n_td, device=dev) < dc.sigma_mix_alpha, log_mix, log_base).exp()
                    else:
                        tsig = log_base.exp()
                    if dc.clamp_train_sigma:
                        tsig = tsig.clamp(min=dc.train_sigma_min)
                    tnoisy = gt_c.unsqueeze(0) + torch.randn(n_td, *gt_c.shape, device=dev) * tsig.view(-1, 1, 1, 1)
                    tden = dh._denoise_samples(tnoisy, tsig, static=static).float()
                    for s in range(n_td):
                        fi, fx = score(tden[s])
                        d, rgr = _shape_probes(tden[s], gt_c, atom_m, asym)
                        rows.append([rid, idx, n_chains, n_tok, "traindraw", float(tsig[s]), fi, fx, d, rgr, r_cc_z])

            wr.writerows(rows)
            fh.flush()
            done += 1
            if done % 5 == 0:
                print(f"[shard {args.shard}] {done}/{args.n}  ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip {idx}] OOM", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[shard {args.shard}] wrote {done} examples → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
