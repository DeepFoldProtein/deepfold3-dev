"""Can the denoiser rescue a wrong pose? (no gradient)

Four questions in one pass, runnable against **any** checkpoint — a DF3 ``.pt`` or
the converted AlphaFold-3 ``.safetensors`` — so the same decomposition can be read
off AF3's own failures and compared with DF3's.

1. **zero-shot** — x̂₀ from the very FIRST denoise call, at σ_max, from pure noise.
   The only thing available to it is the trunk conditioning.
2. **zero-shot vs finished** — does the 20-step rollout move the pose away from
   that first guess, or only polish it?
3. **rescue** — take the finished (possibly wrong) structure, re-noise it to σ_r
   and re-run the sampler from there (SDEdit / restart). Sweeping σ_r tests the
   squeeze that `export/pose-decomposition/` argues makes rescue impossible:
   σ_r low enough to keep the good local structure also keeps the wrong pose,
   σ_r high enough to erase the pose erases everything and re-commits. If some
   intermediate σ_r improves `fape_inter`, that argument is wrong and restart is
   a free inference-time fix.
4. **is it the trunk?** — ``r_cc_z`` (linear decodability of the true cross-chain
   distance from the trunk pair rep) splits each model's failures into
   "conditioning carried it and the denoiser ignored it" vs "conditioning never
   had it", measured in the SAME pass (the crop is redrawn per ``__getitem__``).

Usage:
    # DF3
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_pose_rescue.py --shard 0 --num-shards 8 --n 25
    # AlphaFold-3, its own recycle count, same eval crops
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_pose_rescue.py \
        --ckpt resources/alphafold3.safetensors --recycles 10 --tag af3 ...
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from deepfold3.config_loader import load_config
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.config import ModelConfig, model_cfg_from_metadata
from deepfold3.model.diffusion import _augment_in_place, noise_schedule
from deepfold3.model.model import DeepFold3
from deepfold3.model.precision import upcast_fp32
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.data import _apply_cutoff
from deepfold3.training.loss import fape_loss, permute_gt_for_diffusion

# σ to re-noise the finished structure to before restarting the sampler. The
# rollout schedule spans σ_data·[smax..smin] = [2560 .. 0.006]; these bracket
# "keeps the pose" (≤20, input d_ratio ≈ 1.01) through "erases it" (≥320).
RESCUE_SIGMAS = (5.0, 20.0, 80.0, 320.0, 1280.0)
ROLLOUT_STEPS = 20
MAX_CC = 40_000
FAPE_KW = dict(clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
               all_atom_points=True, mode="both", chunk_size=64)


def ridge_r(x_tr, y_tr, x_te, y_te, lam: float = 10.0) -> float:
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
    p = xe @ w
    if not torch.isfinite(p).all():
        return float("nan")
    p = p - p.mean()
    e = y_te.double() - y_te.double().mean()
    if float(p.norm()) < 1e-6 * float(e.norm().clamp(min=1e-12)):
        return float("nan")
    return max(-1.0, min(1.0, float((p @ e) / (p.norm() * e.norm()))))


def load_any(ckpt: Path, device: str):
    """Load a DF3 ``.pt`` (EMA shadow) or a converted AF3 ``.safetensors``.

    Returns ``(model, model_cfg, train_cfg_or_None)``. AF3 has no training
    config; the caller supplies the dataset settings so both models see
    identical eval crops.
    """
    if ckpt.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(ckpt), "pt") as f:
            meta = f.metadata() or {}
        # ``convert_weights`` embeds the architectural ModelConfig as JSON; without
        # it the defaults would silently disagree with the trained architecture.
        cfg = model_cfg_from_metadata(meta) or ModelConfig()
        model = DeepFold3(cfg)
        missing, unexpected = model.load_state_dict(load_file(str(ckpt)), strict=False)
        print(f"  safetensors: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return model.to(device).eval(), cfg, None

    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg, tcfg = blob["model_cfg"], blob["train_cfg"]
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
    model = DeepFold3(cfg)
    model.load_state_dict(state, strict=False)
    del blob, state
    return model.to(device).eval(), cfg, tcfg


@torch.no_grad()
def run_sampler(dh, static_bs, positions, nls, gammas, start_step, aug_mask):
    """Euler loop of ``_sample_impl`` from ``start_step`` — verbatim, including the
    per-step random augmentation, the gamma churn and the ``step_scale`` update.
    ``positions`` must already be at noise level ``nls[start_step]``."""
    cfg = dh.cfg
    for step in range(start_step, ROLLOUT_STEPS):
        sigma_prev, sigma_next = nls[step], nls[step + 1]
        _augment_in_place(positions, aug_mask, None)
        sigma_hat = sigma_prev * (1 + gammas[step])
        noise_scale = cfg.noise_scale * (sigma_hat**2 - sigma_prev**2).clamp(min=0).sqrt()
        positions_hat = positions + noise_scale * torch.randn_like(positions)
        denoised = dh._denoise_samples(positions_hat, sigma_hat, static_bs=static_bs)
        d_t = sigma_next - sigma_hat
        positions = positions_hat + cfg.step_scale * d_t * (positions_hat - denoised) / sigma_hat
    return positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--dataset-config", default="configs/stage2.yaml", help="dataset/crop settings, shared by both models")
    ap.add_argument("--train-cfg-from", default="runs/df3-stage1-r14/checkpoint_34000.pt",
                    help="checkpoint to borrow TrainingConfig from when --ckpt has none (AF3)")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30")
    ap.add_argument("--out", default="/tmp/pose_rescue")
    ap.add_argument("--tag", default="df3")
    ap.add_argument("--recycles", type=int, default=-1, help="-1 = use the checkpoint's own num_recycles, capped at 10")
    ap.add_argument("--msa-crop", type=int, default=1024)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(7000 + args.shard)
    gen = np.random.default_rng(7000 + args.shard)

    model, model_cfg, train_cfg = load_any(Path(args.ckpt), dev)
    if train_cfg is None:
        blob = torch.load(args.train_cfg_from, map_location="cpu", weights_only=False, mmap=True)
        train_cfg = blob["train_cfg"]
        del blob
    for k, v in load_config(args.dataset_config).get("training", {}).items():
        if hasattr(train_cfg, k) and not isinstance(v, (list, dict)):
            setattr(train_cfg, k, v)
    R = args.recycles if args.recycles >= 0 else min(model_cfg.num_recycles, 10)
    # skip_amp is a per-checkpoint property (r14 True, AF3 False) — respect each
    # model's own setting or the comparison is not like-for-like.
    fp32_denoise = bool(model_cfg.diffusion.skip_amp_sample)
    print(f"[{args.tag} shard {args.shard}] recycles={R} fp32_denoise={fp32_denoise}", flush=True)

    ds = TrainingDataset(
        records=_apply_cutoff(discover_records(Path(args.data_dir)), args.cutoff_date, keep_after=True),
        data_dir=Path(args.data_dir), is_distillation=False, name="patchr_eval",
        max_tokens=train_cfg.max_tokens, max_atoms=train_cfg.max_atoms, max_msa_seqs=train_cfg.max_msa_seqs,
        mmcif_dir=Path(train_cfg.mmcif_dir) if train_cfg.mmcif_dir else None,
        max_templates=model_cfg.data.max_templates, template_cutoff_days=model_cfg.data.template_cutoff_days,
        max_template_date=model_cfg.data.max_template_date,
        default_template_cutoff_date=model_cfg.data.default_template_cutoff_date,
        prefer_monomer_templates=model_cfg.data.prefer_monomer_templates,
        max_neighborhood=model_cfg.data.max_neighborhood,
        contiguous_crop_weight=model_cfg.data.contiguous_crop_weight,
        spatial_crop_weight=model_cfg.data.spatial_crop_weight,
        interface_crop_weight=model_cfg.data.interface_crop_weight,
        interface_distance=model_cfg.data.interface_distance,
        max_chains=model_cfg.data.max_chains, atomize_prob=0.0,
        ccd_cache_path=train_cfg.ccd_cache_path or None,
        pdb_chain_cache_dir=train_cfg.pdb_chain_cache_dir or None,
        pdb_chain_cache_mem=train_cfg.pdb_chain_cache_mem,
        msa_crop_size=args.msa_crop, n_msa_crops=0,
        template_struct_space=True,
    )
    order = np.random.default_rng(args.seed).permutation(len(ds)).tolist()[args.shard :: args.num_shards]

    dh = model.diffusion_head
    gate_k = train_cfg.chain_perm_gate_k
    out_path = Path(args.out) / f"rescue_{args.tag}_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["tag", "idx", "record_id", "n_chains", "n_tokens", "r_cc_z",
                 "zeroshot", "final", "r_zeroshot_final"] + [f"rescue_s{int(s)}" for s in RESCUE_SIGMAS])

    done, t0 = 0, time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        rid = getattr(ds.records[idx], "id", f"idx{idx}")
        features = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if R > 0:
                    emb_det, *_ = model.run_evoformer(features, R - 1, seed=0, crop_offset=0)
                    features["_prev_pair"], features["_prev_single"] = emb_det["pair"], emb_det["single"]
                embeddings, rel_feat, target_feat, seq_mask, _ = model.run_evoformer(features, 0, seed=R, crop_offset=R)
                features.pop("_prev_pair", None)
                features.pop("_prev_single", None)
                embeddings["target_feat"] = target_feat
                ref_kw = model._ref_kwargs(features)

                amp_off = torch.amp.autocast(device_type="cuda", enabled=False)
                ctx = amp_off if fp32_denoise else torch.autocast("cuda", dtype=torch.bfloat16)
                with ctx:
                    emb_u = upcast_fp32(embeddings) if fp32_denoise else embeddings
                    rk_u = {k: (upcast_fp32(v) if fp32_denoise and k in ("ref_pos", "ref_charge", "ref_mask") else v)
                            for k, v in ref_kw.items()}
                    rel_u = upcast_fp32(rel_feat) if fp32_denoise else rel_feat
                    sqm_u = upcast_fp32(seq_mask) if fp32_denoise else seq_mask
                    static = dh.compute_static(embeddings=emb_u, seq_mask=sqm_u, rel_feat=rel_u, **rk_u)
                    S_ROLL = max(1, gate_k)
                    static_bs = static.expand_for_samples(S_ROLL)

                    nls = noise_schedule(torch.linspace(0, 1, ROLLOUT_STEPS + 1, device=dev),
                                         sigma_data=dh.cfg.sigma_data, smin=dh.cfg.smin,
                                         smax=dh.cfg.smax, p=dh.cfg.noise_p)
                    gammas = torch.where(nls[1:] > dh.cfg.gamma_min, dh.cfg.gamma_0, 0.0)
                    aug_mask = ref_kw["ref_mask"].bool()[0] if ref_kw["ref_mask"].dim() == 3 else ref_kw["ref_mask"].bool()

                    N, M = aug_mask.shape
                    pos0 = torch.randn(S_ROLL, N, M, 3, device=dev) * nls[0]
                    # --- zero-shot: the FIRST denoise call, pure noise in ------
                    _augment_in_place(pos0, aug_mask, None)
                    sh0 = nls[0] * (1 + gammas[0])
                    ns0 = dh.cfg.noise_scale * (sh0**2 - nls[0] ** 2).clamp(min=0).sqrt()
                    ph0 = pos0 + ns0 * torch.randn_like(pos0)
                    zero_shot = dh._denoise_samples(ph0, sh0, static_bs=static_bs).float()
                    positions = ph0 + dh.cfg.step_scale * (nls[1] - sh0) * (ph0 - zero_shot) / sh0
                    final = run_sampler(dh, static_bs, positions, nls, gammas, 1, aug_mask).float()

                f0 = slice_batch_dict(features, 0, 1)
                gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(
                    {"mini_rollout_pos": final[:1], "chain_perm_gate_pos": final}, f0, stability_gate_k=gate_k
                )
                asym = f0["asym_id"]
                seq_m = f0["seq_mask"].bool()
                if c_keep is not None:
                    seq_m = seq_m & c_keep
                n_chains = int(asym[seq_m].unique().numel())
                if n_chains < 2:
                    continue

                atom_m = gt_mask.bool()
                mf = atom_m.float().unsqueeze(-1)
                gt_c = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                pm = fm.unsqueeze(-1) & f0["pair_mask"].bool()

                def sc(p, _g=gt_c, _f=f0, _fm=fm, _sm=seq_m, _pm=pm, _am=atom_m, _az=asym):
                    _, v = fape_loss(p.unsqueeze(0).float(), _g, None, None, None,
                                     _f["frame_atom_index"], _f["frame_token_index"], _fm, _sm, _pm,
                                     atom_mask=_am, asym_id=_az, **FAPE_KW)
                    return float(v)

                fi_zero, fi_final = sc(zero_shot[0]), sc(final[0])

                rep = f0["token_to_rep_atom"]
                ar = torch.arange(rep.shape[0], device=dev)
                gt_rep = gt_pos[ar, rep].float()
                d_gt = torch.cdist(gt_rep, gt_rep)
                rep_ok = seq_m & atom_m[ar, rep]
                vld = rep_ok.unsqueeze(1) & rep_ok.unsqueeze(0)
                vld.fill_diagonal_(False)
                cc = vld & (asym.unsqueeze(1) != asym.unsqueeze(0))
                ij = cc.nonzero(as_tuple=False)
                if ij.shape[0] < 500:
                    continue

                def dcc(x, _ij=ij, _ar=ar, _rep=rep):
                    r = x[_ar, _rep].float()
                    return torch.cdist(r, r)[_ij[:, 0], _ij[:, 1]]

                def pear(a, b):
                    a = a.double() - a.double().mean()
                    b = b.double() - b.double().mean()
                    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))

                r_zf = pear(dcc(zero_shot[0]), dcc(final[0]))

                # --- rescue: re-noise the FINISHED structure, restart ---------
                rescued = []
                with ctx:
                    for s_r in RESCUE_SIGMAS:
                        k = int(torch.argmin((nls[:ROLLOUT_STEPS] - s_r).abs()).item())
                        start = final.clone() + nls[k] * torch.randn_like(final)
                        out = run_sampler(dh, static_bs, start, nls, gammas, k, aug_mask).float()
                        rescued.append(sc(out[0]))

                # --- is the trunk the bottleneck for THIS model's failures? ---
                z = embeddings["pair"][0].float()
                n_all = rep.shape[0]
                side = torch.zeros(n_all, dtype=torch.bool, device=dev)
                side[torch.as_tensor(gen.permutation(n_all)[: n_all // 2].copy(), device=dev)] = True

                def sub(m, _gen=gen, _dev=dev):
                    q = m.nonzero(as_tuple=False)
                    if q.shape[0] > MAX_CC:
                        q = q[torch.as_tensor(_gen.permutation(q.shape[0])[:MAX_CC].copy(), device=_dev)]
                    return q

                q_tr = sub(cc & side.unsqueeze(1) & side.unsqueeze(0))
                q_te = sub(cc & (~side).unsqueeze(1) & (~side).unsqueeze(0))
                r_cc_z = ridge_r(z[q_tr[:, 0], q_tr[:, 1]], d_gt[q_tr[:, 0], q_tr[:, 1]],
                                 z[q_te[:, 0], q_te[:, 1]], d_gt[q_te[:, 0], q_te[:, 1]])

            wr.writerow([args.tag, idx, rid, n_chains, int(seq_m.sum()), r_cc_z,
                         fi_zero, fi_final, r_zf, *rescued])
            fh.flush()
            done += 1
            if done % 5 == 0:
                print(f"[{args.tag} shard {args.shard}] {done}/{args.n} ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip {idx}] OOM", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[{args.tag} shard {args.shard}] wrote {done} → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
