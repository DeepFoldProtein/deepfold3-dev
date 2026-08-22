"""When is the inter-chain pose committed during the rollout? (no gradient)

Tests the "it locks in early" hypothesis for the failure class where the trunk
DOES carry the interface but the rollout misposes anyway (``r_cc_z > 0.7`` &
``fape_inter > 0.6`` — 48% of failures, see
``export/pose-decomposition/2026-07-31-onestep-vs-rollout-pose.md`` Result 5).

``_sample_impl`` calls ``_denoise_samples`` exactly once per Euler step, and its
output is the current x̂₀ estimate. Wrapping that call captures the whole
trajectory for free. Per step, on cross-chain rep-atom pairs only:

  r_to_final — Pearson(d_cc at step k, d_cc of the returned structure)
               → WHEN the pose stops moving. If step 1 is already ~1.0 the pose
                 was decided by the first denoiser application and 19 further
                 steps are refinement, not search.
  r_to_gt    — Pearson(d_cc at step k, d_cc of the GT)
               → whether it ever approaches the right answer.
  d_ratio    — Σ‖Δc_pred‖ / Σ‖Δc_gt‖ over chain-centroid pairs.

Runs S=4 independent samples so the within-example spread separates
"deterministically locked by the conditioning" from "stochastic, so
best-of-N / ranking is the lever" — FoldBench oracle_gain is 27.8 pp on
antibody–antigen, so that distinction is not academic.

Also records ``r_cc_z`` (the USE-vs-REPRESENTATION classifier) and per-chain MSA
depth in the SAME pass, because the crop is redrawn on every ``__getitem__``.

Usage (8 GPUs):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 uv run python scripts/probe_pose_commit.py \
        --shard $i --num-shards 8 --n 30 --out /tmp/pose_commit &
    done; wait
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
from deepfold3.model.model import DeepFold3
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.data import _apply_cutoff
from deepfold3.training.loss import fape_loss, permute_gt_for_diffusion

N_TRAJ_SAMPLES = 4
MAX_CC = 40_000
RIDGE_LAM = 10.0
FAPE_KW = dict(clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
               all_atom_points=True, mode="both", chunk_size=64)


def _capture_denoise(orig, sink):
    """Wrap ``_denoise_samples`` so every Euler step's x-hat-0 lands in ``sink``.

    Module level, not a closure inside the example loop: the returned function
    must not capture loop-scoped names (ruff B023).
    """

    def _cap(*a, **k):
        out = orig(*a, **k)
        sink.append(out.detach().float())
        return out

    return _cap


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    d = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / d)


def ridge_r(x_tr, y_tr, x_te, y_te, lam=RIDGE_LAM) -> float:
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
    e = y_te.double() - y_te.double().mean()
    p = p - p.mean()
    if float(p.norm()) < 1e-6 * float(e.norm().clamp(min=1e-12)):
        return float("nan")
    return max(-1.0, min(1.0, float((p @ e) / (p.norm() * e.norm()))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--config", default="configs/stage2.yaml")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30")
    ap.add_argument("--out", default="/tmp/pose_commit")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(5000 + args.shard)
    gen = np.random.default_rng(5000 + args.shard)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
    model = DeepFold3(model_cfg)
    model.load_state_dict(state, strict=False)
    del blob, state
    model.to(dev).eval()
    for k, v in load_config(args.config).get("training", {}).items():
        if hasattr(train_cfg, k) and not isinstance(v, (list, dict)):
            setattr(train_cfg, k, v)

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
        msa_crop_size=model_cfg.msa_crop_size,
        n_msa_crops=(train_cfg.max_recycles + 1) if train_cfg.msa_precrop else 0,
        template_struct_space=True,
    )
    order = np.random.default_rng(args.seed).permutation(len(ds)).tolist()[args.shard :: args.num_shards]

    cap: dict = {}
    _orig_load_msa = ds._load_cropped_msa

    def _capture_msa(record, tokenized, surviving, cols):
        pmsa, umsa = _orig_load_msa(record, tokenized, surviving, cols)
        cap.clear()
        cap["record_id"] = record.id
        cap["unpaired"] = {k: int(v.sequences.shape[0]) for k, v in umsa.items()}
        cap["surviving"] = sorted(int(c) for c in surviving)
        return pmsa, umsa

    ds._load_cropped_msa = _capture_msa

    R, gate_k, dh = train_cfg.max_recycles, train_cfg.chain_perm_gate_k, model.diffusion_head

    ex_path = Path(args.out) / f"commit_ex_{args.shard:02d}.csv"
    tj_path = Path(args.out) / f"commit_traj_{args.shard:02d}.csv"
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    fex, ftj = ex_path.open("w", newline=""), tj_path.open("w", newline="")
    wex, wtj = csv.writer(fex), csv.writer(ftj)
    wex.writerow(["idx", "record_id", "n_chains", "r_cc_z", "unpaired_med",
                  "fape_inter_s0", "fape_inter_best", "fape_inter_worst", "fape_inter_spread"])
    wtj.writerow(["idx", "sample", "step", "sigma", "r_to_final", "r_to_gt", "d_ratio"])

    done, t0 = 0, time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        features = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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

                # rollout 1 — for the chain permutation, exactly as training resolves it
                orig = dh.cfg.steps, dh.cfg.num_samples
                dh.cfg.steps, dh.cfg.num_samples = train_cfg.mini_rollout_steps, gate_k
                try:
                    mini = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)["atom_positions"]
                finally:
                    dh.cfg.steps, dh.cfg.num_samples = orig

                f0 = slice_batch_dict(features, 0, 1)
                gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(
                    {"mini_rollout_pos": mini[0, :1], "chain_perm_gate_pos": mini[0]}, f0, stability_gate_k=gate_k
                )
                asym = f0["asym_id"]
                seq_m = f0["seq_mask"].bool()
                if c_keep is not None:
                    seq_m = seq_m & c_keep
                n_chains = int(asym[seq_m].unique().numel())
                if n_chains < 2:
                    continue

                atom_m = gt_mask.bool()
                rep = f0["token_to_rep_atom"]
                ar = torch.arange(rep.shape[0], device=dev)
                rep_ok = seq_m & atom_m[ar, rep]
                gt_rep = gt_pos[ar, rep].float()
                d_gt = torch.cdist(gt_rep, gt_rep)

                valid = rep_ok.unsqueeze(1) & rep_ok.unsqueeze(0)
                valid.fill_diagonal_(False)
                cc = valid & (asym.unsqueeze(1) != asym.unsqueeze(0))
                ij = cc.nonzero(as_tuple=False)
                if ij.shape[0] < 500:
                    continue
                if ij.shape[0] > MAX_CC:
                    ij = ij[torch.as_tensor(gen.permutation(ij.shape[0])[:MAX_CC].copy(), device=dev)]
                y_gt = d_gt[ij[:, 0], ij[:, 1]]

                # rollout 2 — capture x̂₀ at every Euler step
                traj: list[torch.Tensor] = []
                _orig_den = dh._denoise_samples

                dh.cfg.steps, dh.cfg.num_samples = train_cfg.mini_rollout_steps, N_TRAJ_SAMPLES
                dh._denoise_samples = _capture_denoise(_orig_den, traj)
                try:
                    final = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)["atom_positions"]
                finally:
                    dh._denoise_samples = _orig_den
                    dh.cfg.steps, dh.cfg.num_samples = orig

                sig = dh.cfg.sigma_data  # only for reporting the schedule
                from deepfold3.model.diffusion import noise_schedule
                nl = noise_schedule(torch.linspace(0, 1, train_cfg.mini_rollout_steps + 1, device=dev),
                                    sigma_data=sig, smin=dh.cfg.smin, smax=dh.cfg.smax, p=dh.cfg.noise_p)

                fin_s = [final[0, s][ar, rep].float() for s in range(N_TRAJ_SAMPLES)]
                y_fin = [torch.cdist(f, f)[ij[:, 0], ij[:, 1]] for f in fin_s]

                mf = atom_m.float().unsqueeze(-1)
                gt_c = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                pm = fm.unsqueeze(-1) & f0["pair_mask"].bool()
                fis = []
                for s in range(N_TRAJ_SAMPLES):
                    _, v = fape_loss(final[0, s].unsqueeze(0).float(), gt_c, None, None, None,
                                     f0["frame_atom_index"], f0["frame_token_index"], fm, seq_m, pm,
                                     atom_mask=atom_m, asym_id=asym, **FAPE_KW)
                    fis.append(float(v))

                def cen_ratio(p, asym=asym, rep_ok=rep_ok, gt_rep=gt_rep):
                    num = den = 0.0
                    ks = sorted(asym[rep_ok].unique().tolist())
                    c_p, c_g = {}, {}
                    for c in ks:
                        s_ = (asym == c) & rep_ok
                        if s_.sum() >= 1:
                            c_p[c], c_g[c] = p[s_].mean(0), gt_rep[s_].mean(0)
                    for a_ in range(len(ks)):
                        for b_ in range(a_ + 1, len(ks)):
                            ka, kb = ks[a_], ks[b_]
                            if ka in c_p and kb in c_p:
                                g = float((c_g[ka] - c_g[kb]).norm())
                                if g > 1e-3:
                                    num += float((c_p[ka] - c_p[kb]).norm())
                                    den += g
                    return num / den if den > 1e-3 else float("nan")

                rows = []
                for k in range(len(traj)):
                    for s in range(N_TRAJ_SAMPLES):
                        xr = traj[k][0, s][ar, rep]
                        dk = torch.cdist(xr, xr)[ij[:, 0], ij[:, 1]]
                        rows.append([idx, s, k, float(nl[k]), pearson(dk, y_fin[s]), pearson(dk, y_gt), cen_ratio(xr)])

                # USE-vs-REPRESENTATION classifier, same crop
                z = embeddings["pair"][0].float()
                n_all = rep.shape[0]
                side = torch.zeros(n_all, dtype=torch.bool, device=dev)
                side[torch.as_tensor(gen.permutation(n_all)[: n_all // 2].copy(), device=dev)] = True
                trm = cc & side.unsqueeze(1) & side.unsqueeze(0)
                tem = cc & (~side).unsqueeze(1) & (~side).unsqueeze(0)

                def sub(m):
                    q = m.nonzero(as_tuple=False)
                    if q.shape[0] > MAX_CC:
                        q = q[torch.as_tensor(gen.permutation(q.shape[0])[:MAX_CC].copy(), device=dev)]
                    return q

                q_tr, q_te = sub(trm), sub(tem)
                r_cc_z = ridge_r(z[q_tr[:, 0], q_tr[:, 1]], d_gt[q_tr[:, 0], q_tr[:, 1]],
                                 z[q_te[:, 0], q_te[:, 1]], d_gt[q_te[:, 0], q_te[:, 1]])

            surv = cap.get("surviving", [])
            udep = [cap.get("unpaired", {}).get(c, 0) for c in surv]
            wtj.writerows(rows)
            wex.writerow([idx, cap.get("record_id", f"idx{idx}"), n_chains, r_cc_z,
                          float(np.median(udep)) if udep else "",
                          fis[0], min(fis), max(fis), max(fis) - min(fis)])
            fex.flush()
            ftj.flush()
            done += 1
            if done % 5 == 0:
                print(f"[shard {args.shard}] {done}/{args.n} ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fex.close()
    ftj.close()
    print(f"[shard {args.shard}] wrote {done} → {ex_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
