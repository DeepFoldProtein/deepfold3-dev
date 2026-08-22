"""Does the TRUNK already know the inter-chain pose? (no gradient)

``probe_pose_sigma.py`` established that a single denoise step from pure noise
reproduces the 20-step rollout's inter-chain pose (r = 0.959): the diffusion
trajectory adds nothing to docking, so the pose is fixed by what the trunk hands
the denoiser. That leaves two candidates, which this script separates:

  (a) the trunk does not represent the pose  → trunk / conditioning problem
  (b) the trunk represents it but the denoiser cannot render it → denoiser problem

The distogram head reads the trunk pair representation and nothing else — it
never sees the diffusion noise — so it is a direct read-out of what the trunk
knows. For each example this scores the trunk's **cross-chain** distogram
against the ground truth and correlates it with the rollout's ``fape_inter``.

  high trunk AUC + failed rollout pose  → (b)
  low  trunk AUC + failed rollout pose  → (a)

Metric choice: the distogram saturates at 21.69 Å (last bin), so a plain MAE is
uninformative for the far pairs that matter most to a pose. The headline is
instead a saturation-free **contact discrimination** score — ROC-AUC and average
precision of P(d < 8 Å) against the true cross-chain contact map, i.e. "does the
trunk know which residues form the interface". Intra-chain is the control.

Usage (8 GPUs):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 uv run python scripts/probe_distogram_pose.py \
        --shard $i --num-shards 8 --n 40 --out /tmp/disto_probe &
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

CONTACT_A = 8.0  # cross-chain contact definition
IN_RANGE_A = 20.0  # below the 21.69 Å last-bin saturation
FAPE_KW = dict(
    clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
    all_atom_points=True, mode="both", chunk_size=64,
)


def auc_ap(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """ROC-AUC (Mann-Whitney U) and average precision, no sklearn dependency."""
    n_pos, n_neg = int(label.sum()), int((~label.astype(bool)).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties so AUC is unbiased for the many identical
    # near-zero probabilities the far pairs produce
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    auc = (ranks[label.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    desc = np.argsort(-score, kind="mergesort")
    lab = label.astype(bool)[desc]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    ap = float((prec * lab).sum() / n_pos)
    return float(auc), ap


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap_.add_argument("--config", default="configs/stage2.yaml")
    ap_.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap_.add_argument("--cutoff-date", default="2021-09-30")
    ap_.add_argument("--out", default="/tmp/disto_probe")
    ap_.add_argument("--n", type=int, default=40)
    ap_.add_argument("--shard", type=int, default=0)
    ap_.add_argument("--num-shards", type=int, default=1)
    ap_.add_argument("--seed", type=int, default=101)
    args = ap_.parse_args()

    dev = "cuda:0"
    torch.manual_seed(2000 + args.shard)

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

    R, gate_k, dh = train_cfg.max_recycles, train_cfg.chain_perm_gate_k, model.diffusion_head
    head = model.distogram_head
    breaks = torch.linspace(head.first_break, head.last_break, head.num_bins - 1, device=dev)
    gap = breaks[-1] - breaks[-2]
    lower = torch.cat([torch.zeros(1, device=dev), breaks])
    upper = torch.cat([breaks, breaks[-1:] + gap])
    centers = (lower + upper) * 0.5
    contact_bins = (upper <= CONTACT_A).float()  # P(d < 8 Å) = Σ over fully-inside bins

    out_path = Path(args.out) / f"disto_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow([
        "record_id", "idx", "n_chains", "n_tokens",
        "cc_auc", "cc_ap", "cc_base_rate", "cc_n_contacts", "cc_mae_in", "cc_pred_d_in", "cc_gt_d_in",
        "intra_auc", "intra_mae_in",
        "roll_fape_inter", "roll_fape_intra", "roll_cc_recall", "roll_cc_precision", "roll_d_ratio",
    ])

    done, t0 = 0, time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        rid = batch.get("record_id", [f"idx{idx}"])
        rid = rid[0] if isinstance(rid, (list, tuple)) else str(rid)
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

                orig = dh.cfg.steps, dh.cfg.num_samples
                dh.cfg.steps, dh.cfg.num_samples = train_cfg.mini_rollout_steps, max(train_cfg.mini_rollout_samples, gate_k)
                try:
                    mini = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)["atom_positions"]
                finally:
                    dh.cfg.steps, dh.cfg.num_samples = orig

                pred_b = {"mini_rollout_pos": mini[0, : train_cfg.mini_rollout_samples], "chain_perm_gate_pos": mini[0]}
                f0 = slice_batch_dict(features, 0, 1)
                gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(pred_b, f0, stability_gate_k=gate_k)

                asym = f0["asym_id"]
                seq_m = f0["seq_mask"].bool()
                if c_keep is not None:
                    seq_m = seq_m & c_keep
                n_chains = int(asym[seq_m].unique().numel())
                if n_chains < 2:
                    continue
                n_tok = int(seq_m.sum())

                atom_m = gt_mask.bool()
                # trunk read-out — the distogram never sees the diffusion noise
                logits = head(embeddings["pair"])[0].float()  # [N, N, 64]
                probs = logits.softmax(-1)
                p_contact = (probs * contact_bins).sum(-1)  # [N, N]
                e_dist = (probs * centers).sum(-1)  # [N, N]

                rep = f0["token_to_rep_atom"]
                ar = torch.arange(rep.shape[0], device=dev)
                gt_rep = gt_pos[ar, rep].float()
                rep_ok = seq_m & atom_m[ar, rep]
                d_gt = torch.cdist(gt_rep, gt_rep)
                d_roll = torch.cdist(mini[0, 0][ar, rep].float(), mini[0, 0][ar, rep].float())

                valid = rep_ok.unsqueeze(1) & rep_ok.unsqueeze(0)
                valid.fill_diagonal_(False)
                cc = valid & (asym.unsqueeze(1) != asym.unsqueeze(0))
                ic = valid & (asym.unsqueeze(1) == asym.unsqueeze(0))

                def block(m, d_gt=d_gt, p_contact=p_contact, e_dist=e_dist):
                    lab = (d_gt[m] < CONTACT_A).cpu().numpy()
                    a, p = auc_ap(p_contact[m].cpu().numpy(), lab)
                    inr = m & (d_gt < IN_RANGE_A)
                    mae = float((e_dist[inr] - d_gt[inr]).abs().mean()) if inr.any() else float("nan")
                    return a, p, lab, mae, (float(e_dist[inr].mean()), float(d_gt[inr].mean())) if inr.any() else (float("nan"),) * 2

                cc_auc, cc_ap, cc_lab, cc_mae, (cc_pd, cc_gd) = block(cc)
                i_auc, _, _, i_mae, _ = block(ic)

                # what the rollout actually produced, on the same contact set
                gt_c = (d_gt < CONTACT_A) & cc
                pr_c = (d_roll < CONTACT_A) & cc
                rec = float((gt_c & pr_c).sum() / gt_c.sum().clamp(min=1))
                prc = float((gt_c & pr_c).sum() / pr_c.sum().clamp(min=1))

                fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                pm = fm.unsqueeze(-1) & f0["pair_mask"].bool()
                mf = atom_m.float().unsqueeze(-1)
                gt_ctr = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                fx, fi = fape_loss(
                    mini[0, 0].unsqueeze(0).float(), gt_ctr, None, None, None,
                    f0["frame_atom_index"], f0["frame_token_index"], fm, seq_m, pm,
                    atom_mask=atom_m, asym_id=asym, **FAPE_KW,
                )

                def cen(p, asym=asym, atom_m=atom_m):
                    o = {}
                    for c in asym.unique().tolist():
                        s = asym == c
                        m2 = atom_m[s].float().unsqueeze(-1)
                        if m2.sum() >= 1:
                            o[int(c)] = (p[s].float() * m2).sum((0, 1)) / m2.sum()
                    return o
                cp, cg = cen(mini[0, 0]), cen(gt_ctr)
                ks = sorted(set(cp) & set(cg))
                dp = [float((cp[ks[i]] - cp[ks[j]]).norm()) for i in range(len(ks)) for j in range(i + 1, len(ks))]
                dgv = [float((cg[ks[i]] - cg[ks[j]]).norm()) for i in range(len(ks)) for j in range(i + 1, len(ks))]
                d_ratio = sum(dp) / sum(dgv) if sum(dgv) > 1e-3 else float("nan")

            wr.writerow([rid, idx, n_chains, n_tok,
                         cc_auc, cc_ap, float(cc_lab.mean()), int(cc_lab.sum()), cc_mae, cc_pd, cc_gd,
                         i_auc, i_mae,
                         float(fi), float(fx), rec, prc, d_ratio])
            fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"[shard {args.shard}] {done}/{args.n} ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[shard {args.shard}] wrote {done} → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
