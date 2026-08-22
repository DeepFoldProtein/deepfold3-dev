"""Per-layer activation amplification probe.

Loads a training checkpoint, runs the trunk forward (Evoformer+MSA stack +
PairFormer + Template embedder) on one record, and reports per-block residual
stream RMS plus update RMS. Lets us see WHERE in the trunk the magnitude blows
up (and whether MSA/template embedder injects abnormally large signal).

Outputs a CSV-friendly table per stage.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402
from scripts.triage_inf import _build_cfgs, _build_dataset, _find_idx, _parse_record  # noqa: E402


def _rms(t: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Per-element RMS over the last (channel) axis, masked."""
    x = t.detach().float()
    if mask is not None:
        m = mask.float()
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        x = x * m
        n = m.sum() * x.shape[-1]
    else:
        n = torch.tensor(float(x.numel()))
    sq = (x * x).sum()
    return float((sq / n.clamp_min(1)).sqrt().item())


def _abs_max(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def _mean_per_token_l2(t: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Per-token RMS over the channel axis, mask-averaged.

    This is exactly the ``single_rms`` / ``pair_rms`` metric (training_loss
    ``reg_form="log_rms"`` form). Equivalent to ``sqrt(single_l2)`` / ``sqrt(pair_l2)``
    under the legacy ``reg_form="lp"`` form (``p=2``). Name kept for backward
    compatibility with older callers.
    """
    x = t.detach().float()
    C = x.shape[-1]
    per_token = (x * x).sum(dim=-1) / C
    if mask is not None:
        m = mask.float()
        return float(((per_token * m).sum() / m.sum().clamp_min(1)).sqrt().item())
    return float(per_token.mean().sqrt().item())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="patchr:6nud")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(_REPO / "configs" / "initial.yaml"))
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-recycles", type=int, default=0)
    p.add_argument("--use-templates", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    train_cfg, model_cfg = _build_cfgs(Path(a.config))
    train_cfg.max_tokens = a.max_tokens
    train_cfg.max_recycles = a.max_recycles
    train_cfg.diffusion_batch_size = 1
    train_cfg.batch_size_per_rank = 1
    train_cfg.num_workers = 0
    # We only want trunk forward; template embedder may bring shape problems
    if not a.use_templates:
        # Force-disable templates to avoid template_aatype dependencies
        model_cfg.data.max_templates = 0

    device = torch.device("cuda")
    dtype = torch.bfloat16

    ds_name, rid = _parse_record(a.record)
    print(f"[probe] dataset={ds_name} record={rid} N_max={train_cfg.max_tokens}")
    ds, records = _build_dataset(ds_name, train_cfg, model_cfg)
    idx = _find_idx(records, rid)
    raw = ds._load_sample(idx)
    feat = collate_fn([raw], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
    feat = {k: v.to(device) for k, v in feat.items() if isinstance(v, torch.Tensor)}

    # Build model. Match training configuration (residual scale, fused flags, etc.)
    model = DeepFold3(model_cfg).to(device)
    print(f"[probe] params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("ema_state_dict", ckpt))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[probe] loaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")
    if "global_step" in ckpt:
        print(f"[probe] checkpoint global_step={ckpt['global_step']}")
    model.eval()

    # ---- Build trunk inputs (replicate model.forward_training prelude) ----
    # We use the model's internal helpers to build target_feat, msa_feat, etc.
    # Then call evoformer directly to capture intermediate magnitudes.
    target_feat = model._build_raw_target_feat(feat)  # [B, N, 63]
    # Pad target feat to 447 (raw 63 + evoformer_conditioning 384)
    # Actually model.forward_training uses self.evoformer_conditioning for token act.
    # Easier: just call model._call_trunk... let me check.

    # Inspect model attributes:
    print("[probe] model has methods: ", end="")
    for n in ["_run_trunk", "_call_evoformer", "evoformer", "evoformer_conditioning"]:
        print(f"{n}={'yes' if hasattr(model, n) else 'no'}, ", end="")
    print()

    # Run only the recycling/Evoformer stage from model.forward_training. We
    # extract this by calling the helper that sets up everything and yields
    # `embeddings`. The cleanest path is to call forward_training with
    # mini_rollout_steps=0, diffusion off — but that's not parametrised. So
    # we replicate the trunk prelude manually.

    # Required features (from forward_training body):
    # - target_feat = build_raw + token_act (evoformer_conditioning)
    # - msa_feat
    # - rel_feat
    # - pair_mask, seq_mask, msa_mask
    # - prev_pair=0, prev_single=0
    # We use model.forward(features, num_recycles=0, ...) WITHOUT mini-rollout
    # but the public forward returns final positions, so we need raw embeddings.

    # The cleanest approach: we install hooks on Evoformer block and run the
    # full forward_training. The hooks fire only during the trunk pass; we
    # break out after we capture. Let's do this:

    # We hook each EvoformerBlock and PairFormerBlock to record residual norms
    # at block boundary. We also hook MSA stack output, template embedding,
    # and inner residual updates of one PairFormerBlock for fine-grained probe.

    captured = {
        "msa_emb_init": None,
        "pair_init": None,
        "single_init": None,
        "template": None,  # template embedding output (added to pair)
        "evo_blocks": [],  # list of dicts per layer {single, pair, msa}
        "pf_blocks": [],  # list of dicts per layer {single, pair}
    }

    # We can't pre/post a PairFormerBlock from outside (single & pair are inside its
    # forward args). We hook inputs/outputs as tuples.

    seq_mask = feat.get("seq_mask")
    pair_mask = None
    if seq_mask is not None:
        seq_mask = seq_mask.float()
        pair_mask = (seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)).float()

    def _evo_hook(layer_idx):
        def hook(module, inp, out):
            msa, pair, msa_mask, pmask = inp
            new_msa, new_pair = out
            # RMS per channel — use seq_mask broadcast for valid tokens
            # MSA: [B?, S, N, c]; pair: [B?, N, N, c]
            captured["evo_blocks"].append(
                {
                    "layer": layer_idx,
                    "msa_rms_in": _rms(msa),
                    "msa_rms_out": _rms(new_msa),
                    "pair_rms_in": _rms(pair),
                    "pair_rms_out": _rms(new_pair),
                    "msa_amax_out": _abs_max(new_msa),
                    "pair_amax_out": _abs_max(new_pair),
                }
            )

        return hook

    def _pf_hook(layer_idx):
        def hook(module, inp, out):
            pair, single, pmask, smask = inp
            new_pair, new_single = out
            captured["pf_blocks"].append(
                {
                    "layer": layer_idx,
                    "single_rms_in": _rms(single),
                    "single_rms_out": _rms(new_single),
                    "pair_rms_in": _rms(pair),
                    "pair_rms_out": _rms(new_pair),
                    "single_l2_metric": _mean_per_token_l2(new_single, seq_mask),
                    "pair_l2_metric": _mean_per_token_l2(new_pair, pair_mask),
                    "single_amax_out": _abs_max(new_single),
                    "pair_amax_out": _abs_max(new_pair),
                }
            )

        return hook

    handles = []
    evo = model.evoformer
    for i, blk in enumerate(evo.msa_stack):
        handles.append(blk.register_forward_hook(_evo_hook(i)))
    for i, blk in enumerate(evo.trunk_pairformer):
        handles.append(blk.register_forward_hook(_pf_hook(i)))

    # Hook template embedder if present
    template_capture = {}
    if evo.template_embedding is not None:

        def _tmpl_hook(module, inp, out):
            template_capture["rms"] = _rms(out)
            template_capture["amax"] = _abs_max(out)

        handles.append(evo.template_embedding.register_forward_hook(_tmpl_hook))

    # Run training-style forward pass (we just need trunk forward; recycle=0)
    print("[probe] running forward_training (recycles=0, diffusion_bs=1)…")
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        with torch.no_grad():
            try:
                model.forward_training(
                    feat,
                    num_recycles=0,
                    diffusion_batch_size=1,
                    mini_rollout_steps=1,
                    mini_rollout_samples=1,
                    seed=0,
                    fused_ce=False,
                    diffusion_chunk_size=0,
                    backward_fn=None,
                    loss_cfg=None,
                    compute_confidence=False,
                )
            except Exception as e:
                print(f"[probe] forward raised after capturing: {type(e).__name__}: {e}")

    for h in handles:
        h.remove()

    # ---- Report ----
    out_lines = []

    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit("\n# Layer-wise activation amplification probe")
    emit(f"# checkpoint={a.checkpoint}")
    emit(f"# record={a.record}  step={ckpt.get('global_step', '?')}")
    if "rms" in template_capture:
        emit("\n## TEMPLATE EMBEDDING output (added to pair)")
        emit(f"  rms={template_capture['rms']:.4f}  abs_max={template_capture['amax']:.3e}")

    emit(f"\n## EVOFORMER MSA STACK ({len(captured['evo_blocks'])} blocks)")
    emit(
        f"  {'layer':>5} | {'msa_in':>8} {'msa_out':>8} {'msa×':>6} | {'pair_in':>8} {'pair_out':>8} {'pair×':>6} | {'msa_amax':>10} {'pair_amax':>10}"
    )
    for r in captured["evo_blocks"]:
        msa_amp = r["msa_rms_out"] / max(r["msa_rms_in"], 1e-9)
        pair_amp = r["pair_rms_out"] / max(r["pair_rms_in"], 1e-9)
        emit(
            f"  {r['layer']:5d} | {r['msa_rms_in']:8.4f} {r['msa_rms_out']:8.4f} {msa_amp:6.3f} | "
            f"{r['pair_rms_in']:8.4f} {r['pair_rms_out']:8.4f} {pair_amp:6.3f} | "
            f"{r['msa_amax_out']:10.3e} {r['pair_amax_out']:10.3e}"
        )

    emit(f"\n## PAIRFORMER STACK ({len(captured['pf_blocks'])} blocks)")
    emit(
        f"  {'layer':>5} | {'sing_in':>8} {'sing_out':>8} {'sing×':>6} | {'pair_in':>8} {'pair_out':>8} {'pair×':>6} | {'sing_l2m':>9} {'pair_l2m':>9} | {'s_amax':>10} {'p_amax':>10}"
    )
    prev_s, prev_p = None, None
    for r in captured["pf_blocks"]:
        s_amp = r["single_rms_out"] / max(r["single_rms_in"], 1e-9)
        p_amp = r["pair_rms_out"] / max(r["pair_rms_in"], 1e-9)
        emit(
            f"  {r['layer']:5d} | {r['single_rms_in']:8.4f} {r['single_rms_out']:8.4f} {s_amp:6.3f} | "
            f"{r['pair_rms_in']:8.4f} {r['pair_rms_out']:8.4f} {p_amp:6.3f} | "
            f"{r['single_l2_metric']:9.4f} {r['pair_l2_metric']:9.4f} | "
            f"{r['single_amax_out']:10.3e} {r['pair_amax_out']:10.3e}"
        )

    if captured["pf_blocks"]:
        first = captured["pf_blocks"][0]
        last = captured["pf_blocks"][-1]
        n = len(captured["pf_blocks"])
        cumul_s = last["single_rms_out"] / max(first["single_rms_in"], 1e-9)
        cumul_p = last["pair_rms_out"] / max(first["pair_rms_in"], 1e-9)
        per_layer_s = math.exp(math.log(max(cumul_s, 1e-9)) / n)
        per_layer_p = math.exp(math.log(max(cumul_p, 1e-9)) / n)
        emit(f"\n## CUMULATIVE PAIRFORMER AMPLIFICATION ({n} layers)")
        emit(f"  single: {cumul_s:.3f}× total → {per_layer_s:.4f}× per layer (geom. mean)")
        emit(f"  pair:   {cumul_p:.3f}× total → {per_layer_p:.4f}× per layer (geom. mean)")

    if a.out:
        Path(a.out).write_text("\n".join(out_lines))
        print(f"\n[probe] wrote {a.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
