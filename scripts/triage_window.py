"""Triage: reproduce training-style 4-sample accumulated backward.

Training runs with ``batch_size_per_rank=2 × accumulate_grad_batches=2``
— 4 microbatches per optimizer step — and only inspects gradients AFTER
all four backwards have accumulated. This script loads one rank's actual
4-sample NaN/Inf window (from a training log) and replays the same
sequence on one GPU to see whether the NaN first appears during a
specific microbatch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402
from deepfold3.model.triton import evo_attention as _evo_mod  # noqa: E402
from deepfold3.training.collectives import slice_batch_dict  # noqa: E402
from deepfold3.training.loss import training_loss  # noqa: E402
from scripts.triage_inf import (  # noqa: E402
    HookRecorder,
    _build_cfgs,
    _build_dataset,
    _find_idx,
    _parse_record,
)

# ---------------------------------------------------------------------------
# Kernel-input capture hook
# ---------------------------------------------------------------------------
# Monkey-patches _EvoformerAttention.forward to watch every attention call in
# a target microbatch and keep the one with the largest |output|. Saved in the
# non-pre-transposed layout [B, N_seq, N_res, H, D] so tests/triton/
# test_evo_attn_robustness.py::test_known_failure_replay can load it directly.

_CAPTURE_STATE: dict = {
    "active": False,
    "call_idx": 0,
    "winner": None,  # dict with Q/K/V/res_mask/pair_bias/expected_output/meta
    "winner_max_abs": -1.0,
}
_CAPTURE_ORIG_APPLY = None


def _install_capture_hook() -> None:
    """Wrap _EvoformerAttention.apply so we can (a) score each call by max|out|
    and (b) register a backward hook on the returned grad-tracked tensor to
    capture the incoming dO from the real training backward.

    Hooking `forward` directly does not work: inside Function.forward, `out`
    has requires_grad=False. We need the post-apply tensor, which participates
    in the autograd graph.
    """
    global _CAPTURE_ORIG_APPLY
    if _CAPTURE_ORIG_APPLY is not None:
        return
    _CAPTURE_ORIG_APPLY = _evo_mod._EvoformerAttention.apply

    def _wrapped_apply(*args, **kwargs):
        # Parse args: (Q, K, V, res_mask, pair_bias, pre_transposed=False).
        # Stash input references before calling; they remain unchanged by apply.
        Q, K, V, res_mask, pair_bias = args[:5]
        pre_transposed = args[5] if len(args) >= 6 else kwargs.get("pre_transposed", False)

        out = _CAPTURE_ORIG_APPLY(*args, **kwargs)

        if not _CAPTURE_STATE["active"]:
            return out

        # Normalize layouts to the non-pre_transposed convention.
        out_cap = out.detach()
        Q_in = Q.detach()
        K_in = K.detach()
        V_in = V.detach()
        if pre_transposed:
            Q_in = Q_in.transpose(-2, -3).contiguous()
            K_in = K_in.transpose(-2, -3).contiguous()
            V_in = V_in.transpose(-2, -3).contiguous()
            out_cap = out_cap.transpose(-2, -3).contiguous()
        else:
            Q_in = Q_in.contiguous()
            K_in = K_in.contiguous()
            V_in = V_in.contiguous()
            out_cap = out_cap.contiguous()

        out_finite = bool(torch.isfinite(out_cap).all().item())
        score = float("inf") if not out_finite else float(out_cap.abs().max().item())
        idx = _CAPTURE_STATE["call_idx"]
        _CAPTURE_STATE["call_idx"] = idx + 1

        # Backward hook — fires only if `out` participates in a real backward.
        dO_slot: list = [None]
        if out.requires_grad:

            def _bwd_hook(grad, _slot=dO_slot, _pre_transposed=pre_transposed):
                g = grad.detach()
                if _pre_transposed:
                    g = g.transpose(-2, -3).contiguous()
                else:
                    g = g.contiguous()
                _slot[0] = g.clone()
                return None

            out.register_hook(_bwd_hook)

        if score > _CAPTURE_STATE["winner_max_abs"]:
            _CAPTURE_STATE["winner_max_abs"] = score
            _CAPTURE_STATE["winner"] = {
                "Q": Q_in.clone(),
                "K": K_in.clone(),
                "V": V_in.clone(),
                "res_mask": res_mask.detach().contiguous().clone(),
                "pair_bias": pair_bias.detach().contiguous().clone(),
                "expected_output": out_cap.clone(),
                "_dO_slot": dO_slot,
                "meta": {
                    "call_idx": idx,
                    "pre_transposed_source": bool(pre_transposed),
                    "output_finite": out_finite,
                    "max_abs_output": score,
                    "shape_Q": tuple(Q_in.shape),
                    "shape_pair_bias": tuple(pair_bias.shape),
                    "dtype": str(Q_in.dtype),
                },
            }
        return out

    _evo_mod._EvoformerAttention.apply = _wrapped_apply


def _set_capture_active(on: bool) -> None:
    _CAPTURE_STATE["active"] = on
    if on:
        _CAPTURE_STATE["call_idx"] = 0
        _CAPTURE_STATE["winner"] = None
        _CAPTURE_STATE["winner_max_abs"] = -1.0


def _save_capture(path: Path, extra_meta: dict) -> None:
    w = _CAPTURE_STATE["winner"]
    if w is None:
        print("[capture] no attention calls observed in target microbatch — nothing to save")
        return
    # Resolve dO captured by backward hook (may be None if backward didn't fire).
    dO_slot = w.pop("_dO_slot", [None])
    dO = dO_slot[0]
    if dO is not None:
        w["dO"] = dO
        w["meta"]["dO_max_abs"] = float(dO.abs().max().item())
        w["meta"]["dO_finite"] = bool(torch.isfinite(dO).all().item())
        w["meta"]["dO_shape"] = tuple(dO.shape)
    else:
        w["meta"]["dO_max_abs"] = None
        w["meta"]["dO_finite"] = None
    w["meta"] = {**w["meta"], **extra_meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, path)
    print(f"[capture] saved attention-input fixture: {path}")
    print(
        f"[capture] winner call_idx={w['meta']['call_idx']} "
        f"max_abs_output={w['meta']['max_abs_output']:.3e} "
        f"output_finite={w['meta']['output_finite']} "
        f"shape_Q={w['meta']['shape_Q']} dtype={w['meta']['dtype']}"
    )
    if dO is not None:
        print(f"[capture] dO: max|={w['meta']['dO_max_abs']:.3e}  finite={w['meta']['dO_finite']}")
    else:
        print("[capture] dO: NOT captured (backward did not flow through this attention output)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--records",
        required=True,
        help="Comma-separated list of dataset:record_id — order matches training microbatches.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(_REPO / "configs" / "initial.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--diffusion-bs", type=int, default=8)
    p.add_argument("--max-recycles", type=int, default=0)
    p.add_argument("--no-fused", action="store_true", help="Disable all Triton fused kernels at once.")
    p.add_argument(
        "--disable-fused",
        default="",
        help="Comma-separated fused-flag names to turn off individually — e.g. "
        "'use_fused_transition_pair'. Useful for bisecting which kernel drives the blow-up.",
    )
    p.add_argument("--log", default=None)
    p.add_argument(
        "--capture-kernel-inputs",
        default=None,
        help="Path to save (Q, K, V, res_mask, pair_bias, expected_output) "
        "for the attention call with largest |output| in the target microbatch. "
        "Used by tests/triton/test_evo_attn_robustness.py::test_known_failure_replay.",
    )
    p.add_argument(
        "--capture-microbatch",
        type=int,
        default=-1,
        help="Which microbatch to capture from (0-indexed). -1 = last microbatch (default).",
    )
    a = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    if a.log:

        class _Tee:
            def __init__(self, path):
                self.stdout = sys.stdout
                self.f = open(path, "w", buffering=1)

            def write(self, m):
                self.stdout.write(m)
                self.f.write(m)

            def flush(self):
                self.stdout.flush()
                self.f.flush()

        t = _Tee(a.log)
        sys.stdout = t
        sys.stderr = t

    refs = [r.strip() for r in a.records.split(",") if r.strip()]
    print(f"[window] records ({len(refs)}): {refs}")

    train_cfg, model_cfg = _build_cfgs(Path(a.config))
    train_cfg.diffusion_batch_size = a.diffusion_bs
    train_cfg.diffusion_chunk_size = min(train_cfg.diffusion_chunk_size, a.diffusion_bs)
    train_cfg.max_recycles = a.max_recycles
    train_cfg.batch_size_per_rank = 1
    train_cfg.num_workers = 0
    ALL_FUSED = (
        "use_fused_evo_attention",
        "use_fused_pair_bias",
        "use_fused_swiglu",
        "use_fused_transition_pair",
        "use_fused_transition_single",
        "use_fused_ada_transition",
        "use_fused_triangle_mul",
        "use_fused_ln_transpose",
    )
    disabled: list[str] = []
    if a.no_fused:
        for f in ALL_FUSED:
            if hasattr(model_cfg.optim, f):
                setattr(model_cfg.optim, f, False)
                disabled.append(f)
    if a.disable_fused:
        for f in [x.strip() for x in a.disable_fused.split(",") if x.strip()]:
            if hasattr(model_cfg.optim, f):
                setattr(model_cfg.optim, f, False)
                disabled.append(f)
    if disabled:
        print(f"[window] fused flags disabled: {sorted(set(disabled))}")
    device = torch.device(a.device)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map.get(train_cfg.dtype, torch.float32)
    amp_enabled = amp_dtype is not torch.float32
    print(f"[window] dtype={train_cfg.dtype} amp_enabled={amp_enabled}")

    # Pre-build dataset handles, one per (dataset name).
    ds_cache = {}
    batches = []
    for i, ref in enumerate(refs):
        ds_name, rid = _parse_record(ref)
        if ds_name not in ds_cache:
            ds_cache[ds_name] = _build_dataset(ds_name, train_cfg, model_cfg)
        ds, records = ds_cache[ds_name]
        idx = _find_idx(records, rid)
        raw = ds._load_sample(idx)
        feat = collate_fn([raw], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        feat = {k: v.to(device) for k, v in feat.items() if isinstance(v, torch.Tensor)}
        batches.append((ref, feat))
        print(f"[window] loaded microbatch {i}: {ref}")

    print("[window] building DeepFold3")
    model = DeepFold3(model_cfg).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(
        f"[window] checkpoint global_step={ckpt.get('global_step', '?')} effective_step={ckpt.get('effective_step', '?')}"
    )
    model.train()

    # Optional kernel-input capture: hook fused evo-attention forward so we
    # can save the inputs of the worst attention call (max |output|) in the
    # target microbatch. Useful for tests/triton/test_evo_attn_robustness.py.
    capture_path = Path(a.capture_kernel_inputs) if a.capture_kernel_inputs else None
    capture_micro = a.capture_microbatch if a.capture_microbatch >= 0 else len(batches) - 1
    if capture_path is not None:
        _install_capture_hook()
        print(
            f"[capture] installed hook; target microbatch #{capture_micro} "
            f"({batches[capture_micro][0]}); output path: {capture_path}"
        )

    # Accumulated backward: forward + loss + backward per microbatch, no
    # zero_grad between them. At the end, inspect whether any parameter
    # now has a NaN/Inf grad. Also keep per-microbatch forward hooks to
    # flag the first in-forward blow-up.
    model.zero_grad(set_to_none=True)
    first_bad_micro = None
    for i, (ref, feat) in enumerate(batches):
        print(f"\n========== microbatch {i}: {ref} ==========")
        if capture_path is not None:
            _set_capture_active(i == capture_micro)
        rec = HookRecorder(model)
        rec.install()
        try:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model.forward_training(
                    feat,
                    num_recycles=train_cfg.max_recycles,
                    diffusion_batch_size=train_cfg.diffusion_batch_size,
                    mini_rollout_steps=train_cfg.mini_rollout_steps,
                    mini_rollout_samples=train_cfg.mini_rollout_samples,
                    seed=i,
                    fused_ce=False,
                    diffusion_chunk_size=0,
                    backward_fn=None,
                    loss_cfg=None,
                    compute_confidence=True,
                )
            # Real training loss: slice per-B element (matches step.py's
            # loop), then average. Production batch_size_per_rank=1 here,
            # so single slice = single element.
            pred_b = slice_batch_dict(pred, 0, 1)
            feat_b = slice_batch_dict(feat, 0, 1)
            loss, loss_dict = training_loss(pred_b, feat_b, train_cfg)
            loss_summary = ", ".join(f"{k}={float(v.detach()):.3e}" for k, v in sorted(loss_dict.items()))
            print(f"  training_loss = {loss.item():.3e}  (isfinite={torch.isfinite(loss).item()})")
            print(f"  components    : {loss_summary}")
            (loss / len(batches)).backward()
        finally:
            rec.remove()
            if capture_path is not None:
                _set_capture_active(False)

        if rec.events:
            if first_bad_micro is None:
                first_bad_micro = i
            print(f"  !!! {len(rec.events)} submodule(s) produced NaN/Inf during forward")
            rec.report()
        else:
            print("  forward clean")

        # Per-microbatch param grad scan.
        bad_p = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_p.append((name, tuple(p.shape), float(p.grad.abs().max().item())))
        if bad_p:
            print(f"  accumulated grads NON-FINITE after this microbatch: {len(bad_p)} params (first 10):")
            for n, s, amax in bad_p[:10]:
                print(f"    {n:70s} shape={s!s:20s} abs_max={amax:.3e}")
            if first_bad_micro is None:
                first_bad_micro = i
        else:
            # Report magnitude summary + top-5 parameters by |grad|.
            entries = [
                (name, float(p.grad.abs().max().item()), tuple(p.shape))
                for name, p in model.named_parameters()
                if p.grad is not None
            ]
            entries.sort(key=lambda e: e[1], reverse=True)
            max_g = entries[0][1] if entries else 0.0
            print(f"  accumulated grads finite; max |grad| = {max_g:.3e}")
            if max_g > 1.0:
                print("  top-10 |grad| params:")
                for n, g, s in entries[:10]:
                    print(f"    |grad|={g:.3e}  {n:68s} shape={s}")

    if first_bad_micro is not None:
        print(f"\n[window] FIRST BAD MICROBATCH: #{first_bad_micro} ({batches[first_bad_micro][0]})")
    else:
        print("\n[window] NO microbatch triggered NaN/Inf — combination effect did not reproduce.")

    if capture_path is not None:
        _save_capture(
            capture_path,
            extra_meta={
                "source_run": "df3-ckpt-per-step",
                "checkpoint": a.checkpoint,
                "target_record": batches[capture_micro][0],
                "microbatch_idx_in_window": capture_micro,
                "window_records": [r for r, _ in batches],
            },
        )

    print("\n[window] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
