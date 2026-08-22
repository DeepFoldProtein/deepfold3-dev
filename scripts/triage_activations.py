"""Per-layer activation magnitude probe.

Runs the same single-sample forward that triggers the fused-kernel
gradient blow-up (``patchr:7k6p`` on ``checkpoint_19`` with production
config) and records the ``abs_max`` of every submodule's forward output.
Takes ``--disable-fused`` so we can diff the fused vs non-fused
activation traces and see where they first diverge.
"""

from __future__ import annotations

import argparse
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


def _walk(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="patchr:7k6p")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(_REPO / "configs" / "initial.yaml"))
    p.add_argument("--diffusion-bs", type=int, default=48)
    p.add_argument("--max-recycles", type=int, default=3)
    p.add_argument("--disable-fused", default="")
    p.add_argument(
        "--include", default="", help="Only print modules whose name contains one of these comma-separated substrings."
    )
    p.add_argument("--log", default=None)
    a = p.parse_args()

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

    ds_name, rid = _parse_record(a.record)
    train_cfg, model_cfg = _build_cfgs(Path(a.config))
    train_cfg.diffusion_batch_size = a.diffusion_bs
    train_cfg.diffusion_chunk_size = min(train_cfg.diffusion_chunk_size, a.diffusion_bs)
    train_cfg.max_recycles = a.max_recycles
    train_cfg.batch_size_per_rank = 1
    train_cfg.num_workers = 0

    if a.disable_fused:
        for f in [x.strip() for x in a.disable_fused.split(",") if x.strip()]:
            if hasattr(model_cfg.optim, f):
                setattr(model_cfg.optim, f, False)
        print(f"[act] disabled fused flags: {a.disable_fused}")

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map.get(train_cfg.dtype, torch.float32)
    amp_enabled = amp_dtype is not torch.float32

    device = torch.device("cuda")
    ds, records = _build_dataset(ds_name, train_cfg, model_cfg)
    idx = _find_idx(records, rid)
    raw = ds._load_sample(idx)
    feat = collate_fn([raw], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
    feat = {k: v.to(device) for k, v in feat.items() if isinstance(v, torch.Tensor)}

    model = DeepFold3(model_cfg).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.train()

    include = [s.strip() for s in a.include.split(",") if s.strip()]

    records_out: list[tuple[str, str, float, float, float]] = []

    def make_hook(name: str, module_type: str):
        def hook(_m, _inp, out):
            if include and not any(s in name for s in include):
                return
            amax = amin = mean = float("nan")
            n_nan = n_inf = 0
            for t in _walk(out):
                if not t.is_floating_point():
                    continue
                with torch.no_grad():
                    tf = t.detach()
                    nan_c = int(torch.isnan(tf).sum().item())
                    inf_c = int(torch.isinf(tf).sum().item())
                    n_nan += nan_c
                    n_inf += inf_c
                    if nan_c or inf_c:
                        continue
                    amax = (
                        max(amax, float(tf.abs().max().item())) if not (amax != amax) else float(tf.abs().max().item())
                    )
                    amin = float(tf.min().item())
                    mean = float(tf.mean().item())
            records_out.append((name, module_type, amax, amin, mean))
            if n_nan or n_inf:
                records_out[-1] = (name + f"[NaN={n_nan},Inf={n_inf}]", module_type, amax, amin, mean)

        return hook

    handles = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        handles.append(mod.register_forward_hook(make_hook(name, type(mod).__name__)))

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        with torch.no_grad():
            model.forward_training(
                feat,
                num_recycles=train_cfg.max_recycles,
                diffusion_batch_size=train_cfg.diffusion_batch_size,
                mini_rollout_steps=train_cfg.mini_rollout_steps,
                mini_rollout_samples=train_cfg.mini_rollout_samples,
                seed=0,
                fused_ce=False,
                diffusion_chunk_size=0,
                backward_fn=None,
                loss_cfg=None,
                compute_confidence=True,
            )
    for h in handles:
        h.remove()

    # Print ranked by abs_max.
    records_out.sort(key=lambda r: r[2] if r[2] == r[2] else 0.0, reverse=True)
    print(f"\n[act] top-40 modules by abs_max (of {len(records_out)} total)")
    for name, mtype, amax, amin, mean in records_out[:40]:
        print(f"  |max|={amax:10.3e}  min={amin:11.3e}  mean={mean:11.3e}  {name:75s} ({mtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
