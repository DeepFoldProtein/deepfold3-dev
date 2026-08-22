"""Triage: locate first NaN/Inf-emitting submodule on a pathological sample.

Loads one offender record (default ``patchr:6nud``), builds ``DeepFold3`` from
``configs/initial.yaml`` with *fresh* (randomly initialised) weights, runs a
single forward_training pass in no_grad mode with forward hooks on every
submodule that inspect output tensors for NaN/Inf. Then re-runs under
``torch.autograd.set_detect_anomaly(True)`` to catch the first backward Inf.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from deepfold3.config_loader import load_config  # noqa: E402
from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.data.module.dataset import TrainingDataset  # noqa: E402
from deepfold3.data.module.of3_dataset import OF3Dataset, discover_of3_records  # noqa: E402
from deepfold3.data.prepare.manifest import discover_records  # noqa: E402
from deepfold3.model.config import DatasetConfig, ModelConfig, TrainingConfig  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402

logger = logging.getLogger("triage_inf")

_DATASET_SPECS = {
    "patchr": DatasetConfig(
        name="patchr",
        data_dir="/data/distillation/patchr_training",
        format="df3",
        cutoff_date="2021-09-30",
    ),
    "long_monomers": DatasetConfig(
        name="long_monomers",
        data_dir="/data/distillation/long_monomers/preprocessed",
        format="of3",
        is_distillation=True,
    ),
    "short_monomers": DatasetConfig(
        name="short_monomers",
        data_dir="/data/distillation/short_monomers/preprocessed",
        format="of3",
        is_distillation=True,
    ),
}


def _parse_record(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        return "patchr", ref
    ds, rid = ref.split(":", 1)
    if ds not in _DATASET_SPECS:
        raise ValueError(f"unknown dataset {ds!r}")
    return ds, rid


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    if not t.is_floating_point():
        return {"shape": tuple(t.shape), "dtype": str(t.dtype), "int": True}
    f32 = t.detach().to(torch.float32)
    fin = torch.isfinite(f32)
    has_nan = bool(torch.isnan(f32).any().item())
    has_inf = bool(torch.isinf(f32).any().item())
    if fin.any():
        v = f32[fin]
        amax, amin, mean = float(v.abs().max()), float(v.min()), float(v.mean())
    else:
        amax = amin = mean = math.nan
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "abs_max": amax,
        "min": amin,
        "mean": mean,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _walk(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def _any_bad(obj) -> tuple[bool, bool]:
    """Cheap in-place NaN/Inf scan — avoids fp32 upcasting (which OOMs
    at production shapes). ``torch.isnan/isinf`` work on bf16/fp16
    directly, so we scan in the tensor's own dtype.
    """
    hn = hi = False
    with torch.no_grad():
        for t in _walk(obj):
            if not t.is_floating_point():
                continue
            if torch.isnan(t).any():
                hn = True
            if torch.isinf(t).any():
                hi = True
            if hn and hi:
                break
    return hn, hi


class HookRecorder:
    def __init__(self, model):
        self.model = model
        self.events: list[dict] = []
        self._handles: list = []
        self._step = 0
        self.parent_of: dict[str, str] = {}
        for n, m in model.named_modules():
            for c, _ in m.named_children():
                full = f"{n}.{c}" if n else c
                self.parent_of[full] = n

    def install(self):
        for name, mod in self.model.named_modules():
            if name == "":
                continue

            def make(mod_name, module):
                def hook(m, inputs, output):
                    self._step += 1
                    in_nan, in_inf = _any_bad(inputs)
                    out_nan, out_inf = _any_bad(output)
                    if out_nan or out_inf:
                        self.events.append(
                            {
                                "order": self._step,
                                "module": mod_name,
                                "module_type": type(m).__name__,
                                "in_nan": in_nan,
                                "in_inf": in_inf,
                                "out_nan": out_nan,
                                "out_inf": out_inf,
                                "input_stats": [_tensor_stats(t) for t in _walk(inputs) if t.is_floating_point()][:4],
                                "output_stats": [_tensor_stats(t) for t in _walk(output) if t.is_floating_point()][:4],
                            }
                        )

                return hook

            self._handles.append(mod.register_forward_hook(make(name, mod)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self):
        if not self.events:
            print("\n[hooks] NO submodule reported NaN/Inf in its output.")
            return
        evs = sorted(self.events, key=lambda e: e["order"])
        f = evs[0]
        print("\n========== FIRST BAD MODULE ==========")
        print(f"module      : {f['module']} ({f['module_type']})")
        print(f"out nan/inf : {f['out_nan']}/{f['out_inf']}")
        print(f"in  nan/inf : {f['in_nan']}/{f['in_inf']}")
        print("input stats :")
        for s in f["input_stats"]:
            print(f"  {s}")
        print("output stats:")
        for s in f["output_stats"]:
            print(f"  {s}")
        chain = [f["module"]]
        cur = f["module"]
        while cur in self.parent_of and self.parent_of[cur]:
            cur = self.parent_of[cur]
            chain.append(cur)
        print("ancestors   :", " -> ".join(reversed(chain)))
        if len(evs) > 1:
            print(f"\n---- other bad modules ({min(25, len(evs) - 1)}/{len(evs) - 1}) ----")
            for e in evs[1:25]:
                print(
                    f"  #{e['order']:5d} {e['module']:60s} ({e['module_type']}) nan={e['out_nan']} inf={e['out_inf']}"
                )


def _build_cfgs(config_path: Path):
    train_cfg, model_cfg = TrainingConfig(), ModelConfig()
    data = load_config(config_path)
    for k, v in data.get("training", {}).items():
        if k in ("extra_datasets", "eval_datasets"):
            continue
        if hasattr(train_cfg, k):
            try:
                setattr(train_cfg, k, v)
            except Exception:
                pass

    def apply(cfg, overrides):
        for k, v in overrides.items():
            if k.startswith("_") or not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
                apply(cur, v)
            else:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

    apply(model_cfg, data.get("model", {}))
    # Default to a slimmed-down shape (one noise sample, no recycling) so
    # the fresh-init sanity pass runs on any GPU. For reproducing training
    # inf, the caller passes --full-config which keeps the production
    # diffusion_batch_size / mini_rollout_steps / recycles verbatim.
    train_cfg.batch_size_per_rank = 1
    train_cfg.num_workers = 0
    train_cfg.accumulate_grad_batches = 1
    # Keep production optim flags as-is — we need Triton fused kernels +
    # gradient checkpointing on to match the training regime where the
    # inf was observed. Hook paths become less readable inside fused
    # modules but the global NaN/Inf detection still works.
    # (Disable ``use_torch_compile`` anyway — graph capture breaks hooks.)
    if hasattr(model_cfg.optim, "use_torch_compile"):
        model_cfg.optim.use_torch_compile = False
    model_cfg.diffusion.skip_amp_sample = False
    if hasattr(model_cfg.diffusion, "skip_amp_sample_training"):
        model_cfg.diffusion.skip_amp_sample_training = False
    if hasattr(model_cfg.precision, "fp32_diffusion_sample"):
        model_cfg.precision.fp32_diffusion_sample = False
    if hasattr(model_cfg.precision, "fp32_diffusion_sample_training"):
        model_cfg.precision.fp32_diffusion_sample_training = False
    return train_cfg, model_cfg


def _build_dataset(ds_name, train_cfg, model_cfg):
    spec = _DATASET_SPECS[ds_name]
    data_dir = Path(spec.data_dir)
    common = dict(
        max_tokens=train_cfg.max_tokens,
        max_atoms=train_cfg.max_atoms,
        max_msa_seqs=train_cfg.max_msa_seqs,
        max_neighborhood=model_cfg.data.max_neighborhood,
        contiguous_crop_weight=model_cfg.data.contiguous_crop_weight,
        spatial_crop_weight=model_cfg.data.spatial_crop_weight,
        interface_crop_weight=model_cfg.data.interface_crop_weight,
        interface_distance=model_cfg.data.interface_distance,
    )
    if spec.format == "of3":
        records = discover_of3_records(data_dir)
        ds = OF3Dataset(
            records=records,
            of3_dir=data_dir,
            is_distillation=spec.is_distillation,
            name=spec.name,
            mmcif_dir=None,
            max_templates=model_cfg.data.max_templates,
            **common,
        )
    else:
        records = discover_records(data_dir)
        ds = TrainingDataset(
            records=records,
            data_dir=data_dir,
            is_distillation=spec.is_distillation,
            name=spec.name,
            mmcif_dir=None,
            max_templates=model_cfg.data.max_templates,
            template_cutoff_days=model_cfg.data.template_cutoff_days,
            max_template_date=model_cfg.data.max_template_date,
            prefer_monomer_templates=model_cfg.data.prefer_monomer_templates,
            max_chains=model_cfg.data.max_chains,
            **common,
        )
    return ds, records


def _find_idx(records, rid):
    for i, r in enumerate(records):
        if r.id == rid:
            return i
    for i, r in enumerate(records):
        if r.id.lower() == rid.lower():
            return i
    raise KeyError(f"record {rid!r} not found ({len(records)} records)")


def _print_input(features):
    print("\n========== INPUT FEATURES (floating-point only) ==========")
    bad = []
    for k, v in sorted(features.items()):
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        s = _tensor_stats(v)
        mark = "  <<< BAD" if s.get("has_nan") or s.get("has_inf") else ""
        if mark:
            bad.append(k)
        print(
            f"  {k:28s} shape={s['shape']!s:28s} {s['dtype']:10s} "
            f"abs_max={s['abs_max']:.3e} nan={s['has_nan']} inf={s['has_inf']}{mark}"
        )
    if bad:
        print(f"  !!! features already carry NaN/Inf in: {bad}")


def _print_output(pred):
    print("\n========== MODEL OUTPUT ==========")
    for k in sorted(pred.keys()):
        v = pred[k]
        if isinstance(v, torch.Tensor):
            s = _tensor_stats(v)
            print(
                f"  {k:24s} shape={s['shape']!s:28s} {s['dtype']:10s} "
                f"abs_max={s.get('abs_max', float('nan')):.3e} "
                f"nan={s.get('has_nan')} inf={s.get('has_inf')}"
            )
        elif isinstance(v, dict):
            hn, hi = _any_bad(v)
            print(f"  {k:24s} dict nan={hn} inf={hi}")


def _run_backward_anomaly(model, features, train_cfg, use_anomaly: bool = False):
    """Forward + backward with NaN/Inf detection on both activations and grads.

    Uses full-module hooks on forward (same as the no_grad pass) PLUS grad
    hooks on every tensor that requires grad. Cheaper than
    ``set_detect_anomaly`` (no per-op graph retention) so it fits on one
    B200 at realistic ``diffusion_batch_size``. ``use_anomaly=True`` falls
    back to the PyTorch anomaly detector for callers that want the stack
    trace — usually OOMs at production shapes, so opt-in only.
    """
    print("\n========== BACKWARD PASS (train mode) ==========")
    model.zero_grad(set_to_none=True)

    fwd = HookRecorder(model)
    fwd.install()

    # Grad hooks: every Linear / LayerNorm output tracks its backward NaN.
    bad_grads: list[tuple[str, str, int]] = []
    grad_hook_counter = {"n": 0}

    def grad_hook_factory(name: str, module_type: str):
        def hook(module, grad_input, grad_output):
            for gi in grad_output or ():
                if gi is None or not gi.is_floating_point():
                    continue
                g32 = gi.detach().to(torch.float32)
                if torch.isnan(g32).any() or torch.isinf(g32).any():
                    grad_hook_counter["n"] += 1
                    bad_grads.append((name, module_type, grad_hook_counter["n"]))
                    break

        return hook

    g_handles: list = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        g_handles.append(mod.register_full_backward_hook(grad_hook_factory(name, type(mod).__name__)))

    ctx = torch.autograd.set_detect_anomaly(True) if use_anomaly else nullcontext()
    try:
        with ctx:
            pred = model.forward_training(
                features,
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
            terms = []
            for _k, v in pred.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.requires_grad:
                    terms.append((v.float() ** 2).mean())
            if not terms:
                print("  no differentiable outputs")
                return
            loss = sum(terms) / len(terms)
            print(f"  surrogate loss = {loss.item():.3e}  (isfinite={torch.isfinite(loss).item()})")
            loss.backward()
    except Exception as e:  # noqa: BLE001
        print(f"  forward/backward raised: {type(e).__name__}: {e}")
    finally:
        fwd.remove()
        for h in g_handles:
            h.remove()

    print("\n----- forward NaN/Inf hit list (train mode) -----")
    fwd.report()

    print("\n----- backward NaN/Inf hit list -----")
    if not bad_grads:
        print("  (none) — no submodule emitted a NaN/Inf grad output")
    else:
        # Earliest-fired grad hook == lowest counter == LAST module in forward
        # order (because grad hooks fire in reverse). Report first 20.
        bad_grads.sort(key=lambda t: t[2])
        for name, mtype, order in bad_grads[:20]:
            print(f"  #{order:4d} {name:70s} ({mtype})")
        if len(bad_grads) > 20:
            print(f"  ... {len(bad_grads) - 20} more")

    bad_params = []
    n = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n += 1
        if not torch.isfinite(p.grad).all():
            bad_params.append((name, tuple(p.shape), float(p.grad.abs().max().item())))
    print("\n----- parameter grad scan -----")
    print(f"  params with grad: {n}")
    if bad_params:
        print(f"  params with NaN/Inf grad: {len(bad_params)}")
        for name, shape, amax in bad_params[:20]:
            print(f"    {name:70s} shape={shape!s:20s} abs_max={amax:.3e}")
    else:
        print("  all parameter grads finite")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="patchr:6nud")
    p.add_argument("--config", default=str(_REPO / "configs" / "initial.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-backward", action="store_true")
    p.add_argument(
        "--slim",
        action="store_true",
        help="Override to single diffusion sample / no recycle / fp32 for fresh-init sanity checks.",
    )
    p.add_argument(
        "--train-mode",
        action="store_true",
        help="Keep model in train() mode (dropout active) — reproduces the training forward path.",
    )
    p.add_argument(
        "--diffusion-bs",
        type=int,
        default=None,
        help="Override diffusion_batch_size (smaller values fit backward + anomaly on one B200).",
    )
    p.add_argument(
        "--max-recycles",
        type=int,
        default=None,
        help="Override max_recycles (set 0 to skip recycling during backward triage).",
    )
    p.add_argument("--log", default=None)
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint to load before forward")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

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

    ds_name, rid = _parse_record(a.record)
    print(f"[triage] dataset={ds_name} record={rid} device={a.device} seed={a.seed}")
    train_cfg, model_cfg = _build_cfgs(Path(a.config))
    if a.slim:
        train_cfg.diffusion_batch_size = 1
        train_cfg.diffusion_chunk_size = 0
        train_cfg.mini_rollout_samples = 1
        train_cfg.max_recycles = 0
        train_cfg.dtype = "fp32"
        train_cfg.mini_rollout_steps = 5
    if a.diffusion_bs is not None:
        train_cfg.diffusion_batch_size = a.diffusion_bs
        # Chunk size must evenly divide diffusion_batch_size; align them.
        train_cfg.diffusion_chunk_size = min(train_cfg.diffusion_chunk_size, a.diffusion_bs)
    if a.max_recycles is not None:
        train_cfg.max_recycles = a.max_recycles
    print(
        f"[triage] max_tokens={train_cfg.max_tokens} max_atoms={train_cfg.max_atoms} "
        f"dtype={train_cfg.dtype} diffusion_bs={train_cfg.diffusion_batch_size} "
        f"recycles={train_cfg.max_recycles} mini_rollout={train_cfg.mini_rollout_steps}"
    )
    ds, records = _build_dataset(ds_name, train_cfg, model_cfg)
    print(f"[triage] {len(records)} records in dataset")
    idx = _find_idx(records, rid)
    print(f"[triage] record idx={idx}")

    raw = ds._load_sample(idx)
    features = collate_fn([raw], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
    features = {k: v for k, v in features.items() if isinstance(v, torch.Tensor)}
    device = torch.device(a.device)
    features = {k: v.to(device) for k, v in features.items()}
    _print_input(features)

    print("[triage] building DeepFold3")
    model = DeepFold3(model_cfg).to(device)
    if a.checkpoint:
        print(f"[triage] loading checkpoint {a.checkpoint}")
        ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        if isinstance(state, dict) and "global_step" in ckpt:
            print(f"  checkpoint global_step={ckpt['global_step']} effective_step={ckpt.get('effective_step', '?')}")
        # DDP saves with "module." prefix — strip it.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  missing keys: {len(missing)} (first 5): {missing[:5]}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
    if a.train_mode:
        model.train()
        print("[triage] model.train() — dropout active")
    else:
        model.eval()
    print(f"[triage] params = {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print("\n========== FORWARD (no_grad, hooks) ==========")
    rec = HookRecorder(model)
    rec.install()
    try:
        with torch.no_grad():
            pred = model.forward_training(
                features,
                num_recycles=0,
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
    finally:
        rec.remove()
    _print_output(pred)
    rec.report()

    if not a.skip_backward:
        if not a.train_mode:
            model.train()
        _run_backward_anomaly(model, features, train_cfg)
    print("\n[triage] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
