"""Static analysis of a Schedule-Free training checkpoint.

Reports:
  * param-group norms (decay vs no-decay), top-K largest params
  * SF iterate divergence ||z - <snapshot>|| / ||<snapshot>|| per param, by module class
  * Shadow vs raw-x divergence ||shadow - x|| / ||x||  (when a shadow is present)
  * Adam v (exp_avg_sq) statistics — implied step-size scale 1/sqrt(v)
  * Trunk per-layer growth (PairFormer single transition output, OPM out_w, tri-mul out)

Vocabulary
----------
Two distinct averaging mechanisms can coexist in one checkpoint and the script
keeps them strictly separated:

  - ``x`` ("average")  → SF's averaged iterate. Defined inside the SF update rule
                         as ``x_t = c_t·z_t + (1-c_t)·x_{t-1}``. The paper's
                         inference target. ``ck["model"]`` holds raw ``x``
                         (``CheckpointManager.save`` wraps the dump in
                         ``optimizer_eval_mode``, no shadow overlay). Snapshot
                         kind ``"x"`` for SF, ``"live"`` for plain AdamW.
  - ``shadow``         → an independent EMA over ``x``, maintained externally to
                         SF: ``shadow_t = decay·shadow_{t-1} + (1-decay)·x_t``.
                         Lives in ``ck["ema"]["shadow"]``. Decoupled time
                         constant (~1/(1-decay) step). At decay=0.999 this is
                         ~1000-step smoothing layered *on top of* SF's own
                         averaging window — the two are orthogonal.
  - ``z`` ("fast")     → SF's fast iterate. The SGD-like point that takes the
                         raw step ``z_{t+1} = z_t - γ·Δ_t`` and that ``x``
                         averages over. Lives in ``ck["optimizer"]["state"][pid]["z"]``.
                         SF-only. Tracks the loss landscape directly and
                         oscillates around the river that ``x`` smooths.

Use ``--snapshot {average,shadow,fast}`` to pick which one populates the
per-param weight tensors used in every section below. The SF-divergence label
updates to match (e.g. ``||z - shadow||/||shadow||`` when ``--snapshot shadow``).
The "Shadow vs raw-x" section is independent of ``--snapshot`` and always
compares ``shadow`` against the raw-``x`` baseline when ``ck["ema"]["shadow"]``
is present — it measures how much smoothing the shadow adds *beyond* SF's own
averaging.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch


def _bucket(name: str) -> str:
    if name.startswith("evo_cond_encoder"):
        return "atom_encoder"
    if name.startswith("evoformer.template"):
        return "template"
    if name.startswith("evoformer.msa_stack"):
        return "msa"
    if name.startswith("evoformer.trunk") or name.startswith("evoformer.pf"):
        return "pairformer"
    if name.startswith("evoformer."):
        return "evoformer_other"
    if name.startswith("diffusion"):
        return "diffusion"
    if name.startswith("confidence"):
        return "confidence"
    if name.startswith("distogram"):
        return "distogram"
    return "other"


_LAYER_RE = re.compile(r"\.(\d+)\.")


def _layer_idx(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _select_snapshot(
    ck: dict,
    choice: str,
    pid_of: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Resolve which weight tensor to feed into every analysis section.

    The three SF weight sources are layered (raw fast → SF-averaged → externally
    smoothed), and each tells a different story about training dynamics:

        average  →  ``ck["model"]`` directly. Raw ``x`` for SF (paper inference
                    target), live params for AdamW. Default — preserves existing
                    behaviour.

        shadow   →  ``ck["ema"]["shadow"]`` overlaid on ``ck["model"]``. The
                    shadow only covers ``model.parameters()`` (not buffers like
                    LN ``running_mean``); non-shadowed entries fall through to
                    the raw-``x`` value, which is fine for analysis since buffers
                    don't carry the shadow. Errors if no shadow exists.

        fast     →  Substitute SF's ``z`` from ``ck["optimizer"]["state"][pid]["z"]``
                    for every SF-tracked param. Errors if no SF state exists.
                    Note that ``||z - z||`` is trivially zero, so the SF-divergence
                    section is skipped under this choice.

    Param shapes are identical across all three sources, so name → pid mapping
    (built once from ``ck["model"]``) and the no-decay heuristic both apply
    unchanged.
    """
    raw = dict(ck["model"])
    if choice == "average":
        return raw

    if choice == "shadow":
        ema_block = ck.get("ema") or {}
        shadow = ema_block.get("shadow") or {}
        if not shadow:
            raise SystemExit(
                "--snapshot shadow: checkpoint has no 'ema'/'shadow' entry "
                "(run was launched without ema_decay > 0, or the EMA "
                "callback was disabled)."
            )
        for name, t in shadow.items():
            if name in raw:
                raw[name] = t
        return raw

    if choice == "fast":
        opt = ck.get("optimizer") or {}
        state = opt.get("state") or {}
        if not any(isinstance(s, dict) and "z" in s for s in state.values()):
            raise SystemExit(
                "--snapshot fast: no SF 'z' state in checkpoint (this is a plain AdamW checkpoint; 'fast' is SF-only)."
            )
        substituted = 0
        for name in raw:
            pid = pid_of.get(name)
            if pid is not None:
                z = state.get(pid, {}).get("z")
                if z is not None and z.shape == raw[name].shape:
                    raw[name] = z
                    substituted += 1
        if substituted == 0:
            raise SystemExit(
                "--snapshot fast: pid mapping recovered 0 params with z state; "
                "likely a name-ordering mismatch between the saved optimizer "
                "and the model state_dict."
            )
        return raw

    raise ValueError(f"unknown snapshot choice: {choice!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Static analysis of a Schedule-Free training checkpoint.")
    p.add_argument("checkpoint")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument(
        "--snapshot",
        choices=["average", "shadow", "fast"],
        default="average",
        help=(
            "Which weight snapshot to analyse. 'average' = ck['model'] directly "
            "(SF's averaged iterate x for SF runs, live params for AdamW; this is "
            "the inference-recommended snapshot, default). 'shadow' = "
            "ck['ema']['shadow'] overlaid on ck['model'] (an independent EMA over "
            "x, external to SF; a separate smoothing layer with its own time "
            "constant). 'fast' = SF's z buffer from ck['optimizer']['state'] "
            "(the SGD-like fast iterate that x averages over; SF-only). The SF "
            "divergence section's denominator updates to match the choice; the "
            "shadow-vs-x section is always shown when a shadow is present."
        ),
    )
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_model = ck["model"]  # used for pid mapping + EMA-vs-x baseline
    opt = ck["optimizer"]
    step = ck["global_step"]
    best_eval = ck.get("best_eval_loss", float("nan"))
    saved_kind = ck.get("snapshot_kind", "?")

    print(f"=== {Path(args.checkpoint).name} ===")
    print(f"step={step}  best_eval_loss={best_eval:.4f}  epoch={ck.get('epoch')}  saved_snapshot_kind={saved_kind!r}")
    pg0, pg1 = opt["param_groups"]
    print(
        f"sf weight_sum={pg0.get('weight_sum'):.4g}  scheduled_lr={pg0.get('scheduled_lr'):.2e}  k={pg0.get('k')}  warmup={pg0.get('warmup_steps')}"
    )
    print(f"sf train_mode={pg0.get('train_mode')} (False ⇒ model holds x; optimizer 'z' is the SGD-like iterate)")
    print(
        f"groups: decay={len(pg0['params'])} params (wd={pg0['weight_decay']})  no-decay={len(pg1['params'])} params (wd={pg1['weight_decay']})"
    )

    # Build pid → (name, in_decay)
    decay_pids = set(pg0["params"])
    nodecay_pids = set(pg1["params"])
    state = opt["state"]

    # The optimizer params list maps param-position → pid; we need name → pid via insertion order.
    # Trainer's build_optimizer feeds the decay group first, then no-decay; both in named_parameters order.
    # Recover by walking model keys and assigning the next available pid in the matching group.
    decay_iter = iter(sorted(decay_pids))
    nodecay_iter = iter(sorted(nodecay_pids))
    pid_of: dict[str, int] = {}
    # Walk in the same order trainer would have walked named_parameters.
    # Without the live model we can't perfectly reproduce it. Use a substring heuristic on the
    # no-decay rule from optim.py:_NO_DECAY_NAME_SUBSTRINGS.
    NO_DECAY_TOKENS = (
        "bias",
        "norm",
        ".gate.",
        "to_g",
        "out_gate",
        "cond_scale",
        "cond_bias",
        "zero_cond",
        "single_activations",
        "msa_activations",
        "embed_",
        "left_single",
        "right_single",
        "position_activations",
        "bond_embedding",
        "prev_",
    )

    def is_no_decay(name: str, p: torch.Tensor) -> bool:
        if p.dim() < 2:
            return True
        if p.dim() == 2 and min(p.shape) == 1:
            return True
        return any(tok in name for tok in NO_DECAY_TOKENS)

    for name, w in raw_model.items():
        if not isinstance(w, torch.Tensor):
            continue
        if is_no_decay(name, w):
            try:
                pid_of[name] = next(nodecay_iter)
            except StopIteration:
                pass
        else:
            try:
                pid_of[name] = next(decay_iter)
            except StopIteration:
                pass

    # ---------- Snapshot selection ----------
    # Build pid_of from raw_model first (above) so 'fast' substitution can find
    # z by name → pid → state[pid]['z']. After this point, ``model`` may be
    # raw x (default), the EMA shadow overlay, or SF's z, depending on --snapshot.
    model = _select_snapshot(ck, args.snapshot, pid_of)
    print(f"snapshot=<{args.snapshot}>  ({len(model)} tensors)")

    # ---------- Group norms ----------
    print()
    print("=== param-norm by bucket ===")
    bucket_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "sumsq": 0.0, "n_params": 0, "max": 0.0})
    for name, w in model.items():
        if not isinstance(w, torch.Tensor) or not torch.is_floating_point(w):
            continue
        b = _bucket(name)
        s = bucket_stats[b]
        s["n"] += w.numel()
        s["n_params"] += 1
        sq = float((w.float() ** 2).sum())
        s["sumsq"] += sq
        nrm = sq**0.5
        if nrm > s["max"]:
            s["max"] = nrm
    print(f"{'bucket':<18} {'n_params':>8} {'numel':>12} {'rms(W)':>10} {'max ||W||':>12}")
    for b in sorted(bucket_stats, key=lambda b: -bucket_stats[b]["sumsq"]):
        s = bucket_stats[b]
        rms = (s["sumsq"] / max(s["n"], 1)) ** 0.5
        print(f"{b:<18} {s['n_params']:>8d} {int(s['n']):>12d} {rms:>10.4f} {s['max']:>12.3f}")

    # ---------- z vs <snapshot> divergence ----------
    # The label below tracks --snapshot: 'x' for average, 'shadow' for shadow.
    # With 'fast' the snapshot IS z, so the comparison degenerates to ||z-z|| = 0
    # and the section is skipped. Whatever the snapshot, z is always the SF fast
    # iterate from optimizer state — the gap measures how far the saved
    # inference target lags behind the point that's actually moving. The
    # primary SF dynamics question (fast vs averaged) is answered by
    # --snapshot average; --snapshot shadow lets you see whether the external
    # smoothing layer changes the picture.
    if args.snapshot == "fast":
        print()
        print("=== SF iterate divergence — skipped ===")
        print("  (snapshot=fast: comparing z to itself is ||z-z||/||z|| = 0; rerun")
        print("   with --snapshot average or --snapshot shadow for a meaningful gap.)")
    else:
        snap_label = {"average": "x", "shadow": "shadow"}[args.snapshot]
        print()
        print(f"=== SF iterate divergence  ||z - {snap_label}||/||{snap_label}||  (top-K) ===")
        div_rows: list[tuple[str, float, float, float, float]] = []  # (name, ||snap||, ||z||, ||z-snap||, ratio)
        for name, w in model.items():
            if not isinstance(w, torch.Tensor) or not torch.is_floating_point(w):
                continue
            pid = pid_of.get(name)
            if pid is None:
                continue
            st = state.get(pid, {})
            z = st.get("z")
            if z is None:
                continue
            x = w.float()
            z = z.float()
            if z.shape != x.shape:
                continue
            nx = float(x.norm())
            nz = float(z.norm())
            nd = float((z - x).norm())
            ratio = nd / max(nx, 1e-12)
            div_rows.append((name, nx, nz, nd, ratio))

        if not div_rows:
            print("(no z state matched any param — pid mapping failed)")
        else:
            print(f"matched {len(div_rows)} params with optimizer z state")
            ratios = [r[4] for r in div_rows]
            ratios_sorted = sorted(ratios)
            n = len(ratios_sorted)
            print(
                f"  ||z-{snap_label}||/||{snap_label}||  p10={ratios_sorted[n // 10]:.3f}  p50={ratios_sorted[n // 2]:.3f}  p90={ratios_sorted[9 * n // 10]:.3f}  p99={ratios_sorted[99 * n // 100]:.3f}  max={ratios_sorted[-1]:.3f}"
            )
            print()
            print(f"  by bucket: median ||z-{snap_label}||/||{snap_label}||")
            per_bucket: dict[str, list[float]] = defaultdict(list)
            for name, _nx, _nz, _nd, r in div_rows:
                per_bucket[_bucket(name)].append(r)
            for b in sorted(per_bucket, key=lambda b: -sorted(per_bucket[b])[len(per_bucket[b]) // 2]):
                lst = sorted(per_bucket[b])
                n = len(lst)
                print(f"    {b:<18} n={n:>4}  p50={lst[n // 2]:.3f}  p90={lst[9 * n // 10]:.3f}  max={lst[-1]:.3f}")
            print()
            print(f"  top-{args.top_k} by ratio:")
            div_rows.sort(key=lambda r: -r[4])
            for name, nx, nz, nd, r in div_rows[: args.top_k]:
                print(
                    f"    {r:8.3f}  ||{snap_label}||={nx:7.3f}  ||z||={nz:7.3f}  ||z-{snap_label}||={nd:7.3f}  {name}"
                )

    # ---------- Shadow vs raw-x divergence (snapshot-independent) ----------
    # SF averages over a window controlled by c_t (the x mechanism, internal
    # to SF); the shadow averages over a separate geometric window with time
    # constant ~1/(1-decay) (external EMA, decoupled from SF). When both
    # exist, this section measures the *additional* smoothing the shadow
    # provides on top of x. If ||shadow-x||/||x|| ≪ ||z-x||/||x|| → the
    # shadow is mostly redundant (x is already smoother than z by a wider
    # margin than the shadow adds). If they're comparable → the shadow is
    # doing complementary work, dampening x's residual oscillation.
    ema_block = ck.get("ema") or {}
    shadow = ema_block.get("shadow") or {}
    if shadow:
        print()
        print("=== Shadow vs raw-x divergence  ||shadow - x||/||x||  (top-K) ===")
        ema_decay = ema_block.get("decay")
        if ema_decay is not None:
            print(
                f"  shadow_decay={ema_decay}  (geometric time constant ≈ {1.0 / max(1.0 - ema_decay, 1e-12):.0f} step)"
            )
        shadow_div: list[tuple[str, float, float, float, float]] = []
        for name, x_t in raw_model.items():
            if name not in shadow:
                continue
            if not isinstance(x_t, torch.Tensor) or not torch.is_floating_point(x_t):
                continue
            s_t = shadow[name]
            if not isinstance(s_t, torch.Tensor) or s_t.shape != x_t.shape:
                continue
            x = x_t.float()
            s = s_t.float()
            nx = float(x.norm())
            ns = float(s.norm())
            nd = float((s - x).norm())
            ratio = nd / max(nx, 1e-12)
            shadow_div.append((name, nx, ns, nd, ratio))

        if not shadow_div:
            print("  (no shadow keys overlapped with model state_dict — should not happen)")
        else:
            ratios = sorted(r[4] for r in shadow_div)
            n = len(ratios)
            print(f"  matched {n} shadow params")
            print(
                f"  ||shadow-x||/||x||  p10={ratios[n // 10]:.3f}  p50={ratios[n // 2]:.3f}  p90={ratios[9 * n // 10]:.3f}  p99={ratios[99 * n // 100]:.3f}  max={ratios[-1]:.3f}"
            )
            print()
            print("  by bucket: median ||shadow-x||/||x||")
            per_b: dict[str, list[float]] = defaultdict(list)
            for name, _nx, _ns, _nd, r in shadow_div:
                per_b[_bucket(name)].append(r)
            for b in sorted(per_b, key=lambda b: -sorted(per_b[b])[len(per_b[b]) // 2]):
                lst = sorted(per_b[b])
                n = len(lst)
                print(f"    {b:<18} n={n:>4}  p50={lst[n // 2]:.3f}  p90={lst[9 * n // 10]:.3f}  max={lst[-1]:.3f}")
            print()
            print(f"  top-{args.top_k} by ratio:")
            shadow_div.sort(key=lambda r: -r[4])
            for name, nx, ns, nd, r in shadow_div[: args.top_k]:
                print(f"    {r:8.3f}  ||x||={nx:7.3f}  ||shadow||={ns:7.3f}  ||shadow-x||={nd:7.3f}  {name}")

    # ---------- Adam v statistics ----------
    print()
    print("=== Adam exp_avg_sq (v) — implied step scale 1/sqrt(v+eps) ===")
    eps = pg0.get("eps", 1e-8)
    v_rows: list[tuple[str, float, float]] = []  # (name, sqrt(mean(v)), 1/sqrt(mean(v)+eps))
    for name, w in model.items():
        if not isinstance(w, torch.Tensor) or not torch.is_floating_point(w):
            continue
        pid = pid_of.get(name)
        if pid is None:
            continue
        v = state.get(pid, {}).get("exp_avg_sq")
        if v is None:
            continue
        mv = float(v.float().mean())
        v_rows.append((name, mv**0.5, 1.0 / (mv**0.5 + eps)))
    if v_rows:
        scales = sorted(r[2] for r in v_rows)
        n = len(scales)
        print(
            f"  n={n}  1/sqrt(v):  p10={scales[n // 10]:.4g}  p50={scales[n // 2]:.4g}  p90={scales[9 * n // 10]:.4g}  max={scales[-1]:.4g}"
        )
        print("  (lr×1/sqrt(v) gives the per-element update magnitude before SF averaging)")

    # ---------- PairFormer per-layer growth ----------
    print()
    print("=== PairFormer per-layer key tensor norms ===")
    targets = {
        "single_transition.linear2.weight": "single_transition_out",
        "tri_mul_out.weight": "tri_mul_out",
        "tri_mul_in.weight": "tri_mul_in",
        "single_attention.q_projection.weight": "qkv_q",
    }
    layer_norms: dict[str, dict[int, float]] = defaultdict(dict)
    for name, w in model.items():
        if "trunk" not in name and "evoformer.pf" not in name:
            continue
        for suffix, label in targets.items():
            if name.endswith(suffix):
                li = _layer_idx(name)
                if li is None:
                    continue
                layer_norms[label][li] = float(w.float().norm())
                break
    for label, m in sorted(layer_norms.items()):
        if not m:
            continue
        idxs = sorted(m)
        head = ", ".join(f"{i}:{m[i]:.2f}" for i in idxs[:6])
        tail = ", ".join(f"{i}:{m[i]:.2f}" for i in idxs[-6:])
        first = m[idxs[0]]
        last = m[idxs[-1]]
        print(
            f"  {label:<22} layers {idxs[0]}..{idxs[-1]}  first={first:.3f}  last={last:.3f}  ratio={last / max(first, 1e-9):.2f}×"
        )
        print(f"    first 6: {head}")
        print(f"    last  6: {tail}")

    # ---------- top-K largest weight norms overall ----------
    print()
    print(f"=== top-{args.top_k} largest ||W|| ===")
    rows = [
        (name, float(w.float().norm()))
        for name, w in model.items()
        if isinstance(w, torch.Tensor) and torch.is_floating_point(w)
    ]
    rows.sort(key=lambda r: -r[1])
    for name, nrm in rows[: args.top_k]:
        print(f"  {nrm:9.3f}  {name}")


if __name__ == "__main__":
    main()
