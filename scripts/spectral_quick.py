"""Quick spectral analysis on a SF training checkpoint.

For each 2D matmul weight (decay group) under the trunk + diffusion + heads,
compute:
  - shape, ‖W‖_F (energy)
  - top-k singular values (via torch.linalg.svdvals on float32 W)
  - stable rank ‖W‖_F² / σ_max²
  - effective rank exp(H(σ_normalized))  (entropy-based)
  - condition number σ_max / σ_min
  - dead heuristic: σ_max < 1e-3

Aggregates per-PF-layer + per-bucket. Emits two plots:
  - pf_spectral_curve.png : effective rank / stable rank per PF block, per submodule
  - top_outliers.png      : top-K weights by smallest effective rank (LoRA candidates)

With ``--vs <baseline.pt>`` emits a diff section: per-bucket Δ, PF per-layer Δ,
PF per-submod mean Δ, and top movers by |Δ eff_rank| / |Δ RMS|. Both
checkpoints are analyzed in one invocation (no cache, ~30s × 2 with CUDA).

Static analysis — single forward, no GPU needed. ~1-2 min on CPU for full trunk.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_LAYER_RE = re.compile(r"\.(\d+)\.")


def _layer_idx(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _submod_label(name: str) -> str:
    """Coarse submodule label for legend grouping."""
    for kw in (
        "tri_mul_out",
        "tri_mul_in",
        "pair_attn_row",
        "pair_attn_col",
        "pair_transition",
        "single_to_pair",
        "single_attn",
        "single_transition",
    ):
        if kw in name:
            return kw
    return "other"


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


def spectral_metrics(W: torch.Tensor, device: str = "cpu") -> dict:
    """Return spectral metrics for a 2D matmul weight."""
    W = W.detach().float()
    if device != "cpu":
        W = W.to(device)
    s = torch.linalg.svdvals(W)  # descending
    s2 = (s * s).sum().item()
    s_max = float(s[0])
    s_min = float(s[-1])
    fro2 = float((W * W).sum())
    # entropy-based effective rank: exp(H), H = -Σ p log p, p = σᵢ² / Σσⱼ²
    p = (s * s) / s2
    p = p.clamp_min(1e-20)
    H = -(p * p.log()).sum().item()
    eff_rank = float(math.exp(H))
    stable_rank = fro2 / max(s_max * s_max, 1e-12)
    cond = s_max / max(s_min, 1e-12)
    return {
        "shape": tuple(W.shape),
        "fro": math.sqrt(fro2),
        "rms": math.sqrt(fro2 / W.numel()),
        "s_max": s_max,
        "s_min": s_min,
        "stable_rank": stable_rank,
        "eff_rank": eff_rank,
        "cond": cond,
        "dead": s_max < 1e-3,
    }


def _load_state(path: str) -> tuple[dict, str]:
    """Return (state_dict, step_label)."""
    suffix = Path(path).suffix
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path), "(safetensors)"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"], str(ck.get("global_step", "?"))


def weight_delta_metrics(W_cur: torch.Tensor, W_base: torch.Tensor, device: str = "cpu") -> dict:
    """Direct ΔW measurements (catches rotation that spectral misses).

    - rel_dW = ‖W_cur - W_base‖_F / ‖W_base‖_F  (relative step magnitude, scale-invariant)
    - cos_sim = ⟨W_cur, W_base⟩_F / (‖W_cur‖·‖W_base‖)  (alignment, 1 = no rotation)
    - rotation_angle = arccos(cos_sim)  (in radians; full direction change)
    """
    a = W_cur.detach().float()
    b = W_base.detach().float()
    if device != "cpu":
        a = a.to(device)
        b = b.to(device)
    diff = a - b
    fro_diff = float(torch.linalg.norm(diff))
    fro_base = float(torch.linalg.norm(b))
    fro_cur = float(torch.linalg.norm(a))
    rel_dW = fro_diff / max(fro_base, 1e-12)
    cos_sim = float((a * b).sum()) / max(fro_cur * fro_base, 1e-12)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle = math.acos(cos_sim)
    return {
        "fro_diff": fro_diff,
        "rel_dW": rel_dW,
        "cos_sim": cos_sim,
        "angle_rad": angle,
    }


def analyze(model_state: dict, device: str, skip_buckets: set[str]) -> list[dict]:
    rows: list[dict] = []
    for name, w in model_state.items():
        if not isinstance(w, torch.Tensor) or not torch.is_floating_point(w):
            continue
        if w.dim() != 2 or min(w.shape) < 2:
            continue
        bucket = _bucket(name)
        if bucket in skip_buckets:
            continue
        m = spectral_metrics(w, device=device)
        m["name"] = name
        m["bucket"] = bucket
        m["layer"] = _layer_idx(name)
        m["submod"] = _submod_label(name)
        rows.append(m)
    return rows


def emit_bucket_summary(rows: list[dict]) -> dict[str, dict]:
    """Print bucket summary; return per-bucket p50 dict for diff use."""
    print("\n=== Per-bucket summary ===")
    print(
        f"  {'bucket':<18} {'n':>4} {'rms p50':>10} {'stable_rank p50':>16} "
        f"{'eff_rank p50':>14} {'cond p50':>10} {'dead':>5}"
    )
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    p50: dict[str, dict] = {}
    for b in sorted(by_bucket, key=lambda b: -len(by_bucket[b])):
        bs = by_bucket[b]
        rms = sorted(r["rms"] for r in bs)
        sr = sorted(r["stable_rank"] for r in bs)
        er = sorted(r["eff_rank"] for r in bs)
        cd = sorted(r["cond"] for r in bs)
        dd = sum(1 for r in bs if r["dead"])
        n = len(bs)
        p50[b] = {
            "n": n,
            "rms_p50": rms[n // 2],
            "stable_rank_p50": sr[n // 2],
            "eff_rank_p50": er[n // 2],
            "cond_p50": cd[n // 2],
            "dead": dd,
        }
        print(
            f"  {b:<18} {n:>4} {rms[n // 2]:>10.4f} {sr[n // 2]:>16.2f} {er[n // 2]:>14.2f} {cd[n // 2]:>10.2f} {dd:>5}"
        )
    return p50


def emit_pf_per_layer(rows: list[dict]) -> dict[int, dict]:
    pf_rows = [r for r in rows if r["bucket"] == "pairformer" and r["layer"] is not None]
    if not pf_rows:
        return {}
    print("\n=== Per PF-layer effective rank (avg over submodules) ===")
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for r in pf_rows:
        by_layer[r["layer"]].append(r)
    layers = sorted(by_layer.keys())
    print(f"  {'layer':>5}  {'n':>3}  {'eff_rank avg':>14}  {'stable_rank avg':>16}  {'rms avg':>10}  {'dead':>5}")
    per_layer: dict[int, dict] = {}
    for L in layers:
        bs = by_layer[L]
        ers = [r["eff_rank"] for r in bs]
        srs = [r["stable_rank"] for r in bs]
        rmss = [r["rms"] for r in bs]
        dd = sum(1 for r in bs if r["dead"])
        per_layer[L] = {
            "n": len(bs),
            "eff_rank": sum(ers) / len(ers),
            "stable_rank": sum(srs) / len(srs),
            "rms": sum(rmss) / len(rmss),
            "dead": dd,
        }
        print(
            f"  {L:>5}  {len(bs):>3}  {per_layer[L]['eff_rank']:>14.2f}  "
            f"{per_layer[L]['stable_rank']:>16.2f}  {per_layer[L]['rms']:>10.4f}  {dd:>5}"
        )
    return per_layer


def emit_pf_per_submod(rows: list[dict]) -> dict[str, dict]:
    pf_rows = [r for r in rows if r["bucket"] == "pairformer"]
    if not pf_rows:
        return {}
    print("\n=== Per PF-submodule averages (across layers) ===")
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in pf_rows:
        by_sub[r["submod"]].append(r)
    print(f"  {'submod':<22} {'n':>4} {'eff_rank avg':>14} {'stable_rank avg':>16} {'rms avg':>10}")
    per_sub: dict[str, dict] = {}
    for sub in sorted(by_sub):
        bs = by_sub[sub]
        ers = [r["eff_rank"] for r in bs]
        srs = [r["stable_rank"] for r in bs]
        rmss = [r["rms"] for r in bs]
        per_sub[sub] = {
            "n": len(bs),
            "eff_rank": sum(ers) / len(ers),
            "stable_rank": sum(srs) / len(srs),
            "rms": sum(rmss) / len(rmss),
        }
        print(
            f"  {sub:<22} {len(bs):>4} {per_sub[sub]['eff_rank']:>14.2f} "
            f"{per_sub[sub]['stable_rank']:>16.2f} {per_sub[sub]['rms']:>10.4f}"
        )
    return per_sub


def emit_pf_layer_submod_matrix(rows: list[dict]) -> None:
    """PF eff_rank as (layer × submod) matrix."""
    pf_rows = [r for r in rows if r["bucket"] == "pairformer" and r["layer"] is not None]
    if not pf_rows:
        return
    submods = sorted({r["submod"] for r in pf_rows})
    by_ls: dict[tuple[int, str], dict] = {}
    for r in pf_rows:
        by_ls[(r["layer"], r["submod"])] = r
    layers = sorted({r["layer"] for r in pf_rows})
    print("\n=== PF (layer × submod) effective rank ===")
    header = f"  {'layer':>5}"
    for s in submods:
        header += f"  {s[:14]:>14}"
    print(header)
    for L in layers:
        line = f"  {L:>5}"
        for s in submods:
            r = by_ls.get((L, s))
            line += f"  {r['eff_rank']:>14.2f}" if r is not None else f"  {'-':>14}"
        print(line)


def emit_diff(
    cur_rows: list[dict],
    base_rows: list[dict],
    cur_state: dict,
    base_state: dict,
    cur_step: str,
    base_step: str,
    device: str,
    top_n: int = 20,
) -> None:
    cur_by_name = {r["name"]: r for r in cur_rows}
    base_by_name = {r["name"]: r for r in base_rows}
    common = sorted(set(cur_by_name) & set(base_by_name))
    only_cur = sorted(set(cur_by_name) - set(base_by_name))
    only_base = sorted(set(base_by_name) - set(cur_by_name))

    print(f"\n\n=== Diff: baseline (step={base_step}) → current (step={cur_step}) ===")
    print(f"  common weights : {len(common)}")
    if only_cur:
        print(f"  only in current: {len(only_cur)}")
        for n in only_cur[:10]:
            print(f"    + {n}")
    if only_base:
        print(f"  only in baseline: {len(only_base)}")
        for n in only_base[:10]:
            print(f"    - {n}")

    # Build pairwise rows + per-weight ΔW (direct distance, catches rotation)
    pairs: list[dict] = []
    for n in common:
        c, b = cur_by_name[n], base_by_name[n]
        dw = weight_delta_metrics(cur_state[n], base_state[n], device=device)
        pairs.append(
            {
                "name": n,
                "bucket": c["bucket"],
                "layer": c["layer"],
                "submod": c["submod"],
                "shape": c["shape"],
                "eff_rank_b": b["eff_rank"],
                "eff_rank_c": c["eff_rank"],
                "d_eff": c["eff_rank"] - b["eff_rank"],
                "rms_b": b["rms"],
                "rms_c": c["rms"],
                "d_rms": c["rms"] - b["rms"],
                "rel_d_rms": (c["rms"] - b["rms"]) / max(b["rms"], 1e-12),
                "sr_b": b["stable_rank"],
                "sr_c": c["stable_rank"],
                "d_sr": c["stable_rank"] - b["stable_rank"],
                "fro_b": b["fro"],
                "fro_c": c["fro"],
                "d_fro": c["fro"] - b["fro"],
                "rel_d_fro": (c["fro"] - b["fro"]) / max(b["fro"], 1e-12),
                "rel_dW": dw["rel_dW"],
                "cos_sim": dw["cos_sim"],
                "angle_rad": dw["angle_rad"],
                "angle_deg": math.degrees(dw["angle_rad"]),
            }
        )

    # --- Direct ΔW (catches rotation/direction change that spectral misses) ---
    print("\n=== Direct ΔW: bucket-level p50 / mean ===")
    print(
        f"  {'bucket':<18} {'n':>5} {'rel_dW p50':>12} {'rel_dW mean':>12} "
        f"{'cos_sim p50':>12} {'angle_deg p50':>14} {'angle_deg max':>14}"
    )
    by_b: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_b[p["bucket"]].append(p)
    for b in sorted(by_b, key=lambda b: -len(by_b[b])):
        bs = by_b[b]
        rd = sorted(p["rel_dW"] for p in bs)
        cs = sorted(p["cos_sim"] for p in bs)
        ang = sorted(p["angle_deg"] for p in bs)
        n = len(bs)
        print(
            f"  {b:<18} {n:>5} {rd[n // 2]:>12.4%} {sum(rd) / n:>12.4%} "
            f"{cs[n // 2]:>12.5f} {ang[n // 2]:>14.4f} {ang[-1]:>14.4f}"
        )

    # PF submod-level ΔW
    pf_pairs = [p for p in pairs if p["bucket"] == "pairformer"]
    if pf_pairs:
        print("\n=== Direct ΔW: PF per-submodule mean (across 48 layers) ===")
        print(
            f"  {'submod':<22} {'n':>4} {'rel_dW mean':>12} {'rel_dW max':>12} "
            f"{'cos_sim mean':>13} {'angle_deg mean':>15} {'angle_deg max':>15}"
        )
        by_s: dict[str, list[dict]] = defaultdict(list)
        for p in pf_pairs:
            by_s[p["submod"]].append(p)
        for s in sorted(by_s):
            bs = by_s[s]
            n = len(bs)
            rd_mean = sum(p["rel_dW"] for p in bs) / n
            rd_max = max(p["rel_dW"] for p in bs)
            cs_mean = sum(p["cos_sim"] for p in bs) / n
            ang_mean = sum(p["angle_deg"] for p in bs) / n
            ang_max = max(p["angle_deg"] for p in bs)
            print(
                f"  {s:<22} {n:>4} {rd_mean:>12.4%} {rd_max:>12.4%} {cs_mean:>13.5f} {ang_mean:>15.4f} {ang_max:>15.4f}"
            )

    # --- Bucket-level Δ ---
    print("\n=== Bucket-level Δ p50 ===")
    print(f"  {'bucket':<18} {'eff_rank':>22}  {'rms':>22}  {'stable_rank':>22}  {'fro':>22}")
    print(
        f"  {'':<18} {'base → cur (Δ)':>22}  {'base → cur (Δ %)':>22}  {'base → cur (Δ)':>22}  {'base → cur (Δ %)':>22}"
    )

    def _p50(items: list[dict], key: str) -> float:
        v = sorted(it[key] for it in items)
        return v[len(v) // 2]

    for b in sorted(by_b, key=lambda b: -len(by_b[b])):
        bs = by_b[b]
        er_b = _p50(bs, "eff_rank_b")
        er_c = _p50(bs, "eff_rank_c")
        rms_b = _p50(bs, "rms_b")
        rms_c = _p50(bs, "rms_c")
        sr_b = _p50(bs, "sr_b")
        sr_c = _p50(bs, "sr_c")
        fro_b = _p50(bs, "fro_b")
        fro_c = _p50(bs, "fro_c")
        print(
            f"  {b:<18} "
            f"{er_b:>7.2f}→{er_c:>7.2f} ({er_c - er_b:+6.2f})  "
            f"{rms_b:>7.4f}→{rms_c:>7.4f} ({100 * (rms_c - rms_b) / max(rms_b, 1e-12):+6.2f}%)  "
            f"{sr_b:>7.2f}→{sr_c:>7.2f} ({sr_c - sr_b:+6.2f})  "
            f"{fro_b:>7.2f}→{fro_c:>7.2f} ({100 * (fro_c - fro_b) / max(fro_b, 1e-12):+6.2f}%)"
        )

    # --- PF per-layer Δ ---
    pf = [p for p in pairs if p["bucket"] == "pairformer" and p["layer"] is not None]
    if pf:
        print("\n=== PF per-layer Δ (averaged across 30 weights per layer) ===")
        print(
            f"  {'layer':>5}  {'eff_rank b→c (Δ)':>22}  {'rms b→c (Δ %)':>22}  "
            f"{'stable b→c (Δ)':>22}  {'fro b→c (Δ %)':>22}"
        )
        by_L: dict[int, list[dict]] = defaultdict(list)
        for p in pf:
            by_L[p["layer"]].append(p)
        for L in sorted(by_L):
            bs = by_L[L]
            er_b = sum(p["eff_rank_b"] for p in bs) / len(bs)
            er_c = sum(p["eff_rank_c"] for p in bs) / len(bs)
            rms_b = sum(p["rms_b"] for p in bs) / len(bs)
            rms_c = sum(p["rms_c"] for p in bs) / len(bs)
            sr_b = sum(p["sr_b"] for p in bs) / len(bs)
            sr_c = sum(p["sr_c"] for p in bs) / len(bs)
            fro_b = sum(p["fro_b"] for p in bs) / len(bs)
            fro_c = sum(p["fro_c"] for p in bs) / len(bs)
            print(
                f"  {L:>5}  "
                f"{er_b:>7.2f}→{er_c:>7.2f} ({er_c - er_b:+6.2f})  "
                f"{rms_b:>7.4f}→{rms_c:>7.4f} ({100 * (rms_c - rms_b) / max(rms_b, 1e-12):+6.2f}%)  "
                f"{sr_b:>7.2f}→{sr_c:>7.2f} ({sr_c - sr_b:+6.2f})  "
                f"{fro_b:>7.2f}→{fro_c:>7.2f} ({100 * (fro_c - fro_b) / max(fro_b, 1e-12):+6.2f}%)"
            )

    # --- PF per-submodule Δ (across layers) ---
    if pf:
        print("\n=== PF per-submodule Δ (averaged across 48 layers) ===")
        print(
            f"  {'submod':<22} {'n':>4} "
            f"{'eff_rank b→c (Δ)':>22}  {'rms b→c (Δ %)':>22}  {'stable b→c (Δ)':>22}  {'fro b→c (Δ %)':>22}"
        )
        by_S: dict[str, list[dict]] = defaultdict(list)
        for p in pf:
            by_S[p["submod"]].append(p)
        for s in sorted(by_S):
            bs = by_S[s]
            er_b = sum(p["eff_rank_b"] for p in bs) / len(bs)
            er_c = sum(p["eff_rank_c"] for p in bs) / len(bs)
            rms_b = sum(p["rms_b"] for p in bs) / len(bs)
            rms_c = sum(p["rms_c"] for p in bs) / len(bs)
            sr_b = sum(p["sr_b"] for p in bs) / len(bs)
            sr_c = sum(p["sr_c"] for p in bs) / len(bs)
            fro_b = sum(p["fro_b"] for p in bs) / len(bs)
            fro_c = sum(p["fro_c"] for p in bs) / len(bs)
            print(
                f"  {s:<22} {len(bs):>4} "
                f"{er_b:>7.2f}→{er_c:>7.2f} ({er_c - er_b:+6.2f})  "
                f"{rms_b:>7.4f}→{rms_c:>7.4f} ({100 * (rms_c - rms_b) / max(rms_b, 1e-12):+6.2f}%)  "
                f"{sr_b:>7.2f}→{sr_c:>7.2f} ({sr_c - sr_b:+6.2f})  "
                f"{fro_b:>7.2f}→{fro_c:>7.2f} ({100 * (fro_c - fro_b) / max(fro_b, 1e-12):+6.2f}%)"
            )

    # --- Top movers ---
    print(f"\n=== Top {top_n} movers by |Δ eff_rank| ===")
    pairs.sort(key=lambda p: -abs(p["d_eff"]))
    for p in pairs[:top_n]:
        print(
            f"  {p['name']:<70}  shape={str(p['shape']):<14} "
            f"eff_rank {p['eff_rank_b']:>6.2f} → {p['eff_rank_c']:>6.2f}  Δ={p['d_eff']:+6.2f}"
        )

    print(f"\n=== Top {top_n} movers by |Δ RMS / RMS_base| (%) ===")
    pairs.sort(key=lambda p: -abs(p["rel_d_rms"]))
    for p in pairs[:top_n]:
        print(
            f"  {p['name']:<70}  shape={str(p['shape']):<14} "
            f"rms {p['rms_b']:>8.5f} → {p['rms_c']:>8.5f}  Δ={100 * p['rel_d_rms']:+7.2f}%"
        )

    print(f"\n=== Top {top_n} movers by rel_dW = ‖ΔW‖/‖W_base‖ (direct step size) ===")
    pairs.sort(key=lambda p: -p["rel_dW"])
    for p in pairs[:top_n]:
        print(
            f"  {p['name']:<70}  shape={str(p['shape']):<14} "
            f"rel_dW={p['rel_dW']:>8.4%}  angle={p['angle_deg']:>6.4f}°  "
            f"fro_Δ={100 * p['rel_d_fro']:+6.2f}%"
        )

    print(f"\n=== Top {top_n} movers by rotation angle (cos_sim → arccos) ===")
    pairs.sort(key=lambda p: -p["angle_deg"])
    for p in pairs[:top_n]:
        print(
            f"  {p['name']:<70}  shape={str(p['shape']):<14} "
            f"angle={p['angle_deg']:>7.4f}°  cos_sim={p['cos_sim']:.6f}  "
            f"rel_dW={p['rel_dW']:>8.4%}"
        )

    print(f"\n=== Top {top_n} movers by |Δ fro| (energy change) ===")
    pairs.sort(key=lambda p: -abs(p["d_fro"]))
    for p in pairs[:top_n]:
        print(
            f"  {p['name']:<70}  shape={str(p['shape']):<14} "
            f"fro {p['fro_b']:>8.2f} → {p['fro_c']:>8.2f}  Δ={p['d_fro']:+8.4f} ({100 * p['rel_d_fro']:+6.2f}%)"
        )


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "bucket",
                "layer",
                "submod",
                "shape",
                "rms",
                "fro",
                "s_max",
                "s_min",
                "stable_rank",
                "eff_rank",
                "cond",
                "dead",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["name"],
                    r["bucket"],
                    r["layer"] if r["layer"] is not None else "",
                    r["submod"],
                    "x".join(str(d) for d in r["shape"]),
                    f"{r['rms']:.6f}",
                    f"{r['fro']:.4f}",
                    f"{r['s_max']:.4f}",
                    f"{r['s_min']:.6f}",
                    f"{r['stable_rank']:.4f}",
                    f"{r['eff_rank']:.4f}",
                    f"{r['cond']:.2f}",
                    int(r["dead"]),
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--vs", dest="baseline", default=None, help="Baseline checkpoint for diff section.")
    p.add_argument("--out-dir", default="/tmp/spectral_out")
    p.add_argument("--top-outliers", type=int, default=20)
    p.add_argument("--top-movers", type=int, default=20)
    p.add_argument(
        "--exclude-bucket",
        action="append",
        default=["confidence"],
        help="Buckets to skip (default: confidence, since alpha_confidence=0 → head untrained)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device for SVD (cpu | cuda | cuda:0 …). Default cpu.",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="If set, also write a per-weight CSV alongside stdout (path or 'auto' for out_dir/<name>.csv).",
    )
    p.add_argument(
        "--layer-submod-matrix",
        action="store_true",
        help="Print full PF (layer × submod) eff_rank matrix.",
    )
    args = p.parse_args()
    skip_buckets = set(args.exclude_bucket)
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print(f"[warn] {device} requested but CUDA unavailable; falling back to cpu")
        device = "cpu"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cur_state, cur_step = _load_state(args.checkpoint)
    print(f"=== Spectral analysis: {Path(args.checkpoint).name}  step={cur_step} ===")
    if skip_buckets:
        print(f"  (excluded buckets: {sorted(skip_buckets)})")

    rows = analyze(cur_state, device, skip_buckets)
    print(f"\nAnalyzed {len(rows)} 2D matmul weights")

    dead = [r for r in rows if r["dead"]]
    print(f"\n=== Dead weights (s_max < 1e-3): {len(dead)} ===")
    for r in dead[:30]:
        print(f"  {r['name']:<70}  shape={r['shape']}  s_max={r['s_max']:.2e}")

    emit_bucket_summary(rows)
    emit_pf_per_layer(rows)
    emit_pf_per_submod(rows)
    if args.layer_submod_matrix:
        emit_pf_layer_submod_matrix(rows)

    # Plot per-PF-layer eff_rank per submodule
    pf_rows = [r for r in rows if r["bucket"] == "pairformer" and r["layer"] is not None]
    if pf_rows:
        fig, ax = plt.subplots(figsize=(11, 5))
        by_submod_layer = defaultdict(dict)
        for r in pf_rows:
            by_submod_layer[r["submod"]][r["layer"]] = r["eff_rank"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(by_submod_layer)))
        for (sub, layer_er), col in zip(sorted(by_submod_layer.items()), colors, strict=False):
            xs = sorted(layer_er.keys())
            ys = [layer_er[x] for x in xs]
            ax.plot(xs, ys, marker="o", markersize=4, label=sub, color=col, linewidth=1)
        ax.set_xlabel("PF block index")
        ax.set_ylabel("effective rank (entropy-based)")
        ax.set_title(f"PF effective rank per layer  (step={cur_step})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out / "pf_spectral_curve.png", dpi=140, bbox_inches="tight")
        print(f"\nsaved {out / 'pf_spectral_curve.png'}")

    # Top outliers — smallest eff_rank/min_dim ratio (LoRA candidates)
    print(f"\n=== Top {args.top_outliers} by smallest eff_rank/min_dim (LoRA candidates) ===")
    for r in rows:
        r["eff_ratio"] = r["eff_rank"] / min(r["shape"])
    rows_sorted = sorted(rows, key=lambda r: r["eff_ratio"])
    for r in rows_sorted[: args.top_outliers]:
        if r["dead"]:
            continue
        print(
            f"  {r['name']:<70}  shape={r['shape']}  "
            f"eff_rank={r['eff_rank']:>6.1f}/{min(r['shape']):>4}  "
            f"ratio={r['eff_ratio']:.3f}  s_max={r['s_max']:.3f}"
        )

    if args.baseline:
        print(f"\n\n=== Loading baseline: {args.baseline} ===")
        base_state, base_step = _load_state(args.baseline)
        base_rows = analyze(base_state, device, skip_buckets)
        print(f"Analyzed {len(base_rows)} 2D matmul weights in baseline")
        emit_diff(
            rows,
            base_rows,
            cur_state,
            base_state,
            cur_step,
            base_step,
            device=device,
            top_n=args.top_movers,
        )

    if args.csv:
        if args.csv == "auto":
            csv_path = out / f"{Path(args.checkpoint).stem}_spectral.csv"
        else:
            csv_path = Path(args.csv)
        save_csv(rows, csv_path)
        print(f"\nsaved per-weight CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
