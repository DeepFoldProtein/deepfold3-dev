"""Trunk cost at inference lengths — the number that actually matters.

The training-step measurement (r16 23.12 s/step vs r17 21.66, ~1.07x) understates
what removing triangle attention buys, because a training step is dominated by
the diffusion side: 64 noise samples plus a 20-step mini-rollout, all at the
crop size of 384. Inference is the opposite regime --

  * the trunk runs all ``num_recycles`` passes forward with no backward,
  * at the real sequence length rather than a 384 crop, and
  * triangle multiplication and triangle attention are both O(N^3) while the
    diffusion transformer is O(N^2), so the trunk's share grows with N.

So this measures the trunk alone, forward-only under ``no_grad``, at inference
lengths, with and without triangle attention. Weights are irrelevant to timing,
so no checkpoint is needed.

Usage:
    python export/mhr-trimul/bench_inference_trunk.py
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from deepfold3.config_loader import resolve_config
from deepfold3.model.config import EvoformerConfig, OptimConfig, PairFormerConfig
from deepfold3.model.evoformer import Evoformer

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def build(cfg_path: Path, optim: OptimConfig, device: str) -> tuple[Evoformer, EvoformerConfig, PairFormerConfig]:
    d = resolve_config(cfg_path)["model"]
    evo = EvoformerConfig(**{k: v for k, v in d["evoformer"].items() if k not in ("msa_attention", "pair_attention")})
    pf = PairFormerConfig(**{k: v for k, v in d["pairformer"].items() if k != "pair_attention"})
    evo.msa_attention.num_head = d["evoformer"]["msa_attention"]["num_head"]
    evo.pair_attention.num_head = d["evoformer"]["pair_attention"]["num_head"]
    pf.pair_attention.num_head = d["pairformer"]["pair_attention"]["num_head"]
    # fp32 parameters + autocast, which is how inference actually runs: the
    # trunk LayerNorms are fp32-always, so casting the module to bf16 outright
    # produces a dtype mismatch at the first pair_logits GEMM.
    return Evoformer(evo, pf, optim=optim).eval().to(device), evo, pf


@torch.no_grad()
def time_trunk(model, evo, pf, n: int, n_msa: int, device: str, dtype, iters: int) -> float:
    rel_dim = model.position_activations.weight.shape[1]
    args = (
        torch.randn(n, 447, device=device),
        torch.randn(n_msa, n, 34, device=device),
        torch.randn(n, n, rel_dim, device=device),
        torch.ones(n, n, device=device),
        torch.ones(n, device=device),
        torch.ones(n_msa, n, device=device),
        torch.zeros(n, n, pf.pair_channel, device=device),
        torch.zeros(n, evo.seq_channel, device=device),
    )
    with torch.autocast(device_type="cuda", dtype=dtype):
        for _ in range(2):
            model(*args)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(*args)
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[384, 768, 1280])
    ap.add_argument("--n-msa", type=int, default=128)
    ap.add_argument("--recycles", type=int, default=10, help="trunk passes per prediction")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3, help="interleaved repeats per length")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # r16's fusion flags (configs/initial.yaml:334-352).
    optim = OptimConfig(
        use_fused_evo_attention=True,
        use_fused_pair_bias=False,
        use_fused_swiglu=True,
        use_fused_triangle_mul=True,
        use_fused_transition_pair=True,
    )
    dtype = torch.bfloat16
    rows = []
    # Interleave the two configs at each N. Measuring all of r16 and then all of
    # r17 lets clock/thermal drift masquerade as a speedup — the first pass at a
    # non-monotonic ratio came out that way.
    models = {}
    for tag, cfg in (("r16", "df3-stage0-r16.yaml"), ("r17", "df3-stage0-r17.yaml")):
        models[tag] = build(REPO / "configs" / cfg, optim, args.device)

    for n in args.lengths:
        for rep in range(args.reps):
            for tag in ("r16", "r17") if rep % 2 == 0 else ("r17", "r16"):
                model, evo, pf = models[tag]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    ms = time_trunk(model, evo, pf, n, args.n_msa, args.device, dtype, args.iters)
                except torch.OutOfMemoryError:
                    print(f"{tag} N={n}: OOM", flush=True)
                    continue
                rows.append(
                    dict(
                        run=tag,
                        n=n,
                        rep=rep,
                        trunk_ms=round(ms, 1),
                        per_prediction_s=round(ms * args.recycles / 1000.0, 2),
                        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        trunk_params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                    )
                )
                print(rows[-1], flush=True)

    with open(OUT / "bench_inference_trunk.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def best(tag, n):
        v = [r["trunk_ms"] for r in rows if r["run"] == tag and r["n"] == n]
        return min(v) if v else None  # min over reps: least contaminated by drift

    print("\nN      r16 trunk    r17 trunk   speedup   (min over reps)")
    for n in args.lengths:
        a, b = best("r16", n), best("r17", n)
        if a and b:
            print(f"{n:5d} {a:10.1f}ms {b:10.1f}ms   {a / b:6.2f}x")


if __name__ == "__main__":
    main()
