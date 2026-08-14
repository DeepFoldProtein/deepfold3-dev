"""Pair-track cost of triangle attention, measured with r16's fusion flags.

The r17 proposal removes ``pair_attn_row`` / ``pair_attn_col`` from every
PairFormer block. What that buys is a question about wall-clock, not FLOP
arithmetic: the repo runs fused Triton kernels whose constant factors differ
between triangle multiplication and grid self-attention, so the ratio has to be
measured on the kernels that will actually run.

Flags mirror ``configs/initial.yaml:334-352`` (the ones r16 trains under):
fused evo-attention on, fused pair-bias *off*, fused triangle-mul on.

Caveat: this is a single-block microbenchmark with no activation checkpointing,
no DDP, and no recycling. The end-to-end training speedup is strictly lower --
the MSA stack, atom encoder, diffusion module and confidence head are unchanged.
Measure the full step before committing a run.

Usage:
    python export/mhr-trimul/bench_pair_track.py
"""

from __future__ import annotations

import csv
import time
import types
from pathlib import Path

import torch

from deepfold3.model.attention import GridSelfAttention
from deepfold3.model.config import OptimConfig
from deepfold3.model.evoformer import PairFormerBlock, dropout_add_rowwise
from deepfold3.model.ops import TriangleMulOutgoing

OUT = Path(__file__).resolve().parent
C_Z, C_S, N_HEADS = 256, 384, 8  # r16 widths (configs/df3-stage0-r16.yaml)


def _forward_without_triangle_attention(self, pair, single, pair_mask, seq_mask):
    """PairFormerBlock.forward with the two grid-attention branches deleted."""
    p, tr, uf = self.pair_dropout, self.training, self.use_fused_dropout
    pair = dropout_add_rowwise(pair, self.tri_mul_out(pair, pair_mask), p, tr, uf)
    pair = dropout_add_rowwise(pair, self.tri_mul_in(pair, pair_mask), p, tr, uf)
    pair = pair + self.pair_transition(pair)
    logits = self.single_to_pair(self.single_to_pair_norm(pair)).movedim(-1, -3)
    single = single + self.single_attn(single, seq_mask, logits)
    return pair, single + self.single_transition(single)


def timed(step, warmup: int = 8, iters: int = 10, reps: int = 3) -> float:
    """Milliseconds per call, min over repeats.

    Warmup matters more than it looks: the Triton kernels autotune on first
    sight of a shape and persist the result, so a short warmup charges the
    first-measured variant for tuning that later variants inherit for free.
    An earlier version of this script used two warmup iterations and produced
    ``tri_mul`` slower than ``tri_attn`` at N=384 while the same run's block
    delta said the opposite, and a ``tri_mul`` that grew only 1.3x from N=384 to
    N=640 for an O(N^3) contraction. Taking the min over repeats after a long
    warmup removes both.
    """
    for _ in range(warmup):
        step()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        best = min(best, 1000.0 * (time.time() - t0) / iters)
    return best


def main() -> None:
    optim = OptimConfig(
        use_fused_evo_attention=True,
        use_fused_pair_bias=False,
        use_fused_swiglu=True,
        use_fused_triangle_mul=True,
        use_fused_transition_pair=True,
    )
    dev, dt = "cuda", torch.bfloat16
    rows = []

    for n in (384, 640):
        pair = torch.randn(n, n, C_Z, device=dev, dtype=dt, requires_grad=True)
        single = torch.randn(n, C_S, device=dev, dtype=dt, requires_grad=True)
        pmask = torch.ones(n, n, device=dev, dtype=dt)
        smask = torch.ones(n, device=dev, dtype=dt)
        zb, mb = pair.unsqueeze(0), pmask.unsqueeze(0)

        mods = {
            "tri_mul": TriangleMulOutgoing(C_Z, optim=optim).to(dev).to(dt),
            "tri_attn": GridSelfAttention(C_Z, num_heads=N_HEADS, transpose=False, optim=optim).to(dev).to(dt),
        }
        blocks = {}
        for tag in ("full", "no_triattn"):
            blk = PairFormerBlock(C_Z, C_S, num_pair_heads=N_HEADS, optim=optim).to(dev).to(dt)
            if tag == "no_triattn":
                blk.forward = types.MethodType(_forward_without_triangle_attention, blk)
            blocks[tag] = blk

        def mod_step(m, zb=zb, mb=mb):
            return lambda: m(zb, mb).sum().backward()

        def blk_step(b, pair=pair, single=single, pmask=pmask, smask=smask):
            def go():
                x, y = b(pair, single, pmask, smask)
                (x.sum() + y.sum()).backward()

            return go

        # Autotune every shape before timing anything, so no variant is charged
        # for tuning that the others inherit.
        for m in mods.values():
            for _ in range(3):
                mod_step(m)()
        for b in blocks.values():
            for _ in range(3):
                blk_step(b)()

        for name, m in mods.items():
            rows.append(dict(scope="module", variant=name, n=n, ms=round(timed(mod_step(m)), 2)))
            print(rows[-1], flush=True)
        per_block = {}
        for tag, b in blocks.items():
            per_block[tag] = timed(blk_step(b))
            rows.append(dict(scope="block", variant=tag, n=n, ms=round(per_block[tag], 2)))
            print(rows[-1], flush=True)
        rows.append(
            dict(scope="block", variant="speedup", n=n, ms=round(per_block["full"] / per_block["no_triattn"], 3))
        )
        print(rows[-1], flush=True)

    # Self-consistency: the block delta must agree with 2x the module time, and
    # an O(N^3) op must grow ~(640/384)^3 = 4.63x. Print both rather than
    # silently trusting the numbers.
    def get(scope, variant, n):
        return next(r["ms"] for r in rows if r["scope"] == scope and r["variant"] == variant and r["n"] == n)

    print("")
    print("consistency checks")
    for n in (384, 640):
        delta = get("block", "full", n) - get("block", "no_triattn", n)
        print(f"  N={n}: block delta {delta:7.2f} ms  vs  2 x tri_attn {2 * get('module', 'tri_attn', n):7.2f} ms")
    for v in ("tri_mul", "tri_attn"):
        r = get("module", v, 640) / get("module", v, 384)
        print(f"  {v:9s} N 384->640 grew {r:.2f}x  (O(N^3) predicts 4.63x)")

    with open(OUT / "bench_pair_track.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "variant", "n", "ms"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
