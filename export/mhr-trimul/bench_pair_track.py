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
    python export/mhg-trimul/bench_pair_track.py
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


def timed(step, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / iters


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

        # --- individual modules
        zb = pair.unsqueeze(0)
        mb = pmask.unsqueeze(0)
        for name, mod in (
            ("tri_mul", TriangleMulOutgoing(C_Z, optim=optim)),
            ("tri_attn", GridSelfAttention(C_Z, num_heads=N_HEADS, transpose=False, optim=optim)),
        ):
            mod = mod.to(dev).to(dt)
            ms = timed(lambda mod=mod: mod(zb, mb).sum().backward())
            rows.append(dict(scope="module", variant=name, n=n, ms=round(ms, 2)))

        # --- whole block, with and without triangle attention
        per_block = {}
        for tag in ("full", "no_triattn"):
            blk = PairFormerBlock(C_Z, C_S, num_pair_heads=N_HEADS, optim=optim).to(dev).to(dt)
            if tag == "no_triattn":
                blk.forward = types.MethodType(_forward_without_triangle_attention, blk)

            def step(blk=blk):
                a, b = blk(pair, single, pmask, smask)
                (a.sum() + b.sum()).backward()

            per_block[tag] = timed(step)
            rows.append(dict(scope="block", variant=tag, n=n, ms=round(per_block[tag], 2)))
        rows.append(
            dict(
                scope="block",
                variant="speedup",
                n=n,
                ms=round(per_block["full"] / per_block["no_triattn"], 3),
            )
        )

    with open(OUT / "bench_pair_track.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "variant", "n", "ms"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
