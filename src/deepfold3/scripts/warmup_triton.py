"""Warm the persistent Triton autotune cache for our bucket sizes.

The fused kernels in ``deepfold3.model.triton`` use ``@persistent_autotune``,
which runs ``triton.autotune`` once per (kernel, shape) combination on first
encounter and persists the winner to ``~/.cache/deepfold3-triton/``.

Without warmup, the **first DDP training run** of every fresh-cache machine
pays the autotune cost on every shape it encounters — and worse, runs that
autotune redundantly across all 8 ranks (each writes the same result).

This script populates the cache **on a single rank** by exercising every fused
kernel at every (M, N, K) shape the model actually uses, for every bucket size
in :data:`deepfold3.model.bucket.BUCKET_SIZES` we expect at training time.

Channel dimensions and the transition expansion factor are loaded from the
``ModelConfig`` in a JSON config file (same format as ``deepfold3 train
--config``); when no config is given the defaults from ``model/config.py`` are
used.

Usage::

    # Default: warms all kernels for the production bucket range (≤384) using
    # default channel dims.
    deepfold3 warmup-triton

    # Use shapes from a project config file (recommended).
    deepfold3 warmup-triton --config configs/initial.json

    # Custom bucket subset
    deepfold3 warmup-triton --buckets 128,160,192,224,256,320,384

    # Skip the GEMM kernels (already-cached deployments)
    deepfold3 warmup-triton --skip-gemm

After this script finishes, the cache JSON files in ``~/.cache/deepfold3-triton/``
hold a config for every shape; subsequent training runs on this GPU model load
those configs at decorator-import time and skip the autotune benchmark phase.

The cache is keyed by GPU compute capability (e.g. ``.10.0.json`` for B200),
so warmup must be re-run once per distinct GPU model.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import click
import torch

from deepfold3.model.bucket import BUCKET_SIZES
from deepfold3.model.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WarmupShapes:
    """Channel dims + MSA crop pulled from a ``ModelConfig`` (or defaults).

    Transition expansion factors are not stored on the per-layer configs;
    they are wired directly through ``make_transition_block`` call sites in
    the model. We use the AF3 defaults (4× for the trunk transitions, 2× for
    the diffusion AdaLN FFN) which match every current call site.
    """

    c_pair: int
    c_single: int
    c_msa: int
    c_diff_token: int  # per-token channels in the diffusion transformer
    num_msa: int  # max MSA rows fed to the Evoformer

    # Pair attention head config (used for the evo_attention warmup shapes).
    pair_attn_num_head: int
    pair_attn_head_dim: int

    # Transition expansion factors (constant in the AF3 architecture).
    trunk_transition_factor: int = 4  # pair / single / MSA TransitionBlock
    diff_transition_factor: int = 2  # diffusion AdaTransitionBlock FFN

    @classmethod
    def from_model_config(cls, mcfg: ModelConfig) -> WarmupShapes:
        # AF3 pair attention key/value dim = c_pair / num_head (=128/4=32).
        pair_num_head = mcfg.evoformer.pair_attention.num_head
        return cls(
            c_pair=mcfg.evoformer.pair_channel,
            c_single=mcfg.evoformer.seq_channel,
            c_msa=mcfg.evoformer.msa_channel,
            c_diff_token=mcfg.diffusion.atom_cross_att.per_token_channels,
            num_msa=512,  # featurizer cap at training time
            pair_attn_num_head=pair_num_head,
            pair_attn_head_dim=mcfg.evoformer.pair_channel // pair_num_head,
            diff_transition_factor=mcfg.diffusion.transformer.num_intermediate_factor,
        )


def _load_model_config(config_path: str | None) -> ModelConfig:
    """Load a ``ModelConfig`` from JSON, mirroring ``deepfold3 train --config``.

    Returns defaults when ``config_path`` is None or omits a ``model`` block.
    """
    mcfg = ModelConfig()
    if not config_path:
        return mcfg
    with open(config_path) as f:
        overrides = json.load(f)
    model_overrides = overrides.get("model", {})
    if not model_overrides:
        return mcfg
    # Reuse the same recursive override logic as the train script.
    from deepfold3.scripts.train import _apply_overrides

    _apply_overrides(mcfg, model_overrides)
    return mcfg


def _layernorm_linear_shapes(n_pad: int, batch: int, s: WarmupShapes) -> list[tuple[str, int, int, int]]:
    """Enumerate (label, M, N, K) shapes for ``FusedLayerNormLinear``.

    The leading batch dim ``batch`` is collapsed into ``M`` by the kernel
    wrapper (``x.view(-1, K)``) so we have to warm one shape per
    ``(bucket, batch_size_per_rank)`` combination.

    Covers:
      - triangle multiplication pre-projection (5 * c_z out)
      - pair transition (gate + value concat: 2 * factor * c_z)
      - pair out projection (TriangleMul post-einsum)
      - single transition (gate + value: 2 * factor * c_single)
      - MSA transition (gate + value: 2 * factor * c_msa)
    """
    pair_M = batch * n_pad * n_pad
    msa_M = batch * s.num_msa * n_pad
    single_M = batch * n_pad
    f = s.trunk_transition_factor
    return [
        ("trimul_gate", pair_M, 5 * s.c_pair, s.c_pair),
        ("pair_transition", pair_M, 2 * f * s.c_pair, s.c_pair),
        ("pair_out_proj", pair_M, s.c_pair, s.c_pair),
        ("single_transition", single_M, 2 * f * s.c_single, s.c_single),
        ("msa_transition", msa_M, 2 * f * s.c_msa, s.c_msa),
    ]


def _ada_layernorm_linear_shapes(n_pad: int, batch: int, s: WarmupShapes) -> list[tuple[str, int, int, int]]:
    """Enumerate (label, M, N, K) shapes for the diffusion FFN ``FusedAdaLayerNormLinear``."""
    return [
        ("diffusion_ffn", batch * n_pad, 2 * s.diff_transition_factor * s.c_diff_token, s.c_diff_token),
    ]


def _run_warm(label: str, build_call, device: torch.device) -> None:
    """Build the kernel inputs, run forward + backward, log timing.

    ``build_call`` returns ``(callable, args, requires_grad_outputs)`` where
    the callable is invoked with ``args`` and the (single) output tensor is
    summed into a scalar that gets ``.backward()``-ed.
    """
    t0 = time.time()
    fn, args = build_call()
    out = fn(*args)
    out.float().sum().backward()
    torch.cuda.synchronize()
    logger.info(f"  {label} → {time.time() - t0:.1f}s")


def _make_input(M: int, K: int, device: torch.device) -> torch.Tensor:
    return torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)


def _warm_layernorm_linear(n_pad: int, batch: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive ``FusedLayerNormLinear`` for every shape we need at this bucket."""
    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear

    for label, M, N, K in _layernorm_linear_shapes(n_pad, batch, shapes):
        def build(M=M, N=N, K=K):
            module = FusedLayerNormLinear(K, N, has_linear_bias=False).to(device).to(torch.bfloat16)
            return module, (_make_input(M, K, device),)

        _run_warm(f"ln_linear[{label}] M={M} N={N} K={K}", build, device)


def _warm_ada_layernorm_linear(n_pad: int, batch: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive ``FusedAdaLayerNormLinear`` for every shape we need at this bucket."""
    from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear

    for label, M, N, K in _ada_layernorm_linear_shapes(n_pad, batch, shapes):
        def build(M=M, N=N, K=K):
            module = FusedAdaLayerNormLinear(K, N).to(device).to(torch.bfloat16)
            x = _make_input(M, K, device)
            scale = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            bias = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            return module, (x, scale, bias)

        _run_warm(f"ada_ln_linear[{label}] M={M} N={N} K={K}", build, device)


def _warm_evo_attention(n_pad: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive the AF3 grid attention kernels (forward + 2 backward).

    Layout matches GridSelfAttention: ``q,k,v: [B*N_rows, N_cols, H, D]``,
    with ``B=1, N_rows=N_cols=n_pad``. Cache key is ``SEQ_LEN{n_pad}_DIM{D}``.
    """
    try:
        from deepfold3.model.triton.evo_attention import fused_evo_attention
    except ImportError:
        logger.info("  evo_attention: kernel unavailable, skipping")
        return

    H = shapes.pair_attn_num_head
    D = shapes.pair_attn_head_dim

    def build():
        BN = n_pad  # B=1, N_rows=n_pad
        N_cols = n_pad
        q = torch.randn(BN, N_cols, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(BN, N_cols, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(BN, N_cols, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        pair_bias = torch.randn(1, H, N_cols, N_cols, device=device, dtype=torch.bfloat16)
        mask = torch.ones(BN, N_cols, dtype=torch.bool, device=device)
        return fused_evo_attention, (q, k, v, pair_bias, mask)

    _run_warm(f"evo_attention[N={n_pad}, H={H}, D={D}]", build, device)


def _warm_dual_gemm(n_pad: int, batch: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive the fused dual-GEMM kernel from triangle_mul (already autotuned).

    TriangleMul packs into ``[batch * N_pad², K=c_z, N=2*c_z]`` for the
    gated dual GEMM.
    """
    try:
        from deepfold3.model.triton.fused_gated_dual_gemm import FusedGatedDualGEMM
    except ImportError:
        logger.info("  dual_gemm: kernel unavailable, skipping")
        return

    M = batch * n_pad * n_pad
    K = shapes.c_pair
    N = 2 * shapes.c_pair

    def build():
        module = FusedGatedDualGEMM(K, N, bias=False).to(device).to(torch.bfloat16)
        return module, (_make_input(M, K, device),)

    _run_warm(f"dual_gemm[M={M} N={N} K={K}]", build, device)


def warmup_buckets(
    buckets: Iterable[int],
    batch_sizes: Iterable[int],
    shapes: WarmupShapes,
    device: torch.device,
    *,
    skip_gemm: bool = False,
) -> None:
    """Run all kernels at every (bucket, batch_size_per_rank) combination."""
    from deepfold3.model.triton.cache import CacheManager

    batch_sizes = list(batch_sizes)
    for n_pad in buckets:
        for i, batch in enumerate(batch_sizes):
            logger.info(f"=== bucket N={n_pad}, batch_per_rank={batch} ===")
            if not skip_gemm:
                _warm_layernorm_linear(n_pad, batch, shapes, device)
                _warm_ada_layernorm_linear(n_pad, batch, shapes, device)
            # Evo-attention key is (SEQ_LEN, DIM) — independent of the
            # leading batch dim. Only warm it once per bucket.
            if i == 0:
                _warm_evo_attention(n_pad, shapes, device)
            _warm_dual_gemm(n_pad, batch, shapes, device)

    # Force one final flush so cache JSON is fully written before exit.
    CacheManager.get().save_all()
    logger.info(f"Cache flushed to {CacheManager.get()._cache_dir}")


@click.command(name="warmup-triton")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "JSON config file in the same format as ``deepfold3 train --config``. "
        "Channel dims and transition factors are read from the ``model`` section; "
        "defaults are used when omitted."
    ),
)
@click.option(
    "--buckets",
    type=str,
    default=None,
    help=(
        "Comma-separated bucket sizes to warm. Defaults to all buckets ≤384 "
        "(the production max_tokens). Override for larger ranges."
    ),
)
@click.option(
    "--batch-sizes",
    type=str,
    default="1,2",
    help=(
        "Comma-separated values of ``batch_size_per_rank`` to warm. The kernel "
        "wrapper collapses the batch dim into ``M`` (M = batch * N²), so each "
        "batch size produces a different (M, N, K) cache key. Defaults to "
        "``1,2`` to cover both single-sample and the production batch_per_rank=2."
    ),
)
@click.option("--skip-gemm", is_flag=True, help="Skip the layernorm_linear / ada_layernorm_linear kernels.")
@click.option("--device", type=str, default="cuda:0", help="CUDA device to warm on.")
def main(
    config_path: str | None, buckets: str | None, batch_sizes: str, skip_gemm: bool, device: str
) -> None:
    """Pre-populate the persistent Triton autotune cache for our bucket sizes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if buckets is None:
        bucket_list = [b for b in BUCKET_SIZES if b <= 384]
    else:
        bucket_list = sorted({int(x) for x in buckets.split(",") if x.strip()})

    batch_size_list = sorted({int(x) for x in batch_sizes.split(",") if x.strip()})

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA not available — warmup needs a GPU.")

    mcfg = _load_model_config(config_path)
    shapes = WarmupShapes.from_model_config(mcfg)
    dev = torch.device(device)
    name = torch.cuda.get_device_name(dev)
    logger.info(f"Warming Triton cache on {name} for buckets: {bucket_list}")
    logger.info(f"Batch sizes: {batch_size_list}")
    logger.info(
        f"Channel dims: c_pair={shapes.c_pair} c_single={shapes.c_single} "
        f"c_msa={shapes.c_msa} c_diff_token={shapes.c_diff_token} num_msa={shapes.num_msa}"
    )
    if config_path:
        logger.info(f"Loaded shapes from {config_path}")
    else:
        logger.info("No --config given, using ModelConfig defaults")

    t0 = time.time()
    warmup_buckets(bucket_list, batch_size_list, shapes, dev, skip_gemm=skip_gemm)
    logger.info(f"Total warmup time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
