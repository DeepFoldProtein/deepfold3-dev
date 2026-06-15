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

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import click
import torch

from deepfold3.model.bucket import BUCKET_SIZES
from deepfold3.model.config import ModelConfig
from deepfold3.scripts import cli
from deepfold3.scripts._common import pin_cuda_device

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

    # Conditioning channels (may differ from trunk channels).
    c_pair_cond: int  # DiffusionConditioningBlock pair output
    c_single_cond: int  # DiffusionConditioningBlock single output

    # Transition expansion factors.
    trunk_transition_factor: int = 4  # pair / single / MSA TransitionBlock
    cond_transition_factor: int = 2  # DiffusionConditioningBlock pair/single trans
    diff_transition_factor: int = 2  # diffusion AdaTransitionBlock FFN

    # Diffusion chunk size — determines the B*S multiplier for the main
    # diffusion transformer and atom encoder/decoder shapes during training.
    diffusion_chunk_size: int = 16

    @classmethod
    def from_model_config(cls, mcfg: ModelConfig) -> WarmupShapes:
        pair_num_head = mcfg.evoformer.pair_attention.num_head
        return cls(
            c_pair=mcfg.evoformer.pair_channel,
            c_single=mcfg.evoformer.seq_channel,
            c_msa=mcfg.evoformer.msa_channel,
            c_diff_token=mcfg.diffusion.atom_cross_att.per_token_channels,
            num_msa=512,
            pair_attn_num_head=pair_num_head,
            pair_attn_head_dim=mcfg.evoformer.pair_channel // pair_num_head,
            c_pair_cond=mcfg.diffusion.conditioning_pair_channel,
            c_single_cond=mcfg.diffusion.conditioning_seq_channel,
            cond_transition_factor=mcfg.diffusion.conditioning_num_intermediate_factor,
            diff_transition_factor=mcfg.diffusion.transformer.num_intermediate_factor,
        )


def _coerce_scalar(raw: str) -> bool | int | float | str:
    """Parse a CLI override value string into bool / int / float / str."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _parse_overrides(override: tuple[str, ...]) -> dict:
    """Parse ``--override key=value`` into a nested dict.

    Bare keys land in the ``training`` section (matching ``deepfold3 train``);
    dotted keys address nested config, e.g. ``model.pairformer.num_layer=4``.
    """
    out: dict = {}
    for kv in override:
        if "=" not in kv:
            raise click.BadParameter(f"Override must be key=value, got: {kv}")
        key, raw = kv.split("=", 1)
        parts = key.split(".")
        if len(parts) == 1:
            parts = ["training", parts[0]]  # bare key → training section
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce_scalar(raw)
    return out


def _load_configs(config_path: str | None, override: tuple[str, ...] = ()) -> tuple[ModelConfig, int, dict, set]:
    """Load a ``ModelConfig`` + training-overrides dict + ``diffusion_chunk_size``.

    ``--override key=value`` pairs are merged on top of the config file. Returns
    ``(ModelConfig, chunk_size, training_dict, explicit_training_keys)`` where the
    last element is the set of top-level training keys set via ``--override`` (so
    the caller can honor e.g. an explicit ``max_recycles`` uncapped).
    """
    from deepfold3.config_loader import deep_merge, load_config

    cfg_dict = load_config(config_path) if config_path else {}
    cli_over = _parse_overrides(override)
    cfg_dict = deep_merge(cfg_dict, cli_over)

    # Validate training override keys against TrainingConfig to catch typos.
    train_cfg = cfg_dict.get("training", {}) or {}
    if cli_over.get("training"):
        import dataclasses

        from deepfold3.model.config import TrainingConfig

        valid = {f.name for f in dataclasses.fields(TrainingConfig)}
        for k in cli_over["training"]:
            if k not in valid:
                raise click.BadParameter(f"Unknown training config key: {k}")

    mcfg = ModelConfig()
    model_overrides = cfg_dict.get("model", {})
    if model_overrides:
        from deepfold3.scripts.train import _apply_overrides

        _apply_overrides(mcfg, model_overrides)
        mcfg.__post_init__()  # re-mirror precision flags after overrides
    chunk_size = int(train_cfg.get("diffusion_chunk_size", 16))  # TrainingConfig default
    explicit_training = set(cli_over.get("training", {}))
    return mcfg, chunk_size, train_cfg, explicit_training


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
    fc = s.cond_transition_factor
    shapes = [
        # Trunk (factor=4)
        ("trimul_gate", pair_M, 5 * s.c_pair, s.c_pair),
        ("pair_transition", pair_M, 2 * f * s.c_pair, s.c_pair),
        ("pair_out_proj", pair_M, s.c_pair, s.c_pair),
        ("single_transition", single_M, 2 * f * s.c_single, s.c_single),
        ("msa_transition", msa_M, 2 * f * s.c_msa, s.c_msa),
        # Diffusion conditioning (factor=2) — pair_trans0/1, single_trans0/1.
        # These use the SAME FusedLayerNormLinear kernel at a DIFFERENT output
        # width (2*fc*c vs 2*f*c). Missing these caused the bp=2/seed=42
        # deadlock: ranks that hit a new bucket triggered autotune while the
        # others waited in the next NCCL collective.
        ("cond_pair_trans", pair_M, 2 * fc * s.c_pair_cond, s.c_pair_cond),
        ("cond_single_trans", single_M, 2 * fc * s.c_single_cond, s.c_single_cond),
    ]
    return shapes


def _ada_layernorm_linear_shapes(n_pad: int, batch: int, s: WarmupShapes) -> list[tuple[str, int, int, int]]:
    """Enumerate (label, M, N, K) shapes for the diffusion FFN ``FusedAdaLayerNormLinear``.

    The diffusion transformer FFN sees M = BS × N where BS is batch × chunk_size
    during training (or batch × num_samples during inference). We warm the
    training chunk multiplier so every microstep is a cache hit.
    """
    fd = s.diff_transition_factor
    shapes = [
        # Per-token diffusion FFN at the standard B*N layout
        ("diffusion_ffn", batch * n_pad, 2 * fd * s.c_diff_token, s.c_diff_token),
    ]
    # Also warm the training layout: B * chunk_size * N (collapsed into M).
    chunk_M = batch * s.diffusion_chunk_size * n_pad
    if chunk_M != batch * n_pad:  # avoid duplicate when chunk_size=1
        shapes.append(("diffusion_ffn_chunk", chunk_M, 2 * fd * s.c_diff_token, s.c_diff_token))
    return shapes


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


def _make_weight(*shape: int, device: torch.device) -> torch.Tensor:
    """A grad-requiring weight tensor — the now-stateless fused kernels take
    LN/linear/gate weights as explicit call-time args, so warmup must supply them."""
    return torch.randn(*shape, device=device, dtype=torch.bfloat16, requires_grad=True)


def _warm_layernorm_linear(n_pad: int, batch: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive ``FusedLayerNormLinear`` for every shape we need at this bucket."""
    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear

    for label, M, N, K in _layernorm_linear_shapes(n_pad, batch, shapes):

        def build(M=M, N=N, K=K):
            # FusedLayerNormLinear is stateless — pass LN scale/bias + the
            # [K, N] linear weight at call time (see its forward signature).
            module = FusedLayerNormLinear(K, N, has_linear_bias=False).to(device).to(torch.bfloat16)
            x = _make_input(M, K, device)
            ln_weight = _make_weight(K, device=device)
            ln_bias = _make_weight(K, device=device)
            linear_weight = _make_weight(K, N, device=device)
            return module, (x, ln_weight, ln_bias, linear_weight)

        _run_warm(f"ln_linear[{label}] M={M} N={N} K={K}", build, device)


def _warm_ada_layernorm_linear(n_pad: int, batch: int, shapes: WarmupShapes, device: torch.device) -> None:
    """Drive ``FusedAdaLayerNormLinear`` for every shape we need at this bucket."""
    from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear

    for label, M, N, K in _ada_layernorm_linear_shapes(n_pad, batch, shapes):

        def build(M=M, N=N, K=K):
            # Stateless: (x, conditioning scale [M,K], conditioning bias [M,K],
            # [K, N] linear weight).
            module = FusedAdaLayerNormLinear(K, N).to(device).to(torch.bfloat16)
            x = _make_input(M, K, device)
            scale = _make_weight(M, K, device=device)
            bias = _make_weight(M, K, device=device)
            linear_weight = _make_weight(K, N, device=device)
            return module, (x, scale, bias, linear_weight)

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
        # Stateless: (x, w_gate [N, K], w_proj [N, K]) — standard nn.Linear shape.
        module = FusedGatedDualGEMM(K, N).to(device).to(torch.bfloat16)
        x = _make_input(M, K, device)
        w_gate = _make_weight(N, K, device=device)
        w_proj = _make_weight(N, K, device=device)
        return module, (x, w_gate, w_proj)

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


def _synthetic_training_features(n_pad: int, batch: int, n_msa: int, device: torch.device) -> dict:
    """Batched ``[B, N, ...]`` features (with GT) to drive ``forward_training``.

    Values are random within valid index ranges — only the shapes matter for
    warmup, so this exactly reproduces the (M, N, K) and kernel specializations
    the real training step hits at this bucket, without needing real data.
    """
    from deepfold3.data import const

    B, N, M, S = batch, n_pad, const.MAX_ATOMS_PER_TOKEN, n_msa
    g = torch.Generator().manual_seed(0)
    ref_mask = torch.zeros(B, N, M, dtype=torch.bool)
    ref_mask[..., :5] = True  # 5 heavy atoms per token
    arange_bn = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()

    def ri(hi, *shape):
        return torch.randint(0, hi, shape, generator=g)

    feats = {
        "res_type": ri(20, B, N),
        "residue_index": arange_bn.clone(),
        "token_index": arange_bn.clone(),
        "asym_id": torch.zeros(B, N, dtype=torch.long),
        "entity_id": torch.ones(B, N, dtype=torch.long),
        "sym_id": torch.ones(B, N, dtype=torch.long),
        "seq_mask": torch.ones(B, N, dtype=torch.bool),
        "pair_mask": torch.ones(B, N, N, dtype=torch.bool),
        "ref_pos": torch.randn(B, N, M, 3, generator=g) * ref_mask.unsqueeze(-1).float(),
        "ref_mask": ref_mask,
        "ref_element": ri(8, B, N, M) * ref_mask.long(),
        "ref_charge": torch.zeros(B, N, M),
        "ref_atom_name_chars": torch.zeros(B, N, M, 4, dtype=torch.long),
        "msa": ri(22, B, S, N),
        "msa_mask": torch.ones(B, S, N, dtype=torch.bool),
        "deletion_matrix": torch.zeros(B, S, N, dtype=torch.long),
        "profile": torch.zeros(B, N, 31),
        "deletion_mean": torch.zeros(B, N),
        "atom_gt_positions": torch.randn(B, N, M, 3, generator=g) * ref_mask.unsqueeze(-1).float(),
        "atom_gt_mask": ref_mask.clone(),
        "mol_type": torch.zeros(B, N, dtype=torch.long),
    }
    return {k: v.to(device) for k, v in feats.items()}


def collapse_to_depth_1(mcfg: ModelConfig) -> dict[str, tuple[int, int]]:
    """Reduce every repeated block stack to a single layer for warmup.

    The Triton autotune cache is keyed by ``(kernel, shape)`` — the layer index
    is *not* part of the key, and every layer in a given stack shares identical
    channel dims, hence identical ``(M, N, K)``. So a single layer warms the
    exact same cache entries as the full stack; the remaining layers are pure
    redundant fwd/bwd compute. Collapsing every stack to depth 1 warms the
    identical cache set at a fraction of the memory/time, with no shape coverage
    lost — and removes the OOM pressure that otherwise forces ``--msa-depth``
    down at N=384.

    The ``num_blocks``-dependent ``LinearNoBias`` projections (atom transformer
    ``pair_logits_proj``, diffusion transformer pair-bias proj) change width
    when their block count drops, but those are plain cuBLAS matmuls — not in
    the Triton autotune cache — so the warmed shapes are unaffected.

    Returns ``{dotted_field: (orig, new)}`` for logging.
    """
    changes: dict[str, tuple[int, int]] = {}

    def set1(name: str, obj: object, attr: str) -> None:
        orig = getattr(obj, attr)
        setattr(obj, attr, 1)
        changes[name] = (orig, 1)

    set1("evoformer.num_layer", mcfg.evoformer, "num_layer")
    set1("pairformer.num_layer", mcfg.pairformer, "num_layer")
    set1("confidence.num_layer", mcfg.confidence, "num_layer")
    set1("template.pairformer.num_layer", mcfg.template.pairformer, "num_layer")
    set1(
        "diffusion.atom_cross_att.atom_transformer.num_blocks",
        mcfg.diffusion.atom_cross_att.atom_transformer,
        "num_blocks",
    )
    # Diffusion transformer asserts num_blocks % super_block_size == 0, so drop
    # both to 1 (one block, one super-block). Per-block Triton kernels (AdaLN
    # FFN / attention) are shape-independent of these counts.
    set1("diffusion.transformer.super_block_size", mcfg.diffusion.transformer, "super_block_size")
    set1("diffusion.transformer.num_blocks", mcfg.diffusion.transformer, "num_blocks")
    return changes


def warmup_real_model(
    mcfg,
    buckets: Iterable[int],
    batch: int,
    n_msa: int,
    diffusion_batch_size: int,
    num_recycles: int,
    compute_confidence: bool,
    device: torch.device,
) -> None:
    """Warm the cache by driving the real model's ``forward_training`` + backward.

    Guarantees the cached (M, N, K) shapes and kernel specializations exactly
    match training — no hand-maintained shape list to drift out of sync. ``mcfg``
    is expected to have been passed through :func:`collapse_to_depth_1` by the
    caller (the default for ``--real-model``): one layer per stack warms the
    identical shape-keyed cache entries as the full stack.
    """
    from deepfold3.model.model import DeepFold3
    from deepfold3.model.triton.cache import CacheManager

    logger.info("Real-model warmup: forward_training + backward at the production shapes")
    model = DeepFold3(mcfg).to(device)
    model.train()
    for n_pad in buckets:
        logger.info(
            f"=== real-model: N={n_pad} batch={batch} S_msa={n_msa} "
            f"S_diff={diffusion_batch_size} recycles={num_recycles} conf={compute_confidence} ==="
        )
        feats = _synthetic_training_features(n_pad, batch, n_msa, device)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_training(
                feats,
                num_recycles=num_recycles,
                diffusion_batch_size=diffusion_batch_size,
                mini_rollout_steps=2,
                mini_rollout_samples=1,
                compute_confidence=compute_confidence,
            )
        # Drive a backward so the bwd kernels warm too (they have their own keys).
        loss = sum(
            v.float().pow(2).mean()
            for v in out.values()
            if torch.is_tensor(v) and v.is_floating_point() and v.requires_grad
        )
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        logger.info(f"  fwd+bwd N={n_pad} → {time.time() - t0:.1f}s")
    CacheManager.get().save_all()
    logger.info(f"Cache flushed to {CacheManager.get()._cache_dir}")


@cli.command(name="warmup-triton")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "JSON or YAML config file in the same format as ``deepfold3 train --config``. "
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
@click.option(
    "--real-model",
    is_flag=True,
    help=(
        "Warm by driving the real model's forward_training + backward at the "
        "production shapes (uses --config's max_tokens / batch_size_per_rank / "
        "diffusion params). Guarantees the cached shapes exactly match training "
        "— no synthetic-shape-list drift. Default buckets = [max_tokens]. "
        "Every repeated block stack is collapsed to depth 1 (shape coverage is "
        "identical — see --full-depth to disable)."
    ),
)
@click.option(
    "--full-depth",
    is_flag=True,
    help=(
        "With --real-model, build the full-depth model instead of collapsing "
        "every block stack to a single layer. The Triton cache is shape-keyed "
        "so depth-1 warms the identical entries; this flag is only an escape "
        "hatch and costs ~N× more memory/time for no extra coverage."
    ),
)
@click.option(
    "--msa-depth",
    type=int,
    default=None,
    help=(
        "MSA rows for --real-model synthetic features. Defaults to the config's "
        "model.msa_crop_size — the fixed pool the trunk subsamples to — so the "
        "MSA-track kernels warm at the depth training uses. Lower it if the "
        "warmup forward OOMs."
    ),
)
@click.option("--device", type=str, default="cuda:0", help="CUDA device to warm on.")
@click.option(
    "--override",
    multiple=True,
    help=(
        "Override config: key=value, applied on top of --config. Bare keys go to "
        "the training section (e.g. --override batch_size_per_rank=1 --override "
        "diffusion_chunk_size=16 --override max_recycles=3); dotted keys address "
        "nested config (e.g. --override model.msa_crop_size=2048). For --real-model "
        "timing/coverage sweeps. An explicit max_recycles here is honored uncapped "
        "(the warming default caps recycles at 2 since they add no new shapes)."
    ),
)
def main(
    config_path: str | None,
    buckets: str | None,
    batch_sizes: str,
    skip_gemm: bool,
    real_model: bool,
    full_depth: bool,
    msa_depth: int | None,
    device: str,
    override: tuple[str, ...],
) -> None:
    """Pre-populate the persistent Triton autotune cache for our bucket sizes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA not available — warmup needs a GPU.")
    # Pin the current CUDA device so Triton launches kernels on the same device
    # as the model/features (see pin_cuda_device — non-default --device otherwise
    # trips "cannot be accessed from Triton (cpu tensor?)").
    dev = pin_cuda_device(device)
    name = torch.cuda.get_device_name(dev)
    mcfg, chunk_size, train_cfg, explicit_training = _load_configs(config_path, override)

    if real_model:
        if full_depth:
            logger.info("Full-depth real-model warmup (no stack collapse)")
        else:
            changes = collapse_to_depth_1(mcfg)
            logger.info("Shallow warmup — collapsing block stacks to depth 1 (shape coverage identical):")
            for field_name, (orig, new) in changes.items():
                logger.info(f"  {field_name}: {orig} → {new}")
        max_tokens = int(train_cfg.get("max_tokens", 384))
        # Training fixes N = max_tokens (collate fixed_bucket_n), so default to
        # warming exactly that bucket; --buckets overrides for larger crops.
        bucket_list = (
            sorted({int(x) for x in buckets.split(",") if x.strip()}) if buckets else [max_tokens]
        )
        batch = int(train_cfg.get("batch_size_per_rank", 2))
        # Match training's PER-CHUNK diffusion size, not the full batch. Training
        # splits diffusion_batch_size into diffusion_chunk_size-sized passes, so
        # the diffusion kernels only ever see chunk_size samples per call. Warming
        # the full unchunked batch both OOMs and warms the wrong shape.
        full_diff = int(train_cfg.get("diffusion_batch_size", 8))
        diff_bs = min(chunk_size, full_diff) if chunk_size and chunk_size > 0 else full_diff
        # Recycles don't add new (kernel, shape) entries (they just repeat the
        # trunk), so the warming default caps at 2 for speed. An explicit
        # --override max_recycles=N is honored uncapped — useful for timing.
        cfg_recycles = int(train_cfg.get("max_recycles", 2))
        recycles = cfg_recycles if "max_recycles" in explicit_training else min(2, cfg_recycles)
        compute_conf = float(train_cfg.get("alpha_confidence", 0.0)) > 0
        # Default MSA depth = the fixed pool the trunk subsamples to (model.py).
        n_msa = msa_depth if msa_depth is not None else int(getattr(mcfg, "msa_crop_size", 1024) or 1024)
        logger.info(f"Real-model warmup on {name} for buckets: {bucket_list} (S_msa={n_msa})")
        t0 = time.time()
        warmup_real_model(mcfg, bucket_list, batch, n_msa, diff_bs, recycles, compute_conf, dev)
        logger.info(f"Total warmup time: {time.time() - t0:.1f}s")
        return

    if buckets is None:
        bucket_list = [b for b in BUCKET_SIZES if b <= 384]
    else:
        bucket_list = sorted({int(x) for x in buckets.split(",") if x.strip()})
    batch_size_list = sorted({int(x) for x in batch_sizes.split(",") if x.strip()})

    shapes = WarmupShapes.from_model_config(mcfg)
    shapes.diffusion_chunk_size = chunk_size
    logger.info(f"Warming Triton cache on {name} for buckets: {bucket_list}")
    logger.info(f"Batch sizes: {batch_size_list}")
    logger.info(
        f"Channel dims: c_pair={shapes.c_pair} c_single={shapes.c_single} "
        f"c_msa={shapes.c_msa} c_diff_token={shapes.c_diff_token} "
        f"c_pair_cond={shapes.c_pair_cond} c_single_cond={shapes.c_single_cond} "
        f"diffusion_chunk_size={shapes.diffusion_chunk_size}"
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
