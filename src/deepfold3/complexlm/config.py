"""Configuration for the complex language model.

Everything that could reasonably be swapped is here, including the whole
positional-architecture ablation, which reduces to one string
(:attr:`ComplexLMConfig.layer_pattern`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

#: Layer kinds. The pattern string cycles over these, so the architecture
#: ablation (A/B/C/D of the design note) is a config value, not a code path.
#:
#: ``c`` chain-scoped attention, RoPE on the per-chain residue index
#: ``g`` complex-scoped attention, no positional encoding      (main model, C)
#: ``r`` complex-scoped attention, RoPE on the concatenated complex index (A/B)
#: ``x`` cross-chain-only attention, no positional encoding    (D, secondary)
LAYER_KINDS = "cgrx"


@dataclass
class ComplexLMConfig:
    """Architecture.

    Defaults follow ESMC (bias-free projections, QK layer norm, SwiGLU at
    expansion 8/3 rounded to a multiple of 256, residual scaling by
    ``sqrt(n_layers / 36)``) with two deliberate departures: RMSNorm instead of
    LayerNorm, and no BOS/EOS. A complex is a *set* of chains; a sentinel at
    the front of one serialisation would make the first chain special, which is
    the asymmetry the architecture exists to remove.

    Attributes:
        layer_pattern: Cycled over :data:`LAYER_KINDS`. ``"cg"`` is the main
            model (1 chain-RoPE : 1 global-NoPE); ``"ccg"`` is 2:1; ``"r"`` is
            the global-RoPE baseline; ``"cx"`` is cross-chain-only.
        attn_impl: ``"varlen"`` is the packed FlashAttention path used for
            training. ``"sdpa"`` builds dense masks — the reference for tests,
            the only path that can return attention weights (so the
            attention-sink diagnostic runs there), and the only one that
            supports kind ``x``.
        max_position: Size of the rotary tables. Under ``c`` this bounds the
            longest *chain*, not the longest complex — the point of restricting
            RoPE to chains.
        mup: Maximal-update parameterisation. Off by default; when on, init
            scales, per-group Adam learning rates and the attention logit
            scale all become functions of ``d_model / mup_base_width``, so a
            learning rate tuned on a narrow proxy transfers to a wide model
            without a sweep at the target size. Transfer is valid across
            *width at fixed depth* — a proxy ladder must therefore keep
            ``n_layers`` at the target's value and vary ``d_model`` only.
        mup_base_width: The proxy's ``d_model``. At ``d_model ==
            mup_base_width`` every scale factor is 1, i.e. muP reduces exactly
            to the standard parameterisation, so the two are comparable at the
            base width.
        init_std: Base initialisation standard deviation, before muP scaling.
        chain_signal: Off by default. ``"random"`` adds a gated random vector
            per chain to the embeddings. It exists because of a provable blind
            spot in the main model: under ``g``, softmax over a duplicated key
            set equals softmax over one copy, so an *exact* homomer produces
            token-for-token the same output as its monomer, at any copy number
            (``tests/test_complexlm.py::test_exact_homomer_collapses_to_monomer``).
            A per-chain random vector breaks that without imposing an order —
            the signal is exchangeable across chains, so permutation
            equivariance survives in distribution.
    """

    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 24
    layer_pattern: str = "cg"
    expansion_ratio: float = 8 / 3
    norm: str = "rmsnorm"  # rmsnorm | layernorm
    norm_eps: float = 1e-5
    qk_layernorm: bool = True
    bias: bool = False
    scale_residue: bool = True
    rope_theta: float = 10000.0
    max_position: int = 4096
    vocab_size: int = 33
    attn_impl: str = "varlen"  # varlen | sdpa
    embed_dropout: float = 0.0
    chain_signal: str = "none"  # none | random
    mup: bool = False
    mup_base_width: int = 256
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        bad = set(self.layer_pattern) - set(LAYER_KINDS)
        if bad or not self.layer_pattern:
            raise ValueError(f"layer_pattern {self.layer_pattern!r} must be a non-empty string over {LAYER_KINDS!r}")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.attn_impl not in ("varlen", "sdpa"):
            raise ValueError(f"unknown attn_impl {self.attn_impl!r}")
        if self.chain_signal not in ("none", "random"):
            raise ValueError(f"unknown chain_signal {self.chain_signal!r}")

    @property
    def kinds(self) -> list[str]:
        """Layer kind per layer, the pattern cycled to ``n_layers``."""
        return [self.layer_pattern[i % len(self.layer_pattern)] for i in range(self.n_layers)]

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def width_mult(self) -> float:
        """``d_model / mup_base_width``; 1.0 when muP is off."""
        return self.d_model / self.mup_base_width if self.mup else 1.0

    @property
    def base_head_dim(self) -> int:
        """Head dimension of the proxy, assuming ``n_heads`` is held fixed."""
        return max(self.mup_base_width // self.n_heads, 1)


@dataclass
class SourceConfig:
    """One dataset in the mixture.

    Attributes:
        name: Store directory name under ``data_root``.
        weight: Unnormalised probability of drawing this source, ``p(D)``.
        homo_weight: Relative weight of homomeric complexes *within* this
            source, against ``hetero_weight``. ``None`` keeps the source's
            natural composition. AFDB-multimer is 96% homodimer, so sampling it
            naturally trains a homodimer model; this is the knob that fixes
            that, and ``p(T | D)`` of the design note is exactly this pair.
        hetero_weight: Relative weight of heteromeric complexes.
        cluster_balanced: Draw a cluster uniformly, then a member uniformly.
            Requires a store with cluster ids.
        interface_masking: Use the interface-biased mask rate. Ignored (with a
            warning) for a store that has no interface flags.
        date_max: Keep complexes released on or before this ``YYYYMMDD``. The
            PDB train/validation split is temporal — the same complex is
            re-deposited for years at better resolution, so a sequence split
            leaks — which makes this the single most important filter on any
            PDB-derived source.
        date_min: Keep complexes released strictly after this ``YYYYMMDD``.
        allow_undated: Whether a complex with no release date survives a date
            filter. Default false, i.e. fail closed: an undated PDB entry is
            indistinguishable from a post-cutoff one, and guessing in the
            permissive direction is how a cutoff silently stops holding.
        exclude_ids: Path to a file of complex ids (one per line) to drop —
            for excluding another benchmark's evaluation set.
        holdout_every: Reserve every ``k``-th complex for evaluation. A stopgap
            for the sources that carry no date (AFDB, Teddymer): an index
            holdout is honest about *examples* but not about *sequences*, since
            a near-duplicate can sit on both sides. Replace it with a cluster
            holdout once the sequences are clustered.
        holdout_side: ``"train"`` keeps everything except the held-out slice,
            ``"eval"`` keeps only it.
    """

    name: str
    weight: float = 1.0
    homo_weight: float | None = None
    hetero_weight: float | None = None
    cluster_balanced: bool = True
    interface_masking: bool = True
    max_complexes: int | None = None
    date_max: int | None = None
    date_min: int | None = None
    allow_undated: bool = False
    exclude_ids: str | None = None
    holdout_every: int = 0
    holdout_side: str = "train"


@dataclass
class MaskingConfig:
    """MLM masking policy.

    Defaults are PPLM's: 15% of residues, 30% at interfaces of structural
    complexes, and the BERT 80/10/10 split of masked positions into
    ``<mask>`` / random / unchanged.

    Attributes:
        homomer_paired: Mask the same positions in every copy of an entity.
            Without it, a homodimer's second copy is a verbatim answer key and
            the task collapses to copying; PPLM masks identical positions
            jointly for the same reason.
    """

    rate: float = 0.15
    interface_rate: float = 0.30
    mask_token_frac: float = 0.8
    random_token_frac: float = 0.1
    homomer_paired: bool = True


@dataclass
class DataConfig:
    """Sampling, cropping and packing.

    Attributes:
        token_budget: Residues per packed micro-batch per rank. Packing is by
            token count, not by example count, so a batch of long complexes and
            a batch of short ones cost the same.
        max_chain_len: Longest chain kept; longer chains are cropped. This is
            what the rotary tables must cover.
        max_complex_len: Longest complex kept; larger ones lose chains, then
            residues, biased to keep interfaces.
        permute_chains: Shuffle chain order per sample. Chain order in a
            complex is arbitrary; the model should not learn one.
    """

    data_root: str = "/data/plm/store"
    sources: list[SourceConfig] = field(default_factory=list)
    token_budget: int = 16384
    max_chain_len: int = 2048
    max_complex_len: int = 4096
    max_chains: int = 16
    min_chains: int = 1
    permute_chains: bool = True
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    num_workers: int = 6
    prefetch_factor: int = 4
    seed: int = 0


@dataclass
class TrainConfig:
    """Optimisation and logging."""

    max_steps: int = 100_000
    lr: float = 4e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    grad_clip: float = 1.0
    accum_steps: int = 1
    dtype: str = "bf16"
    compile: bool = False
    log_every: int = 20
    eval_every: int = 2000
    eval_batches: int = 20
    #: Checkpoint cadence. Must be a multiple of ``eval_every``: a checkpoint
    #: on a step that was not evaluated cannot be tied to any metric, and an
    #: eval whose state was never saved cannot be revisited.
    save_every: int = 5000
    #: Drop evaluation complexes whose cluster also occurs in training. A date
    #: cutoff alone is not a clean holdout — measured on PDB, 51.3% of
    #: post-2021-09-30 complexes belong to a family that appears before it.
    eval_cluster_disjoint: bool = True
    keep_last: int = 3
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    diagnostics_every: int = 0  # 0 = only at eval time
    seed: int = 0

    def __post_init__(self) -> None:
        if self.save_every and self.eval_every and self.save_every % self.eval_every:
            raise ValueError(
                f"save_every={self.save_every} must be a multiple of eval_every={self.eval_every} "
                "so that every checkpoint sits on an evaluated step"
            )


@dataclass
class ExperimentConfig:
    model: ComplexLMConfig = field(default_factory=ComplexLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: Evaluation sources. Empty means "the training mixture", which is only
    #: right for a sanity check — the real split is temporal and lives in each
    #: source's own ``date_min`` / ``date_max``.
    eval_sources: list[SourceConfig] = field(default_factory=list)


def _from_dict(cls: type, raw: dict) -> Any:
    """Build a dataclass from a dict, recursing into dataclass-typed fields.

    Field types are resolved with ``get_type_hints`` rather than read off
    ``Field.type``: this module uses postponed annotations, so ``Field.type``
    is the *string* ``"ComplexLMConfig"`` and an ``is_dataclass`` check on it
    silently fails, leaving nested sections as plain dicts.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"{cls.__name__}: unknown config key {key!r}")
        ftype = hints.get(key, known[key].type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        elif key in ("sources", "eval_sources") and isinstance(value, list):
            kwargs[key] = [_from_dict(SourceConfig, v) for v in value]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_experiment(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an experiment config, applying ``a.b.c=value`` overrides."""
    from deepfold3.config_loader import resolve_config

    raw = resolve_config(path) if path else {}
    cfg = _from_dict(ExperimentConfig, raw)
    for dotted, value in (overrides or {}).items():
        node: Any = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        current = getattr(node, parts[-1])
        setattr(node, parts[-1], _coerce(value, current))
    # Re-validate: an override can violate an invariant the constructor checked.
    cfg.model.__post_init__()
    cfg.train.__post_init__()
    return cfg


def _coerce(value: Any, like: Any) -> Any:
    if isinstance(value, str) and not isinstance(like, str):
        if isinstance(like, bool):
            return value.lower() in ("1", "true", "yes")
        if isinstance(like, int):
            return int(value)
        if isinstance(like, float):
            return float(value)
        if like is None:
            return value
    return value


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)
