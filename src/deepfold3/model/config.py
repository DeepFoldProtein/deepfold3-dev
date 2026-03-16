"""Model configuration dataclasses for DeepFold3."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AtomTransformerConfig:
    num_blocks: int = 3
    num_intermediate_factor: int = 2
    num_head: int = 4
    key_dim: int = 128
    value_dim: int = 128


@dataclass
class AtomCrossAttConfig:
    per_token_channels: int = 768  # diffusion transformer channel dim
    per_atom_channels: int = 128
    per_atom_pair_channels: int = 16
    atom_transformer: AtomTransformerConfig = field(default_factory=AtomTransformerConfig)
    queries_subset_size: int = 32  # Q: atoms per query subset (non-overlapping)
    keys_subset_size: int = 128  # K: atoms per key window (overlapping, centered on Q)
    num_element_classes: int = 128  # element one-hot vocabulary size
    num_atom_name_chars: int = 64  # atom name character vocabulary size
    atom_name_length: int = 4  # characters per atom name


@dataclass
class MSAAttentionConfig:
    num_head: int = 8


@dataclass
class GridAttentionConfig:
    num_head: int = 4


@dataclass
class EvoformerConfig:
    num_layer: int = 4
    msa_channel: int = 64
    seq_channel: int = 384
    pair_channel: int = 128
    max_relative_idx: int = 32
    max_relative_chain: int = 2
    msa_attention: MSAAttentionConfig = field(default_factory=MSAAttentionConfig)
    outer_product_num_channel: int = 32
    pair_attention: GridAttentionConfig = field(default_factory=GridAttentionConfig)


@dataclass
class PairFormerConfig:
    num_layer: int = 48
    pair_channel: int = 128
    seq_channel: int = 384
    pair_attention: GridAttentionConfig = field(default_factory=GridAttentionConfig)
    single_attention_num_head: int = 16
    single_key_dim: int = 384
    single_value_dim: int = 384


@dataclass
class DiffusionTransformerConfig:
    num_blocks: int = 24
    num_head: int = 16
    key_dim: int | None = None  # None → use num_channels (c_diff = 768)
    value_dim: int | None = None  # None → use num_channels
    num_intermediate_factor: int = 2
    super_block_size: int = 4


@dataclass
class DiffusionConfig:
    conditioning_pair_channel: int = 128
    conditioning_seq_channel: int = 384
    fourier_dim: int = 256  # Fourier noise embedding dimension
    conditioning_num_intermediate_factor: int = 2  # transition block expansion factor
    # Sampling
    steps: int = 200
    num_samples: int = 5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    # Noise schedule
    sigma_data: float = 16.0
    smin: float = 0.0004
    smax: float = 160.0
    noise_p: float = 7.0
    # Training noise sampling (log-normal): t = sigma_data * exp(p_mean + p_std * z), z ~ N(0,1)
    p_mean: float = -1.2
    p_std: float = 1.5
    transformer: DiffusionTransformerConfig = field(default_factory=DiffusionTransformerConfig)
    atom_cross_att: AtomCrossAttConfig = field(default_factory=AtomCrossAttConfig)


@dataclass
class ConfidenceConfig:
    num_layer: int = 4
    pair_channel: int = 128
    seq_channel: int = 384
    num_plddt_bins: int = 50
    num_distogram_bins: int = 64
    max_error_bin: float = 31.0
    max_atoms_per_token: int = 24  # AF3 dense atom representation max
    pair_attention_num_head: int = 4
    single_attention_num_head: int = 16


@dataclass
class TemplatePairFormerConfig:
    """Config for the simplified PairFormer stack inside template embedding.

    Uses pair-only layers (no single track) with a smaller transition factor.
    """

    num_layer: int = 2
    num_channels: int = 64
    num_heads: int = 4
    num_intermediate_factor: int = 2


@dataclass
class TemplateEmbeddingConfig:
    """Config for template embedding (structural templates → pair embedding).

    Haiku scope: diffuser/evoformer/template_embedding/

    Attributes:
        num_channels: Hidden dimension for template pair features.
        pair_channel: Pair channel from the Evoformer (query embedding dim).
        dgram_num_bins: Number of distance histogram bins.
        dgram_min_bin: Left edge of the first distance bin (Angstroms).
        dgram_max_bin: Left edge of the last distance bin (Angstroms).
        aatype_num_classes: Number of amino acid type classes for one-hot encoding.
        pairformer: Config for the template PairFormer stack.
    """

    num_channels: int = 64
    pair_channel: int = 128
    dgram_num_bins: int = 39
    dgram_min_bin: float = 3.25
    dgram_max_bin: float = 50.75
    aatype_num_classes: int = 31
    pairformer: TemplatePairFormerConfig = field(default_factory=TemplatePairFormerConfig)


@dataclass
class OptimConfig:
    """Toggle individual kernel optimization backends.

    All flags default to False (pure PyTorch). Set to True to use the
    corresponding Triton/fused kernel if available; silently falls back
    to baseline if the backend is not installed.

    Benchmark results (RTX 5090, bf16, N=384-768):
    ──────────────────────────────────────────────────────────────────
    ATTENTION (104 GridSelfAttention modules per forward):
    - use_fused_evo_attention: 2.9-3.3x faster (autotuned Triton flash-attn with pair bias)
    - use_fused_pair_bias: +30% on top of evo_attn (fused LN + tl.dot projection + permute)
      Now supports training (autograd backward via PyTorch ops) + inference (fully fused).
    - Combined evo+pair_bias: 3.7-4.6x faster vs pytorch baseline
    - use_torch_compile: 1.3-2.3x for MSAAttention/SingleAttention (SDPA backend)
    Note: Triton EvoAttention is NOT used for SingleAttention (N_seq=1 overhead dominates).

    TRIANGLE MULTIPLY (96 TriangleMul modules per forward):
    - use_fused_triangle_mul: enables three fusion levels:
      Level 3 (dual GEMM): 1.24-1.32x fwd, 1.18-1.20x full
        Supports TWO_INPUTS, bias, mask fusion, TF32 precision control.
      Level 2 (center proj): fused LN+Linear for post-einsum norm_center+out_proj
      Level 1 (gate): fallback fused LN+packed GEMM+sigmoid postprocess
    - MEMORY: triton_l3 eliminates the Incoming/Outgoing memory asymmetry
      (cuBLAS workspace causes Incoming to use 1.2-1.5x more memory without fusion;
      triton_l3 reduces diff to ~0).
    - Einsum replaced with torch.matmul on permuted views (zero-copy) for both variants.

    TRANSITION (context-sensitive — M size determines whether Triton wins):
    - use_fused_transition_pair: 1.2-1.3x for pair transitions (M=N², C=128) where M is large
    - use_fused_transition_single: SLOWER at M=N, C=384 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ada_transition: SLOWER at M=N, C=768 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ln_transpose: 1.7-2.3x fwd for LN+transpose (column attn, inference only)
    """

    use_fused_evo_attention: bool = False
    use_fused_pair_bias: bool = False
    use_fused_transition_pair: bool = False  # pair transitions (M=N², C=128) — Triton wins
    use_fused_transition_single: bool = False  # single transitions (M=N, C=384) — PyTorch wins, DO NOT enable
    use_fused_ada_transition: bool = False  # diffusion AdaTransitionBlock (M=N, C=768) — PyTorch wins
    use_fused_triangle_mul: bool = False
    use_fused_ln_transpose: bool = False
    use_torch_compile: bool = False

    # Substrings that identify derived fused-kernel parameters in state dicts.
    # These keys are excluded from checkpoints (they are re-synced from base
    # weights on load) and from EMA shadow copies.
    FUSED_STATE_DICT_MARKERS: tuple[str, ...] = (
        "_fused_gate.",
        "_fused_center_proj.",
        "_fused_dual_gemm.",
        "_fused_pair_bias.",
        "_fused_ada_ln_linear.",
        "_fused_swiglu.",
        "_ln_linear.",
        "_ln_transpose.",
        "_swiglu.",
    )

    @classmethod
    def recommended(cls) -> OptimConfig:
        """Best overall: evo_attn + pair_bias + triangle_mul + pair_transition + compile.

        Only enables Triton kernels where they demonstrably outperform PyTorch.
        Single/diffusion transitions are left to PyTorch/compile (they're faster at small M).
        """
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_triangle_mul=True,
            use_fused_transition_pair=True,
            use_torch_compile=True,
        )

    @classmethod
    def inference(cls) -> OptimConfig:
        """Maximum inference speed (all beneficial Triton + forward-only kernels)."""
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_triangle_mul=True,
            use_fused_transition_pair=True,
            use_fused_ln_transpose=True,
        )

    @classmethod
    def triton_only(cls) -> OptimConfig:
        """All beneficial Triton kernels (no torch.compile)."""
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_transition_pair=True,
            use_fused_triangle_mul=True,
            use_fused_ln_transpose=True,
        )

    @classmethod
    def all_enabled(cls) -> OptimConfig:
        """Enable ALL optimizations including slow ones (for testing)."""
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_transition_pair=True,
            use_fused_transition_single=True,
            use_fused_ada_transition=True,
            use_fused_triangle_mul=True,
            use_fused_ln_transpose=True,
            use_torch_compile=True,
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    All variable training settings are collected here; nothing is hardcoded
    in the trainer or loss modules.  Override via JSON config or CLI flags.

    Attributes:
        lr: Peak learning rate for Adam.
        weight_decay: Weight decay (applied to >= 2-D params only).
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        warmup_steps: Linear LR warmup duration.
        start_decay_after: Step at which LR decay begins.
        decay_every: Steps between each LR decay step.
        decay_factor: Multiplicative LR decay factor.
        grad_clip_norm: Max L2 gradient norm (0 disables).
        accumulate_grad_batches: Gradient accumulation steps.
        ema_decay: EMA decay factor (negative disables EMA).
        max_tokens: Maximum tokens per crop.
        max_atoms: Maximum atoms per crop.
        max_msa_seqs: Maximum MSA sequences.
        max_recycles: Upper bound for random recycle count per batch.
        diffusion_batch_size: Number of noise samples per training step.
        mini_rollout_steps: Denoising steps for the confidence mini-rollout.
        mini_rollout_samples: Diffusion samples in the mini-rollout.
        alpha_diffusion: Diffusion loss weight.
        alpha_distogram: Distogram loss weight.
        alpha_confidence: Confidence loss weight.
        alpha_pae: PAE loss multiplier (0 in stage 1, 1 in finetuning).
        alpha_bond: Bond loss multiplier (0 in stage 1).
        lddt_cutoffs: Smooth LDDT distance error thresholds (Angstroms).
        lddt_inclusion_radius: Smooth LDDT pair inclusion radius (Angstroms).
        min_resolution: Minimum resolution for confidence loss gating (Angstroms).
        max_resolution: Maximum resolution for confidence loss gating (Angstroms).
        dtype: Mixed-precision dtype (``"bf16"``, ``"fp16"``, or ``"fp32"``).
        find_unused_parameters: DDP ``find_unused_parameters`` flag.
        static_graph: DDP ``static_graph`` flag.
        max_steps: Total effective training steps.
        seed: Base random seed (offset by rank for DDP).
        checkpoint_interval: Save a checkpoint every N effective steps.
        eval_interval: Run evaluation every N effective steps.
        log_interval: Log metrics every N effective steps.
        output_dir: Output directory for checkpoints and logs.
        data_dir: Root directory containing training data.
        params_path: Path to pretrained weights (safetensors or .pt).
    """

    # Optimizer (Adam)
    lr: float = 0.0018
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # LR schedule (AF3-style: warmup, plateau, step decay)
    warmup_steps: int = 1000
    start_decay_after: int = 50000
    decay_every: int = 50000
    decay_factor: float = 0.95

    # Gradient
    grad_clip_norm: float = 10.0
    accumulate_grad_batches: int = 1

    # EMA
    ema_decay: float = 0.999

    # Data
    max_tokens: int = 384
    max_atoms: int = 4096
    max_msa_seqs: int = 16384

    # Recycling (random 1..max_recycles each batch)
    max_recycles: int = 4

    # Diffusion training
    diffusion_batch_size: int = 48
    mini_rollout_steps: int = 20
    mini_rollout_samples: int = 1

    # Loss weights
    alpha_diffusion: float = 4.0
    alpha_distogram: float = 0.03
    alpha_confidence: float = 1e-4
    alpha_pae: float = 0.0
    alpha_bond: float = 0.0

    # Smooth LDDT thresholds
    lddt_cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    lddt_inclusion_radius: float = 30.0
    lddt_min_distance: float = 1e-3  # minimum GT distance for LDDT pair inclusion

    # Confidence resolution gating
    min_resolution: float = 0.1
    max_resolution: float = 4.0

    # Mixed precision
    dtype: str = "bf16"

    # DataLoader
    num_workers: int = 4
    eval_num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True

    # DDP
    find_unused_parameters: bool = False
    static_graph: bool = True

    # Run control
    max_steps: int = 200000
    seed: int = 42
    checkpoint_interval: int = 1000
    eval_interval: int = 5000
    log_interval: int = 50

    # Paths (set at runtime)
    output_dir: str = "output"
    data_dir: str = ""
    params_path: str = ""

    def __post_init__(self):
        """Validate configuration values."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.start_decay_after < self.warmup_steps:
            raise ValueError(
                f"start_decay_after ({self.start_decay_after}) must be >= warmup_steps ({self.warmup_steps})"
            )
        if self.decay_every <= 0:
            raise ValueError(f"decay_every must be > 0, got {self.decay_every}")
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"dtype must be 'bf16', 'fp16', or 'fp32', got '{self.dtype}'")
        if self.accumulate_grad_batches < 1:
            raise ValueError(f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}")
        if self.max_recycles < 1:
            raise ValueError(f"max_recycles must be >= 1, got {self.max_recycles}")

    @property
    def torch_dtype(self):
        """Returns the ``torch.dtype`` corresponding to ``self.dtype``."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


@dataclass
class DistogramConfig:
    num_bins: int = 64
    first_break: float = 2.3125
    last_break: float = 21.6875


@dataclass
class DataConfig:
    max_templates: int = 4
    max_neighborhood: int = 40
    spatial_crop_prob: float = 0.5


@dataclass
class ModelConfig:
    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    pairformer: PairFormerConfig = field(default_factory=PairFormerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    template: TemplateEmbeddingConfig = field(default_factory=TemplateEmbeddingConfig)
    distogram: DistogramConfig = field(default_factory=DistogramConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    num_recycles: int = 10
    msa_crop_size: int = 1024  # MSA rows sampled per recycle
