"""Model configuration dataclasses for DeepFold3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25


@dataclass
class PairFormerConfig:
    num_layer: int = 48
    pair_channel: int = 128
    seq_channel: int = 384
    pair_attention: GridAttentionConfig = field(default_factory=GridAttentionConfig)
    single_attention_num_head: int = 16
    single_key_dim: int = 384
    single_value_dim: int = 384
    pair_dropout: float = 0.25


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
    # EDM diffusion-MSE weight w = (σ² + σ_d²)/(σ·σ_d)² → 1/σ² as σ→0, so a
    # rare small-σ draw in the log-normal's left tail can concentrate the
    # gradient in one sample and spike grad_norm by orders of magnitude.
    # Clamp σ from below during training sampling to cap w ≤ 1/σ_min² + 1/σ_d².
    # σ_min = 0.1 gives w ≲ 100; tighten/loosen via override.
    clamp_train_sigma: bool = False
    train_sigma_min: float = 0.1
    # Disable autocast around the diffusion sampler (Protenix ``skip_amp``
    # pattern, `configs_base.py:137-145`). When True, the method upcasts
    # all float tensor inputs to fp32 and runs with autocast disabled, so
    # the full denoising rollout executes in fp32. Warning: fused Triton
    # kernels in the diffusion head (``use_fused_ada_transition``, etc.)
    # assume bf16 inputs — disable those when enabling skip-amp.
    skip_amp_sample: bool = False  # inference Euler-Maruyama (sample())
    skip_amp_sample_training: bool = False  # training rollout (sample_training())
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
    pair_dropout: float = 0.25


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
      Supports training (autograd backward) + inference (fully fused).
    - Combined evo+pair_bias: 3.7-4.6x faster vs pytorch baseline
    - use_torch_compile: 1.3-2.3x for MSAAttention/SingleAttention (SDPA backend)
    Note: Triton EvoAttention is NOT used for SingleAttention (N_seq=1 overhead dominates).

    TRIANGLE MULTIPLY (96 TriangleMul modules per forward):
    - use_fused_triangle_mul: enables three fusion levels:
      Level 3 (dual GEMM): 1.24-1.32x fwd, 1.18-1.20x full
        Supports TWO_INPUTS, bias, mask fusion, TF32 precision control.
      Level 2 (center proj): fused LN+Linear for post-einsum norm_center+out_proj
      Level 1 (gate): fallback fused LN+packed GEMM+sigmoid postprocess

    SWIGLU (inside TransitionBlock / AdaTransitionBlock):
    - use_fused_swiglu: 1.4-1.6x fwd, 1.9-2.1x full at pair scale (M=N², C=128)
      Multi-row Triton kernel (128 rows/block) + in-place backward → 2-6x less memory.
      Neutral at single scale (M=N) where launch overhead dominates — still safe to enable
      because the memory saving is always beneficial.

    TRANSITION (context-sensitive — M size determines whether Triton wins):
    - use_fused_transition_pair: 1.2-1.3x for pair transitions (M=N², C=128) where M is large
    - use_fused_transition_single: SLOWER at M=N, C=384 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ada_transition: SLOWER at M=N, C=768 — DO NOT enable (PyTorch 0.8x)
    - use_fused_ln_transpose: 1.7-2.3x fwd for LN+transpose (column attn, inference only)
    """

    use_fused_evo_attention: bool = False
    use_fused_pair_bias: bool = False
    use_fused_swiglu: bool = False  # standalone FusedSwiGLU — wins at M=N² (pair transitions)
    use_fused_transition_pair: bool = False  # pair transitions (M=N², C=128) — Triton wins
    use_fused_transition_single: bool = False  # single transitions (M=N, C=384) — PyTorch wins, DO NOT enable
    use_fused_ada_transition: bool = False  # diffusion AdaTransitionBlock (M=N, C=768) — PyTorch wins
    use_fused_triangle_mul: bool = False
    use_fused_ln_transpose: bool = False
    # dropout_add fuses per-batch row/column-wise dropout with a residual add and
    # samples the mask on the fly via tl.rand (see model/triton/dropout_add.py).
    # Default True preserves current behavior (Triton always active on CUDA);
    # set False to force the unfused PyTorch fallback for A/B diagnostics.
    use_fused_dropout_add: bool = True
    use_torch_compile: bool = False
    # Tiled linear: chunk pair transitions along dim=-2, recompute during backward
    # Saves ~67% of pair transition activation memory at cost of ~2x FLOPs for those layers.
    # Also supports inference tiling for very long sequences (peak memory reduction).
    use_tiled_linear: bool = False  # tiled recomputation for pair-scale transitions
    tiled_linear_shards: int = 0  # 0 = auto heuristic (ceil(spatial/channels))
    tiled_linear_inference: bool = False  # also tile during inference (for long sequences)
    # Tiled outer product mean: chunk rows to avoid [N,N,32,32] intermediate.
    # Saves ~3.6 GB at N=768. 0 = disabled, >0 = chunk size (128 recommended).
    tiled_opm_chunk: int = 0

    # Gradient checkpointing on the 4-layer Evoformer (MSA) stack and the
    # 48-layer PairFormer trunk. Essential for training — without it, 48 layers
    # of [N,N,128] pair activations accumulate and peak memory easily exceeds
    # 100+ GB at N=384. Each block is individually recomputed in the backward
    # pass. Adds ~30% runtime overhead but drops activations from O(L) to O(1).
    # No effect on inference (only applied when grad is enabled).
    use_gradient_checkpointing: bool = False

    # Legacy fused-kernel parameter substrings. Current fused wrappers are
    # stateless (no owned parameters), but older checkpoints may contain
    # these keys; they are filtered out of state-dict loads and EMA shadow.
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
        """Best overall: evo_attn + pair_bias + swiglu + triangle_mul + pair_transition + compile.

        Only enables Triton kernels where they demonstrably outperform PyTorch.
        SwiGLU is always safe (wins at pair scale, neutral+memory-saving at single scale).
        Single/diffusion transitions are left to PyTorch/compile (they're faster at small M).
        """
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_swiglu=True,
            use_fused_triangle_mul=True,
            use_fused_transition_pair=True,
            use_torch_compile=True,
        )

    @classmethod
    def recommended_tiled(cls) -> OptimConfig:
        """Recommended + tiled linear for pair transitions (memory savings).

        Same as recommended() plus tiled chunking for pair-scale transitions.
        Reduces peak memory for pair transition intermediates [N,N,C*4] by
        processing chunks of rows sequentially instead of all at once.
        Applies to both training (recompute backward) and inference (chunked forward).

        torch.compile is disabled: with Triton kernels dominating runtime,
        compile adds no steady-state benefit but doubles peak memory during
        the compilation pass (~14 GB vs ~8.7 GB at N=1280).
        """
        cfg = cls.recommended()
        cfg.use_torch_compile = False
        cfg.use_tiled_linear = True
        cfg.tiled_linear_inference = True
        cfg.tiled_opm_chunk = 128
        return cfg

    @classmethod
    def inference(cls) -> OptimConfig:
        """Maximum inference speed + memory efficiency.

        All beneficial Triton kernels + tiled pair transitions + forward-only
        LN+transpose fusion. No torch.compile (Triton kernels dominate; compile
        adds startup cost with no steady-state benefit).
        """
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_swiglu=True,
            use_fused_triangle_mul=True,
            use_fused_transition_pair=True,
            use_fused_ln_transpose=True,
            use_tiled_linear=True,
            tiled_linear_inference=True,
            tiled_opm_chunk=128,
        )

    @classmethod
    def triton_only(cls) -> OptimConfig:
        """All beneficial Triton kernels without tiling (no torch.compile)."""
        return cls(
            use_fused_evo_attention=True,
            use_fused_pair_bias=True,
            use_fused_swiglu=True,
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
            use_fused_swiglu=True,
            use_fused_transition_pair=True,
            use_fused_transition_single=True,
            use_fused_ada_transition=True,
            use_fused_triangle_mul=True,
            use_fused_ln_transpose=True,
            use_torch_compile=True,
        )


@dataclass
class DatasetConfig:
    """Configuration for a training or evaluation dataset source.

    Attributes:
        name: Identifier for logging.
        data_dir: Path to data directory (structures/ + msa/).
        weight: Sampling probability relative to primary dataset (1.0).
        strategy: ``"weighted"`` (cluster-balanced) or ``"uniform"``.
        is_distillation: If True, disables confidence losses (PDB-only per AF3 §4.3).
            Diffusion and distogram losses still apply.
        format: ``"df3"`` (default) or ``"of3"`` (OpenFold 3 preprocessed).
        cutoff_date: ISO date filter. For training datasets: records released
            AFTER this date are excluded. For eval datasets: only records
            released AFTER this date are included.
    """

    name: str = ""
    data_dir: str = ""
    weight: float = 1.0
    strategy: str = "uniform"
    is_distillation: bool = False
    format: str = "df3"
    cutoff_date: str = ""


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
        deterministic: If True, force deterministic algorithms
            (``torch.use_deterministic_algorithms(True)``,
            ``cudnn.deterministic = True``, ``cudnn.benchmark = False``)
            for bit-exact reproducibility. Incurs a significant perf cost.
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
    # Raise RuntimeError after this many consecutive opt-step skips from
    # non-finite loss/grads. 0 or negative disables the check.
    max_consecutive_nan_skips: int = 5
    # Compute the grad-norm reduction in fp64 (upcast per-param L2² before
    # summing). Removes spurious non-finite grad-norms from fp32 reduction
    # overflow that differs across ranks by summation order. Default off
    # to preserve bit-exact parity with the historical ``clip_grad_norm_``
    # path; enable when training at high grad magnitudes.
    grad_norm_fp64: bool = False
    # When the observed (pre-clip) grad-norm exceeds this threshold, each
    # rank logs the record_ids that contributed to its accumulation window.
    # 0 disables. Useful for identifying examples that drive grad spikes
    # but stay below the NaN/Inf skip trigger.
    grad_norm_log_threshold: float = 0.0
    # If ANY rank's pre-clip grad-norm exceeds this value, skip the whole
    # optimizer step (per-rank outlier guard). Clipping to ``grad_clip_norm``
    # from a huge pre-clip norm produces a near-random direction that
    # degrades update quality and feeds magnitude drift; dropping the step
    # is cleaner than applying garbage. 0 disables.
    grad_norm_skip_threshold: float = 0.0

    # EMA
    ema_decay: float = 0.999

    # Data
    max_tokens: int = 384
    max_atoms: int = 4096
    max_msa_seqs: int = 16384

    # Recycling (random 0..max_recycles each microstep, AF3 convention).
    # Total trunk forwards = num_recycles + 1.
    # max_recycles=3 → up to 4 trunk passes (matches AF3 paper).
    max_recycles: int = 3

    # Diffusion training
    diffusion_batch_size: int = 48
    diffusion_chunk_size: int = 0  # 0 = no chunking; >0 = process S samples in chunks
    mini_rollout_steps: int = 20
    mini_rollout_samples: int = 1

    # Loss weights
    alpha_diffusion: float = 4.0
    alpha_distogram: float = 0.03
    alpha_confidence: float = 1e-4
    alpha_pae: float = 0.0
    alpha_bond: float = 0.0
    # Smooth LDDT weight. Set to None to disable entirely (AF3 Table 6:
    # smooth LDDT is turned off in fine-tune stage 1).
    alpha_smooth_lddt: float | None = 1.0

    # Single-representation magnitude regularization.  IntFold observed
    # activation explosions in later PairFormer transitions preceding the
    # gradient spikes from the diffusion loss backward; a mild penalty on
    # the single repr bounds magnitudes before they escape.
    #   ``single_reg_p``   — per-token norm (1 = Σ|s|/C, 2 = Σs²/C).
    #   ``alpha_single_reg`` — master switch + weight:
    #       * ``None`` → disabled entirely (no compute, no log).
    #       * ``0.0``  → log-only observation (computed under no_grad).
    #       * ``> 0``  → log + add ``alpha * L_p`` to total loss
    #                    (try ~1e-4 with p=2 to start regularizing).
    single_reg_p: int = 2
    alpha_single_reg: float | None = None
    # Same pattern for the pair representation (Pairformer transitions are
    # where IntFold observes the explosion *first*).
    pair_reg_p: int = 2
    alpha_pair_reg: float | None = None

    # Molecule-type loss reweighting (Protenix: DNA/RNA 5×, ligand 10×)
    nucleotide_loss_weight: float = 5.0
    ligand_loss_weight: float = 10.0

    # AF3 §3.7.1 eq. 2: weighted rigid alignment of GT onto denoised before
    # the diffusion MSE. Removes the global-rotation component of the error
    # signal (the augmented GT is only random-rotated; the denoiser's output
    # lives in a different frame when it's imperfect). Default True matches
    # AF3; disable for A/B against the DF3 pre-alignment baseline.
    use_rigid_align: bool = True

    # Region-based loss weighting (region_id → weight multiplier, e.g. {3: 5.0} for CDR-H3)
    region_loss_weights: dict[int, float] = field(default_factory=dict)

    # Smooth LDDT thresholds. AlphaFold 3 supplement eq. 6 / Protenix
    # ``loss.py``: nucleotide-nucleotide pairs use 30 Å, everything else
    # uses 15 Å.
    lddt_cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    lddt_inclusion_radius: float = 15.0
    lddt_nucleotide_inclusion_radius: float = 30.0
    lddt_min_distance: float = 1e-3  # minimum GT distance for LDDT pair inclusion

    # Confidence resolution gating
    min_resolution: float = 0.1
    max_resolution: float = 4.0

    # Fused loss (chunked linear + CE for pair-scale heads)
    use_fused_ce_loss: bool = False  # saves ~650 MB at N=768 (distogram + PAE + PDE)
    fused_ce_chunk_n: int = 32  # rows of N per chunk

    # Mixed precision
    dtype: str = "bf16"

    # DataLoader
    num_workers: int = 4
    eval_num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # DDP
    find_unused_parameters: bool = False
    static_graph: bool = True
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    # NCCL watchdog + ``collective_guard`` barrier deadline. Named sync
    # points in the trainer surface cross-rank desync as ``RuntimeError``
    # with the guard name, instead of an anonymous NCCL timeout.
    collective_timeout_s: float = 600.0

    # Run control
    max_steps: int = 200000
    seed: int = 42
    deterministic: bool = False  # enable torch.use_deterministic_algorithms + cudnn.deterministic
    checkpoint_interval: int = 1000
    eval_interval: int = 5000
    log_interval: int = 50  # in optimizer steps (after grad accumulation)
    log_microsteps: bool = False  # debug: log every microstep instead of every effective step
    uniform_recycle: bool = (
        True  # all DDP ranks use the same num_recycles per microstep (avoids straggler at no_grad warmups)
    )
    batch_size_per_rank: int = 1  # number of samples per rank per microstep (effective = world × batch × accumulate)
    epoch_size: int = 10000  # samples per synthetic epoch (sampler reseed cadence + DataLoader length)
    eval_before: bool = False  # run one eval pass before the training loop starts (also fires on resume)
    max_checkpoints: int = 5  # keep N most recent checkpoints (0 = keep all)
    checkpoint_every_hours: float = 1.0  # time-based checkpoint interval (0 = disabled)
    auto_resume: bool = False  # auto-detect latest checkpoint in output_dir

    # Multi-stage training (use --override or JSON config to set these per stage)
    train_confidence_only: bool = False  # freeze trunk, only train confidence head
    finetune_scheduler: str = "af3"  # "af3" or "cosine"
    cosine_min_lr_ratio: float = 0.1  # min LR = lr * ratio (for cosine scheduler)
    cosine_max_steps: int = 20000  # total steps for cosine annealing

    # Checkpoint loading
    load_strict: bool = True  # strict state_dict loading
    skip_load_optimizer: bool = False
    skip_load_scheduler: bool = False
    skip_load_step: bool = False

    # Experiment tracking
    wandb_enabled: bool = False  # explicit on/off; takes precedence over wandb_project
    wandb_project: str = ""
    wandb_entity: str = ""  # WandB team/org entity; empty → personal workspace
    # Optional explicit run name. ``None`` (or empty string) → derive from
    # ``output_dir`` and append a YYYYMMDD-HHMMSS timestamp so re-runs are
    # never accidentally collapsed under one wandb run.
    wandb_run_name: str | None = None

    # Evaluation
    eval_cutoff_date: str = ""  # ISO date: records released AFTER this → eval set (e.g. "2021-09-30")
    eval_max_date: str = ""  # ISO date: records released AFTER this are excluded from eval
    eval_samples: int = 128  # max records per eval pass (0 = use all; cluster-weighted when clusters available)
    eval_datasets: list[DatasetConfig] = field(default_factory=list)  # YAML eval_datasets section

    # Paths (set at runtime)
    output_dir: str = "output"
    data_dir: str = ""
    params_path: str = ""

    # Templates
    mmcif_dir: str = ""  # PDB mmCIF mirror path; non-empty enables template loading

    # Unified CCD cache (atoms + ref_pos + automorphisms) on the shared PVC.
    # Non-empty enables ligand atom-permutation loss via ``load_ligand_automorphisms``.
    ccd_cache_path: str = ""

    # Distillation dataset (optional, legacy — prefer extra_datasets)
    distillation_data_dir: str = ""  # empty = disabled
    distillation_weight: float = 0.25  # sampling weight relative to PDB (PDB gets 1.0)

    # Multi-dataset training: list of additional datasets beyond --data-dir.
    # Can be defined in YAML/JSON config under ``datasets`` top-level key
    # or via CLI ``--dataset name:path:weight:strategy:distill:format``.
    extra_datasets: list[DatasetConfig] = field(default_factory=list)

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
        if self.dtype not in _COMPUTE_DTYPE_CHOICES:
            raise ValueError(f"dtype must be one of {_COMPUTE_DTYPE_CHOICES}, got '{self.dtype}'")
        if self.accumulate_grad_batches < 1:
            raise ValueError(f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}")
        if self.max_recycles < 0:
            raise ValueError(f"max_recycles must be >= 0, got {self.max_recycles}")

    @property
    def torch_dtype(self):
        """Returns the ``torch.dtype`` corresponding to ``self.dtype``."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


_COMPUTE_DTYPE_CHOICES: tuple[str, ...] = ("bf16", "fp16", "fp32")


@dataclass
class DistogramConfig:
    num_bins: int = 64
    first_break: float = 2.3125
    last_break: float = 21.6875
    use_half_logits_ln: bool = False


@dataclass
class PrecisionConfig:
    """Mixed-precision policy — region guards + non-compute dtype axes.

    Two flavours of knob live here:

    * ``fp32_*`` region flags: disable autocast and upcast that specific
      region's inputs to fp32. Use to guard numerically sensitive
      projections (especially final logit heads) against bf16 overflow.
    * ``{grad_acc,reduce,loss}_dtype`` axes: the DS-inspired policy knobs
      for gradient accumulation / DDP all-reduce / loss computation.
      Compute dtype remains governed by ``TrainingConfig.dtype`` (the
      autocast context); these axes compose on top.

    Defaults preserve the pre-refactor behaviour bit-for-bit: grads
    accumulate and all-reduce in fp32, losses are fp32. ``reduce_dtype
    = "bf16"`` is an opt-in speedup (registers DDP's ``bf16_compress``
    comm hook and halves all-reduce bandwidth at a small numeric cost).
    """

    # Compute ``training_loss`` in fp32.
    fp32_loss: bool = True
    # Run ConfidenceHead final LN+Linear blocks and the distance embedding in fp32.
    fp32_confidence_head: bool = True
    # Run DistogramHead.half_logits in fp32. Off by default; flip on if the
    # distogram loss shows bursts beyond its theoretical max (ln(num_bins)).
    fp32_distogram_head: bool = False
    # Run the diffusion sampler (inference) in fp32. Mirrors
    # ``DiffusionConfig.skip_amp_sample`` — ``ModelConfig.__post_init__``
    # keeps the two in sync.
    fp32_diffusion_sample: bool = False
    # Run the training-time diffusion rollout in fp32. Mirrors
    # ``DiffusionConfig.skip_amp_sample_training``.
    fp32_diffusion_sample_training: bool = False

    # Gradient accumulation dtype. DeepSpeed-inspired contract — master
    # weights stay fp32 (AdamW requirement), grads accumulate across
    # microsteps in this dtype. ``bf16`` is incompatible with
    # ``accumulate_grad_batches > 1`` (bf16 rounding swallows small
    # contributions); asserted at trainer startup.
    grad_acc_dtype: Literal["bf16", "fp32"] = "fp32"
    # DDP all-reduce dtype. ``fp32`` (default) uses PyTorch's stock
    # allreduce. ``bf16`` registers ``bf16_compress_hook``, halving
    # inter-rank bandwidth; the cost is one-extra-cast per bucket and
    # mild numerical drift vs. fp32 allreduce.
    reduce_dtype: Literal["bf16", "fp32"] = "fp32"
    # Loss-computation dtype. Always ``fp32`` today; kept as a knob to
    # make the policy explicit and to reserve the axis for future bf16-
    # loss experiments.
    loss_dtype: Literal["fp32"] = "fp32"

    def __post_init__(self) -> None:
        if self.grad_acc_dtype not in ("bf16", "fp32"):
            raise ValueError(f"grad_acc_dtype must be 'bf16' or 'fp32', got {self.grad_acc_dtype!r}")
        if self.reduce_dtype not in ("bf16", "fp32"):
            raise ValueError(f"reduce_dtype must be 'bf16' or 'fp32', got {self.reduce_dtype!r}")
        if self.loss_dtype != "fp32":
            raise ValueError(f"loss_dtype must be 'fp32' (other values reserved), got {self.loss_dtype!r}")

    @classmethod
    def from_shorthand(cls, shorthand: str, **overrides) -> PrecisionConfig:
        """Build a policy from a single string (sugar for common presets).

        ``"bf16-mixed"`` → grad_acc=fp32, reduce=bf16, loss=fp32 (maximum bandwidth savings).
        ``"bf16-safe"``  → grad_acc=fp32, reduce=fp32, loss=fp32 (current DF3 default).
        ``"fp16-mixed"`` → grad_acc=fp32, reduce=bf16, loss=fp32 (GradScaler required).
        ``"fp32"``       → grad_acc=fp32, reduce=fp32, loss=fp32.

        The shorthand only sets the non-compute axes; compute dtype is
        governed by ``TrainingConfig.dtype``. ``overrides`` override any
        axis post-expansion (including the region-guard flags).
        """
        presets: dict[str, dict[str, str]] = {
            "bf16-mixed": {"grad_acc_dtype": "fp32", "reduce_dtype": "bf16", "loss_dtype": "fp32"},
            "bf16-safe": {"grad_acc_dtype": "fp32", "reduce_dtype": "fp32", "loss_dtype": "fp32"},
            "fp16-mixed": {"grad_acc_dtype": "fp32", "reduce_dtype": "bf16", "loss_dtype": "fp32"},
            "fp32": {"grad_acc_dtype": "fp32", "reduce_dtype": "fp32", "loss_dtype": "fp32"},
        }
        if shorthand not in presets:
            raise ValueError(f"unknown precision shorthand {shorthand!r}; choices: {sorted(presets)}")
        return cls(**{**presets[shorthand], **overrides})


@dataclass
class DataConfig:
    max_templates: int = 4
    max_neighborhood: int = 40
    # Cropping mixture weights (Protenix-style:
    # ContiguousCropping / SpatialCropping / SpatialInterfaceCropping).
    contiguous_crop_weight: float = 0.2
    spatial_crop_weight: float = 0.4
    interface_crop_weight: float = 0.4
    # Distance threshold (Å) for "interface" tokens used by the
    # interface-seeded spatial cropper.
    interface_distance: float = 15.0
    # AlphaFold 3 §2.4 template release-date guard: a template's release
    # date must precede the query example's release by at least this many
    # days (default 60 for PDB training data).
    template_cutoff_days: int = 60
    # Optional global hard cap on template release dates (ISO YYYY-MM-DD).
    # AF3 uses 2018-04-30 for distillation and the training cutoff for
    # inference; leave None to skip the global cap.
    max_template_date: str | None = None
    # Prefer per-sequence monomer templates over per-record multimer
    # templates when both directories are present.
    prefer_monomer_templates: bool = True
    # Maximum polymer chains to keep per sample at training time
    # (AF3 §2.5.4 / Table 6: 20 for initial + fine-tune 1/2, 50 for
    # fine-tune 3).  When a structure exceeds this limit, the same
    # greedy interface-maximising selection from prepare-time is applied
    # at sampling time.  Set 0 to disable.
    max_chains: int = 20


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
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    num_recycles: int = 10
    # MSA rows fed to the Evoformer per recycle. Defaults to AF3's
    # Nmsa = 16384 (the same cap used by the featurizer), so per-recycle
    # subsampling is a no-op when the featurizer already capped.
    msa_crop_size: int = 16_384

    def __post_init__(self):
        # Mirror the legacy ``DiffusionConfig.skip_amp_sample*`` flags onto the
        # unified ``PrecisionConfig`` fields so setting either side has the
        # same effect regardless of entry point.
        def _mirror(pre_field: str, diff_field: str) -> None:
            val = bool(getattr(self.precision, pre_field) or getattr(self.diffusion, diff_field))
            setattr(self.precision, pre_field, val)
            setattr(self.diffusion, diff_field, val)

        _mirror("fp32_diffusion_sample", "skip_amp_sample")
        _mirror("fp32_diffusion_sample_training", "skip_amp_sample_training")
