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

    Benchmark results (RTX 5090, bf16, 2026-03-07):
    - use_fused_evo_attention: 2-3x faster, 2-5x less memory for GridSelfAttention (104 modules)
    - use_torch_compile: 1.3-2.3x faster for MSAAttention/SingleAttention/TriangleMul (156 modules)
    - use_fused_transition: 1.2-1.3x faster for C>=256 + N>=1024 transitions only
    Note: Triton EvoAttention is NOT used for SingleAttention (N_seq=1 overhead dominates).
    """

    use_fused_evo_attention: bool = False
    use_fused_transition: bool = False
    use_torch_compile: bool = False

    @classmethod
    def recommended(cls) -> OptimConfig:
        """Fused evo attention + torch.compile — the two biggest wins."""
        return cls(use_fused_evo_attention=True, use_torch_compile=True)

    @classmethod
    def triton_only(cls) -> OptimConfig:
        """Triton fused kernels only (no torch.compile)."""
        return cls(use_fused_evo_attention=True, use_fused_transition=True)

    @classmethod
    def all_enabled(cls) -> OptimConfig:
        """Enable all optimizations including fused transition."""
        return cls(use_fused_evo_attention=True, use_fused_transition=True, use_torch_compile=True)


@dataclass
class ModelConfig:
    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    pairformer: PairFormerConfig = field(default_factory=PairFormerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    template: TemplateEmbeddingConfig = field(default_factory=TemplateEmbeddingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    num_recycles: int = 10
    msa_crop_size: int = 1024  # MSA rows sampled per recycle
