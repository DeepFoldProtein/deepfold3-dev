# Performance Optimizations

DeepFold3 supports optional Triton kernel replacements and `torch.compile` for key
operations. All Triton kernels are self-contained under `src/deepfold3/model/triton/`.

Optimizations are toggled via `OptimConfig` — each can be independently enabled or disabled.

## Benchmark Results (RTX 5090, bf16, 2026-03-16)

### Winners (enabled by `--optim recommended`)

| Optimization | Speedup (fwd) | Memory Savings | Where Used | Count |
|---|---|---|---|---|
| **Triton EvoAttention (Grid)** | **2.9-3.3x** | **3.4-4.6x** | EvoformerBlock + PairFormerBlock pair attention | 104 attn |
| **Triton FusedPairBias** | **+25-30%** on top of evo | tl.dot GEMM (5-10x vs scalar loop) | GridSelfAttention pair bias | 104 attn |
| **Combined evo+pair_bias** | **3.7-4.6x** | **3.4-4.6x** | GridSelfAttention (row + col) | 104 attn |
| **Triton FusedTriangleMul (L3)** | 1.24-1.32x | Eliminates In/Out memory asymmetry | TriangleMul outgoing + incoming | 96 modules |
| **torch.compile (MSAAttention)** | 1.5-2.3x | 1.6-2.3x | EvoformerBlock (×4) | 4 modules |
| **torch.compile (TriangleMul)** | 1.25-1.55x | ~1.0x | EvoformerBlock + PairFormerBlock | 104 modules |
| **torch.compile (SingleAttention)** | 1.3-1.4x | ~1.0x | PairFormerBlock (×48) | 48 modules |

### TriangleMul Memory Asymmetry (fixed by triton_l3)

Without fusion, TriangleMulIncoming uses **1.2-1.5x more memory** than Outgoing due to
cuBLAS workspace allocation for different contraction stride patterns. The fused dual GEMM
(triton_l3) eliminates this completely:

| N | pytorch Incoming excess | triton_l3 Incoming excess |
|---|---|---|
| 384 | +326 MB (1.52x) | +0 MB |
| 512 | +252 MB (1.18x) | +0 MB |
| 640 | +1128 MB (1.49x) | +0 MB |
| 768 | +1553 MB (1.46x) | +0 MB |

### Grid Column Attention (evo + pair bias)

| N | pytorch | evo | evo+pb | Speedup | Mem reduction |
|---|---|---|---|---|---|
| 384 | 5.8ms / 2.8GB | 2.0ms / 0.9GB | **1.6ms / 1.0GB** | **3.7x / 2.9x** |
| 512 | 13.3ms / 6.5GB | 4.3ms / 1.6GB | **3.2ms / 1.7GB** | **4.1x / 3.8x** |
| 640 | 25.4ms / 12.3GB | 7.7ms / 2.5GB | **5.5ms / 2.6GB** | **4.6x / 4.7x** |

### Conditional winners (enabled by `--optim all`)

| Optimization | Speedup | Condition |
|---|---|---|
| **Triton FusedTransition (C=384)** | 1.2-1.3x full | Only beneficial at C≥256 and N≥1024; smaller shapes are slower |
| **Triton LayerNormLinear (C=384)** | 1.1-1.2x full | Composited inside FusedTransitionBlock; marginal standalone |
| **Triton FusedLNTranspose** | 1.7-2.3x fwd | Column attention LN+transpose; inference only |

### Not recommended (removed from config)

| Optimization | Result | Reason |
|---|---|---|
| Triton EvoAttention (Single) | 0.87-1.03x (slower) | N=1 sequence dim → kernel overhead dominates; torch.compile is better |
| torch.compile (LayerNorm) | 0.3x (3x slower) | Inductor overhead dominates tiny kernel |
| torch.compile (LinearNoBias) | 0.3-0.5x (slower) | cuBLAS already optimal |
| torch.compile (TransitionBlock) | 0.75-0.98x (slower) | Overhead > gain for most shapes |
| torch.compile (AdaTransitionBlock) | 0.96-1.02x (neutral) | No meaningful benefit |
| torch.compile (OuterProductMean) | 1.06-1.25x (marginal) | Minimal speedup, not worth compile cost |
| Triton FusedTransition (C=64/128) | 0.8-1.0x (slower) | Shapes too small for Triton overhead |
| Triton SwiGLU | 0.73-0.92x (slower) | Kernel launch overhead; 1.4-2.3x memory savings but time regression |

### Key Insight

The **dominant optimizations** are:

1. **Triton EvoAttention + FusedPairBias** for GridSelfAttention — **3.7-4.6x faster** with
   **3-5x less memory**. This covers 104 pair attention modules (the N² bottleneck).
   FusedPairBias now supports training (autograd backward) in addition to inference.

2. **Triton FusedTriangleMul (L3)** for TriangleMul — **1.2-1.3x faster** and eliminates
   the Incoming/Outgoing **memory asymmetry** (1553 MB excess at N=768 → 0 MB).

`torch.compile` helps for MSAAttention (1.5-2.3x), TriangleMul (1.3-1.6x), and
SingleAttention (1.3-1.4x) but hurts for small primitives (LN, Linear) and transitions.

**Important**: Triton EvoAttention is NOT beneficial for SingleAttention (N_seq=1 makes
the 5D kernel overhead dominate). Use torch.compile for SingleAttention instead.

## Available Backends

| Backend | Module | Replaces | Source |
| ------- | ------ | -------- | ------ |
| Triton FusedEvoAttention | `_EvoformerAttention` | Flash-attention with pair bias | `triton/evo_attention.py` |
| Triton FusedPairBias | `FusedPairBias` | LN + Linear + permute for pair bias (tl.dot GEMM) | `triton/fused_pair_bias.py` |
| Triton FusedGatedDualGEMM | `FusedGatedDualGEMM` | Dual GEMM + sigmoid gate (TWO_INPUTS, mask, bias) | `triton/fused_gated_dual_gemm.py` |
| Triton FusedTransitionBlock | `FusedTransitionBlock` | `TransitionBlock` (LN + Linear + SwiGLU + Linear) | `triton/fused_transition.py` |
| Triton FusedLayerNormLinear | `FusedLayerNormLinear` | Fused LayerNorm + Linear | `triton/layernorm_linear.py` |
| Triton FusedLayerNormTranspose | `FusedLayerNormTranspose` | LN + dim transpose (col attention) | `triton/layernorm_transpose.py` |
| Triton FusedTriangleMulGate | `FusedTriangleMulGate` | LN + packed 3-projection GEMM + sigmoid | `triton/triangle_mul_gate.py` |
| Persistent Autotune Cache | `CacheManager` | GPU-fingerprinted JSON cache for Triton configs | `triton/cache.py` |
| torch.compile | — | Wraps TriangleMul, MSAAttention, SingleAttention | PyTorch built-in |

## Configuration

### OptimConfig

```python
@dataclass
class OptimConfig:
    use_fused_evo_attention: bool = False   # Triton flash-attn for GridSelfAttention
    use_fused_pair_bias: bool = False       # Triton fused LN+Linear+permute (tl.dot GEMM)
    use_fused_transition_pair: bool = False # Triton fused transition for pair (M=N², C=128)
    use_fused_triangle_mul: bool = False    # Triton dual GEMM + gate fusion for TriangleMul
    use_fused_ln_transpose: bool = False    # Triton LN+transpose for column attention
    use_torch_compile: bool = False         # torch.compile for attention + TriangleMul
```

### Python API

```python
from deepfold3.model.config import ModelConfig, OptimConfig

# Recommended for inference (biggest wins)
config = ModelConfig()
config.optim = OptimConfig.recommended()  # evo attention + compile

# Enable everything (including fused transition for training memory savings)
config.optim = OptimConfig.all_enabled()

# Check what's available on this system
from deepfold3.model.optim import print_availability
print_availability()
```

### CLI

```bash
# Recommended: fused evo attention + torch.compile (best inference speed)
deepfold3 inference --json input.json --output out/ --params weights.safetensors --optim recommended

# All optimizations (adds fused transition — useful for training backward pass memory)
deepfold3 inference --json input.json --output out/ --params weights.safetensors --optim all

# Triton fused kernels only (no torch.compile)
deepfold3 inference --json input.json --output out/ --params weights.safetensors --optim triton
```

### Recommended Configuration

- **For inference**: `--optim recommended` — enables EvoAttention (4-9x on pair attention) and torch.compile (1.3-2.7x on TriangleMul + attention). Best speed/simplicity tradeoff.
- **For training**: `--optim all` — additionally enables FusedTransition for backward pass memory savings on single transitions (C=384).
- **Debugging**: omit `--optim` — pure PyTorch, no compiled graphs, easy to debug.

## Installation

Triton kernels require the `optim` dependency group:

```bash
# Install with optimization dependencies
uv sync --group optim

# torch.compile requires no extra dependencies (PyTorch built-in)
```

All optimizations are optional. If a backend is not installed, the corresponding
flag is silently ignored and the PyTorch baseline is used.

## Architecture

Optimizations are applied at module construction time. `OptimConfig` is threaded through:

```
DeepFold3(cfg.optim)
  -> Evoformer(optim)
       -> EvoformerBlock(optim)
            -> GridSelfAttention(optim)     # Triton EvoAttention (4-9x)
            -> TriangleMulOutgoing(optim)   # torch.compile (1.3-1.6x)
            -> TriangleMulIncoming(optim)   # torch.compile (1.3-1.6x)
            -> MSAAttention(optim)          # torch.compile (1.4-2.7x)
            -> make_transition_block(optim) # FusedTransition (C>=256 only)
       -> PairFormerBlock(optim)
            -> GridSelfAttention(optim)     # Triton EvoAttention (4-9x)
            -> TriangleMulOutgoing(optim)   # torch.compile (1.3-1.6x)
            -> TriangleMulIncoming(optim)   # torch.compile (1.3-1.6x)
            -> SingleAttention(optim)       # torch.compile only (1.3-1.4x)
            -> make_transition_block(optim) # FusedTransition (C>=256 only)
  -> DiffusionHead(optim)
       -> DiffusionConditioningBlock(optim)
            -> make_transition_block(optim)
       -> DiffusionTransformer(optim)
            -> DiffusionTransformerBlock(optim) # torch.compile (×24, 4800 calls/sample)
```

### FusedTransition min-channels guard

`make_transition_block()` only uses `FusedTransitionBlock` when `channels >= 256`.
For smaller channels (C=64 MSA, C=128 pair), benchmarks show Triton overhead makes
it slower than baseline PyTorch.

### torch.compile strategy

`torch.compile` is applied to inner `_compute` methods (not the full `forward`) to
avoid recompilation from dynamic batch dimensions. The `has_batch` unsqueeze/squeeze
wrapper stays outside the compiled region.

### Triton Kernel Files

```
src/deepfold3/model/triton/
├── __init__.py               # Precision enum, cvt_tf32_rn, precision_dot, lazy imports
├── evo_attention.py          # Flash-attention with pair bias (5 Triton JIT kernels)
├── fused_pair_bias.py        # Fused LN + tl.dot projection + permute + mask (cuEQ pattern)
├── fused_gated_dual_gemm.py  # Dual GEMM + sigmoid gate (TWO_INPUTS, bias, mask, TF32)
├── fused_transition.py       # FusedTransitionBlock (compose LN+Linear+SwiGLU+Linear)
├── fused_ada_transition.py   # FusedAdaTransitionBlock (AdaLN + SwiGLU + Linear)
├── layernorm_linear.py       # FusedLayerNormLinear (4 Triton JIT kernels)
├── ada_layernorm_linear.py   # FusedAdaLayerNormLinear (AdaLN + Linear)
├── layernorm_transpose.py    # FusedLayerNormTranspose (LN + dim transpose)
├── triangle_mul_gate.py      # FusedTriangleMulGate (LN + packed 3-GEMM + sigmoid)
├── swiglu.py                 # FusedSwiGLU (silu(a)*b convention)
├── tuning.py                 # Pre-tuned block size configs for CUDA/HIP
└── cache.py                  # Persistent autotuning cache (GPU fingerprint + JSON)
```

### TF32 Precision Control

All GEMM-based Triton kernels support configurable precision via the `Precision` enum
(adapted from NVIDIA cuEquivariance):

| Mode | Value | Behavior |
|------|-------|----------|
| `DEFAULT` | 0 | Hardware default (fastest, least precise) |
| `TF32` | 1 | Explicit TF32 rounding via `cvt_tf32_rn` (deterministic 19-bit mantissa) |
| `TF32x3` | 2 | Triton's built-in tf32x3 accumulation (higher precision than TF32) |
| `IEEE` | 3 | Full IEEE754 float32 (slowest, most precise) |

The `precision_dot(a, b, acc, PRECISION)` JIT helper dispatches `tl.dot` with the
requested precision mode. The pair bias kernel defaults to `TF32x3` for best
accuracy/speed tradeoff.

### Persistent Autotuning Cache

Triton kernel autotuning results are persisted to `~/.cache/deepfold3-triton/` (or
`DEEPFOLD3_TRITON_CACHE_DIR`). The cache:

- **GPU fingerprinting**: Uses `torch.cuda.get_device_properties` (compute capability, memory, SM count)
- **Fuzzy matching**: When switching GPUs within the same architecture family, cached configs
  transfer if memory/SM count are within 20% tolerance
- **Atomic writes**: Thread-safe JSON persistence via temp files + `os.replace`

Environment variables:
- `DEEPFOLD3_TRITON_CACHE_DIR`: Override cache directory
- `DEEPFOLD3_TRITON_NO_CACHE=1`: Disable caching (re-tune every run)

### Fused Pair Bias Design (cuEQ pattern)

The pair bias kernel fuses LayerNorm + Linear projection + layout permute into a single
kernel using NVIDIA cuEquivariance's 3D grid pattern:

```
Grid: (cdiv(V, TILE_V), U, cdiv(H, HEADS_PER_BLK) × B)

Phase 1: LN statistics (mean, rstd) over C in TILE_K tiles
Phase 2: Normalize + project via tl.dot([TILE_V, TILE_K] × [TILE_K, H_BLK])
Phase 3: Optional mask fusion + write to [B, H, U, V] layout
```

Key improvement: The old kernel used a scalar dot-product loop per head
(`tl.sum(xn * w_h)` × H). The new kernel uses `tl.dot` tensor core GEMM,
which is 5-10x faster for C=128, H=4. Supports both training (autograd with
PyTorch backward) and inference (fully fused, no intermediates).

### Fused Gated Dual GEMM Design (cuEQ pattern)

The dual GEMM kernel computes `sigmoid(x₁ @ W₁ᵀ + b₁) × (x₂ @ W₂ᵀ + b₂) × mask`
in a single kernel pass:

- **TWO_INPUTS**: Separate x₁, x₂ for gate and projection paths
- **Bias**: Optional b₁, b₂ added after GEMM accumulation
- **Mask fusion**: Per-row mask (AdaLN-zero gate) applied in epilogue
- **L2 grouping**: GROUP_SIZE_M linearized grid for cache locality (our advantage over cuEQ)
- **Backward**: Pre-GEMM kernel recomputes sigmoid + derives grad factors, then cuBLAS for weight/input gradients

## Attention Tuning Notes

### GridSelfAttention (EvoAttention kernel)

The Triton EvoAttention kernel operates on 5D tensors: `[B, N_seq, N_res, H, D]`.
For GridSelfAttention, `N_seq = N_rows` (row or column dimension) and `N_res = N_cols`
(the attention dimension). The kernel fuses the pair bias addition and avoids
materializing the N^2 attention matrix.

- **Row attention**: `N_seq = N` rows, each attending over `N_res = N` columns
- **Column attention**: transposed — `N_seq = N` columns, each attending over `N_res = N` rows
- **Pair bias**: `[B, 1, H, N, N]` — shared across all rows/columns within a batch

### SingleAttention (torch.compile only)

SingleAttention previously used EvoAttention with `N_seq = 1`, but benchmarks show
this is 0.87-1.03x (slower or neutral) due to kernel launch overhead dominating the
single-sequence case. `torch.compile` on the baseline path gives 1.3-1.4x speedup and
is now the only optimization applied.

---

## Benchmark Manual

### Prerequisites

- CUDA GPU required (all benchmarks check `torch.cuda.is_available()`)
- Install optimization backends you want to benchmark:

```bash
uv sync --group optim   # triton + matplotlib
```

- Check which backends are detected:

```bash
uv run python -c "from deepfold3.model.optim import print_availability; print_availability()"
```

### Quick Start

```bash
# 1. Verify numerical correctness of all kernels (always run first)
uv run python benchmarks/verify.py --verbose

# 2. Run all benchmarks + generate plots
uv run python benchmarks/run_all.py

# 3. Smoke test (tiny shapes, fast — good for CI or checking setup)
uv run python benchmarks/run_all.py --dry-run
```

### Running Individual Benchmarks

Each benchmark script can be run standalone. All accept `--dry-run`, `--warmup`, and `--rep` flags.

```bash
# Primitives: LayerNorm, SwiGLU, LinearNoBias
uv run python benchmarks/bench_primitives.py

# Transition blocks: TransitionBlock, AdaTransitionBlock
uv run python benchmarks/bench_transition.py

# Attention modules: GridSelfAttention, MSAAttention, SingleAttention
uv run python benchmarks/bench_attention.py

# Pair operations: TriangleMul (outgoing/incoming), OuterProductMean
uv run python benchmarks/bench_pair_ops.py
```

### Numerical Verification

```bash
uv run python benchmarks/verify.py --verbose
```

### Auto-Tuner

```bash
# Tune a single shape
uv run python benchmarks/tuner.py --op transition_block --N 512 --C 384

# Sweep multiple common shapes
uv run python benchmarks/tuner.py --op transition_block --sweep

# Tune Triton block sizes for all DeepFold3 common shapes
uv run python benchmarks/tuner.py --triton-tune

# Smoke test
uv run python benchmarks/tuner.py --dry-run
```

### Result Format

Benchmark results are saved as JSON arrays in `benchmarks/results/`. Each entry:

```json
{
  "op_name": "GridSelfAttention",
  "provider": "triton",
  "shape": {"N": 256, "C": 128, "H": 4},
  "mode": "fwd",
  "time_ms": 0.123,
  "peak_memory_mb": 45.6,
  "dtype": "bfloat16",
  "extra": {}
}
```

### Benchmark File Reference

```
benchmarks/
├── __init__.py            # Package marker
├── utils.py               # gpu_timer(), peak_memory_mb(), BenchResult, save/load JSON
├── optim_config.py        # OptimConfig, is_available(), get_optimized_module()
├── verify.py              # Numerical verification (fwd+bwd allclose)
├── tuner.py               # Auto-tuner with --sweep and --triton-tune modes
├── bench_primitives.py    # LayerNorm, SwiGLU, LinearNoBias
├── bench_transition.py    # TransitionBlock, AdaTransitionBlock
├── bench_attention.py     # GridSelfAttention, MSAAttention, SingleAttention
├── bench_pair_ops.py      # TriangleMul, OuterProductMean
├── plot_results.py        # Matplotlib grouped bar charts
├── run_all.py             # Orchestrator: verify -> bench all -> plot
└── results/               # Output directory (JSON + PNG)
```
