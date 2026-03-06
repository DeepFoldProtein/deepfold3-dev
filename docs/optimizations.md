# Performance Optimizations

DeepFold3 supports optional Triton kernel replacements and `torch.compile` for key
operations. All Triton kernels are self-contained under `src/deepfold3/model/triton/`.

Optimizations are toggled via `OptimConfig` — each can be independently enabled or disabled.

## Benchmark Results (RTX 5090, bf16)

### Winners (enabled by `--optim recommended`)

| Optimization | Speedup | Memory Savings | Where Used | Count |
|---|---|---|---|---|
| **Triton EvoAttention (Grid)** | **4-9x time** | **3-10x memory** | EvoformerBlock + PairFormerBlock pair attention | 104 attn |
| **torch.compile (TriangleMul)** | 1.3-1.6x fwd | 1.1-1.2x fwd mem | EvoformerBlock + PairFormerBlock | 104 modules |
| **torch.compile (MSAAttention)** | 1.4-2.7x fwd | 1.6-2.4x mem | EvoformerBlock (x4) | 4 modules |
| **torch.compile (SingleAttention)** | 1.3-1.7x time | 1.0-1.3x mem | PairFormerBlock (x48) | 48 modules |

### Conditional winners (enabled by `--optim all`)

| Optimization | Speedup | Condition |
|---|---|---|
| **Triton FusedTransition (C=384)** | 1.2-1.3x full | Only beneficial for C >= 256; C=64/128 is slower |
| **Triton LayerNormLinear (C=384)** | 1.1-1.4x full | Composited inside FusedTransitionBlock |

### Not recommended (removed from config)

| Optimization | Result | Reason |
|---|---|---|
| torch.compile (LayerNorm) | 0.3x (3x slower) | Inductor overhead dominates tiny kernel |
| torch.compile (LinearNoBias) | 0.3-0.5x (slower) | cuBLAS already optimal |
| torch.compile (TransitionBlock) | 0.7-0.9x (slower) | Overhead > gain for small shapes |
| Triton FusedTransition (C=64/128) | 0.5-0.8x (slower) | Shapes too small for Triton overhead |

### Key Insight

The **dominant optimization** is Triton EvoAttention for GridSelfAttention — it's 4-9x
faster and 3-10x less memory. This covers 104 pair attention modules (the N^2 bottleneck).
Everything else is marginal in comparison.

`torch.compile` helps for attention and TriangleMul (1.3-2.7x) but hurts for small
primitives (LN, Linear). It is applied selectively at the module level.

## Available Backends

| Backend | Module | Replaces | Source |
| ------- | ------ | -------- | ------ |
| Triton FusedEvoAttention | `_EvoformerAttention` | Flash-attention with pair bias | `deepfold3.model.triton.evo_attention` |
| Triton FusedTransitionBlock | `FusedTransitionBlock` | `TransitionBlock` (LN + Linear + SwiGLU + Linear) | `deepfold3.model.triton.fused_transition` |
| Triton FusedLayerNormLinear | `FusedLayerNormLinear` | Fused LayerNorm + Linear | `deepfold3.model.triton.layernorm_linear` |
| torch.compile | — | Wraps TriangleMul, MSAAttention, SingleAttention | PyTorch built-in |

## Configuration

### OptimConfig

```python
@dataclass
class OptimConfig:
    use_fused_evo_attention: bool = False  # Triton flash-attn for Grid/SingleAttention
    use_fused_transition: bool = False     # Triton FusedTransitionBlock (C>=256 only)
    use_torch_compile: bool = False        # torch.compile for attention + TriangleMul
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
            -> SingleAttention(optim)       # Triton EvoAttention or torch.compile
            -> make_transition_block(optim) # FusedTransition (C>=256 only)
  -> DiffusionHead(optim)
       -> DiffusionConditioningBlock(optim)
            -> make_transition_block(optim)
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
├── __init__.py            # Lazy imports
├── swiglu.py              # FusedSwiGLU (silu(a)*b convention)
├── layernorm_linear.py    # FusedLayerNormLinear (4 Triton JIT kernels)
├── fused_transition.py    # FusedTransitionBlock (compose LN+Linear+SwiGLU+Linear)
├── evo_attention.py       # Flash-attention with pair bias (5 Triton JIT kernels)
└── tuning.py              # Pre-tuned block size configs for CUDA/HIP
```

## Attention Tuning Notes

### GridSelfAttention (EvoAttention kernel)

The Triton EvoAttention kernel operates on 5D tensors: `[B, N_seq, N_res, H, D]`.
For GridSelfAttention, `N_seq = N_rows` (row or column dimension) and `N_res = N_cols`
(the attention dimension). The kernel fuses the pair bias addition and avoids
materializing the N^2 attention matrix.

- **Row attention**: `N_seq = N` rows, each attending over `N_res = N` columns
- **Column attention**: transposed — `N_seq = N` columns, each attending over `N_res = N` rows
- **Pair bias**: `[B, 1, H, N, N]` — shared across all rows/columns within a batch

### SingleAttention (EvoAttention or compile)

SingleAttention uses EvoAttention with `N_seq = 1` (single sequence). If EvoAttention
is unavailable, falls back to `torch.compile` on the baseline path (1.3-1.7x speedup).

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
