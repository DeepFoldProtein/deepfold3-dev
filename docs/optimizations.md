# Performance Optimizations

DeepFold3 supports optional Triton kernel replacements and `torch.compile` for key
operations. All Triton kernels are self-contained under `src/deepfold3/model/triton/`.

Optimizations are toggled via `OptimConfig` — each can be independently enabled or disabled.

## B200 End-to-End Benchmark (bf16, 2026-06-10)

The RTX-5090 section below is **forward-only** and does not reflect training, where
each kernel also runs a backward and is recomputed under gradient checkpointing — so
a kernel that wins on the forward can be net-neutral or slower in a training step.
The numbers here are the real training/inference regime on B200, measured with
`scripts/bench_kernels.py` (8 `PairFormerBlock`s — the trunk component that hosts
these kernels). **These are the basis for the finalized flags.**

Baseline = fused evo-attention + fused dropout-add on, the four candidates off.
Speed is ×baseline; memory is Δ vs baseline (8-block trunk peak).

### Training (fwd + bwd, gradient checkpointing)

| Config | N=384 | N=640 | N=768 |
| --- | --- | --- | --- |
| baseline | 1.00× / 3.18 GB | 1.00× / 8.72 GB | 1.00× / 12.52 GB |
| +swiglu | 1.05× / −0.14 | 1.04× / −0.39 | 1.03× / −0.56 |
| +transition_pair | 1.05× / −0.21 | 1.04× / −0.59 | 1.04× / −0.84 |
| +triangle_mul | 1.05× / −0.42 | 1.05× / −1.17 | 1.06× / −1.69 |
| +pair_bias | 1.00× / +0.07 | 1.07× / +0.20 | 1.12× / +0.29 |
| **+3 safe (no pair_bias)** | **1.04× / −0.63** | **1.09× / −1.76** | **1.09× / −2.54** |
| +ALL four | 1.01× / −0.56 | 1.17× / −1.56 | 1.25× / −2.25 |

- **Training default = the safe trio (swiglu + transition_pair + triangle_mul)**, enabled
  in `configs/initial.yaml`: 1.04–1.09× with a ~20% trunk-activation memory cut that
  grows with crop size. Numerics validated (`tests/triton/test_fused_modules_robustness.py`
  46/46; full-model e2e fwd+bwd, 0 non-finite grads).
- **`pair_bias` stays OFF for training.** At N=384 it adds memory for ~0% speed, and its
  bf16 backward amplifies gradients ~1e11× alongside fused evo-attn (the load-bearing
  reason — see `configs/initial.yaml`). But it is a large speed lever at big crops
  (all-four 1.25× @768 vs 1.09×), so **rewriting its fp32-promoted backward is worth
  ~12–16% throughput if training moves to 640/768 crops.**

### Inference (fwd only)

| Config | N=384 | N=640 | N=768 |
| --- | --- | --- | --- |
| baseline | 52.98 ms / 0.9 GB | 164.85 ms / 2.4 GB | 257.34 ms / 3.4 GB |
| +pair_bias | 41.65 ms (1.27×) | 133.11 ms (1.24×) | 202.77 ms (1.27×) |
| +3 safe | 46.95 ms (1.13×) | 146.50 ms (1.13×) | 228.21 ms (1.13×) |
| **+ALL four** | **35.58 ms (1.49×) / 0.7 GB** | **114.78 ms (1.44×) / 1.9 GB** | **173.46 ms (1.48×) / 2.7 GB** |

`OptimConfig.inference()` already enables all four + `ln_transpose` + tiling — confirmed
optimal (1.44–1.49× and lower memory). With no backward, `pair_bias` is safe and is the
single biggest inference lever (1.24–1.27× alone).

### fused_dropout_add (default ON)

Per-call it is marginally slower than unfused PyTorch (72 vs 70 µs at the 384 pair shape)
but never materializes the `[N, N]` mask (~36 GB/step saved across fwd+bwd). The mask seed
is drawn from a dedicated **rank-aware** generator (`_next_mask_seed`) so the ~244 draws/step
neither pollute the global CPU RNG stream nor correlate masks across DDP ranks; a CPU draw
(~2 µs) is cheaper than an on-GPU seed (~10 µs, launch-bound), so the seed is never a
bottleneck.

## Benchmark Results (RTX 5090, bf16, 2026-03-16)

### Winners (enabled by `--optim recommended`)

| Optimization                        | Speedup (fwd)             | Memory Savings                     | Where Used                                      | Count       |
| ----------------------------------- | ------------------------- | ---------------------------------- | ----------------------------------------------- | ----------- |
| **Triton EvoAttention (Grid)**      | **2.9-3.3x**              | **3.4-4.6x**                       | EvoformerBlock + PairFormerBlock pair attention | 104 attn    |
| **Triton FusedPairBias**            | **+25-30%** on top of evo | tl.dot GEMM (5-10x vs scalar loop) | GridSelfAttention pair bias                     | 104 attn    |
| **Combined evo+pair_bias**          | **3.7-4.6x**              | **3.4-4.6x**                       | GridSelfAttention (row + col)                   | 104 attn    |
| **Triton FusedTriangleMul (L3)**    | 1.24-1.32x                | Symmetric memory (both directions) | TriangleMul outgoing + incoming                 | 96 modules  |
| **torch.compile (MSAAttention)**    | 1.5-2.3x                  | 1.6-2.3x                           | EvoformerBlock (×4)                             | 4 modules   |
| **torch.compile (TriangleMul)**     | 1.25-1.55x                | ~1.0x                              | EvoformerBlock + PairFormerBlock                | 104 modules |
| **torch.compile (SingleAttention)** | 1.3-1.4x                  | ~1.0x                              | PairFormerBlock (×48)                           | 48 modules  |

### Grid Column Attention (evo + pair bias)

| N   | pytorch         | evo           | evo+pb            | Speedup         | Mem reduction |
| --- | --------------- | ------------- | ----------------- | --------------- | ------------- |
| 384 | 5.8ms / 2.8GB   | 2.0ms / 0.9GB | **1.6ms / 1.0GB** | **3.7x / 2.9x** |
| 512 | 13.3ms / 6.5GB  | 4.3ms / 1.6GB | **3.2ms / 1.7GB** | **4.1x / 3.8x** |
| 640 | 25.4ms / 12.3GB | 7.7ms / 2.5GB | **5.5ms / 2.6GB** | **4.6x / 4.7x** |

### Conditional winners (enabled by `--optim all`)

| Optimization                       | Speedup                     | Condition                                                          |
| ---------------------------------- | --------------------------- | ------------------------------------------------------------------ |
| **Triton SwiGLU (pair, M=N²)**     | 1.4-1.6x fwd, 1.9-2.1x full | Only at pair-transition scale (M=N²); single transitions too small |
| **Triton FusedTransition (C=384)** | 1.2-1.3x full               | Only beneficial at C≥256 and N≥1024; smaller shapes are slower     |
| **Triton LayerNormLinear (C=384)** | 1.1-1.2x full               | Composited inside FusedTransitionBlock; marginal standalone        |
| **Triton FusedLNTranspose**        | 1.7-2.3x fwd                | Column attention LN+transpose; inference only                      |

### Not recommended (removed from config)

| Optimization                       | Result                | Reason                                                                       |
| ---------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| Triton EvoAttention (Single)       | 0.87-1.03x (slower)   | N=1 sequence dim → kernel overhead dominates; torch.compile is better        |
| torch.compile (LayerNorm)          | 0.3x (3x slower)      | Inductor overhead dominates tiny kernel                                      |
| torch.compile (LinearNoBias)       | 0.3-0.5x (slower)     | cuBLAS already optimal                                                       |
| torch.compile (TransitionBlock)    | 0.75-0.98x (slower)   | Overhead > gain for most shapes                                              |
| torch.compile (AdaTransitionBlock) | 0.96-1.02x (neutral)  | No meaningful benefit                                                        |
| torch.compile (OuterProductMean)   | 1.06-1.25x (marginal) | Minimal speedup, not worth compile cost                                      |
| Triton FusedTransition (C=64/128)  | 0.8-1.0x (slower)     | Shapes too small for Triton overhead                                         |
| Triton SwiGLU (single, M=N)        | 0.5x (slower)         | Kernel launch overhead dominates at tiny M; use only inside pair transitions |

### Key Insight

The **dominant optimizations** are:

1. **Triton EvoAttention + FusedPairBias** for GridSelfAttention — **3.7-4.6x faster** with
   **3-5x less memory**. This covers 104 pair attention modules (the N² bottleneck).
   FusedPairBias now supports training (autograd backward) in addition to inference.

2. **Triton FusedTriangleMul (L3)** for TriangleMul — **1.2-1.3x faster** with symmetric
   memory usage for both Incoming and Outgoing directions.

`torch.compile` helps for MSAAttention (1.5-2.3x), TriangleMul (1.3-1.6x), and
SingleAttention (1.3-1.4x) but hurts for small primitives (LN, Linear) and transitions.

**Important**: Triton EvoAttention is NOT beneficial for SingleAttention (N_seq=1 makes
the 5D kernel overhead dominate). Use torch.compile for SingleAttention instead.

## Available Backends

| Backend                        | Module                    | Replaces                                                  | Source                            |
| ------------------------------ | ------------------------- | --------------------------------------------------------- | --------------------------------- |
| Triton FusedEvoAttention       | `_EvoformerAttention`     | Flash-attention with pair bias                            | `triton/evo_attention.py`         |
| Triton FusedPairBias           | `FusedPairBias`           | LN + Linear + permute for pair bias (tl.dot GEMM)         | `triton/fused_pair_bias.py`       |
| Triton FusedGatedDualGEMM      | `FusedGatedDualGEMM`      | Dual GEMM + sigmoid gate (TWO_INPUTS, mask, bias)         | `triton/fused_gated_dual_gemm.py` |
| Triton FusedTransitionBlock    | `FusedTransitionBlock`    | `TransitionBlock` (LN + Linear + SwiGLU + Linear)         | `triton/fused_transition.py`      |
| Triton FusedLayerNormLinear    | `FusedLayerNormLinear`    | Fused LayerNorm + Linear                                  | `triton/layernorm_linear.py`      |
| Triton FusedLayerNormTranspose | `FusedLayerNormTranspose` | LN + dim transpose (col attention)                        | `triton/layernorm_transpose.py`   |
| Triton FusedTriangleMulGate    | `FusedTriangleMulGate`    | LN + packed 3-projection GEMM + sigmoid                   | `triton/triangle_mul_gate.py`     |
| Persistent Autotune Cache      | `CacheManager`            | GPU-fingerprinted JSON cache for Triton configs           | `triton/cache.py`                 |
| TiledLinear                    | `TiledLinear`             | Memory-efficient tiled recomputation for pair transitions | `tiled_linear.py`                 |
| torch.compile                  | —                         | Wraps TriangleMul, MSAAttention, SingleAttention          | PyTorch built-in                  |

## Configuration

### OptimConfig

```python
@dataclass
class OptimConfig:
    use_fused_evo_attention: bool = False   # Triton flash-attn for GridSelfAttention
    use_fused_pair_bias: bool = False       # Triton fused LN+Linear+permute (tl.dot GEMM)
    use_fused_swiglu: bool = False          # Triton multi-row SwiGLU (1.5-2.1x at pair M=N²)
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
config.optim = OptimConfig.recommended()  # evo attention + swiglu + triangle_mul + compile

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

See the **B200 End-to-End Benchmark** at the top for the measured training/inference numbers these recommendations come from.

- **For training**: the `optim:` block in `configs/initial.yaml` — fused evo-attention + fused dropout-add + the safe pair-scale trio (`use_fused_swiglu`, `use_fused_transition_pair`, `use_fused_triangle_mul`) + gradient checkpointing. `use_fused_pair_bias` stays **off** (bf16 backward gradient blow-up). Do **not** use `--optim all` for training — it enables `pair_bias` (training hazard) and the single/ada transitions (slower at C=384/768).
- **For inference**: `--optim inference` (= `OptimConfig.inference()`) — all four pair-scale kernels + `ln_transpose` + tiled pair transitions. 1.44–1.49× over baseline. `pair_bias` is safe here (no backward).
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
├── layernorm_linear.py       # FusedLayerNormLinear (4 JIT kernels, GEMMs autotuned)
├── ada_layernorm_linear.py   # FusedAdaLayerNormLinear (AdaLN + Linear, GEMMs autotuned)
├── layernorm_transpose.py    # FusedLayerNormTranspose (LN + dim transpose, elementwise)
├── triangle_mul_gate.py      # FusedTriangleMulGate (LN + packed GEMM + sigmoid, K-only)
├── swiglu.py                 # FusedSwiGLU (silu(a)*b convention, elementwise)
├── tuning.py                 # Legacy hand-tuned block configs (kept for fallbacks)
└── cache.py                  # Persistent autotuning cache (GPU-fingerprinted, on disk)
```

**Autotune coverage** — kernels marked "autotuned" use `@persistent_autotune` so block
sizes are picked by `triton.autotune` per (M, N, K) shape, with results persisted to disk
and reloaded on later runs (warm the cache once via `deepfold3 warmup-triton`):

| Kernel                                                                            | Autotune key              | Notes                                                                                                                                     |
| --------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `_attn_fwd`, `_attn_bwd_dq`, `_attn_bwd_dk_dv` (evo_attention)                    | `[SEQ_LEN, DIM]`          | First-class autotune since the original commit                                                                                            |
| `_fused_dual_gemm_fwd_kernel`, `_fused_dual_gemm_bwd_pregemm_kernel`              | `[M, N, K]`               | First-class autotune since the original commit                                                                                            |
| `_second_forward_kernel`, `_first_backward_kernel` (layernorm_linear)             | `[M, N, K]` / `[M, K, N]` | Converted to persistent_autotune; previously used a hand-tuned `tuning.py` table that returned `None` (PyTorch fallback) for buckets ≤224 |
| `_ada_second_forward_kernel`, `_ada_first_backward_kernel` (ada_layernorm_linear) | `[M, N, K]` / `[M, K, N]` | Same conversion as above                                                                                                                  |

The remaining elementwise kernels (`swiglu`, `triangle_mul_gate.{fwd,bwd}`,
`layernorm_transpose`, `fused_pair_bias`) use a single shape-agnostic block-size
heuristic — `BLOCK_SIZE = next_pow2(K)`, `num_warps` derived from `BLOCK_SIZE` — that's
already optimal across all bucket sizes. There's no autotune surface for them.

### TF32 Precision Control

All GEMM-based Triton kernels support configurable precision via the `Precision` enum
(adapted from NVIDIA cuEquivariance):

| Mode      | Value | Behavior                                                                 |
| --------- | ----- | ------------------------------------------------------------------------ |
| `DEFAULT` | 0     | Hardware default (fastest, least precise)                                |
| `TF32`    | 1     | Explicit TF32 rounding via `cvt_tf32_rn` (deterministic 19-bit mantissa) |
| `TF32x3`  | 2     | Triton's built-in tf32x3 accumulation (higher precision than TF32)       |
| `IEEE`    | 3     | Full IEEE754 float32 (slowest, most precise)                             |

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
- **Pre-hook injection**: On first call for a (kernel, shape) pair, the wrapper checks the
  persistent cache and, if a config is present, injects it directly into Triton's internal
  `Autotuner.cache` so the benchmark phase is skipped entirely. This is what makes warmup
  worth running before training.
  > **Note.** A hardcoded `B200_PRETUNED` table used to seed the cache at import so a fresh
  > B200 paid no autotune cost. It was removed (~2,200 lines, incomplete — `_attn_bwd_db`
  > never had configs, crops > 384 weren't covered) in favor of the shared-cache + warmup
  > workflow below. On an empty cache, kernels now autotune on first call and persist.

#### Recommended workflow: warmup once, read from a shared cache

This is the canonical path (it replaced the hardcoded `B200_PRETUNED` table).

1. Point `~/.cache/deepfold3-triton/` (or `DEEPFOLD3_TRITON_CACHE_DIR`) at **shared storage**
   mounted by every node — or bake a warmed cache into the training container image.
2. Run warmup **once, on a single rank**, with `--real-model` (drives the actual model at
   the production shapes, so every cached `(M, N, K)` and kernel specialization matches
   training exactly — no synthetic-shape-list drift):

   ```bash
   deepfold3 warmup-triton --config configs/initial.yaml --real-model
   ```

3. Every node and every later job then **reads** the persisted configs at startup (lazy
   load + pre-hook injection, `cache.py:204`) and skips autotuning entirely. The read is a
   plain JSON load, so it works over NFS/shared FS.

**Caveat — no cache misses during DDP.** Writes happen only on a miss, and many ranks
missing the *same* shape will write the *same* shared JSON concurrently (the file lock is
in-process, not cross-node) — a corruption/race risk. So warmup must cover **all** buckets
you will hit (384 today; also 640/768 if you move to larger crops). After warmup, the DDP
run should only ever read. `DEEPFOLD3_TRITON_NO_CACHE=1` (below) is a blunt escape hatch.

Environment variables:

- `DEEPFOLD3_TRITON_CACHE_DIR`: Override the cache directory (highest precedence).
- `XDG_CACHE_HOME`: If `DEEPFOLD3_TRITON_CACHE_DIR` is unset, the cache lives under
  `$XDG_CACHE_HOME/deepfold3-triton` (then falls back to `~/.cache/deepfold3-triton`).
- `DEEPFOLD3_TRITON_NO_CACHE=1`: Disable caching (re-tune every run).

> **Containers where `$HOME` is ephemeral (e.g. kakao).** The training image runs as a
> user whose real `$HOME` (`/home/mpiuser`) is *not* the persistent volume — the PVC is
> mounted at `/home/jovyan`. `Path.home()/.cache` would therefore land the cache on
> ephemeral per-pod storage, so all 64 ranks would re-autotune and race-write. The cache
> now honors `XDG_CACHE_HOME` (which the trainjob sets to `/home/jovyan/.cache`), and the
> trainjob also sets `DEEPFOLD3_TRITON_CACHE_DIR=/home/jovyan/.cache/deepfold3-triton`
> explicitly. Both point the cache at the shared PVC. Note this is a *different* cache from
> Triton's own `TRITON_CACHE_DIR` (compiled binaries) — they must each be redirected.

#### Warming the cache

With no hardcoded table, **every** GPU (B200 included) starts from an empty cache and
autotunes each shape on first encounter — wasted compute under DDP, where every rank
autotunes the same shapes redundantly and can desync. Warm the cache **on a single rank**
before launching DDP training (and point it at the shared/persistent path so all ranks
reuse it):

There are two modes:

**`--real-model` (recommended).** Builds the actual `DeepFold3` (same `ModelConfig` path as
`deepfold3 train`) and runs `forward_training` + backward at `fixed_bucket_n` (= `max_tokens`,
default 384) with `batch_size_per_rank` and the diffusion params read from `--config`. Every
cached shape and kernel specialization therefore matches training exactly — there is no
hand-maintained shape list to drift out of sync (the gap that previously left `dual_gemm`
uncached). It warms the per-chunk diffusion size (`min(diffusion_chunk_size, diffusion_batch_size)`).

By default every repeated block stack is **collapsed to depth 1** (`collapse_to_depth_1`):
PairFormer 48→1, diffusion transformer 24→1, Evoformer / confidence / template / atom
transformer →1. The Triton autotune cache is keyed by `(kernel, shape)` — the layer index is
not part of the key and every layer in a stack shares identical channel dims — so one layer
warms the *identical* set of cache entries as the full stack. Coverage is unchanged; the other
47 PairFormer layers were pure redundant fwd/bwd. This keeps activation memory at ~a single
layer (so warming at the production `--msa-depth` no longer OOMs at N=384) and runs in seconds.
Pass `--full-depth` to build the whole model instead — only an escape hatch; it costs ~N× more
memory/time for no extra coverage. `--msa-depth` defaults to `model.msa_crop_size`; lower it
(or `--buckets`) only if the forward still OOMs.

```bash
deepfold3 warmup-triton --config configs/initial.yaml --real-model
```

**Synthetic (default, lighter).** Drives each fused kernel at hand-enumerated `(M, N, K)`
shapes without building the model — faster and lower-memory, but the shape list must be kept
in sync with the model (it can miss shapes/specializations the real run hits):

```bash
deepfold3 warmup-triton --config configs/initial.yaml
```

The synthetic script:

1. Loads channel dimensions from the model config (falls back to defaults).
2. Iterates over every bucket size in `bucket.BUCKET_SIZES ≤ 384` (pass `--buckets` to add
   larger crops, e.g. 640/768, if you train at those sizes).
3. For each bucket, drives `FusedLayerNormLinear`, `FusedAdaLayerNormLinear`,
   `fused_evo_attention`, and `FusedGatedDualGEMM` at the (M, N, K) shapes the model uses.
   (These kernels are stateless — the script passes LN/linear/gate weights at call time.)
4. The first call for each shape triggers `triton.autotune`; the result is persisted to
   `$CACHE/{kernel}.{major}.{minor}.json` keyed by GPU fingerprint.
5. Subsequent processes (training ranks) load the persisted cache via the pre-hook
   injection and skip the benchmark phase.

Total warmup cost on B200: order of minutes (one-time per GPU model, per shape set). The
candidate config space is the `_GEMM_CONFIGS` list in `triton/layernorm_linear.py`.

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

## TiledLinear (Memory-Efficient Pair Transitions)

Pair transition blocks (`TransitionBlock` with C=64/128) operate on `[N, N, C]` tensors.
At N=384 with 4x expansion, each stores `[384, 384, 512]` intermediates for backward —
~150 MB per transition in bf16. With ~52 pair transitions in the Evoformer/PairFormer
trunk, this totals ~8 GB in activation memory.

`TiledLinear` wraps any transition block and chunks the input along `dim=-2` (pair rows).
Each chunk is processed independently, and intermediates are recomputed during backward
instead of being stored.

**How it works:**

- **Training**: Custom `torch.autograd.Function` runs forward under `torch.no_grad()`,
  saves only the input. During backward, recomputes each chunk's forward with
  `torch.enable_grad()` and accumulates parameter gradients across chunks.
- **Inference**: Optionally chunks the forward pass for peak memory reduction on very
  long sequences (enabled via `tiled_linear_inference=True`).

**Why it's exact:** All operations in `TransitionBlock` (LayerNorm, Linear, SwiGLU) act
on `dim=-1` (channels) independently per spatial position. Chunking along `dim=-2` produces
bitwise identical forward results. Backward input gradients are also bitwise identical.
Parameter gradients may differ at ULP level due to fp32 accumulation order (~1e-6 max diff).

**Memory savings:** At N=384 with 3 shards (auto heuristic): ~67% reduction in pair
transition activation memory (~5 GB saved). Cost: ~2x FLOPs for pair transitions only.

**Config:**

```python
OptimConfig(
    use_tiled_linear=True,        # Enable for pair-scale transitions (C=64/128)
    tiled_linear_shards=0,        # 0 = auto: ceil(spatial_dim / channel_dim)
    tiled_linear_inference=False,  # Also tile during inference (long sequences)
)
```

Enabled by default in `OptimConfig.recommended()`. Composable with Triton fused kernels —
the inner module can be `FusedTransitionBlock` with `FusedSwiGLU`.

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
  "shape": { "N": 256, "C": 128, "H": 4 },
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
├── plot_results.py        # Matplotlib grouped bar charts (per-suite PNGs)
├── optimization_report.py # Slide-deck PDF: summary, verification, per-op charts, table
├── run_all.py             # Orchestrator: verify -> bench all -> plot + report
└── results/               # Output directory (JSON + PNG + PDF)
```
