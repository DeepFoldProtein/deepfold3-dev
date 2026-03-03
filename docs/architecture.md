# Architecture Deep-Dive

DeepFold3 is a diffusion-based generative model for biomolecular structure prediction, architecturally equivalent to AlphaFold 3. This document covers the forward pass, module internals, and key design decisions.

## Forward Pass Overview

```
Features dict (featurizer output)
  │
  ├─ create_relative_encoding()        → rel_feat    [N, N, 139]
  ├─ one_hot(res_type) + profile       → raw_feat    [N,  63]
  │    + deletion_mean
  ├─ EvoformerConditioningEncoder()     → evo_cond    [N, 384]
  │    (ref atoms → atom transformer
  │     → masked mean pool to tokens)
  │
  ├─ concat [raw_feat, evo_cond]       → target_feat [N, 447]
  ├─ MSA feat (one_hot + has_deletion  → msa_feat    [S, N,  34]
  │    + deletion_value)
  │
  ╰─ Evoformer trunk (10 recycles)
       ├─ Input projections
       │    single = proj(target_feat)               [N, 384]
       │    pair   = left(tf)⊗right(tf) + rel_feat   [N, N, 128]
       │    msa    = proj(msa_feat) + proj(tf)        [S, N,  64]
       ├─ + recycled prev_single, prev_pair (.detach())
       ├─ MSA stack:       4 × EvoformerBlock
       ╰─ PairFormer stack: 48 × PairFormerBlock
              → single [N, 384]  +  pair [N, N, 128]

  DiffusionHead.sample()
       200-step denoising with random augmentation
       → atom_positions [S, N, M, 3]     (S=num_samples, M=24)

  ConfidenceHead (per sample)
       4-layer PairFormer on embeddings + dgram from predicted positions
       → pLDDT [S, N, M*50],  PAE [S, N, N, 64]
         PDE   [S, N, N, 64], exp_resolved [S, N, M*2]

  DistogramHead
       half_logits(pair) + transpose  →  [N, N, 64]
```

Key constants (`model/model.py`):
- `_RAW_TARGET_FEAT_DIM = 63` (31 one-hot + 31 profile + 1 deletion_mean)
- `_AATYPE_NUM_CLASSES = 31`
- `_MSA_FEAT_DIM = 34` (32 one-hot + 1 has_deletion + 1 deletion_value)
- `_MSA_NUM_CLASSES = 32`

## Module-by-Module Breakdown

### `primitives.py` — Building Blocks

| Class | Description |
|---|---|
| `LinearNoBias` | `nn.Linear(bias=False)` — matches AF3's `hm.Linear` default (`use_bias=False`) |
| `LayerNorm` | `nn.LayerNorm(elementwise_affine=True)` — always has learnable scale+offset |
| `AdaLayerNorm` | Adaptive LN: base LN (no learnable params) + conditioning → sigmoid(scale) * x + bias |
| `SwiGLU` | `swish(a) * b` where `[a, b] = linear(x).chunk(2)` |
| `TransitionBlock` | LN → SwiGLU feed-forward; `linear1: [c → 2*c*factor]`, `linear2: [c*factor → c]`; default factor=4 |
| `AdaTransitionBlock` | AdaLN-zero conditioned FF for diffusion; `zero_cond` gate init at bias=-2; default factor=2 |

### `config.py` — Configuration Hierarchy

`ModelConfig` is the top-level config, containing nested dataclasses:

| Config | Key Defaults |
|---|---|
| `ModelConfig` | `num_recycles=10` |
| `EvoformerConfig` | `num_layer=4`, `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `num_msa=1024`, `max_relative_idx=32`, `max_relative_chain=2` |
| `PairFormerConfig` | `num_layer=48`, `pair_channel=128`, `seq_channel=384`, `single_attention_num_head=16` |
| `AtomCrossAttConfig` | `per_token_channels=768`, `per_atom_channels=128`, `per_atom_pair_channels=16`, `queries_subset_size=32`, `keys_subset_size=128` |
| `AtomTransformerConfig` | `num_blocks=3`, `num_head=4`, `key_dim=128`, `value_dim=128` |
| `DiffusionTransformerConfig` | `num_blocks=24`, `num_head=16`, `super_block_size=4`, `num_intermediate_factor=2` |
| `DiffusionConfig` | `steps=200`, `num_samples=5`, `sigma_data=16.0`, `smin=0.0004`, `smax=160.0`, `noise_p=7.0`, `gamma_0=0.8`, `gamma_min=1.0`, `noise_scale=1.003`, `step_scale=1.5` |
| `ConfidenceConfig` | `num_layer=4`, `num_plddt_bins=50`, `num_distogram_bins=64`, `max_error_bin=31.0`, `max_atoms_per_token=24` |

### `embedders.py` — Relative Encoding

`create_relative_encoding(token_index, residue_index, asym_id, entity_id, sym_id)` → `[N, N, 139]`

Components:
- **rel_pos** (66 dims): residue_index diff, clamped to `±max_relative_idx=32`, one-hot + extra bin for cross-chain
- **rel_token** (66 dims): token_index diff within same residue, same encoding
- **entity_same** (1 dim): `entity_id[i] == entity_id[j]` as float
- **rel_chain** (6 dims): sym_id diff, clamped to `±max_relative_chain=2`, one-hot + extra bin for cross-entity

### `attention.py` — Three Attention Variants

**`GridSelfAttention`** — Row/column attention on pair representation `[N, N, C]`
- `transpose=False` for row-wise, `transpose=True` for column-wise
- 4 heads, `head_dim = max(c // num_heads, 16)`
- Pair bias: `pair → LinearNoBias → [H, N, N]` broadcast over all rows (`pb.unsqueeze(0)`)
- Gated output: `out * sigmoid(to_g(x))`

**`MSAAttention`** — Row-wise MSA attention `[S, N, msa_channel]`
- 8 heads; **no Q/K projections** — attention weights derived from pair activations
- `pair_logits(pair_norm(pair))` → shared attention weights `[H, N, N]` across all S sequences
- `einsum("bhqk,bskhc→bsqhc", w, v)` applies shared weights to per-sequence values

**`SingleAttention`** — Self-attention on single representation `[N, 384]`
- 16 heads, `key_dim=384`, `value_dim=384`
- Receives **pre-computed** `pair_logits [H, N, N]` from `PairFormerBlock.single_to_pair`
- Note: `to_q` uses `nn.Linear` (with bias); all other projections use `LinearNoBias`

### `ops.py` — Triangle Operations

**`TriangleMul`** — Triangle multiplicative update
- Outgoing: `equation="ikc,jkc->ijc"` (default); Incoming: `equation="kic,kjc->ijc"`
- LayerNorm → project to 2×channels → sigmoid gate → einsum → LayerNorm → project back + sigmoid gate

**`OuterProductMean`** — MSA-to-pair information flow
- MSA `[S, N, c_msa]` → two projections `[S, N, 32]` → outer product → mean over S → pair `[N, N, c_pair]`

### `evoformer.py` — Trunk

**`EvoformerBlock`** (×4): MSA column attention → MSA row attention (pair-biased) → MSA transition → OuterProductMean → pair TriangleMul (out+in) → pair GridSelfAttention (row+col) → pair transition

**`PairFormerBlock`** (×48): pair TriangleMul (out+in) → pair GridSelfAttention (row+col) → pair transition → SingleAttention (pair-biased) → single transition

**`Evoformer`** input projections:
- `target_feat [N, 447]` → single `[N, 384]`, pair `[N, N, 128]` (outer sum of left/right projections)
- `msa_feat [S, N, 34]` → msa `[S, N, 64]` (+ broadcast target_feat projection)
- `rel_feat [N, N, 139]` → pair (additive)
- Recycled `prev_pair` and `prev_single` added through LayerNorm + projection

### `atom_cross_attn.py` — Atom-Level Processing

**`AtomSubsets`** — Flat atom layout with rectangular block attention:
- Packs real atoms from `[N, M=24]` grid into flat array, padded to multiple of Q=32
- Non-overlapping **query subsets** of Q=32 atoms (~6 residues each)
- Overlapping **key windows** of K=128 atoms centered on each query subset
- Fields: `ta_to_q_idx [S,Q]`, `q_to_k_idx [S,K]`, `token_for_query [S,Q]`, `token_for_key [S,K]`, etc.

**`EvoformerConditioningEncoder`**:
- Ref atom features (element, charge, name_chars, mask) → per-atom embeddings
- 3-block atom transformer with pair conditioning
- Masked mean pool over atoms per token → `[N, per_token_channels=384]`
- This becomes `evo_cond`, concatenated with `raw_feat` to form `target_feat [N, 447]`

**`AtomCrossAttEncoder`** / **`AtomCrossAttDecoder`**:
- Encoder: maps noisy atom positions to flat subset space, conditions with trunk single/pair
- Decoder: maps denoised subset features back to `[N, M, 3]` atom positions
- Trunk pair broadcast: `pair[token_for_query, token_for_key]` → `[S, Q, K, pair_channels]`

### `diffusion.py` — Denoising

**`DiffusionTransformer`** — 24 blocks, organized in super-blocks of 4:
- Each block: AdaLN → self-attention (pair-biased) → AdaTransitionBlock → atom cross-att encode/decode
- Conditioning: `single_cond [N, 384+447=831]`, `pair_cond [N, N, 128+139=267]`, noise embedding (Fourier, 256-dim)
- EDM-style skip/output scaling: `result = skip_scaling * noisy_pos + out_scaling * pos_update`

**`DiffusionHead.sample()`** — 200-step reverse process:
1. `t = linspace(0, 1, 201)` → `sigma = noise_schedule(t)` (cosine schedule with `sigma_data=16.0`)
2. Initial noise: `x ~ N(0, smax^2) * atom_mask`, shape `[S, N, M, 3]`
3. Each step: `_random_augmentation()` → denoise → Euler update with stochastic gamma injection
4. Gamma: `gamma_0=0.8` if `sigma_next > gamma_min=1.0`, else 0 (no stochasticity near convergence)

### `confidence.py` — Confidence Head

4-layer PairFormer operating on:
- **Pair input**: target_feat outer-sum `[N,N,128]` + distogram from predicted positions (39 bins, 3.25-50.75 Å range)
- **Single input**: single representation `[N, 384]`

Output projections (each preceded by LayerNorm):
- **pLDDT**: `LinearNoBias(384, M*50=1200)` → `[S, N, 1200]`
- **PAE**: `LinearNoBias(128, 64)` → `[S, N, N, 64]`
- **PDE**: `LinearNoBias(128, 64)` → symmetrized → `[S, N, N, 64]`
- **exp_resolved**: `LinearNoBias(384, M*2=48)` → `[S, N, 48]`

Confidence is computed **per diffusion sample** — the head runs S times.

### `model.py` — Top-Level Orchestrator

**`DeepFold3`**: wires all modules together in `forward()`. Orchestrates recycling, calls evoformer, diffusion, confidence, and distogram heads.

**`DeepFold3Output`** — 7 fields:
- `atom_positions [S, N, M, 3]`, `atom_mask [N, M]`
- `plddt [S, N, M*50]`, `pae [S, N, N, 64]`, `pde [S, N, N, 64]`, `exp_resolved [S, N, M*2]`
- `distogram [N, N, 64]`

**`DistogramHead`**: single `LinearNoBias(128, 64)` + symmetrize. 64 bins from 2.3125 to 21.6875 Å.

Also exposes `compute_distogram(features, num_recycles)` which runs only the evoformer (no diffusion) for fast validation.

## Atom Representation

Two parallel layouts coexist throughout the model:

**Token-level** `[N]` — One representative atom per token:
- Protein: Cα (one token per residue)
- Nucleic acid: C1' (one token per residue)
- Ligand: the single atom itself (one token per atom)

**Token-atoms** `[N, M=24]` — All heavy atoms per token, padded to `MAX_ATOMS_PER_TOKEN=24`:
- `ref_pos [N, M, 3]`: ideal CCD coordinates, randomly rotated during training
- `ref_mask [N, M]`: which atom slots are occupied
- `ref_element [N, M]`, `ref_charge [N, M]`, `ref_atom_name_chars [N, M, 4]`

**Flat subset mapping** (`AtomSubsets`):
- The `[N, M]` grid is flattened to a 1D array of real atoms (mask=True), padded to a multiple of Q=32
- Partitioned into non-overlapping query subsets of Q=32 atoms
- Each query subset gets an overlapping key window of K=128 atoms (centered on the query)
- `build_atom_subsets(ref_mask, Q=32, K=128)` computes all index tensors
- Used by atom cross-attention in both encoder and decoder

## Recycling

- Default `num_recycles=10`; the evoformer runs `num_recycles + 1` times total
- `prev_pair [N, N, 128]` and `prev_single [N, 384]` carry information across iterations
- `.detach()` between recycles prevents gradient flow through the recycling chain
- Recycled representations are added to fresh projections via LayerNorm + LinearNoBias

## Diffusion Sampling

The diffusion head implements a 200-step reverse denoising process:

**Noise schedule**: `sigma(t) = sigma_data * (smax^(1/p) + t * (smin^(1/p) - smax^(1/p)))^p`
- `sigma_data=16.0`, `smin=0.0004`, `smax=160.0`, `p=7.0`

**Random augmentation** (critical — applied at every denoising step):
1. Center: subtract masked center of mass
2. Rotate: Gram-Schmidt orthogonalization from two random N(0,1) vectors → SO(3)
3. Translate: add random N(0,1) vector
4. Zero out masked atoms

**Stochastic sampling**:
- `gamma = gamma_0 (0.8)` when `sigma_next > gamma_min (1.0)`, else 0
- `sigma_hat = sigma * (1 + gamma)` — inflated noise level
- Inject noise: `x_hat = x + noise_scale * sqrt(sigma_hat^2 - sigma^2) * noise`
- Euler step: `x_next = x_hat + step_scale * (sigma_next - sigma_hat) * (x_hat - denoised) / sigma_hat`

**Output**: `[num_samples, N, M, 3]` atom positions (default `num_samples=5`).

## Batch Dimension Support

All core model modules support both unbatched and batched (leading B dim) inputs:
- Pattern: `has_batch = x.dim() == expected_batched_dim; if not: unsqueeze(0) → compute → squeeze(0)`
- Supported: attention, ops, evoformer, diffusion, confidence, embedders
- **Not yet batched**: `atom_cross_attn.py` (subset building is per-sample)

## Weight Compatibility

- All weights load from AF3 Haiku params via `convert/af3.py` (Haiku scope → PyTorch state_dict)
- Zero missing / zero unexpected keys (validated in `test_weight_loading.py`)
- Only skipped: **template embedding** (not yet implemented, listed in `skip_prefixes`)
- Haiku `[in, out]` transposed to PyTorch `[out, in]` during conversion
- Each module's docstring documents its corresponding Haiku scope hierarchy
