# Weight Initialization

DeepFold3 follows AlphaFold 3's initialization scheme. This document describes the scheme and compares it with Protenix and Boltz-1.

## Initializer Types

DeepFold3 implements AF3's three named initializers in `af3_linear_init_()` (`model/primitives.py`):

| Name       | Distribution             | Stddev                    | Use                                                     |
| ---------- | ------------------------ | ------------------------- | ------------------------------------------------------- |
| `'linear'` | Truncated normal [-2, 2] | sqrt(1 / fan_in) / 0.8796 | Default for all projections                             |
| `'relu'`   | Truncated normal [-2, 2] | sqrt(2 / fan_in) / 0.8796 | Pre-nonlinearity layers (SwiGLU first linear, pair MLP) |
| `'zeros'`  | Constant 0               | —                         | Output projections, gating, conditioning, recycling     |

The truncation factor 0.87962566 is `scipy.stats.truncnorm.std(a=-2, b=2)` — it adjusts the requested stddev so the actual variance of the truncated distribution equals the intended variance.

`LinearNoBias` accepts `initializer` as a constructor argument (default `'linear'`). For `nn.Linear` with bias, `af3_linear_init_()` is called on the weight and `nn.init.zeros_()` on the bias.

## Per-Module Initialization

### Evoformer Trunk

| Module                | Layer                                                                                                                                     | Init     | Rationale                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------- |
| **Input embeddings**  | `left_single`, `right_single`, `position_activations`, `single_activations`, `msa_activations`, `extra_msa_target_feat`, `bond_embedding` | `linear` | Standard feature projections                        |
| **Recycling**         | `prev_embedding`, `prev_single_embedding`                                                                                                 | `zeros`  | First recycle contributes nothing; identity at init |
| **TransitionBlock**   | `linear1` (SwiGLU)                                                                                                                        | `relu`   | 2x variance for gated nonlinearity                  |
| **TransitionBlock**   | `linear2` (output)                                                                                                                        | `zeros`  | Residual block starts as identity                   |
| **GridSelfAttention** | `to_q`, `to_k`, `to_v`, `pair_bias`                                                                                                       | `linear` | Standard projections                                |
| **GridSelfAttention** | `to_g` (gating)                                                                                                                           | `zeros`  | sigmoid(0) = 0.5; gate starts half-open             |
| **GridSelfAttention** | `out_proj` (output)                                                                                                                       | `zeros`  | Residual block starts as identity                   |
| **MSAAttention**      | `to_v`, `pair_logits`                                                                                                                     | `linear` | Standard projections                                |
| **MSAAttention**      | `to_g`                                                                                                                                    | `zeros`  | Gating                                              |
| **MSAAttention**      | `out_proj`                                                                                                                                | `zeros`  | Output                                              |
| **SingleAttention**   | `to_q` (with bias)                                                                                                                        | `linear` | Bias initialized to zero                            |
| **SingleAttention**   | `to_k`, `to_v`                                                                                                                            | `linear` | Standard projections                                |
| **SingleAttention**   | `to_g`                                                                                                                                    | `zeros`  | Gating                                              |
| **SingleAttention**   | `out_proj`                                                                                                                                | `zeros`  | Output (single_cond=None path)                      |
| **TriangleMul**       | `proj` (projection)                                                                                                                       | `linear` | Standard projection                                 |
| **TriangleMul**       | `gate`                                                                                                                                    | `zeros`  | AF3 `final_init`                                    |
| **TriangleMul**       | `out_proj`                                                                                                                                | `zeros`  | Output                                              |
| **TriangleMul**       | `out_gate`                                                                                                                                | `zeros`  | Gating                                              |
| **OuterProductMean**  | `proj_a`, `proj_b`                                                                                                                        | `linear` | Standard projections                                |
| **OuterProductMean**  | `out_w`, `out_b`                                                                                                                          | `zeros`  | Output                                              |

### Template Embedding

| Layer                                               | Init                     | Rationale                                     |
| --------------------------------------------------- | ------------------------ | --------------------------------------------- |
| `embed_dgram` through `embed_query` (9 projections) | `relu`                   | AF3 `template_pair_embedding_*` uses `'relu'` |
| `output_proj`                                       | `relu`                   | AF3 `output_linear` uses `'relu'`             |
| PairFormer blocks inside templates                  | Same as trunk PairFormer | Shared architecture                           |

### Diffusion Transformer

| Module                         | Layer                                                  | Init                        | Rationale                                            |
| ------------------------------ | ------------------------------------------------------ | --------------------------- | ---------------------------------------------------- |
| **AdaLN**                      | `cond_scale` (with bias)                               | `zeros`                     | Neutral conditioning at init: sigmoid(0) = 0.5 scale |
| **AdaLN**                      | `cond_bias`                                            | `zeros`                     | Zero bias at init                                    |
| **DiffusionTransformerBlock**  | `to_q` (with bias)                                     | `linear`                    | Bias initialized to zero                             |
| **DiffusionTransformerBlock**  | `to_k`, `to_v`                                         | `linear`                    | Standard projections                                 |
| **DiffusionTransformerBlock**  | `to_g`                                                 | `zeros`                     | Gating                                               |
| **DiffusionTransformerBlock**  | `out_proj`                                             | `linear`                    | Default — AdaLN-zero gate handles output scaling     |
| **DiffusionTransformerBlock**  | `zero_cond`                                            | `zeros` weight, bias = -2.0 | sigmoid(-2) ~ 0.12; near-zero gate at init           |
| **DiffusionTransformerBlock**  | `ffn_linear1`                                          | `relu`                      | SwiGLU first layer                                   |
| **DiffusionTransformerBlock**  | `ffn_linear2`                                          | `linear`                    | Default — gate handles output scaling                |
| **DiffusionTransformerBlock**  | `ffn_zero_cond`                                        | `zeros` weight, bias = -2.0 | Near-zero gate at init                               |
| **DiffusionConditioningBlock** | `pair_cond_proj`, `single_cond_proj`, `noise_emb_proj` | `linear`                    | Standard projections                                 |
| **DiffusionHead**              | `single_cond_embedding_projection`                     | `zeros`                     | AF3 `final_init`                                     |

### Atom Cross-Attention

| Module                      | Layer                                             | Init                        | Rationale                                            |
| --------------------------- | ------------------------------------------------- | --------------------------- | ---------------------------------------------------- |
| **AtomPerAtomConditioning** | `embed_ref_*`, `single_to_pair_*`, `embed_pair_*` | `linear`                    | Standard feature projections                         |
| **AtomPerAtomConditioning** | `pair_mlp1`, `pair_mlp2`                          | `relu`                      | Pre-ReLU layers                                      |
| **AtomPerAtomConditioning** | `pair_mlp3`                                       | `zeros`                     | Residual MLP output                                  |
| **AtomCrossAttTransformer** | `to_q` (with bias)                                | `linear`                    | Bias initialized to zero                             |
| **AtomCrossAttTransformer** | `to_k`, `to_v`                                    | `linear`                    | Standard projections                                 |
| **AtomCrossAttTransformer** | `to_g`                                            | `zeros`                     | Gating                                               |
| **AtomCrossAttTransformer** | `out_proj`, `ffn linear2`                         | `linear`                    | Default — gate handles output                        |
| **AtomCrossAttTransformer** | `zero_cond`                                       | `zeros` weight, bias = -2.0 | Near-zero gate                                       |
| **AtomCrossAttTransformer** | `ffn linear1`                                     | `relu`                      | SwiGLU first layer                                   |
| **AtomCrossAttTransformer** | all `*_cond_scale`, `*_cond_bias`                 | `zeros`                     | Neutral AdaLN conditioning                           |
| **AtomEncoder**             | `embed_trunk_single`                              | `zeros`                     | AF3 `final_init` — trunk conditioning starts at zero |
| **AtomEncoder**             | `embed_trunk_pair`                                | `zeros`                     | AF3 `final_init`                                     |
| **AtomDecoder**             | `pos_update`                                      | `zeros`                     | Position updates start at zero                       |

### Output Heads

| Module             | Layer                                                      | Init     | Rationale                      |
| ------------------ | ---------------------------------------------------------- | -------- | ------------------------------ |
| **DistogramHead**  | `half_logits`                                              | `zeros`  | Uniform distance prior at init |
| **ConfidenceHead** | `left/right_target_feat_project`, `distogram_feat_project` | `linear` | Feature projections            |
| **ConfidenceHead** | `plddt_logits`                                             | `zeros`  | Uniform confidence prior       |
| **ConfidenceHead** | `pae_logits`                                               | `zeros`  | Uniform error prior            |
| **ConfidenceHead** | `pde_half_logits`                                          | `zeros`  | Uniform distance error prior   |
| **ConfidenceHead** | `exp_resolved_logits`                                      | `zeros`  | Uniform resolved prior         |

## Design Rationale

The initialization scheme serves three purposes:

1. **Residual identity at init.** Zero-initialized output projections ensure each residual block computes `x + 0 = x` at initialization. In a 48-layer PairFormer with 10 recycles (480 effective layers), random output init would inject noise at every layer, making early gradients chaotic.

2. **Neutral conditioning at init.** Zero-initialized AdaLN projections give `sigmoid(0) = 0.5` scale and zero bias — the conditioning starts neutral and learns to modulate. Zero-initialized `zero_cond` with `bias = -2.0` gives `sigmoid(-2) ~ 0.12` — the diffusion transformer blocks barely contribute at init, ramping up as training progresses.

3. **Appropriate variance scaling.** Fan_in-scaled truncated normal keeps activation variance stable across layers. The `'relu'` variant (2x scale) compensates for the variance reduction caused by gated nonlinearities (SwiGLU).

## Comparison with Other Implementations

### Protenix

Protenix uses an identical scheme — custom `Linear` / `LinearNoBias` classes with an `initializer` parameter (`'default'` = AF3 `'linear'`, `'relu'`, `'zeros'`). The `trunc_normal_init_` function matches AF3's formula exactly. The `BiasInitLinear` class handles `zero_cond`-style layers with `biasinit=-2.0`.

### Boltz-1

Boltz-1 (`src/boltz/model/layers/initialize.py`) defines named initializer functions:

| Boltz-1                | AF3 Equivalent             | Notes                            |
| ---------------------- | -------------------------- | -------------------------------- |
| `lecun_normal_init_`   | `'linear'`                 | Same formula                     |
| `he_normal_init_`      | `'relu'`                   | Same formula                     |
| `final_init_`          | `'zeros'`                  | `weights.fill_(0.0)`             |
| `gating_init_`         | `'zeros'` (no bias in AF3) | Boltz-1 adds explicit `bias=1.0` |
| `glorot_uniform_init_` | —                          | Xavier uniform; not used by AF3  |

Key differences from AF3:

| Layer                      | AF3                            | Boltz-1                                 |
| -------------------------- | ------------------------------ | --------------------------------------- |
| Triangular attention Q/K/V | `'linear'` (truncated normal)  | `glorot_uniform` (Xavier)               |
| TriangleMul gates          | `final_init`, `use_bias=False` | `gating_init_` with explicit `bias=1.0` |
| Transition fc1/fc2         | `'relu'` / `'zeros'`           | `lecun_normal` / `lecun_normal`         |

Despite these differences, Boltz-1 follows the same core principles — zero output projections, zero gating weights, zero recycling projections.

### Cross-Implementation Consensus

All four implementations agree on:

1. **Output projections**: zeros (residual identity)
2. **Gating weights**: zeros
3. **Recycling projections**: zeros (clean first pass)
4. **AdaLN conditioning**: zeros (neutral start)
5. **Default projections**: fan_in-scaled truncated normal
6. **Pre-nonlinearity layers**: 2x variance scaling
