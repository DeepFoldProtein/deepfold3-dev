# Single-track stability — experiment log (2026-05-03)

Working notebook for the ROS / pair_bias_gate / distogram-init / orthogonal-init /
AdaLN cond_bias experiments aimed at controlling trunk single representation growth.

Written so that **a future operator can resume the investigation from a cold
read**: every intervention, the code path that implements it, the YAML that
turned it on, the kakao run name, and the empirical takeaway.

---

## 1. Problem statement

Production schedule-free training (`schedule_free.yaml` on top of
`initial.yaml`, schedule-free AdamW, lr=1e-3, wd=0.05, 12 nodes × 8 B200,
batch_size_per_rank=3, accumulate_grad_batches=1) showed monotonic
divergence of trunk single representation magnitude across PairFormer
layers. By step 4300 of the deepnet baseline, trunk-end RMS(`single`)
reached ~232 (vs ~6 at step 500), driving downstream loss instability.

Pair representation was substantially more stable. The instability is
**single-specific**.

The pre-existing remediation was `pairformer.deepnet_residual_scale=true`
(per-branch DeepNet scaling: 1/√(2L) for tri-mul/single-attn/single-ffn,
1/√L for pair-attn/pair-transition). It only delayed the explosion.

## 2. Diagnostic toolkit

### 2.1 Per-layer trunk single dynamics probe

Hook `PairFormerBlock.forward` per layer and capture, for each of the two
single-track residual updates (`single_attn`, `single_transition`):

- `s_k` (residual stream before the update)
- `u_k` (raw F-output of the branch)
- `γ_k` (effective scale = deepnet_branch_factor × ROS_factor)
- `RMS(u_k)`, `RMS(γ_k · u_k)`
- `cos(s_k, u_k)` — alignment of the update with the residual

Synthetic-input version of this script lives at
`/tmp/probe_single_trunk.py` (reproduced inline below; copy to a permanent
location if needed):

```python
# Synthetic features feed evoformer trunk; record per-layer
# {RMS(s), RMS(u_attn), RMS(u_tr), cos(s,u_attn), cos(s,u_tr)}.
# See git history of this doc for the full hook implementation.
```

### 2.2 Weight-magnitude probes

Per-layer `out_std = std(x @ W.T)` for unit-variance Gaussian input,
covering the "wake-up" output projections (zero-init by AF3 convention):

- `single_transition.linear2` (the SwiGLU output projection, alias `linear_out`)
- `single_attn.out_proj`
- `pair_transition.linear2`
- DiffusionTransformer per-block `attn_cond_bias` / `ffn_cond_bias`

Reference scripts: `/tmp/probe_cond_bias.py`, `/tmp/probe_linear_out.py`,
`/tmp/probe_adab_100.py`. Pattern is the same — load checkpoint, walk a
fixed list of layer indices, report `std(W)`, `frobenius(W)`,
`out_std(x @ W.T)` for `x ~ N(0, I)`.

## 3. Reference dynamics (deepnet baseline)

### 3.1 Trunk-end RMS(single) trajectory

Probed on `df3-12n-dnet-wd05-lr1e3-0502-081150` checkpoints (synthetic
input N=64, S=16):

| step | trunk-end RMS(s) |
| ---- | ---------------- |
| 500  | 6.17             |
| 1000 | 39.87            |
| 4300 | 231.92           |

### 3.2 Per-layer alignment cos(s, u) (transition; attention is similar but slightly weaker)

| step \ layer | L0    | L5    | L10   | L20   | L30   | L47   |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- |
| 500          | +0.01 | +0.02 | +0.10 | +0.43 | +0.81 | +0.96 |
| 1000         | -0.06 | +0.66 | +0.82 | +0.98 | +0.99 | +0.98 |
| 4300         | +0.25 | +0.92 | +0.95 | +0.96 | +0.96 | +0.95 |

Reading: alignment "wave" forms in the deepest layers first (already
+0.96 at L47 by step 500), propagates upward. By step 4300 every layer
is locked in at cos > +0.92.

In the cos→1 regime, `||s+γu||² ≈ ||s||² + 2γ⟨s,u⟩` — the cross term
dominates and growth is quasi-linear in `Σ γ_k · ||u_k||` (not the safer
`Σ γ_k² · ||u_k||²`). **ROS or deepnet forward suppression cannot stop
the alignment dynamic; only the per-step amplitude is reduced.**

### 3.3 Output-projection wake-up (vs zeros init)

Same checkpoints, `out_std` per layer (typical layer band L0..L47):

| step | single_transition.linear2 | single_attn.out_proj | pair_transition.linear2 | attn_cond_bias | ffn_cond_bias |
| ---- | ------------------------- | -------------------- | ----------------------- | -------------- | ------------- |
| 100  | 0.014~0.021               | 0.008~0.009          | 0.011~0.014             | 0.007~0.008    | 0.010~0.011   |
| 200  | 0.034~0.040               | 0.014~0.024          | 0.029~0.031             | 0.012~0.013    | 0.015~0.016   |
| 300  | 0.051~0.055               | 0.021~0.030          | 0.050~0.053             | 0.018~0.019    | 0.018~0.023   |

Reference for sizing init-based interventions: pick a `gain`/`β` close
to the prev-run magnitude at the step you want to "skip past".

### 3.4 Diffusion conditioning forward gain (step 4300)

| layer | σ(attn_cond_scale · LN(single_cond)) | σ(zero_cond)  |
| ----- | ------------------------------------ | ------------- |
| L00   | 0.499 ± 0.103                        | 0.127 ± 0.052 |
| L08   | 0.500 ± 0.113                        | 0.127 ± 0.051 |
| L12   | 0.502 ± 0.119                        | 0.126 ± 0.052 |
| L20   | 0.499 ± 0.113                        | 0.126 ± 0.050 |

→ AdaLN scale stays ~0.5, gate stays ~0.12 across all 24 blocks. Diffusion
forward conditioning is weak; **diffusion is unlikely to be the direct
forward amplifier of single growth**, but its gradient through trunk
single weights still pushes the alignment dynamic.

## 4. Code changes (cumulative, all behind config flags, default-off)

All flags default to the AF3 convention — turning all flags off
reproduces the pre-2026-05-03 architecture bit-for-bit (verified by
`tests/test_evoformer.py::TestLnOutScaleWiring::test_default_off_keeps_unity`
and similar).

### 4.1 Residual Output Scaling (ROS) — `evoformer.py` / `primitives.py` / configs

`x ← x + s · F(x)` on every residual sub-module of every PairFormer-like
stack (Evoformer MSA, PairFormer trunk, ConfidenceHead PairFormer,
Template PairFormer).

Two mutually exclusive policies:

- `residual_output_scaling: bool = False` — per-layer `s = 1/√(layer_idx+1)`
- `uniform_residual_scale: bool = False` — depth-flat `s = 1/√num_layer`

Helper: `block_residual_scale(layer_idx, num_layer, per_layer, uniform)` in
`primitives.py`. Composes multiplicatively with `deepnet_residual_scale` if
both are on.

`single_to_pair_norm`, `MSAAttention.pair_norm`, `_TriangleMulBase.norm_center`
are **not scaled** (side-input LNs, not residual junctions).

`LayerNorm.output_scale` exists as a primitive but no production config
sets it; it's currently unused (left in place for future).

### 4.2 Per-head learnable PairBias gate — `attention.py`, `PairFormerConfig`

In `SingleAttention` only:

- `pair_bias_gate: bool = False`
- `pair_bias_gate_init_tau: float = 0.25`

Adds `pair_bias_gate_a ∈ R^[num_heads]` parameter, `τ_h = exp(a_h)`, init
`a_h = log(τ_0)`. Inside `_compute_baseline`:

```python
pair_logits = pair_logits * exp(pair_bias_gate_a).view(1, H, 1, 1)
```

Lets each head re-balance PairBias-vs-QK contribution to attention
softmax. 1-D parameter so auto-routes to the no-decay AdamW group via
`_NO_DECAY_NAME_SUBSTRINGS` 1-D rule.

### 4.3 Distogram half_logits small-normal init — `model.py`, `DistogramConfig`

`DistogramHead.half_logits` was AF3 zeros init; flag:

- `half_logits_init_alpha: float = 0.05` (default 0.05 — non-zero by
  default after this change)

Init: `W ~ trunc_normal(0, (α/√fan_in)²)`. Output stddev ≈ α. α=0
recovers AF3 zeros init. Reference: pair_l2 stays bounded under the
distogram CE because the head has non-zero gradient back to `pair` from
step 0; α=0 (zeros init) leaves `pair` unconstrained until the head
wakes up (~hundreds of steps).

### 4.4 SingleTransition `linear_out` orthogonal init — `primitives.py` / `optim.py` / `PairFormerConfig`

`TransitionBlock` (under aliases: `linear1` = `linear_gate`, `linear2` =
`linear_out`, internal attribute names preserved for state-dict compat):

- `linear_out_init_gain: float | None = None`

If set, replaces zeros init on `linear2.weight` with
`nn.init.orthogonal_(W, gain=...)` (row-orthonormal, scaled by gain).

Plumbed via `make_transition_block(linear_out_init_gain=...)`. Fused
transition kernels reject this combination.

`PairFormerConfig` and `ConfidenceConfig` expose
`single_transition_linear_out_init_gain: float | None = None`.
`PairFormerBlock` applies it **only to `single_transition`**; pair/msa
transitions stay zero-init.

Rationale: zero-init has the linear_out wake up in a single direction
biased by the loss gradient, which empirically becomes alignment-favouring.
Random orthogonal start gives backward signal in all directions
immediately so the wake-up direction is not biased.

### 4.5 DiffusionTransformer AdaLN cond_bias trunc-normal init — `diffusion.py`, `DiffusionTransformerConfig`

`DiffusionTransformerBlock`:

- `cond_bias_init_beta: float | None = None`

If set, override AF3 zeros init on **both** `attn_cond_bias.weight` and
`ffn_cond_bias.weight` with `trunc_normal(0, (β/√fan_in)²)` so output
stddev ≈ β. Other AdaLN cond projections (`*_cond_scale`, `zero_cond`,
`ffn_zero_cond`) remain zeros init.

Plumbed via `DiffusionTransformer(cond_bias_init_beta=...)` →
`DiffusionHead` reads `cfg.transformer.cond_bias_init_beta`.

`AtomCrossAttTransformer` has identical AdaLN bias paths (q/k/ffn) but
is **not yet wired** — straightforward extension if needed.

### 4.6 kakao submit script tightening — `kakao/submit.sh`

JOB_NAME truncation cap reduced 52 → 45 chars. JobSet pod naming is
`<jobset>-<launcher|node>-<jobIdx>-<podIdx>-<5char-hash>` adding up to
~18 chars; 45 + 18 = 63 (DNS-1123 label limit). Anything longer is
rejected by the JobSet admission webhook with
`JobSet name is too long, pod names ... will exceed 63 characters`.

## 5. YAML configurations

All inherit `schedule_free.yaml` which inherits `initial.yaml`.

### 5.1 `configs/ros_scheduledfree.yaml`

Per-layer ROS (`residual_output_scaling: true`) on all four stacks +
`pair_bias_gate: true` (τ₀=0.25) on PairFormer/Confidence +
`single_transition_linear_out_init_gain: 0.01` +
`distogram.half_logits_init_alpha: 0.05`.

### 5.2 `configs/uniros_scheduledfree.yaml`

Same as 5.1 but `uniform_residual_scale: true` instead of per-layer.
**Not yet run.**

### 5.3 `configs/ros_scheduledfree_adabias.yaml`

`base: ros_scheduledfree.yaml` + `diffusion.transformer.cond_bias_init_beta`.
Currently set to `1.0e-2` (was `1.0e-3` — too small per probe results).

## 6. Run log

Newest first.

| Run name                                         | Started          | Killed at      | Config                                             | Notes                                                                                            |
| ------------------------------------------------ | ---------------- | -------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `df3-12n-ros-otho-adab1e2-0503-170646`           | 2026-05-03 17:06 | running        | ros_scheduledfree_adabias.yaml (β=1e-2)            | Current; β bumped 10× from prior run                                                             |
| `df3-12n-ros-pbg-d05-otho-adab-0503-163023`      | 2026-05-03 16:30 | step ~85       | ros_scheduledfree_adabias.yaml (β=1e-3)            | β too small — single_l2 ≈ vanilla; cond_bias only +10% over zero-wake-up                         |
| `df3-12n-ros-pbg-d05-otho-0503-154102`           | 2026-05-03 15:41 | <100 step      | ros_scheduledfree.yaml + ortho init                | Killed before useful data; superseded by adab variant                                            |
| `df3-12n-ros-pbg-d05-0503-132755`                | 2026-05-03 13:27 | step 300       | ros_scheduledfree.yaml (no ortho/adab)             | Reference for ortho/adab comparisons; checkpoint_100/200/300 retained                            |
| `df3-12n-ros-pbg-disto05-wd05-lr1e3-0503-132312` | 2026-05-03 13:23 | rejected       | (run name 46 chars)                                | JobSet admission rejected — too long; triggered submit.sh cap fix                                |
| `df3-12n-ros-pbg-wd05-lr1e3-0503-100513`         | 2026-05-03 10:05 | (kept briefly) | ros + pair_bias_gate (no ortho/adab/disto changes) | First ROS+gate run                                                                               |
| `df3-12n-lns-pbg-wd05-lr1e3-0503-114456`         | 2026-05-03 11:44 | replaced       | LNS naming intermediate                            | Renamed to ROS shortly after                                                                     |
| `df3-12n-lns-wd05-lr1e3-0503-100513`             | 2026-05-03 10:05 | replaced       | per-layer LNS first cut                            | `pair_logits *= s²` since removed                                                                |
| `df3-12n-lns-wd05-lr1e3-0503-090929`             | 2026-05-03 09:09 | replaced       | LNS pre-PairBias-gate iteration                    | Killed for naming/algorithm refactor                                                             |
| `df3-12n-dnet-wd05-lr1e3-0502-081150`            | 2026-05-02 08:11 | step 4300      | schedule_free.yaml (deepnet only)                  | **Reference baseline.** All probe data in §3 comes from here. checkpoint_500/1000/4300 retained. |

Run-name rules:

- DNS-1123 sanitisation in `kakao/submit.sh`, cap 45 chars.
- Trailing `MMDD-HHMMSS` is the launch timestamp (always 11 chars).
- Effective payload budget: 33 chars for the descriptive prefix.

## 7. Comparison snapshots

### 7.1 step 80 wandb-style metrics (rank 0)

| metric        | prev (no ortho/adabias) `…0503-132755` | new (ortho + adab β=1e-3) `…0503-163023` |
| ------------- | -------------------------------------- | ---------------------------------------- |
| distogram     | 2.11                                   | 2.10                                     |
| pair_l2       | 20.09                                  | 19.87                                    |
| **single_l2** | **14.96**                              | **15.67**                                |
| smooth_lddt   | 0.665                                  | 0.665                                    |
| grad_norm     | 5.57                                   | 5.28                                     |

→ Single-track behaviour effectively unchanged. Step 60 had +27%
favourable for new, but by step 80 the gap closed and reversed.

### 7.2 step 100 weight magnitudes (`out_std` of `x @ W.T` for unit input)

| module                    | prev step 100 (vanilla) | new step 100 (ortho + adab β=1e-3) | Δ    |
| ------------------------- | ----------------------- | ---------------------------------- | ---- |
| single_transition.linear2 | 0.014~0.021             | 0.018~0.024                        | +25% |
| single_attn.out_proj      | 0.008~0.009             | 0.007~0.008                        | -10% |
| pair_transition.linear2   | 0.011~0.014             | 0.011~0.014                        | 0%   |
| attn_cond_bias            | 0.007~0.008             | 0.008~0.009                        | +10% |
| ffn_cond_bias             | 0.010~0.011             | 0.011~0.012                        | +10% |

→ ortho gain=0.01 held a +25% headstart (real but modest); cond_bias
β=1e-3 was overrun by natural wake-up by step ~30 (only +10% margin
remaining at step 100).

## 8. Open questions / next experiments

1. **β=1e-2 cond_bias** (run `df3-12n-ros-otho-adab1e2-0503-170646` —
   in flight). Expected to put cond_bias init at the prev-step-100
   magnitude. Decision criteria at step 100/200:

   - **Case A** — single_l2 trajectory ≥ 30 % below prev at step 200 →
     init-based intervention works; consider further bumps (β=3e-2) or
     queue uniform-ROS variant.
   - **Case B** — trajectory matches prev → init alone is not enough;
     proceed to direct interventions (next item).
   - **Case C** — pair / distogram regressions → roll back, identify
     offending knob.

2. **Direct magnitude penalty (`alpha_single_reg`)** if Case B obtains.
   `TrainingConfig.alpha_single_reg: float | None = None` is wired
   already; setting it to e.g. 1e-4 with `single_reg_p=2` adds a
   `Σ‖single‖²` penalty. This attacks the alignment dynamic by capping
   the residual norm directly rather than slowing wake-up.

3. **`single_attn.out_proj` orthogonal init**. Mirror the
   `single_transition.linear_out` change but on the attention output
   projection. Code change is small (add a kwarg to `SingleAttention`,
   wire through `PairFormerBlock` and configs). Step 100 reference: prev
   wakes up to 0.008 → start at gain ≈ 0.005 to give a comparable
   headstart.

4. **AtomCrossAttTransformer cond_bias** (analogous to §4.5 for the
   3-block atom transformers). Smaller modules — secondary priority.

5. **Uniform ROS variant** (`uniros_scheduledfree.yaml`). Would
   demonstrate whether the per-layer 1/√ℓ schedule is worse than
   depth-flat 1/√L. Currently per-layer ROS gives γ=1.0 at L0, which
   probe data suggested **accelerates** alignment formation.

## 9. Test coverage

`uv run pytest tests/test_primitives.py tests/test_evoformer.py tests/test_diffusion.py -x -q`
covers:

- `LayerNorm.output_scale` (unused but tested).
- `block_residual_scale` policy resolution + mutual-exclusion.
- ROS wiring — every residual-stream LN stays at output_scale=1.0; per-block
  `residual_out_scale` propagates the right scalar.
- `pair_bias_gate` — default off, init magnitude, `pair_logits` multiplied
  by `τ_h` per head, 1-D shape (no-decay routing).
- `single_transition.linear_out` orthogonal init magnitude + fused-kernel
  guard.
- DiffusionTransformer `cond_bias_init_beta` — default zeros, output stddev
  ≈ β, scope limited to `*_cond_bias` (other AdaLN projections untouched),
  propagation to all blocks of the transformer, β≤0 rejected.

Total: ~88 passing assertions across the three test modules at the time
of writing.

## 10. Files of record

- This document: `docs/single-track-stability-experiments.md`
- Configs: `configs/ros_scheduledfree.yaml`, `configs/ros_scheduledfree_adabias.yaml`, `configs/uniros_scheduledfree.yaml`, `configs/sts_{A,B,C}.yaml`
- Code: `src/deepfold3/model/{primitives,attention,evoformer,diffusion,model,confidence,template_embedding,optim,config}.py`, `src/deepfold3/training/single_track_probe.py`
- Tests: `tests/test_primitives.py`, `tests/test_evoformer.py`, `tests/test_diffusion.py`
- Submit: `kakao/submit.sh`, `kakao/trainjob-multinode.yaml`
- Probe scripts (transient, recreate from this doc as needed): `/tmp/probe_*.py`

## 11. 32-record overfit ablation (2026-05-03)

Goal: separate **transient** trunk single magnitude growth from a
**load-bearing aligned-gain branch** that fits loss. On the full PATCHR
manifest the two are confounded; a 32-record overfit set drives loss
down quickly and exposes whether `RMS(s)` follows.

### 11.1 Aligned-gain failure mode

The DeepNet (`1/√(2L)`) and per-layer ROS (`1/√(ℓ+1)`) schedules assume
**independent** updates: ‖x_{k+1}‖² ≈ ‖x_k‖² + γ_k² ‖u_k‖². Empirically
`cos(s, u) > 0` per block — the `single_transition` update reinforces
the residual direction. Under aligned gain the recursion is radial:

    x_{k+1} ≈ (1 + γ_k α_k) · x_k        with α_k = (s_k · u_k) / ‖s_k‖²

so the budget for non-divergence is `Σ γ_k α_k`, **not** `Σ γ_k²`. A
`1/√(ℓ+1)` schedule that's safe in the independent-update regime can
still let the radial branch grow geometrically.

#### Bernstein-style budget switch

* Orthogonal updates (ρ = cos(s, u) ≈ 0) — variance regime:
  Δ‖s‖² ≈ γ²‖u‖², controlled by `Σ γ_k²`. Then `γ_k = k^{-1/2}` is
  appropriate (`Σ k^{-1} = O(log K)`).
* Aligned updates (ρ ≫ 0) — radial regime:
  ‖s_{k+1}‖ ≈ ‖s_k‖ + γ_k ‖u_k‖ ρ_k, and if `u_k ≈ α_k s_k` then
  `log ‖s_K‖ ≈ Σ γ_k α_k`. The relevant budget becomes `Σ γ_k`,
  controlled by `γ_k = k^{-1}` (`Σ k^{-1} = O(log K)`) — or, equivalently,
  by an η-damped `γ_k = η · k^{-1/2}` with small enough η.

The `single_transition` branch lives in the **aligned** regime: the
trunk's per-block `cos(s, u)` reaches 0.7–0.95 within ~200 steps of
training (see §11.4 probe data, peaks around step 500). The pair branch
stays approximately orthogonal (it has no equivalent radial growth in
the data we've seen), so the `pair_transition` brake is generally not
needed once single is bounded.

#### Empirical η anchor

To match DeepNet's flat-`1/√(2L)` budget on the trunk PairFormer (L=48),
solve `η · Σ_{k=1}^{48} k^{-1/2} = 48 · (2L)^{-1/2}`:

    Σ_{k=1}^{48} k^{-1/2} ≈ 12.43        (Euler-Maclaurin: 2√48 + ζ(1/2))
    DeepNet flat budget   = 48 · 1/√96 ≈ 4.90
    η_required            = 4.90 / 12.43 ≈ 0.39

Empirically, `η = 0.35` (sts_B) lands just below this threshold and
yields the tightest single_l2 plateau without measurable fit-speed
cost. `η = 0.5` (sts_D) overshoots the budget; the prediction
`|s_K|_{η=0.5} / |s_K|_{η=0.35} = exp(α · 0.15 · 12.43) ≈ 1.32` matches
the observed single_l2 ratio of 44.2/30.7 ≈ 1.44 to within α-uncertainty
(observed slightly higher because larger η also slightly increases α
through second-order training dynamics).

For the per-layer-extra schedule (sts_C, `γ_k = 1/(ℓ+1)`):
`Σ_{k=1}^{48} 1/k ≈ ln 48 + γ_E ≈ 4.45`, very close to η=0.35's `Σγ = 4.35`.
The observed plateaus (B 30.7 vs C 42.6) differ more than the budget
identity predicts because C is depth-imbalanced (γ=1 at L=1, γ=0.02 at
L=48) while B is uniformly damped — and the aligned regime forms
strongest at deep layers where C's γ is already small.

### 11.2 Knobs

`PairFormerConfig` and `ConfidenceConfig` ([`config.py`](../src/deepfold3/model/config.py)):

- `single_transition_residual_gain: float | None` — constant multiplier
  on `s_single_ffn`. `None` ≡ 1.0.
- `single_transition_residual_per_layer_extra: bool` — when True,
  multiplies by an extra `1/√(ℓ+1)` per block.
- `pair_transition_residual_gain: float | None` — same dial on the
  `pair_transition` branch. (Added 2026-05-04 alongside the DeepNet
  removal; not yet used by an arm but available for next-round sweeps.)
- `pair_transition_residual_per_layer_extra: bool` — same boolean on
  the pair_transition branch.

All four resolve via `_resolve_branch_residual_gain` in
[evoformer.py](../src/deepfold3/model/evoformer.py). Confidence head
mirrors the same knobs; note that the confidence path is detached from
the trunk in `model.py:single_real.detach()`, so confidence-head settings
only affect the head's own dynamics (not the trunk we're studying here).

> **DeepNet removed (2026-05-04).** The legacy `deepnet_residual_scale`
> flag and its `1/√(2L)` per-branch schedule were dropped from
> `PairFormerConfig` / `ConfidenceConfig`. The ablation arms below all
> assume per-layer ROS (`residual_output_scaling: true`) as the baseline.

### 11.3 Three arms

All three inherit per-layer ROS, so `s_ros = 1/√(ℓ+1)`. The only thing
that varies between arms is the extra **gain factor** multiplied on top
of `s_ros` for the single FFN branch:

    s_single_ffn = gain · s_ros

| Arm | gain | `s_single_ffn` (resolved) | Knob |
| --- | --- | --- | --- |
| A | 1                  | `s_ros = 1/√(ℓ+1)`         | (defaults) |
| B | 0.35               | `0.35 · s_ros = 0.35/√(ℓ+1)` | `single_transition_residual_gain: 0.35` |
| C | `1/√(ℓ+1)`         | `(1/√(ℓ+1)) · s_ros = 1/(ℓ+1)` | `single_transition_residual_per_layer_extra: true` |
| D | 0.5                | `0.5 · s_ros = 0.5/√(ℓ+1)`  | `single_transition_residual_gain: 0.5` |

D is launched **into the same run root** as A/B/C (same data, same seed
schedule, same launch tooling) so the η-sweep B(0.35) → D(0.5) is
directly comparable. Add D to a mid-flight run with:

```bash
bash scripts/run_sts_arm.sh D 3 runs/sts_<DATE>
```

Inherits `ros_scheduledfree_adabias.yaml` (per-layer ROS, PairBias gate,
distogram α=0.05, AdaLN cond_bias β=1e-2,
`single_transition.linear2` init gain=0.01).

### 11.4 SingleTrackProbe

[`src/deepfold3/training/single_track_probe.py`](../src/deepfold3/training/single_track_probe.py).
Attaches to every PairFormer trunk block; on selected steps captures
per-block `(rms_s, rms_u, rms_gu, cos_su, alpha_su, gamma)` plus
weight Frobenius norms `w_single_trans_l2_*`, `w_single_attn_out_*`.
Writes one JSONL row per probed step to
`output_dir/single_track_probe.jsonl` (rank-0 only, no WandB).

Trainer flags:
- `training.single_track_probe: bool = False`
- `training.single_track_probe_interval: int = 50`

### 11.5 Setup

**Mini dataset** ([`scripts/build_sts_minidata.py`](../scripts/build_sts_minidata.py)):
32 records from `/data/data/patchr_training`, 1–2 chains, 100–400 residues,
deterministic by `md5(record_id)`. Symlinks structures / masks /
templates / MSAs into `runs/sts_minidata`. Avoids known PATCHR
multi-chain ligand assembly spike records.

**Training**: 1 GPU per arm (3 GPUs total, no DDP). `bp=2`, `accum=1`,
`max_tokens=256`. `lr = 1.8e-3 · √(2/256) ≈ 1.6e-4` (sqrt-scaling from
production batch ≈ 256). Schedule-Free AdamW, wd=0.05, warmup=200,
`max_steps=5000`, no eval. Probe captures every 5 steps
(`single_track_probe_interval=5` → 1000 rows / arm).

**Launch** ([`scripts/run_sts_ab_c.sh`](../scripts/run_sts_ab_c.sh)):
```bash
uv run python scripts/build_sts_minidata.py \
    --src /data/data/patchr_training --dst runs/sts_minidata --num 32
bash scripts/run_sts_ab_c.sh runs/sts_minidata
```

Each arm writes `runs/sts_<DATE>/{A,B,C}/`:
- `stdout.log` — training log every 50 steps.
- `single_track_probe.jsonl` — per-block stats every 50 steps.
- `checkpoint_*.pt`, `config.yaml`, `provenance.yaml`.

### 11.5b Live plotting

[`scripts/plot_sts_probe.py`](../scripts/plot_sts_probe.py) renders three
PNG pages from the per-arm probe JSONLs. Probe rows now include all
loss components (`total`, `diffusion_mse`, `distogram`, `smooth_lddt`,
`single_l2`, `pair_l2`, …), so every panel + scatter is plotted at the
probe interval (5 step) without log_interval aliasing.

```bash
# One-shot render
uv run python scripts/plot_sts_probe.py runs/sts_<DATE>

# Live monitor — re-render every 60s
uv run python scripts/plot_sts_probe.py runs/sts_<DATE> --watch 60
```

| File | Contents |
| --- | --- |
| `plots/overview.png` | Loss / fit progress (3×3): `total`, `diffusion_mse`, `distogram`, `smooth_lddt`, `single_l2`, `pair_l2`, plus three scatter decision plots (`single_l2 vs RMS(s_l_last)`, `L_diff vs RMS(s_l_last)`, `cos(s,u)_mean vs L_diff`). |
| `plots/depth.png` | Depth profile (3×3): `RMS(s)`, `RMS(γ·u)`, `cos(s,u)` at L=0 / mid / last. |
| `plots/wakeup.png` | Wake-up & gain (2×2): `‖W₂(single_trans)‖` and `‖Wo(single_attn)‖` at L=0 vs Llast (overlay), `α_su` mean / max overlay, per-step `Σ_k γ_k·α_k` aligned-gain budget. |

The most direct decision metric is the per-step `Σ_k γ_k·α_k`: under
aligned gain ‖s‖ grows by `Π_k (1 + γ_k·α_k) ≈ exp(Σ_k γ_k·α_k)` per
forward, so a B/C arm that drops this sum at matched loss has bounded
the radial branch.

### 11.6 Decision rule

Joint trajectory of **`L_diff` vs `rms_s_l_last`** (single norm at the
end of the trunk, captured at the same step):

| Pattern | Read |
| --- | --- |
| `L↓`, `rms_s_l_last↑` (lockstep) | Aligned gain is load-bearing → B or C should bound `rms_s_l_last` at matched loss. |
| `L↓`, `rms_s_l_last` plateau | Transient only — A is fine, no intervention. |
| `L` plateau, `rms_s_l_last↑` | Pathological drift — stronger brake or wd needed. |

`cos_su_mean` ≈ 1 across late blocks ⇒ aligned-gain regime;
`cos_su_mean` ≈ 0 ⇒ independent-update regime where the conventional
budget calculation applies.

### 11.7 Status

- 2026-05-03: implementation complete (knobs + probe + configs +
  data-dir builder + launch script).
- 2026-05-04: 4-arm sweep complete (A/B/C through step ~4230 of 5000;
  D added mid-flight at η=0.5 reaches step ~1275). Findings:
  - **B (η=0.35) is the production winner.** Tightest single_l2
    plateau (mean 30.7, σ=5.2), tightest pair_l2 plateau (mean 43.4,
    declining late), distogram and smooth_lddt indistinguishable from
    other arms.
  - **A (η=1) is unbounded.** single_l2 climbs monotonically from 15
    to 275+ across all observed windows; pair plateau forms only
    around step 3000 (latest of all arms).
  - **C (per-layer extra) is intermediate** (single 42.6, pair 55.9
    σ=9.3). Depth-imbalanced brake transfers volatility to the pair
    track (range 37–80 vs B's 31–55).
  - **D (η=0.5) is too weak.** Single_l2 plateau 44.2 vs B's 30.7
    (1.44× higher), with no fit-speed advantage — η scales single
    magnitude near-linearly, in agreement with the budget theory in
    §11.1.
  - The Bernstein budget switch (`Σγ²` → `Σγ` under alignment) gives
    a clean theoretical anchor for η ≈ 0.39; the empirical 0.35
    matches.
- Production deployment: `configs/ros_scheduledfree_adabias_s035.yaml`
  (single brake at 0.35, pair untouched). Currently running on kakao
  as `df3-12n-s035-0504-032348` (12 nodes × 8 = 96 ranks).

## 12. Layer-0 dominance (s035 production run, 2026-05-07)

`df3-12n-s035-lr2-5e4-wd10-0504-181609` (s035 with lr halved to 2.5e-4
and wd doubled to 0.10) reached step 10.6k with `single_l2` 7→411,
trunk weight ||W|| stable, `smooth_lddt` 0.669→0.458 (still descending),
but p50 `grad_norm` 8.55→43.2 and 92.7% of recent steps clipped. This
section documents the checkpoint-level diagnosis confirming §8.5's
hypothesis ("per-layer ROS γ=1.0 at L0 accelerates alignment").

### 12.1 Probe data — per-layer single amplification across training

Probed via [`scripts/probe_layer_amp.py`](../scripts/probe_layer_amp.py)
on record `patchr:6nud` at three checkpoints. Per-block
`||single_out|| / ||single_in||`:

| layer | step 1k | step 5k | step 10.5k |
| ----: | ------: | ------: | ---------: |
| **0** | **1.020** | **2.621** | **4.970** |
|     4 |   1.019 |   1.073 |     1.083 |
|     8 |   1.016 |   1.034 |     1.042 |
|    12 |   1.013 |   1.019 |     1.025 |
|    16 |   1.013 |   1.013 |     1.016 |
| 20–47 | 1.007–1.011 | 1.007–1.011 | 1.008–1.015 |
| **cumul (L0→L47)** | **1.75×** | **9.34×** | **18.08×** |

Pair amplification is roughly stationary in time (cumul 1.83 → 2.87 →
2.45). The drift is single-specific, and concentrated at L0.

### 12.2 Mechanism — uniform gate wake-up × asymmetric ROS scale

Checkpoint-diff (1k → 10.5k) of zero-init residual outputs. **The
wake-up itself is layer-uniform:**

| param | layer 0 (1k → 10.5k) | layer 24 (1k → 10.5k) |
| --- | --- | --- |
| `tri_mul_in.gate.weight`     | 0.56 → 2.25 (4.0×) | 0.46 → 2.36 (5.1×) |
| `tri_mul_out.gate.weight`    | 0.57 → 2.30 (4.0×) | 0.45 → 2.39 (5.4×) |
| `tri_mul_in.out_gate.weight` | 0.44 → 1.59 (3.6×) | 0.34 → 1.67 (4.9×) |
| `tri_mul_out.out_gate.weight`| 0.42 → 1.70 (4.0×) | 0.35 → 1.98 (5.7×) |

What differs between L0 and L24 is the residual scale: per-layer ROS
gives `1/√(L+1)` so L0=1.0, L24=0.2. With s035's η=0.35 brake on
single FFN: L0_eff=0.35, L24_eff=0.07. **Same wake-up × 5× larger
residual scale = L0 dominates the single stream.** Trunk weights are
otherwise unchanged across layers (weight ||W|| ratio 0.87× across
L0..L47).

The most affected LayerNorm bias confirms the picture: top-15
`||Δθ||/||θ||` (per 100 steps) entries are all `single_to_pair_norm.bias`
with the largest at L0 (||x||=0.008, SF ||z||=11.3, ratio 1192×).
That's textbook SF averaging cancelling a high-amplitude oscillation
near zero — SF is working as advertised; the dynamics it's averaging
are abnormal.

### 12.3 Why schedule changes alone won't fix it

The natural prescription from §8.5 is uniros (depth-flat 1/√L_total),
which would drop L0's scale from 1.0 to 1/√48 ≈ 0.144. **But §3
already showed DeepNet (depth-flat 1/√(2L) ≈ 0.102 single-branch
scale, more aggressive than uniros) failed at step ~4300 with
trunk-end RMS(single)=232.** §11.1 is explicit: "ROS or deepnet
forward suppression cannot stop the alignment dynamic; only the
per-step amplitude is reduced."

So the layer-0 finding identifies the *amplification site*, not a fix
path. The next intervention should target the alignment dynamic
itself, e.g. §8.2's `alpha_single_reg` direct magnitude penalty, not
another schedule reshape. uniros remains worth running as a
diagnostic (it would isolate L0's contribution from the rest of the
trunk) but should not be expected to plateau single_l2 on its own.

### 12.4 Optimizer regime caveat

In the same window (steps 8k–10.6k), 92.7% of grad-norm samples
exceeded the clip threshold of 10. `clip_grad_norm_` runs in-place
*before* `scaler.step(optimizer)` (see
[`optim.py:485-486`](../src/deepfold3/training/optim.py#L485-L486) and
[`optim.py:603`](../src/deepfold3/training/optim.py#L603)), so Adam's
`exp_avg_sq` accumulates clipped-direction noise rather than raw
gradient curvature. The optimizer in this regime is closer to
"SF + normalized-direction descent" than to SF-AdamW; `1/√v`
percentiles should not be read as curvature estimates. SF's `x`
average remains lightly perturbed (||Δx||/||x|| 0.1–0.9% per 100
steps for most groups), which is why eval metrics keep improving
despite the spike density.

### 12.5 Files of record (added 2026-05-07)

- [`scripts/analyze_checkpoint.py`](../scripts/analyze_checkpoint.py) —
  per-bucket param norms, SF ||z-x||/||x|| divergence, Adam v stats,
  PF per-layer key-tensor norms. Used for §12.2.
- [`scripts/probe_layer_amp.py`](../scripts/probe_layer_amp.py) —
  pre-existing forward-pass per-layer amplification probe. Used for §12.1.
- Run: `runs/df3-12n-s035-lr2-5e4-wd10-0504-181609/checkpoint_{1000,5000,10500,10600}.pt`

## 13. uniros + PairFormer residual-branch RMSNorm (in flight, 2026-05-07)

`df3-12n-uniros-rmsn-lr1-5e4-0507-103830` (12 nodes × 8 = 96 ranks) launched
as the layer-0-dominance follow-up to s035. Goal: directly test whether
removing the per-layer ROS asymmetry (uniform ROS) plus forcing every
PairFormer residual branch to unit-RMS (architectural intervention)
plateaus the single-track magnitude — or whether the alignment dynamic
(§11.1) is the dominant cause and only direct magnitude penalty
(`alpha_single_reg`, §8.2) will do.

### 13.1 Active interventions vs s035_lr2_5e4_wd10

| Knob | s035 | uniros + RMSNorm |
| --- | --- | --- |
| ROS schedule | per-layer `1/√(ℓ+1)` (L0=1.000) | depth-flat `1/√48 ≈ 0.144` |
| Single FFN brake | `single_transition_residual_gain=0.35` | none (mutually exclusive with RMSNorm) |
| PairFormer residual-branch normalization | none | non-learnable RMSNorm at every branch output |
| `lr` | 2.5e-4 | 1.5e-4 |
| `grad_clip_norm` | 10.0 | 20.0 |
| `wd` | 0.10 | 0.10 (retained) |
| `reg_form` (logging only) | `lp` (`single_l2` / `pair_l2`) | `log_rms` (`single_rms` / `pair_rms`) |
| `alpha_{single,pair}_reg` | None | 0.0 (record-only, no gradient) |
| `pair_bias_gate` | False | False (default) |
| `single_to_pair_norm` | LayerNorm (default) | LayerNorm (untouched — PairBias side input, not residual) |

Config: [`configs/uniros_rmsn_lr1_5e4.yaml`](../configs/uniros_rmsn_lr1_5e4.yaml).

### 13.2 Code changes that landed for this arm

- `deepfold3.model.primitives.rms_normalize(x, eps=1e-6)` — non-learnable
  RMSNorm helper, fp32 rsqrt with bf16 passthrough.
- `PairFormerConfig.residual_branch_rmsnorm: bool = False` — when True,
  wraps 7 PairFormer residual branches (`tri_mul_in/out`,
  `pair_attn_row/col`, `pair_transition`, `single_attn`,
  `single_transition`) with `rms_normalize` before the per-block γ
  multiplier. `single_to_pair` left alone (not a residual branch).
  `__post_init__` rejects the combo with any per-branch residual gain
  knob (RMSNorm makes them redundant).
- `TrainingConfig.reg_form: Literal["lp", "log_rms"] = "lp"` — selects the
  representation regularization form. `log_rms` logs `single_rms`/`pair_rms`
  (= sqrt of the legacy `single_l2`/`pair_l2`) and uses
  `α · log(RMS + 1e-6)` as the loss term.
- Tests: `tests/test_primitives.py::TestRmsNormalize`,
  `tests/test_evoformer.py::TestPairFormerResidualBranchRMSNorm`,
  `tests/test_evoformer.py::TestPairFormerConfigRmsnormValidation`,
  `tests/test_ligand_loss_mask.py::test_log_rms_metric_matches_sqrt_of_l2`.

### 13.3 Decision criteria

Compare the new run's `single_rms` trajectory against s035's
`single_l2 = single_rms²` baseline at matched effective steps. Convert
s035 numbers via `single_rms ≈ sqrt(single_l2)` for direct comparison:
s035 step 1k → 7.6, step 5k → 13.2, step 10.5k → 20.3.

| Outcome at step ~10k | Reading |
| --- | --- |
| `single_rms` plateaus at ~3-5 | Layer-0 dominance + RMSNorm bound is dominant; the alignment-dynamic warning of §11.1 was overstated for our setting |
| `single_rms` plateaus at ~10-15 (lower than s035's 20.3 but rising) | RMSNorm helps but alignment is still active; queue an `α_single_reg > 0` follow-up |
| `single_rms` ≥ 20 (matches/exceeds s035) | RMSNorm did not bound the single track; alignment dynamic dominates → §8.2 path is the only remaining knob |
| Eval (`smooth_lddt` on `x` average) regresses vs s035 | RMSNorm is over-bounding capacity; dial back to subset of branches (e.g., single only) |

Probe `runs/.../checkpoint_{1k,5k,10k}.pt` with `scripts/probe_layer_amp.py`
to confirm: per-block single amplification should now be near-uniform
across layers (`1.0 + γ_k ≈ 1.144` for every layer) instead of
concentrating at layer 0.

### 13.4 Operational notes

- 12-node submit at the time of launch was within the cluster's B200
  capacity envelope (per `reference_kakao_node_capacity.md`); pre-flight
  check before resubmission.
- Watch: `kubectl get trainjob df3-12n-uniros-rmsn-lr1-5e4-0507-103830`.
- WandB: `DeepFoldProtein/deepfold3-stage0`.


---

## 14. uniros+RMSN long-run pathologies and pivot to AF3 baseline (2026-05-12)

The uniros+RMSN stack (§13) launched as `df3-12n-uniros-rmsn-lr1-5e4-0507-103830`
and graduated through `df3-12n-uniros-rmsn-lr2e4-clip70-cont` and
`df3-2e4-c70-rsf-c200-ema` (predecessor) over ~16000 steps. By step 16000 the
trunk had stabilised (single_rms ≈ 12.2, pair_rms ≈ 12.4) but multiple
secondary pathologies became visible:

### 14.1 PF eff_rank collapse — single-track specific

`scripts/spectral_quick.py` + `scripts/plot_pf_collapse.py` per-submodule
breakdown of PairFormer eff_rank across the full lineage (step 1000 → 16800):

| submod | step 1000 | peak @ step 8000 | step 16800 | Δ peak→16800 |
| --- | --- | --- | --- | --- |
| **single_transition** | 187.5 | 187.7 | **178.4** | **−9.2** |
| **single_attn** | 141.5 | 139.4 | **131.4** | **−8.0** |
| tri_mul_in | 40.6 | 42.9 | 42.3 | −0.6 |
| tri_mul_out | 40.7 | 43.3 | 42.7 | −0.7 |
| pair_attn_row | 44.3 | 45.9 | 45.6 | −0.3 |
| pair_attn_col | 44.3 | 45.4 | 45.3 | −0.2 |
| pair_transition | 73.9 | 77.8 | 77.3 | −0.6 |
| single_to_pair | 15.0 | 14.9 | 14.6 | −0.3 |

Single-track (`single_attn`, `single_transition`) collapses by 8-9 effective
ranks from peak. Pair-track is essentially stable.

**Cause**: the single-track matrices are large (384×384, 3072×384) so the
scale-invariant equilibrium |w|²* ~ η/(2λ_w)·E|Pg|² ends up with many more
"unused" small singular values under the wd=0.10 pressure than the smaller
pair-track matrices (128×128) which are already near full rank at peak.
PF eff_rank collapse rate (−2e-4/step) is **independent of α_distogram**
(both 0.03 and ×100 alpha give identical rates) — confirms AdamW
preconditioner P-normalises any loss scaling, so the issue is the wd/η
balance, not gradient magnitude.

### 14.2 The ablation chain (all from `runs/df3-2e4-c70-rsf-c200-ema/checkpoint_16000.pt`)

| arm | hypothesis | result | verdict |
| --- | --- | --- | --- |
| `df3-dist3` | α_distogram ×100 → restore PF gradient (RMSN was blocking) | AdamW absorbed scaling — ΔW magnitude/angle identical to predecessor. But metrics improved via head fro growth + msa direction reversal + activation regime shift. Distogram CE dropped to 1.78 (s035 level, −0.04 vs predecessor). | Useful but not via hypothesised mechanism. |
| `df3-eos` | lr 2e-4 → 1e-4 + clip 70 → 400 (lr × clip = 0.040, proper EoS regime) | r_clip ≈ 5% achieved. tm/itm slightly improved (best for itm 0.110), single_rms more stable. smooth_lddt **regressed** toward predecessor plateau (0.468 → 0.471). | lr halving cost > ponderomotive benefit. |
| `df3-c400` | lr 2e-4 + clip 400 (lr × clip = 0.080, untested envelope) | RMSN+uniros stabilised lr × clip 2.2× above the pre-RMSN abort boundary (0.036). Multimer best (itm 0.110); smooth_lddt regressed to 0.475 (above predecessor 0.472). | Clip regime change tilts metric preference — multimer ↑, smooth_lddt ↓. |
| `df3-c400r` | Same as c400 + `skip_load_optimizer=true` + `warmup_steps=20` (optimizer reset for fast ablation) | smooth_lddt collapsed to 0.55 within 200 steps; tm dropped to 0.17. v=0 + short warmup transient mishandled. | Killed. |

### 14.3 Key empirical principles surfaced

1. **AdamW absorbs loss-weight scaling**. χ = η(1-β)(μ_max+λ_w)/2 is alpha-invariant.
   See `memory/reference_sf_adamw_phase_diagram.md`.

2. **Clip is a metric-preference axis**. r_clip 30-50% (norm-constrained) favours
   smooth_lddt; r_clip 1-10% (proper EoS) favours multimer. Predecessor and
   df3-dist3 both ran in clip-dominated regime — smooth_lddt was *protected*
   by clip, not just safe under it.

3. **PF single-track collapse is `wd × |Pg|²` balance**, not gradient starvation.
   Adding or removing RMSNorm doesn't change the rate. The fix has to be wd
   reduction or per-bucket wd, not optimizer regime.

4. **Refined SF `k` continued at step 16k makes x heavily inertial** (~800-step
   window). Ablations have ~1500-2000 step verdict horizon when resuming with
   k continued. See `memory/reference_optimizer_resume_semantics.md`.

### 14.4 The pivot — `df3-af3r`

After four ablations on the uniros+RMSN stack hit different walls, the cleanest
test was: **strip the architecture modifications, keep only the optimizer change
that motivated this whole investigation (refined SF-AdamW)**.

Configuration ([`configs/af3_refined_sf.yaml`](../configs/af3_refined_sf.yaml)):

- `residual_branch_rmsnorm: false` (default — AF3 baseline)
- `uniform_residual_scale: false` (default — AF3 baseline)
- `lr: 1.8e-3` (AF3 paper)
- `weight_decay: 0.0` (AF3 baseline — smallest possible start)
- `warmup_steps: 1000`
- `grad_clip_norm: 10.0` (lr × clip = 0.018)
- `batch_size_per_rank: 3, accumulate_grad_batches: 1` (288 effective, matches s035)
- `average_concentration: 200` (refined SF — sole non-AF3 change)
- `ema_decay: 0.999` (passive shadow control; cf. `feedback_sf_ema_redundancy.md`)
- Fresh from random init (no resume; uniros+RMSN ckpts are not weight-compatible
  with the AF3 architecture in forward).

Watch points:

- **PF eff_rank trajectory** — does it reach AF3-class > 60 without trunk drift?
- **single_rms** — does AF3 baseline + refined SF + wd=0 hold the trunk, or does
  the s035-era single-track divergence return?
- **Loss/metric** vs s035 at matched effective step.

If df3-af3r is stable and reaches AF3-class spectra, the uniros+RMSN stack was a
net negative (it solved trunk drift but introduced smooth_lddt entanglement and
single-track collapse). If it shows trunk drift like s035 did at step 4k+, then
some stabilisation mechanism is still needed — but RMSN's particular formulation
is the wrong one.
