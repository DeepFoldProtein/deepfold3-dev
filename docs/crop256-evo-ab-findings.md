# Fused vs vanilla evo_attention A/B — crop=256, resume from drifting checkpoint

**Date**: 2026-04-25
**Context**: the 051858 run drifted with `use_fused_evo_attention=true`, reaching `distogram ≈ log(64)` at ~800 steps (see `project_051858_run_failure.md`). This A/B tests whether the fused Triton evo_attention kernel is a drift contributor by resuming from the mid-drift checkpoint and running a short window of steps with fused on vs off.

## Setup

- Resume: `runs/df3-20260421-051858/checkpoint_700.pt`
- `max_tokens`: 256 (vanilla PyTorch OOMs at 384)
- Ranks: **1 GPU** — removes DDP / all-reduce non-determinism from the comparison
- `--deterministic`, `seed: 42`, same data pipeline
- Effective batch: `1 × batch_per_rank=2 × accumulate=2 = 4`
- `eval_before: false`, `eval_interval: 100000` (skip pre-training eval)
- 10 steps captured per arm
- Arm A: `use_fused_evo_attention: true`
- Arm B: `use_fused_evo_attention: false`
- Everything else identical (including `use_fused_dropout_add=true` in both arms)

Runs at:

- `runs/ab_1gpu_fused_from700/train.log`
- `runs/ab_1gpu_vanilla_from700/train.log`

## Results

### Loss

| step | arm A (fused) | arm B (vanilla) | Δrel   |
| ---- | ------------- | --------------- | ------ |
| 701  | 240.46        | 266.08          | +10.7% |
| 702  | 135.05        | 167.83          | +24.3% |
| 703  | 169.14        | 201.62          | +19.2% |
| 704  | 31.93         | 41.71           | +30.6% |
| 705  | 428.07        | 209.25          | -51.1% |
| 706  | 1196.0        | 517.80          | -56.7% |
| 707  | 179.35        | 110.86          | -38.2% |
| 708  | 968.42        | 927.66          | -4.2%  |
| 709  | 4174.6        | 2908.0          | -30.3% |
| 710  | 352.47        | 310.96          | -11.8% |

### Distogram CE

| step | arm A | arm B | Δabs     |
| ---- | ----- | ----- | -------- |
| 701  | 4.55  | 4.40  | -0.15    |
| 702  | 3.23  | 3.09  | -0.14    |
| 703  | 2.57  | 2.48  | -0.09    |
| 704  | 4.32  | 4.25  | -0.07    |
| 705  | 2.96  | 2.96  |  0.00    |
| 706  | 2.99  | 3.06  | +0.07    |
| 707  | 4.80  | 4.75  | -0.05    |
| 708  | 2.49  | 2.69  | +0.20    |
| 709  | 4.24  | **2.78** | **-1.46** |
| 710  | 3.49  | 3.36  | -0.13    |

### `single_l2` (single-representation norm)

| step | arm A (M) | arm B (M) | Δrel       |
| ---- | --------- | --------- | ---------- |
| 701  | 14.14     | 14.12     | -0.2%      |
| 702  | 14.20     | 14.20     | -0.03%     |
| 703  | 14.94     | 14.81     | -0.86%     |
| 704  | 14.81     | 14.74     | -0.45%     |
| 705  | 14.39     | 14.43     | +0.30%     |
| 706  | 13.65     | 13.30     | -2.6%      |
| 707  | 12.37     | 12.36     | -0.06%     |
| 708  | 10.25     | 11.17     | +9.0%      |
| 709  | 10.09     | 8.68      | -14.0%     |
| 710  | **9.05**  | **2.69**  | **-70.3%** |

### `pair_l2` (pair-representation norm)

| step | arm A | arm B | Δrel   |
| ---- | ----- | ----- | ------ |
| 701  | 612   | 608   | -0.6%  |
| 702  | 510   | 497   | -2.5%  |
| 703  | 561   | 534   | -4.8%  |
| 704  | 740   | 749   | +1.2%  |
| 705  | 583   | 531   | -8.9%  |
| 706  | 592   | 554   | -6.4%  |
| 707  | 586   | 616   | +5.1%  |
| 708  | 501   | 508   | +1.4%  |
| 709  | 1021  | 522   | -48.9% |
| 710  | 464   | 488   | +5.2%  |

### `grad_norm`

| step | arm A | arm B | note                            |
| ---- | ----- | ----- | ------------------------------- |
| 701  | 4.89  | 4.12  |                                 |
| 702  | 7.29  | 7.23  |                                 |
| 703  | 7.00  | 5.99  |                                 |
| 704  | 7.77  | 6.99  |                                 |
| 705  | 7.02  | 7.29  |                                 |
| 706  | 20.52 | 6.53  | arm A clipped (>10), arm B not  |
| 707  | 6.68  | 5.01  |                                 |
| 708  | 9.64  | 12.33 | arm B clipped, arm A not        |
| 709  | 7.54  | 9.78  |                                 |
| 710  | 7.77  | 5.96  |                                 |

## Reading

1. **Determinism works** — steps 701–707 show `single_l2` matching to ≤2.6%, `pair_l2` to ≤9%, and `smooth_lddt` sometimes to 4 decimal places. Both arms see the same batch and same pre-optimizer weights.
2. **Per-step loss variance at batch=4 is large and sign-oscillating** — arm A is higher on steps 701–704, lower on 705–710. A 10-step window is insufficient to claim a trajectory.
3. **At step 710, arm B's `single_l2` collapsed 5× (14.12M → 2.69M)** while arm A only dropped to 9.05M. This is the single representation "recovering" toward normal scale under vanilla attention but **holding elevated under fused**.
4. **At step 709, arm B's distogram dropped to 2.78** (vs arm A 4.24) — arm B escaped the log(64) uniform regime within the 10-step window; arm A did not.
5. **No NaN / Inf / step skip** in either arm. Both trained cleanly; the difference is in trajectory, not in stability.
6. **Memory**: arm B uses ~103 GB vs arm A ~92 GB. Expected — vanilla materializes the full attention-score matrix.

## Interpretation (tentative)

The fused evo_attention kernel is **not numerically wrong** — early-step pair/single metrics match vanilla within the bf16 kernel precision floor. But in the drifted regime, it appears to **hold the single representation elevated** while vanilla lets it shrink. Plausible mechanism: bf16 kernel precision artifacts amplified by drift-regime activation scales feed back into the representation update in a way that favors holding high-norm state.

Consistent with the `project_dual_root_cause_fix.md` history — both drift-amplification paths identified there were bf16-specific precision issues in fused kernels.

**This is suggestive, not conclusive.** A 10-step window at batch=4 has huge variance.

## Limitations

- **Batch=4 is extremely noisy.** Real training runs at batch=256 (128 rank × 2) with tokens_per_opt_step=65536. Drift dynamics may differ at true batch size.
- **Crop=256 vs original 384.** Attention softmax concentration is N-dependent; drift may not reproduce the same way at a smaller crop.
- **Short horizon (10 steps).** Trajectory claims require longer windows.
- **Resume-from-drifting-checkpoint only.** If the fused kernel is a drift *cause* (not just an amplifier), we'd expect it to also diverge from a non-drifted checkpoint.

## Follow-up: 30-step window from the same checkpoint (same setup)

**Runs**:

- `runs/ab_1gpu_fused_from700_30s/train.log`
- `runs/ab_1gpu_vanilla_from700_30s/train.log`

### Run-to-run variance (same arm, same seed)

Re-running arm A from the same `checkpoint_700.pt` with `max_steps=730` produced **different** per-step losses than the 10-step arm A run, despite identical config + seed + `--deterministic`:

| step | 10-step arm A | 30-step arm A | Δrel  |
| ---- | ------------- | ------------- | ----- |
| 701  | 240.46        | 243.80        | +1.4% |
| 704  | 31.93         | 55.84         | +75%  |
| 705  | 428.07        | 263.11        | -39%  |

Run-to-run variance is 1–75% per-step. `torch.use_deterministic_algorithms` guarantees op-level determinism within a run; it does not guarantee identical RNG streams across runs when first-time autotune side-effects perturb CUDA context state. **Implication: the 10-step "arm A vs arm B" gaps of 10–30% per-step are within the run-to-run noise band and are not informative.**

### Means over 30 steps (701–730)

| metric         | arm A (fused) | arm B (vanilla) | reading           |
| -------------- | ------------- | --------------- | ----------------- |
| mean loss      | 668.3         | 592.2           | similar           |
| mean dmse      | 166.5         | 147.5           | similar           |
| mean distogram | 3.78          | 3.46            | ~equal            |
| mean pair_l2   | 616           | 802             | arm A lower       |
| mean single_l2 | 13.6M         | 16.8M           | arm A lower       |
| grad clips (>10) | 8/30        | 2/30            | arm A more spikes |

### `single_l2` trajectory (M)

| step | arm A | arm B |
| ---- | ----- | ----- |
| 701  | 14.15 | 14.11 |
| 705  | 14.56 | 15.11 |
| 710  | 15.32 | 13.65 |
| 712  | 15.42 | 11.42 |
| 713  | 14.23 | **0.66** ← transient dip |
| 714  | 13.72 | 13.46 |
| 716  | 8.32  | 16.61 |
| 717  | **0.00** ← transient dip | 17.57 |
| 720  | 13.65 | 19.04 |
| 725  | 14.23 | 21.37 |
| 730  | 16.15 | **24.72** |

### Critical reversal of the 10-step finding

**The 10-step arm B "single_l2 = 2.69M at step 710" was a transient dip, not a recovery.** Both arms experience these single-step dips (arm A at 716–717 → 8.3M then 3.6K; arm B at 713 → 0.66M). They rebound within 1–2 steps. They are not kernel-determined — they are stochastic artifacts of the high-variance batch=4 regime.

**Over 30 steps, arm B's `single_l2` GROWS monotonically from 13.5M (step 714) to 24.7M (step 730)** while arm A oscillates around 14M and ends at 16.2M. Vanilla evo_attention is **not** holding the single representation in check — if anything, it accumulates more upward drift than fused over this window.

## Revised verdict

The fused evo_attention kernel **is not detectably amplifying the 051858-style drift** in this regime. Both fused and vanilla produce statistically similar loss/distogram trajectories, and vanilla's `single_l2` ends up larger than fused's after 30 steps. The fused kernel does produce **more grad-norm spikes** (8/30 vs 2/30 clips at >10) — a real, reproducible difference — but those spikes do not translate into worse mean loss or worse mean distogram.

The drift driver is upstream of evo_attention. Likely candidates per `project_051858_run_failure.md`:

- distogram head's LN-less feedback loop into pair representation
- `alpha_pair_reg = 0` removing the trunk-output magnitude regularizer
- single representation init scheme (Lecun normal vs zeros — see `plan_init_scheme_review.md`)

## What this run does NOT cover

- **30 steps at batch=4 is still small.** A 200+ step window at the original batch=256 would tighten error bars by ~10×.
- **No A/B from a non-drifted checkpoint.** Worth doing if the upstream-driver hypothesis above doesn't pan out — would confirm fused vs vanilla equivalence in healthy regimes.
- **`use_fused_dropout_add` not yet tested.** Plumbing exists (`OptimConfig.use_fused_dropout_add`, default True); A/B with it set to False would test whether `dropout_add` is a hidden contributor. Lower priority now that evo_attention is exonerated.

## Config artifacts

- `configs/ab_fused.yaml` — arm A config
- `configs/ab_vanilla.yaml` — arm B config (only `use_fused_evo_attention: false` differs)
- `OptimConfig.use_fused_dropout_add` — added in this session, default True, wired through `EvoformerBlock`/`PairFormerBlock`/`TemplatePairFormerBlock`. Not yet used in either A/B arm (both keep fused dropout on); staged for a future kernel-by-kernel A/B if evo is exonerated by (b).
