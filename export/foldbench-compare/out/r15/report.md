# DeepFold3 r15 (step 14k) vs FoldBench — width scale-up, first read

**Date:** 2026-08-08
**Model:** `df3-stage0-r15` (bench series `df3v5r15`), checkpoint step **14,000**; training live at ~14.6k.
**Config:** `configs/stage0_v2.yaml` on top of `configs/initial.yaml` — the **Protenix-v2 width scale-up** arm.
**Baseline:** FoldBench official rows (AF3, Boltz-1/2, Protenix v1/v2, OpenFold-3 preview-2, ESMFold2), plus **r14 at the same step** as the internal control.
**Figures:** `confidence_calibration.png`, `foldbench_models_ppi_agab.png` (+`_post2023-06-01`), `foldbench_ranked_ppi_agab.png` (+`_post2023-06-01`), `foldbench_conf_models.png`
**Data:** `confidence_data.csv`, `foldbench_models_data.csv`, `foldbench_conf_data.csv`, `foldbench_ranked_data.csv` (+ `_post2023-06-01` variants)

Regenerate: `../../update_foldbench.sh --run r15` (auto-detects the latest benched step).

## What r15 changes

| | r14 | r15 |
|---|---|---|
| `pair_channel` (trunk / template / diffusion cond.) | 128 | **256** |
| `msa_channel` | 64 | **128** |
| attention heads (pairformer / MSA / template / diffusion) | 8 / 4 / 4 / 4 | **16 / 8 / 8 / 8** |
| `diffusion_batch_size` | 48 | **64** |
| batch composition | 2 per rank × 1 accum | 1 per rank × 2 accum |
| `fape_mode` | — | `pooled` (`alpha_fape_inter=0.0`) |

**The step axis is exactly comparable.** Both runs report `world_size=112,
effective_batch_size=224, tokens_per_opt_step=86016`. r15's batch is re-split (1×2 instead of
2×1) but the per-optimizer-step example and token counts are identical, so "r15 @ 14k vs r14 @
14k" is a like-for-like comparison of the same training budget — not a proxy. Wall-clock is
*not* matched: the wider model costs more per step.

## Headline

Per-target DockQ success rate, all annotated interfaces, 20 predictions/target for every model
(`foldbench_ranked_data.csv`; r14 columns recomputed at the matching step):

| | | **r15 @14k** | r14 @14k | r14 @49k | AF3 | PXv2 |
|---|---|---|---|---|---|---|
| **protein–protein** | rank-1 | 56.5 | 61.7 | 62.7 | 67.0 | 71.3 |
| (n=209) | mean | 58.4 | 60.3 | 62.2 | 65.1 | 71.8 |
| | oracle | 64.1 | 67.0 | 69.9 | 72.2 | 78.0 |
| **antibody–antigen** | rank-1 | **29.7** | 28.6 | 25.3 | 48.4 | 67.0 |
| (n=91) | mean | **25.3** | 23.1 | 25.3 | 47.3 | 58.2 |
| | oracle | **52.7** | 39.6 | 45.1 | 64.8 | 75.8 |

Two clean, opposite results at the same training budget:

- **Antibody–antigen is the win.** r15 @14k reaches **52.7 oracle** — +13.1 over r14 at the same
  step, and **+7.6 over r14 at step 49k**, i.e. it passes r14's entire 49k-step trajectory in
  under a third of the steps. rank-1 (29.7) also beats r14 at both steps. On the post-2023-06-01
  cutoff control the ordering holds (r15 oracle 52.9 vs r14 47.1).
- **Protein–protein is behind** at the same step: −2.9 oracle, −5.2 rank-1 vs r14 @14k, and it has
  not yet reached r14's 14k number, let alone its 49k one. Whether this is a slower start for the
  wider model or a real regression cannot be settled at 14k — r14's ppi oracle moved only
  67.0→69.9 across 35k further steps, so the gap r15 has to close is small but its rate is
  currently unknown.

Against the external field, r15 @14k sits last among the ranking-capable models on both
categories by rank-1, but its antibody–antigen *oracle* (52.7) has already caught Boltz-2
(52.7) and is within a point of OpenFold3-p2 (53.8) and ESMFold2 (54.9). AF3
remains 8.1 points ahead on ppi oracle and 12.1 on agab oracle. Nothing here changes the
standing conclusion that the deficit is concentrated in assembly, not in folding.

## Confidence is not yet calibrated on protein–protein

From `confidence_data.csv` (within-target Pearson r between `ranking_score` and per-sample
quality, Fisher-z averaged, 95% CI):

| category | r15 @14k mean → rank-1 | within-target r (95% CI) | r14 @14k r | r14 @49k r |
|---|---|---|---|---|
| protein–protein | 0.583 → **0.554** | **0.062 [−0.050, 0.172]** | 0.219 | 0.303 |
| antibody–antigen | 0.225 → **0.268** | 0.145 [0.036, 0.250] | 0.045 | 0.160 |
| monomer protein | 0.931 → 0.930 | 0.036 [−0.029, 0.101] | 0.108 | 0.163 |

On protein–protein r15's rank-1 lands **below its own sample mean** (0.554 vs 0.583) and the
calibration CI spans zero — the ranking is not merely uninformative there, selecting by it is
currently worse than taking a random sample. Antibody–antigen is the opposite: significant
positive calibration and rank-1 above mean. This is worth watching rather than acting on;
r14 showed the same ppi calibration climbing steadily with training (0.219 @14k → 0.303 @49k),
and `alpha_confidence` behaviour is unchanged between the arms.

The oracle→rank-1 gap on antibody–antigen is **23.0 points** (52.7 oracle vs 29.7 rank-1). The
structures are being generated; the ranking cannot find them.

## Per-category snapshot (`confidence_data.csv`)

| category | n | r15 mean | r15 rank-1 | r15 oracle | r14 @14k oracle |
|---|---|---|---|---|---|
| monomer protein (TM) | 261 | 0.931 | 0.930 | 0.942 | 0.942 |
| monomer RNA (TM) | 15 | 0.300 | 0.300 | 0.349 | 0.338 |
| monomer DNA (TM) | 11 | 0.192 | 0.190 | 0.227 | 0.223 |
| protein–protein (DockQ succ.) | 229 | 0.583 | 0.554 | 0.654 | 0.688 |
| antibody–antigen | 137 | 0.225 | 0.268 | **0.515** | 0.377 |
| protein–peptide | 31 | 0.800 | 0.800 | 0.943 | 0.943 |
| protein–DNA | 18 | 0.400 | 0.350 | 0.500 | 0.421 |
| protein–RNA | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

Monomer protein is at parity with r14 @14k (0.942 oracle) and with r14 @49k (0.944) — the width
scale-up neither helps nor hurts single-chain folding, consistent with folding being solved and
docking being the bottleneck. The small-n categories (RNA, DNA, peptide, protein–RNA) carry no
usable signal at these counts and are listed for completeness only.

## Caveats

- **One checkpoint, mid-training.** r15 is 14k steps into a run whose comparison point ran to
  49k. Every r15 number here is a point on a rising curve, and the ppi/agab split may not survive.
- **n differs slightly between tables.** The ranked figure pools all annotated interfaces
  (ppi n=209, agab n=91); the confidence table counts scored interfaces (229 / 137); the
  models figure uses the single designated interface on the ∩ of all models (207 / 90). These
  are three different conventions on purpose — do not mix them.
- **rank-1 tie-break.** `ranking_score` is stored to 4 dp and r15's per-target scores cluster
  tightly, so tied top scores are common; rank-1 here is the mean over the tied set. See
  `../../README.md` § "rank-1 ties are not a detail". The rule changed on 2026-08-08, so r14
  numbers quoted above are recomputed under it and differ by up to 3.4 pts in tier split from
  what `out/r14/report.md` records inline.
- **Templates.** Bench inputs are template-free. That cost AF3 3.8 ppi / 9.9 agab points when
  measured through our harness, and there is no reason to think it costs r15 nothing — so the
  gap to the FoldBench-official AF3 rows overstates the model gap by an unknown amount.
