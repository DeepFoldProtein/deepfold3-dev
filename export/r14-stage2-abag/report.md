# r14 stage 2: no measurable effect, and why antibody-antigen is stuck

2026-07-29. Run `df3-stage2-r14` (steps 25000–30000), bench series `df3v4r14`.

Three questions, answered in order, each one forced by the previous answer:

1. Did stochastic atomization (stage 2) change anything? → No, and the noise floor says why it could not have been seen.
2. Antibody-antigen rank-1 trails oracle by 20+ points — can a better sample selector close it? → No. The remaining gap is mostly a lottery no selector can win.
3. Then what is actually wrong with antibody-antigen? → The paratope is right, the epitope is wrong, and the wrong epitope is not reproducible across entries.

Raw tool output for every number below is in `raw/`; the scripts that produced it are in `scripts/`. Per-step bench aggregates are in `steps_summary.csv`.

---

## 1. Stage 2 (stochastic atomization) — no effect over 5000 steps

`runs/df3-stage2-r14` resumed from `runs/df3-stage1-r14/checkpoint_25000` with `configs/stage2.yaml`
(`atomize_prob 0.01`, `source interface`, `max_residues 8`, `atomize_eval false`).
The two runs' dumped resolved configs differ **only** in that block.

**Confound, recorded on purpose:** the two checkpoints were also produced by different code.
Stage 1 launched 10:20 UTC; the of3 template-placement fix (`be801d2`) landed 15:46; stage 2 launched 16:47.
`long_monomers` + `short_monomers` are 25% of the training mix and their templates were being placed
at crop-shifted residues before that fix. So "stage 2" here = atomization **+** corrected of3 templates
**+** more steps. There is no no-atomize control past 25000.

### Per-target DockQ, 6 checkpoints

| category | 25k | 26k | 27k | 28k | 29k | 30k |
|---|---|---|---|---|---|---|
| ppi mean | 0.4890 | 0.4905 | 0.4976 | 0.4905 | 0.4869 | 0.4878 |
| ppi oracle | 0.5798 | 0.5725 | 0.5966 | 0.5768 | 0.5644 | 0.5718 |
| AbAg mean | 0.2091 | 0.2082 | 0.1960 | 0.1997 | 0.2103 | 0.1984 |
| AbAg oracle | 0.3467 | 0.3378 | 0.3192 | 0.3236 | 0.3420 | 0.3418 |
| monomer lDDT | 0.8659 | 0.8674 | 0.8667 | 0.8657 | 0.8643 | 0.8655 |

Pooled windows, paired bootstrap over targets (`raw/stage2_window.txt`):

| comparison | delta | 95% CI |
|---|---|---|
| ppi mean, {25+26k} vs {29+30k} | −0.0024 | [−0.0169, +0.0120] |
| ppi oracle | −0.0080 | [−0.0259, +0.0098] |
| AbAg mean | −0.0043 | [−0.0266, +0.0168] |
| AbAg oracle | −0.0004 | [−0.0212, +0.0208] |

Six-point per-target slope: ppi −0.00068/1k steps [−0.0039, +0.0026], AbAg −0.00124 [−0.0063, +0.0036].

### The number that matters: churn

Mean |ΔDockQ| per target between **adjacent** checkpoints:

| | 25→26k | 26→27k | 27→28k | 28→29k | 29→30k |
|---|---|---|---|---|---|
| ppi | 0.0347 | 0.0385 | 0.0356 | 0.0470 | 0.0333 |
| AbAg | 0.0406 | 0.0492 | 0.0388 | 0.0512 | 0.0385 |

Constant across all five intervals and not shrinking with training. Every effect measured above is
±0.01, i.e. a quarter of the churn. Two-checkpoint windows still give CI half-widths of ±0.015 (ppi)
and ±0.022 (AbAg); four-checkpoint windows would give ~±0.010. **Single-checkpoint comparisons in this
regime carry no information** — step 27000 produced a ppi oracle gain that was "significant"
(+3.75 pt, CI [+0.42, +7.08]) and step 28000 took it all back with an equally "significant" loss.

Individual targets swing far harder than the aggregate: 8pyr C-D went 0.678 (15/20 passing) → 0.914
(20/20) → 0.018 (0/20) across 25k/26k/27k, with iLDDT following (0.779 → 0.848 → 0.599) and per-sample
sd staying ~0.02. Whole-complex pose bistability, not sampling noise.

**Verdict: atomization neither helped nor hurt over 5000 steps.** Deciding it needs either 4+ checkpoint
windows or a no-atomize control arm.

**Boundary for later readers of series `df3v4r14`:** checkpoints up to **31000** come from
`runs/df3-stage2-r14` (atomization ON); **32000 onward** come from `runs/df3-stage1-r14`, restarted
2026-07-30 06:53 UTC from the 31k weights with `atomize_prob: 0.0`. The series is continuous in step
number but not in recipe. Note this is a sequential switch, not a parallel control — 32k+ vs 25–30k still
confounds atomization with training time.

### The distribution, not just the mean

Means hide two things worth seeing (`ilddt_distribution.png`, `ilddt_quantiles.csv`): whether the *bottom*
of the distribution moves, and what shape it has. Inter-chain lDDT per interface (mean over its 20 samples),
14 checkpoints from 20k:

| | 20k | 25k | 30k | 31k | 32k (off) | 33k (off) |
|---|---|---|---|---|---|---|
| AbAg Q1 | 0.527 | 0.550 | 0.536 | 0.531 | 0.551 | 0.543 |
| AbAg median | 0.629 | 0.628 | 0.619 | 0.632 | 0.622 | 0.628 |
| ppi Q1 | 0.150 | 0.148 | 0.142 | 0.153 | 0.161 | 0.140 |
| ppi median | 0.648 | 0.613 | 0.650 | 0.651 | 0.639 | 0.621 |

AbAg's lower quartile oscillates in a 0.527–0.555 band across all 14 points; bootstrap deltas over targets
are 25k→30k −0.014 [−0.067, +0.008], 25k→33k −0.007 [−0.037, +0.026], 31k→33k +0.012 [−0.016, +0.045] —
**no rise at the bottom**, and the 31k→32k step that looks like one is the same amplitude as the 23k→30k
fall. p10 swings twice as wide (0.088–0.141) but is a thin tail with a CI of ±0.17.

The shape differs sharply between categories, and that is the real story for ppi. Its distribution is
**bimodal** — 57 of 240 interfaces below 0.1 (no interface formed at all) and a tall mode at 0.85–0.9 —
so a flat mean means *the two populations are not exchanging members*, not that everything is stuck in the
middle. This is the same structure the DockQ pass counts show (129 of 240 pass on all 20 samples, 67 on
none). AbAg instead sits in one broad mode at 0.55–0.75 with a small dead tail (~19 interfaces below 0.3):
the interface region is modelled, the pose is wrong — which is what §3 dissects.

---

## 2. Sample selection is not the bottleneck

Antibody-antigen at step 30000: random pick 28.3%, rank-1 31.9%, oracle 51.5%. The 20-point gap looks
like the biggest single win available. It is not.

### Every available confidence signal ranks equally badly

Within-target Spearman against DockQ, and the success rate of picking by each signal
(`raw/selector_comparison.txt`, `raw/iface_plddt_selector.txt`):

| selector | AbAg 30k ρ | AbAg 30k success | ppi 30k ρ | ppi 30k success |
|---|---|---|---|---|
| global `ranking_score` (0.8·ipTM+0.2·pTM) | +0.083 | 31.88% | +0.113 | 65.83% |
| per-chain-pair ipTM (`602e548`) | +0.105 | 34.06% | +0.109 | 66.25% |
| interface pLDDT (contact residues) | +0.131 | 31.90% | — | — |
| — random pick | | 28.26% | | 65.00% |
| — oracle | | 51.45% | | 72.08% |

Across-target ρ is +0.678 (AbAg) / +0.534 (ppi): **confidence knows which targets are hard, not which
sample is right.**

Per-chain-pair ipTM was added on the hypothesis that pooling every interface into one score cannot rank
a prediction whose interfaces disagree (AbAg averages 1.52 scored interfaces per prediction vs 1.15 for
ppi). Measured over three checkpoints the difference vs the global score is −1.45 / +2.17 / +2.17 pt on
138 interfaces — inside the churn — and ρ is unchanged. **The hypothesis was wrong: the bottleneck is the
head's within-target resolution, not the pooling.** The feature is still worth keeping as a per-interface
report to users; it is not a ranking fix.

(Joining bench interfaces to `chain_pair_iptm` requires OST's per-sample `chain_mapping`: OST names
interfaces by *reference* chain, the confidence JSON by *model* chain. Name-matching silently drops
22/144 AbAg and 57/240 ppi interfaces.)

### The remaining headroom is a lottery

Passing samples per target, AbAg step 30000 (`raw/headroom_shape.txt`):

| passing of 20 | targets |
|---|---|
| 0 | **67 (48.6%)** |
| 1–2 | 16 |
| 3–5 | 9 |
| 6–10 | 12 |
| 11–15 | 9 |
| 16–19 | 13 |
| 20 | 12 |

Ceiling for a selector that always finds a passing sample when ≥m of 20 pass:

| m≥1 (oracle) | m≥2 | m≥3 | m≥5 | m≥8 |
|---|---|---|---|---|
| 51.45% | 44.20% | 39.86% | 34.78% | **31.88%** |

Current rank-1 sits exactly at m≥8. A perfect-but-realistic selector (finds it when ≥3 of 20 pass) buys
+8 pt; the rest of the oracle is 1–2/20 events. ppi is worse still: 129 of 240 targets pass on all 20
samples and 67 on none, leaving 18 in between — rank-1 65.8% against an oracle of 72.1%.

---

## 3. What antibody-antigen actually gets wrong

Interfaces split by how many of 20 samples pass, step 30000. `fnat`/`fnonnat` come from OST's own DockQ
decomposition (`raw/abag_failure_groups.txt`):

| | never 0/20 (67) | partial 1–15/20 (46) | solved ≥16/20 (25) |
|---|---|---|---|
| fnat (native contacts recovered) | **0.029** (med 0.006) | 0.212 | 0.680 |
| fnat < 0.05 | **79%** | 15% | 0% |
| fnonnat (invented contacts) | **0.961** | 0.779 | 0.415 |
| iRMSD | 14.15 Å | 7.76 Å | 2.31 Å |
| LRMSD | 47.23 Å | 21.89 Å | 7.20 Å |
| whole-model lDDT | 0.823 | 0.859 | 0.887 |
| iLDDT | 0.472 | 0.615 | 0.752 |
| cross-sample contact Jaccard | 0.447 | 0.506 | 0.754 |
| **sd(DockQ) over 20 samples** | **0.0162** | 0.1483 | 0.1221 |

Three facts:

- **The chains fold.** lDDT 0.82 against 0.89 for solved targets. This is a docking failure, not a folding one.
- **It is a different epitope, not a shifted one.** 3% of native contacts, 96% of predicted contacts invented, LRMSD 47 Å.
- **All 20 samples agree on it.** sd(DockQ) 0.016 — an eighth of the solved group's. More samples cannot help; these targets are confidently wrong, not diverse.

### Paratope vs epitope

Split the interface by side, joining GT to model by ordinal position through OST's per-chain sequence
alignment (`raw/paratope_vs_epitope.txt`; GT chains must be looked up by `label_asym_id` — for 7txw the
interface (A, C) lives on auth chains L and A):

| | never (67) | partial (46) | solved (25) |
|---|---|---|---|
| paratope recall (antibody side) | **0.695** | 0.758 | 0.906 |
| epitope recall (antigen side) | **0.208** | 0.495 | 0.841 |

The antibody uses 70% of its real paratope even when the answer is completely wrong. never → solved moves
the paratope 1.3× and the epitope 4×. Caveat: CDR loops are the protruding tip of an Fv, so *any* docking
pose contacts them — read 0.695 as "the paratope is not the failure", not as "paratope selection is correct".

### Is the epitope chosen from the antigen alone?

Seven antigens recur across bench entries at 0.98–1.00 sequence identity, bound by different antibodies
(12 interface pairs). If the model picked a patch from antigen surface properties alone, predicted
epitopes would agree with each other *more* than the real ones do (`raw/same_antigen.txt`):

| | mean Jaccard |
|---|---|
| GT epitope vs GT epitope | 0.229 |
| predicted vs predicted | **0.101** |

The opposite. On the 7yds/8aok/8aom triple the real epitopes overlap 0.44–0.54 while the predictions
overlap 0.00 / 0.23 / 0.00. Cross-recall (predicted epitope vs the *partner's* real epitope) is ~0, so it
is not copying the wrong answer from a similar complex either — it is a third location.

**So the failure is not a collapsed prior and not a surface bias.** The mapping from (CDR geometry,
antigen surface) to epitope reacts strongly to its inputs but is not the right function: deterministic
within a target, arbitrary across targets.

---

## Standing implications

- Antibody-antigen is the one interface class with no co-evolutionary signal available in principle —
  antibodies and antigens do not co-evolve — so the MSA route that carries other interfaces is empty here.
  Nothing in this investigation suggests a short experiment flips it.
- Do not spend effort on sample selection for either category: +8 pt is the realistic ceiling for AbAg and
  ~+2 pt for ppi, both requiring a ranker far better than anything the head currently emits.
- If confidence-head work is revisited anyway, the lever is `mini_rollout_samples: 1` — one rollout per
  example means the loss never asks the head to compare two poses of the same target, which is exactly the
  discrimination that is missing (across-target ρ 0.68 vs within-target ρ 0.10). `alpha_confidence: 1e-4`
  is a second candidate.
- Any future stage comparison on this bench needs ≥4-checkpoint windows or a control arm. The per-checkpoint
  churn (0.035–0.050 DockQ) is a property of the model, not of the eval.

## Files

| path | contents |
|---|---|
| `steps_summary.csv` | per-category, per-step bench aggregates, 25k–30k, 8 categories |
| `ilddt_quantiles.csv` | inter-chain lDDT p10/Q1/median/Q3/p90 per step, 20k–33k, both interface categories |
| `ilddt_distribution.png` | quantile fan + step-33k histogram (the bimodality) |
| `raw/stage2_window.txt` | 6-point trajectories, pooled-window bootstrap, churn |
| `raw/headroom_shape.txt` | pass-count histogram + selector ceiling curve |
| `raw/selector_comparison.txt` | chain-pair ipTM vs global ranking_score, 3 checkpoints |
| `raw/iface_plddt_selector.txt` | interface-pLDDT selector |
| `raw/abag_failure_groups.txt` | never/partial/solved profile with fnat, fnonnat, spread |
| `raw/paratope_vs_epitope.txt` | paratope and epitope recall by group |
| `raw/same_antigen.txt` | shared-antigen epitope agreement |
| `scripts/` | the script behind each `raw/` file (run from the repo root) |

Related: `bench/scripts/step_compare.py` (`9c3589c`) for per-target checkpoint diffs;
`export/af3-vs-r14-foldbench/` for the AF3 comparison this continues.
