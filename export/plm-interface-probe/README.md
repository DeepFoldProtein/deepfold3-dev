# PLM interface probe — is there interface signal in a frozen ESM2 embedding?

**Date:** 2026-08-13 · **Status:** interim; no training arm launched
**Code:** [`probe.py`](probe.py), [`layer_sweep.py`](layer_sweep.py)
**Raw data:** `probe_results_{esm2,saesm2}.csv`, `probe_per_target_{esm2,saesm2}.csv`, `layer_sweep_{esm2,saesm2}.csv`

## Why measure this first

r16's remaining deficit is **pose selection**. On protein–protein it has passed r14's plateau on DockQ and interface lDDT (Welch t = +3.90 and +2.67 over 9k–17k) but not on success rate (t = −1.47). The pose is fixed at the step-0 denoise from `z`, and failure tracks how much cross-chain information `z` carries, so improving it means putting more of that information into `z`.

Conditioning the pair track on a protein language model ([`lm_pair.LMPairShim`](../../src/deepfold3/model/lm_pair.py)) is one candidate. But **if the embedding carries no interface information, there is nothing for the shim to project.** This probe answers that before any training runs.

## Method

### Labels — taken straight from the CIF

Coordinates come from `bench/data/ground_truths/{pdb}-assembly1.cif` via gemmi: Cβ, falling back to Cα for glycine, first altloc. Nothing predicted or annotated enters the label. For residue $i$ of chain $A$ and residue $j$ of chain $B$,

$$
y_{ij} \;=\; \mathbb{1}\!\left[\, \lVert \mathbf{c}^A_i - \mathbf{c}^B_j \rVert_2 < 8\,\text{Å} \,\right]
$$

and an interface residue is the cross-chain OR of that:

$$
s_i \;=\; \max_{B \neq A} \; \max_j \; y_{ij}
$$

Positive rates are $\bar{s} \approx 0.25$ (interface residues) and $\bar{y} \approx 0.001$ (contact pairs).

**Independent check.** Every one of the 72 mappable interface pairs named in FoldBench's own `data/targets/*.csv` has contacts under this computation — **zero disagreements** — so the chain mapping and the contact criterion agree with the benchmark's annotation. All 80 targets have real inter-chain contact; none is a spurious singleton.

### Three probes

Write $h_i \in \mathbb{R}^{1280}$ for the frozen PLM hidden state and $e_i \in \{0,1\}^{20}$ for the amino-acid one-hot.

**(1) Interface residue** — logistic regression:

$$
\hat{s}_i = \sigma(w^\top h_i + b), \qquad
\mathcal{L} = -\sum_i \left[ \lambda\, s_i \log \hat{s}_i + (1-s_i)\log(1-\hat{s}_i) \right] + \gamma \lVert w \rVert^2
$$

with $\lambda = (1-\bar{s})/\bar{s}$. **Null:** the same regression on $e_i$, which learns nothing but **20 per-amino-acid interface propensities**.

**(2) Contact pair, zero-shot** — cosine similarity, no fitting:

$$
\text{score}_{ij} = \frac{h_i^\top h_j}{\lVert h_i \rVert \lVert h_j \rVert}
$$

**(3) Contact pair, trained** — low-rank bilinear form, $U, V \in \mathbb{R}^{r \times 1280}$, $r = 32$:

$$
\text{score}_{ij} = (U h_i)^\top (V h_j) + b
$$

**Null:** the same form on $e_i$, i.e. a **learned 20 × 20 contact-propensity matrix**. The rank is capped deliberately — the question is whether the information is linearly accessible, not how much a large probe can fit.

### Scoring

AUROC from ranks; with $n_+$ positives, $n_-$ negatives and $r_k$ the ascending rank of score $k$,

$$
\mathrm{AUC} = \frac{\sum_{k:\, y_k=1} r_k - \tfrac{1}{2} n_+(n_+ + 1)}{n_+ n_-}
$$

**Splits are by target** (first 20 train, last 20 test). Splitting by residue would put the same complex on both sides.

### Two corrections applied after a first pass

Both changed the numbers, so the first-pass results are superseded.

**Rotary spacing.** Unresolved residues have no coordinates and are dropped, which splices the sequence: **84 of 234 chains (36%) have breaks in author numbering**, 13 of them five or more. Feeding contiguous positions makes RoPE report a separation of 1 across a twenty-residue gap. The probe and [`plm.py`](../../src/deepfold3/model/plm.py) now pass explicit `position_ids` carrying true numbering.

Chain base depends on the join. Under `per_chain` the offset provably cancels — RoPE is relative and attention is block-diagonal, measured at max |Δ| = 4e-6 for a 5000-residue shift — so each chain restarts at 0, which keeps positions inside the range ESM2 was pre-trained on (author numbering runs from −3 in 8osw and can reach the thousands). Under `joined` the chains share one position axis, so restarting would give residue $k$ of chain A and residue $k$ of chain B the same phase, i.e. a relative distance of zero across the interface; there the axis keeps running.

**Degenerate targets.** At an 8 Å cutoff a peptide complex has every residue at the interface: `8zmm` (chains of 20 and 21) is **100%**, `8zmw` 98.6%, `8qab` 94.9%. Those AUCs are undefined (single class) or trivially high — the first pass silently dropped one PPI target to NaN and kept the rest. Targets are now restricted to interface fraction in [5%, 50%].

## Result 1 — interface residues yes, contact pairs no

Final layer (33), `per_chain`, 40 targets per category, after both corrections:

| | Interface residue AUC | | | Contact pair AUC | | |
|---|---|---|---|---|---|---|
| | ESM2 | SaESM2 | AA null | cosine (ESM2 / SaESM2) | bilinear (ESM2 / SaESM2) | AA null |
| **PPI** | 0.666 | **0.698** | 0.515 | 0.649 / **0.708** | 0.547 / 0.535 | 0.518 |
| **AbAg** | 0.881 | **0.888** | 0.577 | 0.686 / **0.786** | 0.785 / **0.802** | 0.576 |

- **Interface residues are encoded.** PPI at 0.67–0.70 against a null of 0.51 is information beyond residue identity — though weaker than the 0.71–0.76 the uncorrected pass reported.
- **PPI contact pairs essentially are not** — bilinear 0.53–0.55 against a null of 0.52. The embedding knows which residues sit at an interface but not **which pairs with which**.
- **SaESM2 is better on PPI** (+0.032 interface, +0.059 cosine) and marginal on AbAg. Its config is identical to ESM2's, so it drops into the same loader unchanged.

## Result 2 — the AbAg number is the antibody side (the important finding)

Splitting AbAg interface AUC by chain role inverts the reading:

| Layer | ESM2 antibody | ESM2 **antigen** | SaESM2 antibody | SaESM2 **antigen** |
|---|---|---|---|---|
| 0 | 0.597 | 0.519 | 0.597 | 0.519 |
| 8 | 0.881 | 0.588 | 0.886 | 0.555 |
| 16 | 0.924 | 0.473 | 0.920 | 0.533 |
| 24 | 0.928 | 0.554 | 0.924 | 0.438 |
| 33 | **0.938** | **0.599** | **0.944** | **0.598** |

**The antigen side sits at 0.44–0.61 at every depth — near chance** — rising only from 0.52 at layer 0 to ~0.60 at the last layer. The antibody side reaches 0.94.

The pooled 0.88 is an average dominated by the antibody: **9,110 antibody residues against 2,578 antigen residues, so 78% of the pool is antibody.** Most of what it measures is **"find the CDRs"**, which sequence alone nearly determines.

**But the epitope is what r16 gets wrong.** The r14 stage-2 analysis concluded the paratope is right, the epitope is wrong, and the wrong epitope does not reproduce across entries. The probe finds no signal in exactly that place.

## Result 3 — depth and chain joining

**Depth.** PPI climbs 0.52 (layer 0) → 0.62 (layer 12) → **0.67–0.70 (final layer)**, non-monotonically, with a dip through the middle (0.55–0.60). Contrary to the usual expectation that structural information peaks mid-stack, **the last layer is best here.** The shim's zero-initialized softmax mixture over all hidden states is the right construction for this and should end up weighting the top.

**Joining.** With correct positions, `joined` is now clearly *worse*, not ambiguous: AbAg interface drops **0.881 → 0.730** (ESM2) and **0.888 → 0.756** (SaESM2), and AbAg contact cosine collapses **0.686 → 0.547**. In the uncorrected pass `joined` looked marginally better; that was an artifact of contiguous positions keeping the packed sequence inside ESM2's trained range. Once the second chain's positions genuinely run past the first, the model is out of distribution and degrades. **Fix `join="per_chain"`.**

## Result 4 — the signal is where we do not need it

Interface prediction is only useful if it is informative on the targets r16 gets wrong. Per-target interface AUC (2-fold cross-fitted, so no target scores its own probe) against r16's per-target mean DockQ at step 17000, PPI, n = 65 targets — code in [`vs_r16.py`](vs_r16.py), data in `vs_r16_*_step17000.csv`:

| | Spearman ρ | p | probe AUC on r16's failing half | on its succeeding half |
|---|---|---|---|---|
| ESM2 | +0.107 | 0.395 | 0.620 | 0.641 |
| **SaESM2** | **+0.338** | **0.006** | **0.603** | **0.673** |

**The sign is the finding.** A *negative* correlation would mean the PLM reads interfaces well exactly where r16 is lost — complementary information, the case for adding it. The correlation is **positive**: SaESM2 reads the interface best on the targets r16 already folds correctly, and drops to 0.603 on the half r16 fails.

This is not a difficulty artifact. Controlling for interface fraction or mean chain length — both proxies for how hard a target is — leaves it intact:

$$
\rho_{\text{AUC},\,\text{DockQ}} = +0.338 \;\longrightarrow\; \rho_{\text{partial}} = +0.329 \;(\text{interface fraction}), \quad +0.342 \;(\text{chain length})
$$

Neither covariate correlates much with DockQ on its own ($\rho = +0.083$ and $+0.043$), so there is little for them to explain away.

ESM2's ρ = +0.107 at p = 0.395 is simply no relationship: its interface signal is uninformative about r16's outcome either way.

## Weaknesses — what this probe cannot answer

**1. Every example is a complex that binds.** The benchmark targets are all real complexes; there are no non-binding pairs. The probe therefore **never tests specificity**. "Does this surface have an interface-like patch" and "which of several candidate patches is the real one" are different questions, and the second is where we fail. An interface-residue AUC of 0.70 speaks only to the first.

**2. No partner discrimination.** Predicting that residue $i$ of chain $A$ is at an interface says nothing about **whether it binds chain $B$ or chain $C$**. With the label OR-ed as $\max_{B \neq A}$, multi-chain targets make the task easier, not harder.

**3. The antibody/antigen split uses an ad-hoc heuristic I wrote.** `is_antibody()` is "length 95–260, a `W` between 10 and 25 residues after the first `C`, at least two cysteines." It is not ANARCI or any standard numbering. Across 40 AbAg targets it flags 86 chains — 2.15 per target against an expected 2 for H+L, which looks about right — but **the misclassification rate was never checked, and Result 2 rests on this rule.**

**4. Pooled AUC overstates confidence.** Pooling ~16,000 residues looks like a large sample, but residues within one complex are strongly correlated, so **the effective sample size is the 20 test targets.** Per-target AUC standard deviation is 0.10–0.13, so differences below ~0.02 should not be read. Per-target values are in `probe_per_target_*.csv` for paired tests.

**5. Labels inherit whatever assembly was deposited.** Crystal contacts and over-expanded assemblies contaminate the positives — `8wmt` is already known to be 2× over-expanded and to score a crystal contact. The 5–50% interface-fraction filter removes the worst cases but not this one.

**6. The probe is not the shim.** A low-rank bilinear form is neither an upper nor a lower bound. The real shim uses the outer product **and** the difference and is followed by 48 pairformer layers, so it could extract more; conversely it never sees the contact label, only diffusion and distogram gradient, so it could extract less.

**7. Overlap with the PLMs' training data is unverified.** ESM2 was trained on UniRef50 and these targets' sequences may be in it. There are no structural labels there, but sequence-level memorization is not ruled out — and SaESM2, being structure-aware, carries a more direct version of this concern.

## Interim conclusions

| Question | Answer |
|---|---|
| Is interface information in the embedding? | **Yes** (PPI 0.70 vs 0.52 null) |
| Is cross-chain contact-pair information there? | **Essentially not for PPI** (0.55 vs 0.52 null) |
| Can it find the epitope? | **No** (antigen side 0.44–0.61, near chance) |
| Does joining chains help? | **No — it hurts** (AbAg 0.88 → 0.73) |
| Is SaESM2 better than ESM2? | **Yes on PPI** (+0.03 to +0.06); marginal on AbAg, epitope stays at chance |
| Is the signal informative where r16 fails? | **No — it is anti-informative** (ρ = +0.34, strong where r16 already wins) |

Three findings compound into one recommendation. The embedding supplies an **interface-residue prior**, not contact predictions. It has **no epitope signal**, which is r16's actual AbAg failure. And on protein–protein its prior is **strongest exactly where r16 already succeeds** and weakest where it fails.

**Do not launch the shim arm on this evidence.** The infrastructure ([`esm2.py`](../../src/deepfold3/model/esm2.py), [`plm.py`](../../src/deepfold3/model/plm.py), [`lm_pair.py`](../../src/deepfold3/model/lm_pair.py)) is committed, tested and backend-agnostic, so it costs nothing to leave idle until a PLM appears that clears these bars — a complex-trained model is exactly the case that could, since every failure above traces back to ESM2 having seen only single chains.

## Next

1. **Address weaknesses 1 and 2** by adding non-binding chain pairs (random pairings) as negatives. Specificity is the one property this probe never measured, and Result 4 makes it the deciding question: a prior that is redundant on binders might still discriminate binders from non-binders.
2. **Replace the antibody heuristic with ANARCI** and re-confirm Result 2, which currently rests on a hand-written rule.
3. **Re-run Result 4 against AbAg DockQ**, not just PPI. The epitope null and the redundancy finding are separate claims and only the second has been tested on PPI.
4. The **paired-MSA depth A/B** remains the cheaper and more direct test of the same underlying hypothesis — that `z` lacks cross-chain information — and is now the clear next experiment.
