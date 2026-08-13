# PLM interface probe — is there interface signal in a frozen ESM2 embedding?

**Date:** 2026-08-13 · **Status:** interim; no training arm launched
**Code:** [`probe.py`](probe.py), [`layer_sweep.py`](layer_sweep.py), [`vs_r16.py`](vs_r16.py), [`decompose.py`](decompose.py), [`probe_pplm.py`](probe_pplm.py)
**Raw data:** `probe_results_{esm2,saesm2}.csv`, `probe_per_target_{esm2,saesm2}.csv`, `layer_sweep_{esm2,saesm2}.csv`, `vs_r16_*_step17000.csv`, `decompose_pair_signal.csv`, `probe_pplm.csv`

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
- **The contact-pair row is superseded by Result 5.** The rank-32 bilinear probe reported here (0.53–0.55) failed to fit; the marginal product reaches 0.69–0.75 on the same task. The conclusion survives in a sharper form — the pair signal exists but is rank-1, carrying no pairing information.
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

## Result 5 — why each model fails, and why the reasons differ

Code: [`decompose.py`](decompose.py) · data: `decompose_pair_signal.csv`, `probe_pplm.csv`

### The pair signal is real, but it is rank-1

Result 1's rank-32 bilinear probe reported ~0.53 on contact pairs. That was a **probe failure**: the plain product of the two marginal interface probabilities scores **0.69 (ESM2) / 0.74 (SaESM2)** on the same task, and its rank correlation with the fitted bilinear score is ~0.00 — the bilinear form never found the solution. Result 1's "contact pairs are not encoded" is withdrawn; the correct statement is below.

Conditioning on the marginals settles what kind of signal it is:

| | marginals only $(\text{logit}\,p_i,\ \text{logit}\,p_j)$ | + cosine | cosine alone | **increment** |
|---|---|---|---|---|
| ESM2 | 0.6926 | 0.7028 | 0.6492 | **+0.0102** |
| SaESM2 | 0.7439 | 0.7510 | 0.7079 | **+0.0071** |

Given both residues' interface probabilities, the embedding cosine adds **one AUC point**. So

$$
z_{ij} \;\approx\; \phi(p_i)\,\psi(p_j)
$$

— **rank-1 along the interface-propensity direction.** The pair representation says "both of these look like interface residues," never "these two touch each other." That is the ceiling of an outer product over two chains embedded independently: no co-evolutionary term is ever formed, so none can be projected.

### ESM2 — the signal is generic, and misses the target

Interface AUC 0.666, and **ρ = +0.107 (p = 0.395)** against r16's DockQ — no relationship in either direction.

A UniRef50 masked LM learns conservation and hydrophobicity, which produces a **generic surface-property readout**. Interfaces are enriched in conserved hydrophobic patches, hence 0.67 pooled. But the same signature fires on active sites, crystal contacts, and exposed core — it is not interface-specific, so *which* targets it reads correctly is close to random. Attaching it is harmless, and harmlessly uninformative.

### SaESM2 — the signal is real, and we already have it

Interface AUC 0.698 (+0.032 over ESM2), cosine 0.708 (+0.059), and **ρ = +0.338 (p = 0.006)**, with probe AUC 0.603 on r16's failing half against 0.673 on its succeeding half.

Structure-aware pre-training genuinely buys a better surface readout. The problem is that this is what r16 **already does well**: monomer lDDT 0.875 sits +6.5σ above r14's plateau, and PPI interface lDDT is above plateau too. Folding and surface geometry are solved; SaESM2 duplicates them. Hence the positive correlation — both are strong on well-behaved targets and both fail on the same hard ones — and it survives controlling for interface fraction and chain length, so it is not merely a difficulty effect.

### AbAg — an information ceiling, not a model limitation

The antigen side sits at chance for both models at every depth. This is expected and no single-chain PLM can fix it: **an epitope is not an intrinsic property of the antigen sequence.** The same antigen surface is bound at different sites by different antibodies, so there is no function from antigen sequence to epitope to learn.

The paratope *is* intrinsic — CDRs are the binding site by construction — which is exactly why that side reaches 0.94. Reading an epitope requires seeing the antibody and the antigen together, which is what `joined` attempted; ESM2 has never seen a multi-chain input, and it degraded (0.88 → 0.73).

### Summary

| Model | Why it fails | Character |
|---|---|---|
| **ESM2** | encodes generic surface properties, not interface specificity → target-level near-random | **harmless noise** |
| **SaESM2** | real structural signal, but duplicates what r16 already has → weakest where r16 fails | **redundant** |
| **both** | independent per-chain embeddings make the outer product rank-1; pairing adds +0.01 | **structural ceiling** |
| **AbAg** | epitope is not a function of antigen sequence | **information ceiling** |

Three of the four reduce to one fact: **these are single-chain models.** A complex-trained PLM is aimed at exactly that, which is why the backend-agnostic plumbing was worth keeping.

## Result 6 — PPLM, a pair-trained LM, does not break the ceiling either

Code: [`probe_pplm.py`](probe_pplm.py) · data: `probe_pplm.csv`

Result 5 traced three of the four failure modes to the models being single-chain, which makes a pair-trained LM the obvious test. [PPLM](https://github.com/junliu621/PPLM) (Liu, Chen & Zhang, *Nat. Commun.* 2026) is exactly that: same shape as ESM2-650M (33 layers, 1280 wide, 20 heads) but taking two sequences at once with a dedicated inter-chain attention path, gated per layer by a learned `inter_attn_weight` and exposed as **660 genuine $L_A \times L_B$ maps** — not an outer product. Weights load with zero missing or unexpected keys.

| | interface AUC | antigen side | marginals only | + inter-attn (660 ch.) | inter-attn alone | **increment** |
|---|---|---|---|---|---|---|
| PPI | 0.616 | — | 0.6885 | 0.6924 | 0.6449 | **+0.0039** |
| AbAg | 0.818 | **0.569** | 0.8736 | 0.8738 | 0.6837 | **+0.0002** |

**The increment is no better than ESM2's** (+0.0102) or SaESM2's (+0.0071) — on antibody–antigen it is zero. The inter-chain attention does carry real signal on its own (0.64 / 0.68, well above chance) but it is **entirely redundant with the two marginals**. Purpose-built cross-chain attention lands in the same place as an outer product of independent embeddings.

The antigen side stays at **0.569**, inside the 0.44–0.61 band the single-chain models occupy, even though PPLM sees the antibody while embedding the antigen. The epitope ceiling is therefore not about architecture.

Interface AUC on PPI is *lower* than either ESM2 (0.666) or SaESM2 (0.698) — the pair-trained representation is specialized in a way that costs it on the per-residue task.

**The strongest evidence is PPLM's own contact predictor.** `run_pplm-contact.py` does not feed the language model to a head and stop. It runs HHblits against UniRef, extracts HHM profiles and DCA couplings, builds a **paired MSA**, extracts features from that too, and only then adds the PPLM attention as one input among several. A pair language model with a released interface-contact predictor still needs co-evolution to predict contacts.

That is the cleanest statement of what all six results converge on: **pairing information lives in the paired MSA, not in sequence-model weights.** DF3 already has a paired-MSA channel; the question was never whether to add a new source but whether that channel is being used to its depth.

## Result 7 — retraction: the probe cannot support the claim it made about PPLM

Result 6 concluded that PPLM's inter-chain attention carries no pairing information, on the strength of a +0.0039 AUC increment. **The PPLM paper's own ablation says the opposite, and it is the better evidence.**

From [Liu, Chen & Zhang (*Nat. Commun.* 2026)](https://www.nature.com/articles/s41467-026-70457-5), Fig. 4D–E, on the 2PMY example:

| variant | top-N precision | top-L precision |
|---|---|---|
| **w/o PPLM features** | 16.7% | 20.8% |
| **w/o MSA features** | 78.2% | 100% |
| full model | 81.3% | 100% |

and on Homodimer300, PPLM-Contact reaches 77.8% top-L against DeepInter 68.9%, CDPred 63.0%, DeepHomo2.0 51.4%. Removing the language model collapses it; removing coevolution barely moves it. In their architecture the PPLM features carry the contact signal and the MSA is nearly dispensable — the reverse of what Result 6 asserted.

**The discrepancy is not the metric and not feature scaling.** Both were checked:

| | AUC | top-L precision |
|---|---|---|
| marginals only | 0.6884 | 0.6% |
| + inter-attn (raw) | 0.6923 | 0.7% |
| inter-attn, channel z-score | 0.6542 | 1.2% |
| inter-attn, log | 0.7312 | 1.0% |
| inter-attn, APC-corrected | 0.5353 | 0.8% |
| inter-attn, z-score + APC | 0.4959 | 0.2% |

Top-L precision tops out at 1.2% against their 77.8%. No normalization rescues it.

**The difference is probe capacity.** `PPLM_Contact` is a 12-block network: `Inter_ResNet` over the 806-channel pair input (660 of which are PPLM attention), then triangle-multiplication blocks coupling the inter-chain map to both intra-chain maps, then transformer blocks. Triangle multiplication is exactly the operation that composes "$i$ contacts $k$" and "$k$ contacts $j$" into a statement about $(i,j)$ — a higher-order function of the pair map that **a logistic regression on raw attention values cannot express, at any normalization.**

This is the third time in this investigation that an underpowered probe returned a misleading negative: the rank-32 bilinear form in Result 1 (0.53 where the marginal product gave 0.70), the mean/max summary of `inter_attn` in Result 6's first pass, and now the linear readout itself.

### What survives and what does not

**Retracted:** "PPLM's inter-chain attention carries no pairing information", and with it the sweeping form of Result 6's conclusion — *pairing information lives in the paired MSA, not in sequence-model weights*. Their w/o-MSA ablation retaining 78.2% top-N is direct evidence against it.

**Still standing**, because these are per-residue linear probes on embeddings, which is what a linear probe is for:

- interface-residue AUC and its amino-acid nulls (Result 1)
- the antigen side sitting at chance across every depth and all three models (Results 2, 6)
- partner-specific AUC and the zero partner-to-partner variance of per-chain models (Result 7 table below)
- the positive correlation with r16's per-target DockQ (Result 4) — the signal is redundant with what r16 already does well

**Now reopened:** whether a PLM pair channel could help DF3. The mechanism their result depends on — triangle multiplication over the pair map — is not exotic; **DF3's trunk already has 48 layers of it.** A shim feeding `lm_z` into `z` puts the PLM's pair signal exactly where triangle updates can act on it, which is the setting their ablation says works and the setting my probe could not emulate.

## Result 7b — partner specificity, on a common footing

The interface numbers in Results 1 and 6 used different labels: `probe.py` asked "does residue $i$ touch *any* other chain", `probe_pplm.py` asked "does it touch *this partner*". On the harder, partner-specific label, all three models on identical targets:

| | partner-specific AUC | partner-to-partner score sd | contact: marginals | + pair feature | increment |
|---|---|---|---|---|---|
| ESM2 | 0.6296 | **0.0** | 0.6941 | 0.7019 (cosine) | +0.0078 |
| SaESM2 | **0.6584** | **0.0** | **0.7446** | **0.7496** (cosine) | +0.0050 |
| PPLM | 0.6159 | **0.1818** | 0.6885 | 0.6924 (660 ch.) | +0.0039 |
| PPLM | — | — | 0.6885 | 0.6917 (cosine) | +0.0032 |

ESM2's earlier 0.666 was on the easy label; on the hard one it is 0.630, so the three models are much closer than Result 6 implied.

**The `score_sd_across_partners` column is structural, not empirical.** A per-chain model returns one embedding per chain, so its prediction for chain A is identical no matter who the partner is — the variance is exactly zero by construction, and it *cannot* discriminate partners. PPLM re-embeds against each partner and its predictions do move (0.18). Under a linear readout that movement does not improve the AUC; whether a triangle-update trunk could use it is precisely the question Result 7 reopens.

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
| Is cross-chain *pairing* information there? | **No** — pair score is rank-1; cosine adds +0.01 over the two marginals |
| Can it find the epitope? | **No** (antigen side 0.44–0.61, near chance) |
| Does joining chains help? | **No — it hurts** (AbAg 0.88 → 0.73) |
| Is SaESM2 better than ESM2? | **Yes on PPI** (+0.03 to +0.06); marginal on AbAg, epitope stays at chance |
| Does a pair-trained LM (PPLM) help? | **Unresolved** — my linear probe says no (+0.004), their trained ablation says yes (w/o PPLM: 100% → 20.8% top-L). See Result 7 |
| Is the signal informative where r16 fails? | **No — it is anti-informative** (ρ = +0.34, strong where r16 already wins) |

Three findings compound into one recommendation. The embedding supplies an **interface-residue prior**, not contact predictions. It has **no epitope signal**, which is r16's actual AbAg failure. And on protein–protein its prior is **strongest exactly where r16 already succeeds** and weakest where it fails.

**Do not launch the shim arm on this evidence.** The infrastructure ([`esm2.py`](../../src/deepfold3/model/esm2.py), [`plm.py`](../../src/deepfold3/model/plm.py), [`lm_pair.py`](../../src/deepfold3/model/lm_pair.py)) is committed, tested and backend-agnostic, so it costs nothing to leave idle.

Result 7 is a retraction of the strongest version of that. My probe found PPLM's inter-chain attention adds +0.004, but PPLM's own ablation shows the opposite — remove the language-model features and top-L precision collapses from 100% to 20.8%, while removing MSA barely moves it. The gap is probe capacity: their contact head runs triangle multiplication over the pair map, and a logistic regression cannot express that at any normalization (checked across five feature transforms; top-L never exceeds 1.2% against their 77.8%).

So the shim arm is **not ruled out** — it is unevaluated. What my evidence does establish is narrower and still worth acting on: the *per-residue* interface prior these models supply is redundant with what r16 already does well (ρ = +0.34 against per-target DockQ), and the antigen side sits at chance for every model and depth tested.

The honest ordering is therefore: **the paired-MSA depth A/B first**, because it is cheap and aims at the same channel; and if a PLM arm is run afterwards, feed `lm_z` into `z` where DF3's 48 triangle-multiplication layers can act on it, which is the setting their ablation says works and mine could not emulate.

## Next

1. **The paired-MSA depth A/B is now the experiment**, not a fallback. Six results point at the same channel, and DF3 already has it: `patchr` carries 411 paired shards against 256 unpaired. What has never been measured is how much of r16's cross-chain deficit is depth-limited there — the monomer A/B found half of r14's monomer failures were MSA-limited, and the interface equivalent is untested.
2. **Address weaknesses 1 and 2** by adding non-binding chain pairs as negatives. Specificity is the one property no probe here measured, and Result 4 makes it the open question: a prior that is redundant on binders might still separate binders from non-binders.
3. **Replace the antibody heuristic with ANARCI** and re-confirm Result 2, which rests on a hand-written rule.
4. **Re-run Result 4 against AbAg DockQ.** The epitope null and the redundancy finding are separate claims and only the second has been tested, on PPI.
