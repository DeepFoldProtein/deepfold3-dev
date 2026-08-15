# MHR-TriMul and the triangle-attention removal — design note

**MHR = Multi-Head *Routed* triangle multiplication.** Not "gated": every gate
in the module — $\sigma(G_a)$, $\sigma(G_b)$, $\sigma(G_o)$ — is AF3's and
unchanged. What is new is the router, and a softmax router is not a gate. A gate
is an independent per-channel multiplier in $(0,1)$; the router is bound by
$\sum_h g^{(h)} = H$, so it **reallocates across heads rather than gating
them**. That constraint is the design: the router answers *which* head explains
a pair, and the retained output gate answers *whether* the module should fire at
all.

Two independent changes to the PairFormer pair track, planned as two
from-scratch arms:

| arm | change | question it answers |
| --- | --- | --- |
| **r17** | delete `pair_attn_row` / `pair_attn_col` from every block, **and** add the MHR router (per-head norm off) | is triangle attention worth 2.4x the block cost, and does target-pair-conditioned routing over relation subspaces help? |
| **r18** | r17 + per-head normalization (`head_rms`) | does giving the router sole authority over head magnitude complete the mechanism? |

r17 is a combined arm by choice, but only one of its two changes carries risk at
initialization: the router is zero-initialized to `g == 1`, so MHR-TriMul is
*bitwise* AF3 TriMul at step 0. Per-head normalization is held back to r18 for
the attribution reason in §4, not because it is dangerous.

Both keep r16's widths: `c_z = 256`, `c_s = 384`, `H = 8`, `d = c_z / H = 32`.

---

## 1. What triangle attention costs

All numbers below were taken on an **idle GPU**, with every shape autotuned
before timing and the minimum taken over repeats. Earlier revisions of this note
carried figures that were wrong by ~2x; see *Measurement pitfalls* at the end of
this section for what went wrong, because the same traps are easy to re-enter.

### Training shape (forward + backward, one PairFormer block, c_z=256)

| N | `tri_mul` | `tri_attn` | full block | block w/o tri-attn | speedup |
| --- | --- | --- | --- | --- | --- |
| 384 | 4.50 ms | 8.22 ms | 33.36 ms | 12.46 ms | **2.68x** |
| 640 | 12.45 ms | 33.52 ms | 103.76 ms | 34.83 ms | **2.98x** |

### Inference shape (forward only, whole trunk, 4 Evoformer + 48 PairFormer, 128 MSA rows)

| N | r16 trunk | r17 trunk | speedup | peak r16 | peak r17 |
| --- | --- | --- | --- | --- | --- |
| 384 | 451.5 ms | 401.5 ms | **1.12x** | 3.87 GiB | 3.68 GiB |
| 768 | 2648.5 ms | 1672.7 ms | **1.58x** | 9.18 GiB | 8.57 GiB |
| 1280 | 13933.7 ms | 8831.9 ms | **1.58x** | 21.76 GiB | 20.15 GiB |

**Forward-only buys far less than forward+backward** — 1.58x against 2.98x — for
two compounding reasons. Triangle attention's backward is disproportionately
expensive relative to triangle multiplication's, and much of the trunk is
untouched by the arm and dilutes the win: MSA row attention, outer-product mean,
the MSA transition, both pair transitions, and the PairFormer single track.
(The Evoformer stack's *pair* sub-track is **not** in that list — it changed
exactly as the PairFormer's did; see the scope table below.) Quote 2.7-3.0x for
training and 1.6x for the trunk at inference; they are different quantities.

**The trunk is not all of inference.** Diffusion sampling (200 steps x 5
samples) is untouched by this arm, so end-to-end inference speedup is strictly
below 1.58x. That measurement has not been made.

### Measurement pitfalls (all of these produced wrong numbers here first)

1. **Autotune contamination.** The Triton kernels tune on first sight of a shape
   and persist the result, so a short warmup charges the first-measured variant
   for tuning the rest inherit free. This produced `tri_mul` *slower* than
   `tri_attn` at N=384 while the same run's block delta said the opposite. Fix:
   autotune every shape before timing anything; min over repeats.
2. **GPU sharing.** An 8-GPU inference job on the same host inflated an earlier
   set by ~2x and made the speedup non-monotonic in N. Check `nvidia-smi` for
   *other* processes, not just free memory.
3. **Measuring one config then the other.** Clock and thermal drift masquerade as
   a speedup. `bench_inference_trunk.py` interleaves at each length.
4. **Casting the model to bf16.** The trunk LayerNorms are fp32-always, so
   `.to(bfloat16)` raises a dtype mismatch at the first `pair_logits` GEMM.
   Inference runs fp32 parameters under `autocast`; the benchmark must too.

### Consistency checks (these are in `bench_pair_track.py`, and they matter)

* N=640: block delta 68.93 ms vs 2 x `tri_attn` 67.04 ms — 2.8% apart.
* N=384: 20.90 vs 16.44 ms — 27% apart. Expected: the block runs a row *and* a
  transposed column attention, and only the row variant is timed standalone. The
  gap closes at N=640, so the transpose penalty is relatively smaller there.
* `tri_mul` grew **2.77x** from N=384 to 640. The naive O(N^3) expectation
  (4.63x) is **wrong for these sizes**: at c_z=256 the projections cost
  N^2 x 6C^2 = 5.8e10 against a contraction of N^3 x C = 1.45e10, so triangle
  multiplication is projection-dominated here and should scale as N^2 — which
  predicts 2.78x. It is width, not length, that sets its cost in this regime.
* `tri_attn` grew 4.08x, near the O(N^3) prediction, as a contraction-dominated
  op should.
* Both trunks grow faster than N^3 between 768 and 1280 (5.26x / 5.28x vs 4.63x),
  and r16's N=1280 repeats were still descending (15.6 -> 14.8 -> 13.9 s), so the
  1280 row is the least settled of the three.

## 2. Why removing triangle attention is a real subtraction, not a simplification

The two pair operators differ in *which index selects `k`*:

$$\text{TriMul: } T_{ij}=\sum_k a_{ik}\odot b_{jk}
\qquad
\text{TriAttn: } z_{ij}\leftarrow\sum_k \operatorname{softmax}_k\!\big(q_{ij}\!\cdot\!k_{ik}+b_{jk}\big)\,v_{ik}$$

In triangle multiplication the weight on intermediate node `k` is a function of
the pairs $(i,k)$ and $(j,k)$ only — **never of the target pair $(i,j)$**. In
triangle attention the target pair *is* the query. Triangle attention is
therefore the only module in the trunk that performs target-pair-conditioned
selection over intermediate nodes, and it is the only pair operator whose
mixing over `k` is normalized (competitive) rather than an unnormalized dense
sum.

MHR-TriMul does **not** restore this. Its router selects *which relation
subspace* explains a pair, not *which intermediate node* does. The two are
orthogonal axes. This is why the arms are sequential: if a combined
`MHR + no-TriAttn` arm underperforms r16, the loss cannot be attributed.

### Pre-registered diagnostic

If the removal hurts, it should hurt **beta-sheet topology first**. Beta pairing
is a long-range register problem — residue `i` pairs with a specific `j` far in
sequence, and identifying it means selecting the right intermediate `k`, which
is exactly the capability being deleted. `df3v1` already measures a
beta-topology bottleneck (48% of sheets unformed) with helix formation intact,
so the metric is sensitive and the baseline is known.

Watch, in order of expected sensitivity:

1. `dssp_f1.py` — F1_beta (recall side), pooled over the FoldBench monomer set
2. `ss_confusion.py` — 3/8-state confusion, the beta recall cell specifically
3. the BP1/BP2 four-way strand-fate breakdown and the sliding-window coherence
   length (`reference_coherence_length_method`)
4. only then whole-model lDDT / TM

A drop confined to (1)-(3) with (4) intact says triangle attention is doing
strand register and nothing else, which is actionable. A uniform drop says it
is doing general pair reasoning and the FLOPs are earned.

## 3. Why a router at all

The arm exists for four reasons, and they are worth separating because the
measurements bear on them unequally.

1. **Homomeric and heteromeric interfaces bind by the same physicochemistry but
   arrive with very different input priors** — above all in coevolution. Two
   copies of the same entity have a paired MSA that is the same sequence twice,
   so the cross-chain covariance signal is degenerate; two different entities
   have a genuine one. A single composition rule has to serve both. This is the
   sharpest of the four because it is *not* a claim about physics, only about
   what the model is given, and it is directly testable with the `entity_id`
   split in §5.
2. **Triangle attention may not be needed.** Other pair tracks (Pairmixer, MSA
   Pairformer) drop or replace it. r17 tests that empirically rather than
   arguing it — see §2 for what is being given up and the pre-registered
   beta-topology diagnostic for how a loss would show itself.
3. **Not every interaction is the same kind of interaction.** Protein-protein,
   protein-nucleic and protein-ligand contacts differ in chemistry, in
   geometry, and in which input features carry the evidence. §5 measures whether
   the router actually separates them; the answer so far is "partly, and along a
   coarser axis than modality identity".
4. **Triangle multiplication is already router-like.** It composes $i\!-\!j$
   through an intermediate $k$, weighting each $k$ by a learned, input-dependent
   gate — a soft selection. What it cannot do is condition that selection on the
   target pair (§2). Adding an explicit, target-conditioned router over relation
   subspaces extends a mechanism the module already has rather than bolting on a
   foreign one.

## 4. MHR-TriMul

### Definition

Inputs $z\in\mathbb{R}^{N\times N\times c_z}$, mask $M\in\{0,1\}^{N\times N}$,
heads $H$, head width $d=c_z/H$.

$$x_{ij} = \operatorname{LayerNorm}(z_{ij})$$

**Operands** — unchanged from AF3, full-width input gates, mask applied before
the contraction so padding cannot enter $\sum_k$:

$$a_{ik}=M_{ik}\,\sigma\!\big(G_a(x_{ik})\big)\odot A(x_{ik}),
\qquad
b_{jk}=M_{jk}\,\sigma\!\big(G_b(x_{jk})\big)\odot B(x_{jk})$$

**Dense contraction** — unchanged, and identical for every head. Outgoing
$T_{ij}=\sum_k a_{ik}\odot b_{jk}$; incoming $T_{ij}=\sum_k a_{ki}\odot b_{kj}$.
Note that $\operatorname{einsum}(\texttt{ikhd,jkhd->ijhd})$ is *bitwise* equal
to $\operatorname{einsum}(\texttt{ikc,jkc->ijc})$ under $c=(h,d)$ — only $k$ is
contracted, so the head axis adds no structure here. The head partition earns
its keep in steps 4-5, not in the contraction.

**Per-head normalization** — the first departure from AF3, which normalizes
over all of $c_z$:

$$\hat T^{(h)}_{ij}=\operatorname{RMSNorm}_d\!\big(T^{(h)}_{ij}\big),
\qquad h=1,\dots,H$$

**Router**, conditioned on the *target* pair, sum-normalized so it reallocates
across heads without changing global scale:

$$g_{ij}=H\cdot\operatorname{softmax}_h\big(R(x_{ij})\big),
\qquad \tfrac1H\textstyle\sum_h g^{(h)}_{ij}=1$$

**Output** — output gate unchanged, applied after $W_o$ as in AF3:

$$\boxed{\;\Delta z_{ij}=\sigma\!\big(G_o(x_{ij})\big)\odot
W_o\Big[g^{(1)}_{ij}\hat T^{(1)}_{ij}\,\Big\Vert\,\cdots\,\Big\Vert\,g^{(H)}_{ij}\hat T^{(H)}_{ij}\Big]\;}$$

### Why the router is outside AF3's function class

Write $W_o^{(h)}\in\mathbb{R}^{c_z\times d}$ for the $h$-th column block of
$W_o$. Then

$$\Delta z_{ij}=\sigma(G_o x_{ij})\odot\sum_h g^{(h)}_{ij}W_o^{(h)}\hat T^{(h)}_{ij}$$

so the router makes the output projection input-dependent,
$W_o^{\text{eff}}(x_{ij})=\big[g^{(1)}W_o^{(1)}\,|\cdots|\,g^{(H)}W_o^{(H)}\big]$
— a rank-$H$ hypernetwork over $W_o$. AF3's output gate sits *after* $W_o$ and
can only scale output channels, applying one common factor to every head's
contribution to a given channel. The router sits *before* $W_o$ and can zero
head 3 while keeping head 5 even though both write to the same output channels.
Pre-multiplying by a block-constant diagonal is not expressible as
post-multiplying by a diagonal unless $W_o$ is block-diagonal.

This is the property v1 of the proposal lacked. There the router multiplied the
operands $a_{ik}, b_{jk}$, alongside full-width free gates $\sigma(G_a)$ whose
range is already $(0,1)$ — so $\sigma(G_a)\sigma(R)$ spans exactly the same set,
and the change was a reparameterization with zero expressivity gain.

### Design points worth stating explicitly

**Division of labor, and why it is now a config choice.** The softmax
constraint $\sum_h g^{(h)}=H$ means the router answers *which* subspace, never
*whether* the module should fire at all; the retained output gate
$\sigma(G_o)$ answers *whether*.

That constraint has a cost the r17 measurement made visible. It is an
architectural claim — that heads **compete for a fixed budget** — and for
relation subspaces that is probably wrong: two heads capturing complementary
relations (local backbone geometry, long-range contact) should be able to be on
for the same pair at once. Worse, with no temperature and no regularizer (and
r17 runs `weight_decay = 0.0`), a vertex is the *fixed point*: as
$\lVert W_r\rVert$ grows the softmax sharpens monotonically, so collapse to one
head is the default outcome rather than a finding about the task. Under softmax
"route harder" and "use fewer heads" are the same operation, which confounds the
two.

`trimul_router_kind="sigmoid"` gives $g^{(h)} = 2\sigma(R(x))^{(h)}$ instead:
independent per head, bounded in $(0,2)$, so complementary subspaces can
co-activate and there is no winner-take-all attractor. **The novelty claim does
not depend on this choice** — being outside AF3's function class comes from
acting *before* $W_o$ on its column blocks, not from the normalization. Its cost
is that $\mathrm{mean}_h\, g$ may drift from 1, so the branch magnitude drifts
and `out_gate` has to absorb it; `router/g_mean` logs exactly that (pinned at
1.0 for softmax, free for sigmoid).

This is **not** the v1 pathology. v1 stacked sigmoids on the *operands*
$a_{ik}, b_{jk}$ alongside free full-width gates, so it started at
$0.25^2 = 1/16$ and was absorbed by those gates. Here the gate is
post-contraction and pre-$W_o$, starts at exactly 1, and is not absorbable.

**A load-balancing loss is deliberately not offered.** In MoE, balance exists for
a *systems* reason — routing sends each token to $k$ of $N$ experts, so
imbalance idles devices and drops tokens. MHR computes **all $H$ heads for every
pair regardless of $g$**: no capacity, no dropping, no device imbalance. The only
remaining reason to want balance would be a belief that more subspaces is
better, which nothing here supports, and the loss would push heads on against
what the main objective says. Removing the competition is the cause-level fix;
balancing it is a counterforce.

Default is `softmax`, which is what r17 trains.

**Initialization.** $W_R=0$ gives $g^{(h)}\equiv 1$, so the router is an exact
no-op at step 0. Its gradient is nonetheless healthy: at $p=1/H$,
$\partial g^{(h)}/\partial r^{(h')}=\delta_{hh'}-1/H$, which is $O(1)$ — unlike
v1, where the routing signal passed through four stacked sigmoids and started at
1/16 of the ungated magnitude.

**RMSNorm rather than LayerNorm over $d$.** At $d=32$ the LayerNorm mean is
estimated from 32 samples, so it carries $\sigma/\sqrt{32}\approx 0.18\sigma$ of
noise and is subtracted from the signal; the variance estimate has relative
error $\sqrt{2/(d-1)}\approx 25\%$. RMSNorm drops the mean subtraction and
halves the noise sources. The repo already has RMSNorm branch machinery
(`residual_branch_rmsnorm`).

`head_rms` is RMSNorm proper — gain, no shift — so `norm_center` is built
bias-free in that mode rather than carrying a parameter with no consumer (which
DDP reports as unused). This is also aligned with the measured trunk-bias
finding: the LN and linear biases wander with no compensating benefit. The cost
is that `head_rms` then differs from `head_ln` by two things rather than only
the mean subtraction, so the B-vs-C comparison is slightly less clean; the shift
contributes $\sigma(G_o(x_{ij}))\odot(W_o c)$, a pair-dependent scaling of one
fixed direction, so the impurity is rank-1 and small.

**Head normalization and the router are coupled, not independent.**
$\operatorname{RMSNorm}_d$ forces $\lVert\hat T^{(h)}_{ij}\rVert=\sqrt d$ for
every head and every pair, which *destroys* the signal "this head has nothing to
say about this pair" — a signal AF3's $\operatorname{LN}_{c_z}$ preserves as
relative head magnitude. After per-head normalization every head speaks at the
same volume and only the router can turn one down. Per-head norm therefore
creates the problem the router solves; neither is expected to be useful alone.

**Cost.** $W_R$ is $c_z\times H = 2048$ parameters per direction per layer,
about 0.5% of a triangle-multiply module's ~393k. The extra activation is
$[N,N,H]$. The real cost is the loss of the `FusedLayerNormLinear` fusion of
`norm_center` + `out_proj` ([ops.py:135-141](../../src/deepfold3/model/ops.py#L135-L141)):
per-head norm is cheap as a view + `F.layer_norm`/RMS, but norm and linear no
longer fuse.

### Ablation grid

Three effects — head decomposition, independent normalization, conditional
routing — separate as a 2x2, and the informative cell is the one that is
*predicted to be bad*:

| | no router | router |
| --- | --- | --- |
| $\operatorname{LN}_{c_z}$ (AF3) | plain no-TriAttn baseline | **A = r17** — routing only; exactly identity at init |
| $\operatorname{RMSNorm}_d$ | **B** — predicted harmful (dead heads amplified to unit scale) | **C = r18** — full MHR-TriMul |

If C beats the baseline but B also matches it, the per-head norm is inert and
all of the gain is A. If B is *not* worse than the baseline, the coupling story
above is wrong.

Arm A is what r17 runs. Arm C (= r18) warm-starts from an r17 checkpoint with
`load_strict=false`, but not for free:
$\operatorname{LN}_{c_z}\neq\operatorname{RMSNorm}_{d}$ is a function change,
so it needs re-equilibration rather than reading in the ~2-3k an
identity-initialized change would.

## 5. r17 is the combined arm — what can and cannot be recovered afterward

r17 ships two changes at once (`configs/df3-stage0-r17.yaml`, from scratch,
depth 48 and crop held at r16's values). That buys one run instead of two and
costs attribution: a shortfall against r16 cannot be split between the deleted
attention and the new routing. How much of that cost is recoverable depends
entirely on *which* pieces are in the arm, and the pieces are not symmetric —
which is why r17 carries the router but not the per-head norm.

**The router is instrumented, and separately lesionable.**

- *Free, from step 0.* Two statistics are logged on the activation-meter window
  (`collect_router_stats` in `ops.py`, wired in `trainer.py`), and **entropy
  alone is not enough**:

  - `router/entropy` = $H(p)/\log H$ averaged over pairs and layers. 1.0 means
    the router never left uniform — inert, functionally AF3 TriMul.
  - `router/g_std` = the **across-pair** standard deviation of $g$, averaged
    over heads. This is the one that decides whether the router is *earning* its
    place. A router that is sharp but the same for every pair scales each column
    block of $W_o$ by a constant, which is a reparameterization of $W_o$ and
    therefore inside AF3's function class — it contributes nothing, yet its
    entropy is low and would read as "active". Only the pair-conditional part
    is outside the class, and only `g_std` sees it. Since
    $\tfrac1H\sum_h g^{(h)}=1$ by construction, 0.1 is a modest effect.

  | `entropy` | `g_std` | reading |
  | --- | --- | --- |
  | $\approx 1$ | $\approx 0$ | inert — the router never engaged |
  | $< 1$ | $\approx 0$ | active but **redundant** with $W_o$ |
  | $< 1$ | $> 0$ | genuinely pair-conditional routing |

  `router/{entropy,g_std}_by_module/*` give the depth profile, so "only the late
  layers route" is visible too.

  Two caveats on `g_std`. It is accumulated over every forward in the window and
  reported as a mean — recycling means several forwards per step, so this is not
  "whichever forward was last". And it measures only *within-example* variation:
  a router that varied across examples while staying constant within one would
  also be outside the function class and would read as 0 here, so `g_std` is a
  **conservative lower bound** on the conditioning the router carries.
- *One eval.* Zero the router in a trained checkpoint and re-bench:
  ```python
  for m in model.modules():
      if getattr(m, "router", None) is not None:
          m.router.weight.zero_()          # g ≡ 1
  ```
  This is a **lesion**, not a training ablation. It measures how much the
  trained model *depends* on routing, which is not the same as whether training
  with a router produced a better model — a network trained with a router can
  route its computation through it so thoroughly that removing it is
  catastrophic even where training without one would have reached the same
  place by another path. Directional evidence only.
- Note where the lesion lands: zeroing the router in full MHR leaves
  $\operatorname{RMSNorm}_d$ with no router, which is **cell B** — the cell
  predicted to be harmful. A large drop is consistent with the coupling story
  (the router is holding up a representation that per-head norm flattened). No
  drop means the router was inert, and by the same coupling argument the
  per-head norm is then doing nothing either.

**The per-head norm is not lesionable at all**, and that is why it is not in
r17. There is no setting of the parameters that turns
$\operatorname{RMSNorm}_d$ back into $\operatorname{LN}_{c_z}$ — they are
different functions with different statistics. Shipping it in r17 would make a
shortfall a three-way confound {triangle attention, per-head norm, router} that
no logging resolves. Holding it back keeps the failure case clean: **r17 loses
and `router/entropy` is 1.0 ⇒ the loss is the triangle-attention removal,
single cause, established from inside the run.**

The cost of that choice is real. Without per-head norm the router's authority is
partial — head magnitudes at a given $(i,j)$ still vary freely, so scaling a
head that is already small does little. The two pieces were designed as one
mechanism ("head magnitude is decided by the router, not by the contraction")
and r17 runs half of it, which may under-deliver and read as a weak negative on
routing that a fuller version would not have produced.

That asymmetry is the practical shape of the ablation grid: the router column is
cheap to probe post hoc, the norm row is not. To keep the
follow-ups cheap, every cell of the grid carries **identical `state_dict` keys**
(the per-head gain reuses `norm_center.weight` viewed as `[H, d]`, `router.weight`
is the only extra tensor, and `head_rms` alone drops `norm_center.bias`), so any
cell warm-starts from any other with `load_strict=false`.

## 6. Measured: the router is pair-conditional (r17, step 1500)

`probe_router.py` on 8 real examples from the run's own eval set
(`patchr_training`, post-2021-09-30, 153-384 tokens). Raw per-module numbers in
`router_stats_step1500.csv`.

| | value |
| --- | --- |
| `router/entropy` (mean over 104 modules) | 0.607 (min 0.045) |
| **`router/g_std`** | **0.396** (max 1.098) |
| `router/occupancy_max` | 0.973 |

Since $\tfrac1H\sum_h g^{(h)} = 1$ by construction, a `g_std` of 0.396 means the
per-head multipliers swing by about ±40% of their mean from pair to pair. The
router is **not** a constant rescale, so it is not a reparameterization of $W_o$
— this is the component that AF3's function class cannot express. Third row of
the table in §4.

**Routing is a late-trunk phenomenon**, and both statistics agree on that,
as does the growth of $\lVert W_r\rVert$ (PF mean 0.324 -> 0.720 -> 1.157 at
steps 500 / 1000 / 1500, with late layers 3.2x early):

| PairFormer band | entropy | `g_std` |
| --- | --- | --- |
| L0-11 | 0.856 | 0.212 |
| L12-23 | 0.637 | 0.313 |
| L24-35 | 0.456 | 0.402 |
| L36-47 | 0.388 | 0.730 |

**Head collapse, and what it implies for r18.** 12 of 104 modules sit below
entropy 0.20 — effectively single-head — concentrated in L30-38, and 40 of 104
are below 0.5. This is direct evidence for the prediction in §4 that cell **B**
(per-head norm, *no* router) is harmful: where a module has settled on one head,
$\operatorname{RMSNorm}_d$ would rescale the seven idle heads to unit norm with
nothing left to turn them down. It also qualifies MHR's own premise — in the
late trunk many modules are using roughly one reasoning subspace, not $H$.
Whether that is wasted capacity or the right answer for those layers is not
decidable from this measurement.

**What it routes *on*: chain relationship.** The pooled `g_std` says the router
is pair-conditional but not what it conditions on. Splitting the pairs by
`asym_id` (12 examples: 3 monomers, 9 multimers of 2-25 chains) gives, per
module, the per-head mean of $g$ on intra- vs inter-chain pairs, and the $L_1$
distance between those two profiles:

| comparison | mean $L_1$ | median |
| --- | --- | --- |
| inter-chain vs **all** intra-chain | 1.398 | 1.193 |
| **inter-chain vs *distant* intra-chain** ($\lvert i-j\rvert > 32$) | **1.078** | 0.926 |
| distant vs near intra-chain (pure sequence-separation effect) | 0.462 | 0.252 |

The middle row is the controlled number. Intra- and inter-chain pairs differ in
more than chain identity — most obviously sequence separation, since
intra-chain includes every $\lvert i-j\rvert = 1$ neighbour while inter-chain has
no sequence relation at all — so a router keyed on separation would show the
same split. Restricting the intra-chain side to distant pairs retains **77%** of
the raw separation, and the pure separation effect is less than half the size.
The specialization is by chain relationship, not by a sequence-distance proxy.

Individual heads split cleanly. L46 `tri_mul_in`, per-head $g$ (mean 1 per head
by construction):

| head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intra-chain | 2.42 | **3.59** | 0.04 | 0.13 | 0.06 | 1.06 | 0.54 | 0.15 |
| inter-chain | **5.19** | 1.54 | 0.01 | 0.02 | 0.01 | 1.09 | 0.11 | 0.02 |

Head 0 is the interface head, head 1 the intra-chain head. The five most
chain-selective modules are all in **L44-47** — the same late band where `g_std`
peaks. This is MHR's "relation subspace" premise doing what it was designed to
do, via a mechanism AF3 has no equivalent for.

Measurement noise: across two runs the mean separation moved 1.453 -> 1.398
(4%) and `g_std` 0.396 / 0.388 / 0.391 across three (2%). Crop-anchor randomness
changes the drawn examples slightly between runs.

Remaining confound: inter-chain pairs also differ from distant intra-chain pairs
in contact probability and in whether paired-MSA coevolution signal exists. The
cleaner control would compare contacting interfaces against non-contacting chain
pairs; not done.

**Specialization or head death?** Low `entropy` alone cannot tell them apart —
each pair committing to a *different* head, and the same few heads winning
everywhere, both drive it down. `marginal_entropy` = $H(\mathbb{E}_{ij}[p])/\log H$
swaps the averaging order, and by Jensen the gap
`marginal_entropy - entropy` is exactly $I(\text{pair};\text{head})/\log H$.
At step 2000, in effective heads ($8^{\text{entropy}}$):

| band | per pair | used at all | dead | $I$(pair; head) |
| --- | --- | --- | --- | --- |
| L0-11 | 6.16 | 6.54 | 1.46 | 0.09 bits |
| L12-23 | 4.14 | 4.55 | 3.45 | 0.13 bits |
| L24-35 | 2.83 | 3.41 | 4.59 | 0.25 bits |
| L36-47 | 2.20 | 3.24 | **4.76** | 0.54 bits |

**Death dominates.** In the late trunk roughly 4.8 of 8 heads are unused *at
all* — not selected against per pair, simply off. Overall the router carries
0.25 bits about the pair out of a possible 3, which is 27% of what a perfect
intra/inter-chain split would yield at the observed 34% inter-chain rate. So the
chain specialization reported above is real but is a split among the ~3
surviving heads, not an 8-way one. Specialization does grow with depth (0.09 ->
0.54 bits), but so does death (1.5 -> 4.8 heads), in the same layers.

This settles what the sigmoid arm is for: mostly **reclaiming dead capacity**,
not extending specialization. Whether those 4.8 heads died because softmax
forced a budget, or because three subspaces is genuinely enough for those
layers, is exactly what removing the competition would decide.

**Modality (12 nucleic-containing complexes, step 2000).** The first probe drew
12 protein-only examples — patchr is only 4.6% nucleic+protein — so it never
tested the motivation MHR was designed around. Selecting for nucleic content
(11 with nucleic, 6 with ligand, 1-24 chains) gives, per module, the $L_1$
distance between per-head profiles of each mol-type pair class:

| | prot_prot | prot_nucl | nucl_nucl | prot_lig |
| --- | --- | --- | --- | --- |
| **prot_prot** | — | 1.572 | **3.843** | 1.719 |
| **prot_nucl** | | — | **4.046** | 0.826 |
| **nucl_nucl** | | | — | **4.362** |

Chain relationship in the same sample, sequence-controlled, is 1.557 — the
reference scale.

`nucl_nucl` is the strongest routing signal measured anywhere in this probe, at
~2.5x the chain effect. But the router does **not** appear to code modality
*identity*: `prot_nucl` vs `prot_lig` is only 0.826, smaller than either is from
`prot_prot`. The separation instead orders by **how many partners are
non-protein** (0 / 1 / 2), which is a coarser thing than "protein, nucleic,
ligand and ion each want their own composition rule".

Weakest number in the table, and it is the biggest one: nucleic tokens run 2-41
out of 384, so `nucl_nucl` can be as few as 4 pairs in an example against 147k
total, and every example was weighted equally regardless. `nucl_nucl` pairs are
also spatially clustered (same strand, or complementary strands of a duplex), so
the same distance control applied to the chain result has not been applied here.
Re-measure on nucleic-rich records with a distance control before relying on it.

**Homomeric vs heteromeric interfaces (motivation 1).** Splitting inter-chain
pairs by `entity_id` — same entity across different chains is a homomeric
interface, different entities a heteromeric one — separates the per-head
profiles by **0.739** (20 examples, shadow weights, step 2500), against a
sequence-controlled chain-relationship reference of 1.009 in the same run. Real,
and about three quarters the size of the chain effect.

| comparison | sep |
| --- | --- |
| `nucl_nucl` vs everything else | 2.97 - 3.35 |
| intra- vs inter-chain (sequence-controlled) | 1.009 |
| `prot_prot` vs `prot_nucl` | 1.136 |
| **`inter_homo` vs `inter_hetero`** | **0.739** |
| `prot_lig` vs `inter_hetero` | 0.168 |

**Read this carefully: the router did not *discover* the distinction.**
`entity_id_same` is an explicit channel of the relative-position encoding
(`embedders.py`), and `asym_id_same` feeds `rel_pos` and `rel_chain`, so both the
homo/hetero and the intra/inter splits are flags the input hands to the router
rather than structure it inferred. What the measurement does establish is
narrower and still worth having: the router is **using a given prior to select a
composition rule**. AF3 receives the same flags, but all it can do with them is
change the pair representation; MHR lets them choose how $W_o$'s column blocks
are mixed, which by the argument in §4 is outside AF3's function class. That is
the correct strength of support for motivation 1 — the prior is exploited, not
learned.

`nucl_nucl` is different in kind. `mol_type` has no dedicated pair channel; it
enters only through `res_type` at the token embedding. So the nucleic separation
is closer to the router reading something the trunk *composed*. The sample
caveat above still applies, and it is the weakest number in the table.

**`prot_lig` vs `inter_hetero` is 0.168** — protein-ligand pairs are routed
essentially the same as heteromeric protein interfaces. That is evidence
*against* motivation 3, at least for ligands: the router is not applying a
distinct rule to them.

**What the router actually keys on: contact geometry, not molecule class.**
Classing pairs by ground-truth token-centre distance instead of by mol type
(20 examples, 17 with ligand, shadow weights, step 2500). A within-class null —
the separation between two random halves of the *same* class, accumulated
identically — gives the noise floor; it is conservative by about $\sqrt2$ since
each half holds half the pairs.

| comparison | sep | null | ratio |
| --- | --- | --- | --- |
| `geo_contact` (<8 Å) vs `geo_far` (≥15 Å) | **4.967** | 0.001 | ~5000x |
| `geo_contact` vs `geo_near` (8-15 Å) | 3.770 | 0.013 | ~290x |
| `geo_near` vs `geo_far` | 2.399 | 0.004 | ~600x |
| **`pp_iface_contact` vs `pp_iface_far`** | **2.125** | 0.057 | **37x** |
| `homo_pp` vs `hetero_pl` | 0.790 | 0.003 | ~260x |
| `hetero_pp` vs `hetero_pl` | 0.595 | 0.003 | ~200x |
| `homo_pp` vs `hetero_pp` | 0.392 | 0.003 | ~130x |

The decisive row is `pp_iface_contact` vs `pp_iface_far`: distance bands *within*
protein-protein interfaces, so mol type and chain relationship are both held
fixed and only distance varies. 2.125 against a 0.057 floor.

**Distance beats molecule class by 3.6-5x**, and at 31% of the maximum possible
separation (2H = 16) `geo_contact` vs `geo_far` is the largest effect measured
anywhere in this probe. Molecule-class effects are real — every one is over 100x
its null — but an order of magnitude smaller. So the router's primary axis is
the physics of the contact, not the identity of the molecules, which reframes
motivation 3: grouping a protein-ligand pocket with a heteromeric protein
interface may be the router getting it *right*, since both are non-covalent
interfaces of buried surface, while nucleic base-pairing genuinely is a
different regime.

**Two corrections this forced.** First, the earlier `prot_lig` vs
`inter_hetero` = 0.168, which I read as evidence against motivation 3, was an
artifact of overlapping masks: a ligand is its own chain with its own entity, so
nearly every `prot_lig` pair is *also* an `inter_hetero` pair, and in a
monomer+ligand example the two classes coincide. On disjoint classes the same
comparison is **0.595** — larger than the homo/hetero separation, not smaller.
Second, `inter_homo` vs `inter_hetero` fell from 0.739 to **0.494** once the
classes were made disjoint. The overlap biased the two comparisons in opposite
directions.

**Still uncontrolled:** the chain-relationship (1.38) and homo/hetero (0.49)
figures may themselves be partly distance effects, since intra-chain pairs are
on average much closer and only *sequence* separation was controlled, not 3D
distance. The clean version compares mol-type classes *within* a distance band.

**Drawn out per layer, the router does nothing over the first half of the
trunk.** `plot_router_map.py` renders, for one target at a time, the top-1 head
$\arg\max_h g_{ij}^{(h)}$ and the per-pair entropy $H(p_{ij})/\log H$ as $N\times N$
maps over all 48x2 modules, plus a depth curve and a reference figure (GT
contact, *predicted* contact from the distogram, GT distance, chain and entity
blocks). Outputs in `router_maps/` for a monomer (8vc8), a homomer (8s9q) and a
heteromer (7ykg) at step 2500 on shadow weights.

The depth curve is the decisive one: **per-pair and marginal entropy lie on top
of each other from L0 to about L25**, so the gap — $I(\text{pair};\text{head})/\log H$ —
is ~0 there, and only from L25 does it separate, reaching ~0.2 by L44-47. Over
the first half of the trunk the router carries no pair information at all,
whether its entropy is high (uniform, a no-op) or low (collapsed, a constant
per-head rescale — which §4 shows is *inside* AF3's function class). **The router
does real work only in roughly the last twenty layers.** Band averages hid this.

**This corrects the "effective heads" reading.** L0-11 was reported as using 6.5
of 8 heads against 3.2 late, which invited the conclusion that the early trunk
uses its subspaces more richly. The maps say otherwise: early layers are
speckled at pixel granularity with no spatial structure, while late layers show
few colours in large coherent regions that resemble contact patterns (L39,
L44-47 especially). High entropy there is *undifferentiated*, not richly used —
so what looked like head death late may be the specialization itself. The gap
was already saying this; leading with effective-head counts obscured it.

Three things visible in the maps and consistent with the numbers: many panels
give the near-diagonal a distinct colour (L18/L20/L22/L28/L30 out), matching
distance dominance; single-colour panels cluster in the middle of the trunk and
overwhelmingly on the `in` direction, matching the measured collapse asymmetry
(9/48 `in` vs 3/48 `out`); and the small chain of 7ykg appears as an edge strip
that takes its own head in some layers.

Caveat on the heteromer: 7ykg is 250 tokens with chain B at ~20 of them, so it
is a large chain plus a short peptide rather than a balanced heterodimer.

**The axes the router separates, in order of size** (step 3000, shadow, 16
examples; nulls are the within-class random-half floor):

| axis | comparison | sep | null |
| --- | --- | --- | --- |
| **1. distance** | `geo_contact` (<8 Å) vs `geo_far` (≥15 Å) | **5.86** | 0.002-0.017 |
| **2. chain** | inter-chain contact vs intra-chain contact | **2.98-3.43** | 0.065 |
| **3. secondary structure** | contact classes against each other | **0.66-1.26** | 0.07-0.12 |

Secondary structure is assigned by the P-SEA criterion — CA(i)-CA(i+2,3,4)
must *simultaneously* match (5.5, 5.3, 6.4) Å for helix or (6.7, 9.9, 12.4) Å for
strand. A first attempt used a single d(i,i+4) band and produced impossible
assignments (73% helix with 4% strand on one target); with three distances the
spread across targets — 63/2, 39/18, 23/21, 18/24, 4/46 — is ordinary fold
diversity, all-alpha through all-beta.

Holding contact fixed, `hh_contact` vs `ss_pair` is 1.258, `hh_contact` vs
`hs_contact` 1.087, `ss_pair` vs `hs_contact` 0.657 — all 8-13x their nulls. So
the router does give the diagonal, helix packing and sheet pairing different
heads, which is what the routing maps look like; but the effect is a fifth of
the distance axis.

**This is negative for the beta diagnostic.** `ss_pair` — the beta-sheet
register — is the *least* distinctive of the contact classes, not the most.
There is no sign of a dedicated rule for it, so MHR is not compensating for the
target-pair-conditioned k-selection that triangle attention supplied (§2). That
is consistent in direction with the benchmark deficit, though it is
circumstantial; the actual test is DSSP F1.

Two numbers in that table must not be quoted as secondary-structure results:
`ss_pair` vs `ss_far` (5.36) and `hh_local` vs `geo_far` (6.53) are contact
versus non-contact, i.e. the distance axis measured again.

**What this does not show.** That the router *helps*. A network will route
through whatever capacity it is given; being used is necessary for the mechanism
to matter, not sufficient. Only the benchmark answers that.

## 7. How the router numbers are computed

Every router figure in §5 comes from `probe_router.py`. The procedure and the
reason for each choice:

### Capture point

Inside `_TriangleMulBase._mhr_head_mix`, immediately after the gate is formed
from the router logits $r_{ij} = W_r\,\operatorname{LN}(z)_{ij}$:

$$g_{ij}^{(h)} = H\cdot\operatorname{softmax}_h(r_{ij}) \qquad [B, N, N, H]$$

$g$ is measured rather than $r$ or the product $g\hat T$ because $g$ is the
quantity that actually multiplies the head block — it is the causal variable.
Logits are pre-nonlinearity and have no scale interpretation; the product mixes
in the variance of $\hat T$.

### Per-call statistics

Over the $P = B\!\cdot\!N\!\cdot\!N$ pairs of one forward:

$$p_{ij}^{(h)} = \frac{g_{ij}^{(h)}}{\sum_{h'} g_{ij}^{(h')}}
\qquad
\texttt{entropy} = \frac1P\sum_{ij}\frac{H(p_{ij})}{\log H}
\qquad
\texttt{marginal} = \frac{H(\bar p)}{\log H},\ \ \bar p^{(h)} = \frac1P\sum_{ij} p_{ij}^{(h)}$$

$$\texttt{g\_std} = \frac1H\sum_h \operatorname{std}_{ij}\big(g_{ij}^{(h)}\big)$$

Normalizing by $\log H$ makes the entropies comparable across head counts and
across `router_kind`. Deriving $p$ as $g/\sum_h g$ reduces **exactly** to the
softmax probabilities when $\sum_h g = H$, and gives the sigmoid variant the same
ruler. `g_std` is taken on $g$ rather than $p$ because $g$ is what multiplies the
block and its scale is pinned — under softmax $\operatorname{mean}_h g = 1$, so
0.5 reads directly as "swings by half its mean".

### Why the gap is a mutual information, not an ad-hoc index

Let the pair index be a uniform random variable $J$ and the head $Y \sim p_J$.
Then $H(Y\mid J) = \mathbb{E}_J[H(p_J)] = \texttt{entropy}$ and
$H(Y) = H(\mathbb{E}_J[p_J]) = \texttt{marginal}$, so

$$\texttt{marginal} - \texttt{entropy} \;=\; H(Y) - H(Y\mid J) \;=\; \frac{I(J;Y)}{\log H}$$

This is an identity — it holds whatever the router is doing, with no modelling
assumption, and Jensen guarantees it is non-negative.

### Aggregation

Each call adds its scalars to a running sum with a counter;
`_flush_router_stats` divides and resets, so a reported value is the mean over
**every forward in the window** rather than whichever forward happened to be
last (recycling means several per example). `collect_router_stats` then takes an
arithmetic mean over the 104 router-bearing modules (PairFormer 48x2 +
Evoformer 4x2); only `occupancy_max/min` are extrema.

### Derived quantities

**Effective heads** $= H^{\texttt{entropy}}$. With $e = H_{\text{nats}}/\ln H$,
perplexity $= \exp(H_{\text{nats}}) = H^{e}$ — the number of equally likely heads
with the same entropy. Band figures are the mean of $H^{e}$, not $H^{\text{mean }e}$;
by Jensen these differ.

**Bits** $=$ gap $\times \log_2 H =$ gap $\times 3$.

**Pair-class separation.** A wrapper around `_record_router_stats` receives the
same $g$ and averages it over the pairs selected by $[N,N]$ masks built from
`asym_id` and `mol_type`:

$$\texttt{sep}(A,B) = \sum_h \big\lvert \bar g_A^{(h)} - \bar g_B^{(h)} \big\rvert,
\qquad \bar g_A^{(h)} = \operatorname{mean}_{(i,j)\in A} g_{ij}^{(h)}$$

$L_1$ because $g$'s scale is pinned at 1 per head, so the sum of absolute
differences reads as "how many units of gate got reallocated between the two
classes".

### What the procedure gets wrong

Two of these were found only while writing this section, and both have been
fixed in the script. **They matter: re-measuring step 2500 on the corrected
settings moved `g_std` from 0.583 to 0.385 (-34%) and the specialization gap
from 0.0997 to 0.0539 (-46%)** — far outside the ~2-4% run-to-run noise. Every
router number in §5 predates the fix and is being re-measured; treat those
figures as provisional and the *relative* depth and pair-class patterns as more
trustworthy than the absolute levels.

The direction is what you would expect. At `decay = 0.999` the EMA shadow is an
average over roughly the last thousand updates, and $\lVert W_r\rVert$ is growing
fast at this stage (PF mean 0.32 -> 0.72 -> 1.16 over steps 500 / 1000 / 1500),
so the shadow's router is systematically smaller, hence flatter, than the live
one. Both are real; they answer different questions. **Shadow** is the network
the `eval/*` metrics describe and the one that would be deployed. **Live** is
what the model currently is, and during rapid growth it leads.

1. **EMA snapshot (fixed).** The run is AdamW with `ema_decay = 0.999` and
   `eval_snapshot: average`, which for a non-Schedule-Free optimizer with EMA
   present is upgraded to the **shadow** — so every `eval/*` metric quoted for
   r17 comes from the shadow weights. The probe loaded `blob["model"]`, the raw
   live weights, and was therefore measuring a different network from the one
   the eval numbers describe. `--snapshot` now defaults to `shadow`.
2. **Recycle count (fixed).** The run's eval uses
   `num_recycles = min(max_recycles, 4) = 3`, i.e. 4 trunk passes. The probe
   used 4, i.e. 5 passes, so its last pass saw a pair representation one stage
   more developed than training or eval ever produces. `--recycles` now
   defaults to 3.
3. **Recycle stages are pooled.** Every forward gets equal weight, so a reported
   value averages over recycle depth as well as over pairs. Early recycles have
   a less-developed $z$ and therefore less for the router to condition on.
4. **Example size is ignored.** Pairs are averaged *within* a forward before
   forwards are averaged, so a 153-token example counts as much as a 384-token
   one despite having six times fewer pairs.
5. **Class size is ignored**, which is the flaw that lands on the largest number
   in the modality table: `nucl_nucl` can be 4 pairs in one example and 1681 in
   another, weighted equally.
6. **Module weight is uniform.** A module whose branch barely reaches the
   residual stream votes as loudly as a dominant one.
7. **Sample.** 12 examples, one draw per checkpoint. Measured run-to-run noise
   is ~2% on `g_std` and ~4% on `sep`.

Padding is *not* a problem here, which was checked: examples are collated one at
a time and no bucket padding is applied (token counts came out 153 / 204 / 303 /
306 / 384), so no invalid pairs enter the statistics.

## 8. Sequencing, and the confound to avoid

Running `MHR + no-TriAttn` as a single arm is cheaper only if it works. If it
does not, the loss is unattributable and decomposing it costs further runs. That
was the argument for splitting r17/r18; the arm was combined anyway, so the
mitigation is §4 — instrument the half that can be instrumented, and accept
that the norm half needs a run.

**Decided: depth 48 and crop 384 held at r16's values.** The speedup is banked
as steps, and the r16 comparison is to be reported at **both** iso-step and
iso-wall-clock — they will disagree, and both are honest. The alternatives, for
the record: Three options, and they
answer different questions.

- *Iso-compute depth (48 -> ~114 at N=384).* Tests "is triangle attention worth
  its FLOPs" head-on, but 114 residual blocks is deep for a stack that already
  shows layer-0 amplitude dominance and an aligned-branch budget
  $\rho_l\propto 1/(c+l)$.
- *Spend on crop instead (384 -> ~640).* A no-TriAttn block at N=640 costs
  78.7 ms, about what a full block costs at N=384. Closest to the measured
  deficit — PPI difficulty is log-linear in BSA/L and the AF3 gap is a pure
  1.15-1.38x horizontal shift along that axis — but it changes the data
  distribution, which confounds the architecture comparison in a different way.

The surplus gets spent at r18, once r17 has said whether the pair track
survives without triangle attention.

## 9. Where it lives

| | |
| --- | --- |
| module | `TriangleMulOutgoing` / `TriangleMulIncoming` (`model/ops.py`), `num_heads` / `head_norm` / `router` |
| block | `use_triangle_attention` on `PairFormerBlock` and `EvoformerBlock` (`model/evoformer.py`) |
| config | `use_triangle_attention`, `trimul_num_heads`, `trimul_head_norm`, `trimul_router` on `EvoformerConfig` / `PairFormerConfig` |
| run | `configs/df3-stage0-r17.yaml` (arm A), `configs/df3-stage0-r18.yaml` (arm C, warm-started from r17) |
| grid cells B and the plain baseline | `--override model.{evoformer,pairformer}.trimul_router=false` on r18 / r17 — no config file |
| read-out | `router/entropy*`, `router/occupancy_*` on the `act_rms` log window |
| tests | `tests/test_mhr_trimul.py` |
| known pre-existing failure | `tests/triton/test_evo_attn_robustness.py::test_row_norm_heterogeneity` (`fail_frac=4.69e-2` vs a `1e-2` limit). Confirmed identical, bit for bit, at `023c064` — before any of this work — so it is unrelated to the pair-track changes and is *not* fixed here. |
| benchmark | `bench_pair_track.py` -> `bench_pair_track.csv` |

### Scope

Verified against the live r17 checkpoint at step 1500 (`pair_attn` tensors
remaining: 96, all outside the trunk):

| stack | changed | untouched |
| --- | --- | --- |
| `msa_stack` (Evoformer, 4 blocks) | `pair_attn` **removed**; `tri_mul` -> MHR (8 routers) | `row_attn`, `outer_product`, `msa_transition`, `pair_transition` |
| `trunk_pairformer` (48 blocks) | `pair_attn` **removed**; `tri_mul` -> MHR (96 routers) | everything else |
| template stack (2 blocks) | — | `pair_attn` (32 tensors), AF3 `TriangleMul` |
| confidence head (4 blocks) | — | `pair_attn` (64 tensors), AF3 `TriangleMul` |

The Evoformer stack's pair sub-track changed **identically** to the
PairFormer's. This is deliberate: the pair representation is shared between the
two stacks, so leaving triangle attention in one would be a hybrid. In cost
terms it is minor — 4 blocks against 48.

It does carry a distinct risk, though. The MSA stack is where the pair
representation is first formed, out of the outer-product mean; removing triangle
attention there hands the PairFormer a less-refined pair prior, which is not the
same kind of change as removing it from a 48-block stack that already has one.

The read-out heads are left at AF3 so they are not confounded with the trunk
arm. The consequence is that the confidence head now has a different pair
architecture from the trunk it reads; if the arm succeeds, r18+ should make them
consistent.
