# MHG-TriMul and the triangle-attention removal — design note

Two independent changes to the PairFormer pair track, planned as two
from-scratch arms:

| arm | change | question it answers |
| --- | --- | --- |
| **r17** | delete `pair_attn_row` / `pair_attn_col` from every block | is triangle attention worth 2.4x the block cost? |
| **r18** | r17 + replace `TriangleMul` with `MHG-TriMul` | does target-pair-conditioned routing over relation subspaces recover what was lost, and more? |

Both keep r16's widths: `c_z = 256`, `c_s = 384`, `H = 8`, `d = c_z / H = 32`.

---

## 1. What triangle attention costs

Measured with r16's exact fusion flags (`configs/initial.yaml:334-352` — fused
evo-attention on, fused pair-bias off, fused triangle-mul on), bf16, forward +
backward, single GPU. Reproduce with `bench_pair_track.py`; raw numbers in
`bench_pair_track.csv`.

| N | `tri_mul` | `tri_attn` | full block | block w/o tri-attn | speedup |
| --- | --- | --- | --- | --- | --- |
| 384 | 10.7 ms | 19.2 ms | 68.7 ms | 29.2 ms | **2.35x** |
| 640 | 29.5 ms | 78.0 ms | 240.5 ms | 78.7 ms | **3.06x** |

The ratio grows with N because triangle attention's softmax over `k` is
$O(N^3)$ with a larger constant than the triangle-multiply GEMM, and its
backward materializes more.

**Caveat, and it is a large one.** This is a single-block microbenchmark: no
activation checkpointing, no DDP, no recycling, and the MSA stack, atom
encoder, diffusion module and confidence head are untouched. The end-to-end
step speedup is strictly lower than 2.35x and must be measured on the real
config before a run is committed.

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

MHG-TriMul does **not** restore this. Its router selects *which relation
subspace* explains a pair, not *which intermediate node* does. The two are
orthogonal axes. This is why the arms are sequential: if a combined
`MHG + no-TriAttn` arm underperforms r16, the loss cannot be attributed.

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

## 3. MHG-TriMul

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

**Division of labor.** The softmax constraint $\sum_h g^{(h)}=H$ means the
router answers *which* subspace, never *whether* the module should fire at all.
The retained output gate $\sigma(G_o)$ answers *whether*. Keeping both is
deliberate; a sigmoid router would conflate them and reintroduce the
multiplicative-suppression pathology of v1.

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
| $\operatorname{LN}_{c_z}$ (AF3) | r17 baseline | **A** — routing only; exactly identity at init |
| $\operatorname{RMSNorm}_d$ | **B** — predicted harmful (dead heads amplified to unit scale) | **C** — full MHG-TriMul |

If C beats the baseline but B also matches it, the per-head norm is inert and
all of the gain is A. If B is *not* worse than the baseline, the coupling story
above is wrong.

Arm A is exactly function-identical to r17's `TriangleMul` at initialization
($g\equiv1$, $\operatorname{LN}_{c_z}$ retained), so it warm-starts from an r17
checkpoint with `load_strict=false` and reads in ~2-3k steps. Arm C does not:
$\operatorname{LN}_{c_z}\neq\operatorname{RMSNorm}_{d}$, so it needs
re-equilibration.

## 4. Sequencing, and the confound to avoid

Running `MHG + no-TriAttn` as a single first arm is cheaper only if it works.
If it does not, the loss is unattributable and decomposing it costs two further
runs — worst case three full runs against the sequential plan's one full run
plus short warm-started branches. The sequential order also makes r17 a
deliverable on its own (a 2.35x cheaper pair track at crop 384), and lets r18's
arm A warm-start from it for free.

**Open decision: what to do with the freed compute.** Three options, and they
answer different questions.

- *Keep depth 48 and crop 384.* Cleanest attribution; the speedup shows up as
  more steps in the same wall-clock. Report the r16 comparison at **both**
  iso-step and iso-wall-clock — they will disagree, and both are honest.
- *Iso-compute depth (48 -> ~114 at N=384).* Tests "is triangle attention worth
  its FLOPs" head-on, but 114 residual blocks is deep for a stack that already
  shows layer-0 amplitude dominance and an aligned-branch budget
  $\rho_l\propto 1/(c+l)$.
- *Spend on crop instead (384 -> ~640).* A no-TriAttn block at N=640 costs
  78.7 ms, about what a full block costs at N=384. Closest to the measured
  deficit — PPI difficulty is log-linear in BSA/L and the AF3 gap is a pure
  1.15-1.38x horizontal shift along that axis — but it changes the data
  distribution, which confounds the architecture comparison in a different way.

Recommendation: keep depth and crop fixed for r17 so the subtraction is clean,
bank the speedup as steps, and decide how to spend the surplus at r18.
