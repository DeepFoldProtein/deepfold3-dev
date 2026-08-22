# Complex language model

A masked language model over protein *complexes*: chain-wise RoPE attention
alternating with complex-wise NoPE attention, trained on MLM alone.

Package: `src/deepfold3/complexlm/`. Research notes, measurements and the
rationale for each decision: [`export/complex-lm/README.md`](../export/complex-lm/README.md).
Not to be confused with `model/plm.py`, which is the *frozen* single-chain PLM
frontend feeding DF3's pair track.

## Architecture

A complex is a set of chains. Inside a chain there is a residue order and a
meaningful sequence distance; between chains there is neither. The model
follows that split literally:

```
packed tokens ──┬── chain_cu    → chain-scoped attention, RoPE on the per-chain index
                └── complex_cu  → complex-scoped attention, no positional encoding
```

Both layer kinds read the same packed hidden tensor; only the cumulative
sequence-length vector handed to `torch.nn.attention.varlen.varlen_attn`
changes. No padding, no re-gathering between kinds.

`model.layer_pattern` is cycled over the layers and selects the kind of each:

| kind | attention scope | positional encoding |
| ---- | --------------- | ------------------- |
| `c`  | within a chain | RoPE, index restarts at 0 per chain |
| `g`  | within a complex | none |
| `r`  | within a complex | RoPE over the concatenated complex |
| `x`  | other chains only | none (dense path only — not a varlen mask) |

So `cg` is the main model, `ccg` the 2:1 arm, `r` the global-RoPE baseline,
`cx` cross-chain-only. Blocks follow ESMC — bias-free projections, QK layer
norm, SwiGLU at expansion 8/3 rounded to a multiple of 256, residual scaling
`sqrt(n_layers/36)` — with RMSNorm by default and **no BOS/EOS/sink token**.
Token ids are ESM2's, so a warm start is a plain embedding copy.

`attn_impl` is `varlen` (training) or `sdpa` (dense reference; the only path
that returns attention weights, and the only one supporting kind `x`). The
dense path takes over automatically on CPU or in fp32, which are kernel
constraints rather than preferences.

**Sizes.** `configs/plm-base.yaml` is ESM2-650M's shape (33 × 1280, 20 heads →
672M), the shape PPLM uses too; `configs/plm-{8m,35m,150m}.yaml` reproduce
ESM2's smaller members to within 5%. Measured on one B200 at
`token_budget=16384`: 672M runs 267 ms/step (59k tok/s, 52.6 GiB), so 100k
steps on 8 GPUs is ~7.4 h and 12.6 B tokens.

**Learning rate.** Measured at the production batch on a narrow proxy and
transferred: `mup: true`, `mup_base_width: 320`, `lr: 9.8e-4`. `plm-base.yaml`
still ships ESM2-650M's 4e-4 with muP off, which is an inherited value, not a
measured one — see `export/complex-lm/README.md`.

**muP.** `model.mup` (default off) puts the model in maximal-update
parameterisation: hidden and readout matrices take init variance and Adam LR
scaled by `d_model / mup_base_width`, weight decay scaled up so `lr × wd` stays
width-invariant, attention logits scaled `1/d_head`, readout output divided by
the width multiple. At `d_model == mup_base_width` every factor is 1 and the
model is identical to the standard parameterisation. Transfer is valid across
**width at fixed depth**, so a proxy sweep must hold `n_layers` at the target's
value — ESM2's ladder varies both and is not a valid proxy ladder. Verify with
`export/complex-lm/mup_coord_check.py` before trusting a transferred LR.

**Known blind spot.** Under `g`, an *exact* homomer produces token-for-token
the monomer's output at any copy number — softmax over a `k`-fold duplicated
key set equals softmax over one copy. See Result 1 of the research note;
`model.chain_signal: random` is the (default-off) counter-measure.

## Store format

One format for every source, under `{data_root}/{source}/`, all flat `.npy`
opened with `mmap_mode="r"`:

| file | dtype | meaning |
| ---- | ----- | ------- |
| `seq.npy` | uint8 `[R]` | unique chain sequences as token ids, concatenated |
| `chain_off.npy` | int64 `[C+1]` | offsets into `seq` |
| `chain_cluster.npy` | int32 `[C]` | sequence-cluster id, `-1` if unclustered |
| `slot_chain.npy` | int32 `[S]` | chain row of each complex-chain slot |
| `slot_iface.npy` | uint8 `[Rs]` | per-residue interface flag (structural sources) |
| `cx_off.npy` | int64 `[N+1]` | offsets into `slot_chain` |
| `cx_cluster.npy` | int32 `[N]` | complex-level cluster id |
| `cx_date.npy` | int32 `[N]` | release date `YYYYMMDD`, 0 if unknown |
| `ids.txt` | | complex ids, one per line |

Sequences are deduplicated globally, so **two slots are the same entity exactly
when they share a chain row** — homomer detection, homomer-paired masking and
the homo/hetero split all read off that. Interface flags are per *slot*: one
sequence has different interface residues in different complexes.

## Data pipeline

Draw a complex → permute its chains → crop to the length budgets → mask →
pack into a flat token axis until `token_budget` is reached.

- **Sampling** (`sampler.py`) is hierarchical: source, then homo/hetero within
  it, then a cluster, then a member. DDP needs no coordination — i.i.d. draws
  already shard.
- **Cropping** (`crop.py`) gives each chain a proportional share of
  `max_complex_len`, capped at `max_chain_len`, in a window anchored on an
  interface residue. Copies of one entity get the *same* window.
- **Masking** (`masking.py`) is PPLM's policy: 15% of residues, 30% at
  interfaces of structural sources, BERT 80/10/10. With `homomer_paired`,
  copies of an entity share **one** selection *and* one corruption draw — an
  OR of independent draws would raise the marginal rate to `1-(1-p)^k`.
- **Evaluation** masks are seeded from the complex index, so two checkpoints
  are scored on identical masked positions.

### PDB date cutoff, and why it is not enough

Every record carries a release date and the cutoff is applied at open time, so
moving it needs no rebuild. Dates come from `data/mmcif_meta.json`. The filter
is **fail-closed**: an undated complex is dropped unless `allow_undated: true`.
Train on `date_max`, evaluate on `date_min` over the same store.

A date cutoff alone is a leaky holdout: 51.3% of post-2021-09-30 PDB complexes
belong to a family that also appears before it. `train.eval_cluster_disjoint`
(default on) additionally drops every evaluation complex whose complex-cluster
occurs anywhere in the training mixture. Complex-cluster ids are positions in
one table built across all stores of a clustering run, so each store records
that run's token and only stores agreeing on it are compared.

## Metrics

`train/accuracy` and every `*/accuracy` are top-1 over the **scored** positions
— the ~16% the masker selected, the same set the loss uses, including the 10%
that 80/10/10 leaves unchanged (so it reads above a strict masked-token
accuracy, as in ESM2 and BERT). Baseline: always predicting the most frequent
residue gives 9.7% on this mixture.

`*_single` is the same masked positions scored with the model seeing **only
that residue's own chain** — the same packed batch with `complex_cu :=
chain_cu`, one extra forward pass. So `delta = loss_single − loss_full` is what
the partner chains were worth.

Groups: `all`, `homomer`/`heteromer`, `interface`/`non_interface`/`unlabelled`,
`homomer_interface`, `heteromer_interface`, `heteromer_non_interface`, and one
per source.

Two readings that are easy to get wrong:

- **`homomer/delta` is zero by construction**, not by failure — an exact
  homomer *is* its monomer under global NoPE, so hiding its partners changes
  nothing. In bf16 it shows as ~1e-4. Partner signal lives in `heteromer_*`.
- **`unlabelled` exists because a sequence-only source has no interface
  labels.** Counting its all-zero flags as "non-interface" turns the interface
  comparison into a source comparison; runs started before `iface_known` was
  added have that contamination and are correctable post hoc (see
  `export/complex-lm/README.md`).

Also logged at every eval: **permutation sensitivity** (loss change and
token/pooled cosine under a chain re-ordering) and **attention-sink**
statistics per layer and kind. Compare `x_uniform` / `frac_of_ceiling` /
`top5_mass`, **not** raw `max_mass`: the uniform baseline is `1/T` for both
kinds but a chain-scoped token's share is capped at `L_c/T` against 1 for a
complex-scoped one, so raw `max_mass` differs by a factor of the chain count
before any behavioural difference. `export/complex-lm/sink_probe.py` does the
full version (is it one token, the same token across layers, at a boundary).

## Commands

```bash
# Build stores (see /data/plm/build_stores.sh for the full set)
deepfold3 plm-prepare structure --data-dir /data/distillation/afdb_nvda \
    --output /data/plm/store/afdb --source afdb --workers 64
deepfold3 plm-prepare structure --data-dir /data/distillation/patchr_training \
    --output /data/plm/store/pdb --source pdb --mmcif-meta data/mmcif_meta.json --workers 64
deepfold3 plm-prepare pairs --fasta /data/plm/raw/STRING_seqs.fasta \
    --pairs /data/plm/raw/trainset.STRING.txt --output /data/plm/store/string --source string
deepfold3 plm-prepare stats --store /data/plm/store/pdb

# Cluster every store together; writes ids back and reports source overlap
deepfold3 plm-prepare cluster --store /data/plm/store/pdb --store /data/plm/store/afdb \
    --work-dir /data/plm/clust --threads 64

# Train
torchrun --nproc-per-node=8 -m deepfold3.scripts.plm_train \
    --config configs/plm-base.yaml --output runs/plm-r1
deepfold3 plm-train --config configs/plm-smoke.yaml --output /tmp/smoke --dry-run
bash kakao/plm-submit.sh plm-r1                                  # 1 node x 8 B200
bash kakao/plm-submit.sh plm-ccg "--override model.layer_pattern=ccg"
```

## Configs

`configs/plm-base.yaml` is the main model; `configs/plm-smoke.yaml` inherits
from it for a minutes-long end-to-end check. Ablations are overrides:

```bash
--override model.layer_pattern=ccg      # 2 chain : 1 global
--override model.layer_pattern=r        # global-RoPE baseline
--override model.chain_signal=random    # break the exact-homomer collapse
--override data.max_complex_len=2048    # context arm (the one that matters)
--override data.masking.interface_rate=0.15   # uniform-masking arm
--override data.masking.homomer_paired=false  # copying-shortcut arm
```
