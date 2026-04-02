# Training Pipeline Comparison: Boltz vs Protenix vs DeepFold3

## 1. Architecture Overview

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Framework | PyTorch Lightning | Native PyTorch | Native PyTorch |
| Multi-GPU | DDP (Lightning) | DDP (manual) | DDP (manual) |
| FSDP | No | No | No |
| Config system | Hydra YAML + `_target_` | Nested dict + CLI merge | Dataclass + JSON |
| Experiment tracking | WandB (optional) | WandB (optional) | **None** |
| Batch size | 1 per GPU | 1 per GPU | 1 per GPU |
| Grad accumulation | 128 (via Lightning) | configurable `iters_to_accumulate` | configurable |
| Mixed precision | fp32 default | bf16 default | bf16/fp16/fp32 |
| Gradient clipping | 10.0 | 10.0 | 10.0 |

## 2. Data Preparation Pipeline

### 2.1 Structure Processing

| Step | Boltz | Protenix | DeepFold3 |
|------|-------|----------|-----------|
| mmCIF parsing | gemmi → structured `.npz` | biotite `MMCIFParser` → pickle | gemmi → structured `.npz` |
| Assembly | biological assembly 1 | biological assembly 1 | `gemmi.make_assembly()` (assembly 1) |
| Storage format | `.npz` (structured arrays) | `.pkl.gz` (gzip pickle) | `.npz` (structured arrays) |
| Metadata | resolution, method, dates | resolution, method, dates | resolution, method, dates |
| Quality filters | resolution, size, date | resolution, size, date | min/max length, unknown, CA dist, clashing |

### 2.2 Sequence Clustering

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Method | MMseqs2 `easy-cluster` 40% | Pre-computed PDB clusters (40%) | **Not implemented** |
| Protein | MMseqs2 at 40% seq identity | `clusters-by-entity-40.txt` from wwPDB | SHA256 hash placeholder |
| Nucleic acid | Each unique sequence = own cluster | Same PDB cluster file | N/A |
| Ligand | Each CCD code = own cluster | N/A (handled by entity) | N/A |
| Output | `clustering.json` | CSV with cluster_id per chain | `cluster_id` field in ChainInfo (unused) |

### 2.3 MSA Generation

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Search tool | External (pre-computed A3M) | JackHMMER/HHBlits (AF2-style) | MMseqs2 (ColabFold API or local) |
| Storage | `.npz` per chain | A3M files per chain | `.npz` per chain |
| Max sequences | 16384 | 16384 | 16384 |
| Pairing | Taxonomy-based (Redis) | UniRef-based pairing/non-pairing DBs | **Not implemented for training** |
| RNA MSA | Not mentioned | Separate RNA MSA pipeline | Not implemented |

### 2.4 Template Search

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Search method | Not detailed in training scripts | HHsearch / hmmsearch | **Not implemented for training** |
| Date filtering | Not detailed | release_date cutoff + obsolete check | N/A |
| Template dropout | Not mentioned | 0.25 during training | N/A |
| Storage | Pre-computed | HHR/A3M files alongside MSAs | N/A |

## 3. Training Data Loading

### 3.1 Sampling Strategy

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Sampler | `ClusterSampler` (infinite iterator) | `DistributedWeightedSampler` | `DistributedSampler` (uniform) |
| Weighting | `β × Σ(α × count) / cluster_size` | `β × Σ(α × count) / (cluster_size + ε)` | **No weighting** |
| α weights | prot=3, nucl=3, lig=1 | prot=3, nucl=3, lig=1 | N/A |
| β weights | chain=0.5, interface=1.0 | chain=0.5, interface=1.0 | N/A |
| Epoch size | 100,000 synthetic | 10,000 synthetic | len(records) |
| Interface bias | Yes (β_interface > β_chain) | Yes (same scheme) | No |

### 3.2 Cropping

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Methods | Hybrid spatial-contiguous | 3 methods (weighted random) | 2 methods (coin flip) |
| Spatial | KD-tree by 3D distance | KD-tree by 3D distance | KD-tree by 3D distance |
| Contiguous | Chain-walk with neighborhood window | AF-Multimer Algorithm 1 | Random chain, random start, walk forward |
| Interface-seeded | Via `interface_id` parameter | `SpatialInterfaceCropping` (40% weight) | No |
| Train weights | Unified (neighborhood_size controls blend) | contiguous=0.2, spatial=0.4, interface=0.4 | spatial_prob=0.5 |
| Max tokens | configurable | 384 (train) | 384 (configurable) |
| Ligand handling | Not detailed | Complete-ligand enforcement | No special handling |

### 3.3 Featurization

Both Boltz and Protenix produce equivalent feature sets (res_type, MSA, templates, ref atoms, GT positions). DeepFold3 produces all features except:

- **`resolution`**: never added (confidence loss gate is dead code)
- **Paired MSA**: always empty dict (no cross-chain coevolution)
- **Templates**: not loaded during training

## 4. Model Training

### 4.1 Recycling

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Max recycles (train) | 3 | 4 (N_cycle=4) | configurable (default 10) |
| Random recycles | `randint(0, max)` per step | fixed N_cycle | `randint(1, max)` per step |
| Gradient | Last recycle only | Last recycle only | Last recycle only |
| Implementation | Correct (loop with `set_grad_enabled`) | Correct (loop with `no_grad` wrapper) | **BUG**: `_prev_pair`/`_prev_single` written to features dict but `run_evoformer()` always initializes from zeros — multi-recycle is effectively 1 recycle |

### 4.2 Diffusion Training

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Noise distribution | `σ_data × exp(N(-1.2, 1.5²))` | `σ_data × exp(N(-1.2, 1.5²))` | Same |
| σ_data | 16.0 | 16.0 | 16.0 |
| Diffusion batch size | 16 | 48 | 48 |
| Augmentation | `center_random_augmentation` per step | `centre_random_augmentation` per step | `_random_augmentation` per step |
| Alignment | Kabsch (SVD, fp32) | Weighted rigid alignment (fp32) | Kabsch alignment |
| Preconditioning | EDM (c_skip, c_out, c_in) | EDM (c_skip, c_out, c_in) | EDM |
| Chunking | Not mentioned | `diffusion_chunk_size=4` | Not implemented |

### 4.3 Mini-Rollout (for Confidence)

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Steps | 20 | 20 | 20 |
| Samples | 1 | 1 | 1 |
| Gradient | No | No | No |
| Purpose | Predicted positions for confidence head | Same | Same |

### 4.4 Loss Functions

| Loss | Boltz | Protenix | DeepFold3 |
|------|-------|----------|-----------|
| Diffusion MSE | ✓ (EDM-weighted) | ✓ (EDM-weighted) | ✓ (EDM-weighted) |
| Smooth LDDT | ✓ (optional, default on) | ✓ (on in stage 1) | ✓ |
| Bond violation | ✗ | ✓ (stage 2, α=1.0) | **Config exists, not implemented** |
| Distogram | ✓ (CE, α=0.03) | ✓ (CE, α=0.03) | ✓ (CE, α=0.03) |
| pLDDT | ✓ (CE, 50 bins) | ✓ (CE, 50 bins) | ✓ (CE, 50 bins) |
| PAE | ✓ (CE, gated by α_pae) | ✓ (off stage 1, on stage 3) | ✓ (α_pae=0 default) |
| PDE | ✓ | ✓ | ✓ |
| Exp resolved | ✓ (BCE) | ✓ (BCE) | ✓ (BCE) |
| Resolution gate | Not mentioned | ✓ (0.1–4.0 Å) | Dead code (resolution not in features) |

**Molecule-type loss reweighting:**

| Mol type | Boltz MSE weight | Protenix MSE weight | DeepFold3 |
|----------|-----------------|---------------------|-----------|
| Protein | 1× | 1× | 1× |
| DNA | Not detailed | 5× | 1× (no reweighting) |
| RNA | Not detailed | 5× | 1× (no reweighting) |
| Ligand | 10× | 10× | 1× (no reweighting) |
| Nucleotide (Boltz) | 5× | — | — |

### 4.5 Loss Weights (Combined)

```
total_loss = α_diffusion × (MSE + smooth_LDDT [+ bond])
           + α_distogram × distogram_CE
           + α_confidence × (pLDDT + PDE + exp_resolved + α_pae × PAE)
```

| Weight | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| α_diffusion | 4.0 | 4.0 | 4.0 |
| α_distogram | 0.03 | 0.03 | 0.03 |
| α_confidence | 3e-3 (full) / 1e-4 (struct) | 1e-4 | 1e-4 |
| α_pae | 0 (stage 1) → 1 | 0 (stage 1) → 1 (stage 3) | 0 (default) |
| α_bond | N/A | 0 (stage 1) → 1 (stage 2) | 0 (unimplemented) |

## 5. Optimizer & Schedule

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Optimizer | Adam | Adam (AdamW optional) | Adam (AdamW-style weight decay on ≥2D) |
| Learning rate | 0.0018 | 0.0018 | 0.0018 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| ε | 1e-8 | 1e-8 | 1e-8 |
| Weight decay | Not mentioned | 1e-8 (optional) | 0.0 (default) |
| Warmup steps | 1000 | 2000 (demo) / 10 (config) | 1000 |
| Decay start | 50,000 | After warmup | 50,000 |
| Decay schedule | `0.95^(step/50000)` | `0.95^(step/50000)` | `0.95^(step/50000)` |
| Finetune schedule | Not detailed | Cosine annealing + warmup | Not implemented |

## 6. EMA

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Decay | configurable | 0.999 | 0.999 |
| Warm-start | `min(decay, (1+n)/(10+n))` | Not mentioned | `min(decay, (1+n)/(10+n))` |
| Apply for eval | Swap weights | Swap weights | Swap weights |
| Checkpoint | Saved in main checkpoint | Separate `{step}_ema_{decay}.pt` | Saved in main checkpoint |
| Selective update | No | Yes (`mutable_param_keywords`) | No |

## 7. Training Stages

### Boltz

Three config files suggest stages but details are sparse:

1. **Structure training** (`structure.yaml`): diffusion + distogram, confidence at 1e-4
2. **Confidence training** (`confidence.yaml`): confidence at 3e-3, structure frozen
3. **Full training** (`full.yaml`): joint diffusion + confidence at 3e-3

### Protenix (well-documented)

1. **Stage 1 — Initial**: α_diffusion=4.0, smooth_lddt=1.0, α_bond=0.0, α_pae=0.0. N_cycle=4. Full model trains.
2. **Stage 2 — Finetune (structure)**: α_bond=1.0, smooth_lddt=0.0. Bond geometry replaces smooth LDDT. Cosine annealing LR.
3. **Stage 3 — Finetune (confidence)**: train_confidence_only=True. Freezes trunk. α_pae=1.0. α_diffusion=0, α_distogram=0.

### DeepFold3

- **Single stage only**. `TrainingConfig` has `alpha_pae=0.0` (stage 1 behavior) but no mechanism for stage transitions, confidence-only training, or fine-tune LR schedules.

## 8. Checkpointing

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Format | Lightning `.ckpt` | `{step}.pt` | `checkpoint_{step}.pt` |
| Contents | model + optimizer + scheduler + EMA | model + optimizer + scheduler + step | model + optimizer + scheduler + EMA + step |
| Rotation | `save_top_k` (configurable) | All kept | All kept |
| Resume | `ckpt_path` to `trainer.fit()` | `load_checkpoint_path` + fine-grained flags | `--resume` flag |
| Partial load | via `strict_loading` flag | `load_strict`, `skip_load_optimizer/scheduler` | Not supported |
| Confidence from trunk | Special trunk→confidence weight copy | `load_strict=False` + new params | Not supported |

## 9. Evaluation

| Aspect | Boltz | Protenix | DeepFold3 |
|--------|-------|----------|-----------|
| Metrics | Per-type lDDT, RMSD, confidence MAE | lDDT, TM-score, DockQ (via external) | **None** (eval loop is a stub) |
| EMA for eval | Yes (weight swap) | Yes (weight swap) | Yes (weight swap) |
| Eval frequency | Every epoch | Every `eval_interval` steps | Every `eval_every` steps |
| Eval data | Separate validation set | Separate validation set | 5% random split |

## 10. Code Style Comparison

### Boltz Style

- **Lightning-based**: `training_step()`, `validation_step()`, `configure_optimizers()` — framework handles DDP, logging, checkpointing
- **ABC hierarchies**: `Cropper`, `Sampler`, `Tokenizer`, `Featurizer` — clean interfaces
- **Hydra config**: YAML with `_target_` for object instantiation
- **Composition**: Components injected into Dataset via config
- **Moderate size**: Well-factored but `training_step`/`validation_step` can be long
- **Naming**: snake_case, descriptive class names (`BoltzCropper`, `ClusterSampler`)

### Protenix Style

- **Monolithic trainer**: Single `AF3Trainer` class with `run()` loop — all training logic in one place
- **Dict-based config**: Nested Python dicts, merged with CLI, `GlobalConfigValue` for cross-references
- **Dense code**: Long methods, many inline conditions, extensive inline comments
- **Feature-rich**: NaN handling, distributed NaN sync, gradient accumulation, multiple LR schedulers
- **Direct**: Less abstraction, more explicit control flow

### DeepFold3 Style (current)

- **Native DDP**: Manual `Trainer` class, no framework dependency
- **Dataclass config**: `TrainingConfig`, `ModelConfig` — type-safe, validated in `__post_init__`
- **Clean separation**: `trainer.py`, `loss.py`, `ema.py`, `scheduler.py` are independent modules
- **Minimal**: Lean implementation, but some features are stubs

## 11. Gap Analysis: What DeepFold3 Needs

### Resolved

| Gap | Resolution |
|-----|------------|
| Recycling bug | Fixed — `run_evoformer()` reads `_prev_pair`/`_prev_single` from features |
| No cluster-weighted sampling | `WeightedClusterSampler` with AF3-style weights |
| No paired MSA | `ChainInfo.paired_msa_id` + `TrainingDataset` loads both paired and unpaired |
| No molecule-type MSE reweighting | DNA/RNA 5×, ligand 10× (configurable) |
| No interface-seeded cropping | 3-strategy crop: contiguous 20%, spatial 40%, interface 40% |
| No bond violation loss | `bond_loss()` + bond feature pipeline (backbone + ligand bonds, GT distances) |
| No training stages | Config override system (`--config` JSON + `--override key=value`) |
| No resolution feature | Added to feature dict from manifest metadata |
| No experiment tracking | WandB integration (`--wandb project`) |
| No eval metrics | `_evaluate()` runs `forward_training` + `training_loss` with EMA weights |
| No checkpoint rotation | `max_checkpoints` config (default 5) |
| No finetune LR schedule | `CosineAnnealingScheduler` + `build_scheduler()` factory |
| No NaN detection | Distributed NaN sync via all-reduce |

### Remaining

| Gap | Impact | Reference |
|-----|--------|-----------|
| **No template loading** | No structural homolog signal during training | Protenix loads + dropout 0.25 |
| **No nucleic backbone bonds** | O3'→P bonds missing from bond loss | Protenix covers all backbone |
| **No bucket padding** | `torch.compile` recompiles per sequence length | Low priority |
