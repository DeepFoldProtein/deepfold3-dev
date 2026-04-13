# Training Pipeline Comparison: Boltz vs Protenix vs OpenFold 3 vs DeepFold3

## 1. Architecture Overview

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Framework | PyTorch Lightning | Native PyTorch | PyTorch Lightning | Native PyTorch |
| Multi-GPU | DDP (Lightning) | DDP (manual) | DDP (Lightning) | DDP (manual) |
| FSDP | No | No | No | No |
| Config system | Hydra YAML + `_target_` | Nested dict + CLI merge | YAML + dataclasses | Dataclass + JSON |
| Experiment tracking | WandB (optional) | WandB (optional) | WandB (optional) | WandB (optional) |
| Batch size | 1 per GPU | 1 per GPU | configurable | 1 per GPU |
| Grad accumulation | 128 (via Lightning) | configurable `iters_to_accumulate` | configurable | configurable |
| Mixed precision | fp32 default | bf16 default | configurable | bf16/fp16/fp32 |
| Gradient clipping | 10.0 | 10.0 | 10.0 | 10.0 |
| License | MIT | Apache 2.0 (ByteDance) | Apache 2.0 (AlQuraishi Lab) | — |

## 2. Data Preparation Pipeline

### 2.1 Structure Processing

| Step | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|------|-------|----------|------------|-----------|
| mmCIF parsing | gemmi → structured `.npz` | biotite `MMCIFParser` → pickle | biotite → AtomArray `.npz` | gemmi → structured `.npz` |
| Assembly | biological assembly 1 | biological assembly 1 | biological assembly 1 | `gemmi.make_assembly()` (assembly 1) |
| Storage format | `.npz` (structured arrays) | `.pkl.gz` (gzip pickle) | `.npz` (biotite AtomArray) | `.npz` (structured arrays) |
| Metadata | resolution, method, dates | resolution, method, dates | JSON/LMDB dataset cache | resolution, method, dates |
| Quality filters | resolution, size, date | resolution, size, date | resolution, size, date | min/max length, unknown, CA dist, clashing |
| Import from OF3 | — | — | — | `deepfold3 import-of3` converts biotite NPZ → DF3 format |

### 2.2 Sequence Clustering

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Method | MMseqs2 `easy-cluster` 40% | Pre-computed PDB clusters (40%) | Pre-computed PDB clusters (40%) | Pre-computed PDB clusters or MMseqs2 |
| Protein | MMseqs2 at 40% seq identity | `clusters-by-entity-40.txt` from wwPDB | Per-chain `cluster_id` in dataset cache | `clusters-by-entity-40.txt` from wwPDB |
| Nucleic acid | Each unique sequence = own cluster | Same PDB cluster file | Same scheme | Same PDB cluster file |
| Ligand | Each CCD code = own cluster | N/A (handled by entity) | N/A | N/A (handled by entity) |
| Output | `clustering.json` | CSV with cluster_id per chain | JSON/LMDB dataset cache | `cluster_id` in ChainInfo → `WeightedClusterSampler` |

### 2.3 MSA Generation

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Search tool | External (pre-computed A3M) | JackHMMER/HHBlits (AF2-style) | ColabFold server, JackHMMER, hhblits | MMseqs2 (ColabFold API or local) |
| Storage | `.npz` per chain | A3M files per chain | A3M/STO or pre-parsed NPZ or LMDB | `.npz` per chain |
| Max sequences | 16384 | 16384 | configurable (`max_rows`) | 16384 |
| Pairing | Taxonomy-based (Redis) | UniRef-based pairing/non-pairing DBs | Paired + unpaired with `max_rows_paired` | Per-chain `paired_msa_id` in ChainInfo |
| RNA MSA | Not mentioned | Separate RNA MSA pipeline | Supported | Not implemented |

### 2.4 Template Search

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Search method | Not detailed in training scripts | HHsearch / hmmsearch | CIF/BCIF structures + pre-parsed NPZ | **Not implemented for training** |
| Date filtering | Not detailed | release_date cutoff + obsolete check | Per-dataset date cutoffs | N/A |
| Template dropout | Not mentioned | 0.0 default (configurable) | Not detailed | N/A |
| Storage | Pre-computed | HHR/A3M files alongside MSAs | CIF + template_structure_array NPZ | N/A |
| Template distogram bins | — | 39 | 39 | — |

## 3. Training Data Loading

### 3.1 Sampling Strategy

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Sampler | `ClusterSampler` (infinite iterator) | `DistributedWeightedSampler` | `OF3DistributedSampler` (multi-dataset multinomial) | `WeightedClusterSampler` (falls back to `DistributedSampler`) |
| Weighting | `β × Σ(α × count) / cluster_size` | `β × Σ(α × count) / (cluster_size + ε)` | `β × Σ(α × count) / cluster_size` (AF3 Algorithm 1) | `β × Σ(α × count) / cluster_size` (AF3 formula) |
| α weights | prot=3, nucl=3, lig=1 | prot=3, nucl=3, lig=1 | configurable `a_prot, a_nuc, a_ligand` | prot=3, nucl=3, lig=1 |
| β weights | chain=0.5, interface=1.0 | chain=0.5, interface=1.0 | configurable `w_chain, w_interface` | chain=0.5, interface=1.0 |
| Epoch size | 100,000 synthetic | 10,000 synthetic | configurable `epoch_len` | 10,000 synthetic (configurable) |
| Multi-dataset | No | Yes (per-dataset weights) | Yes (`SamplerDataset` with dataset probabilities) | No |
| Interface bias | Yes (β_interface > β_chain) | Yes (same scheme) | Yes (same scheme) | Yes (same scheme) |

### 3.2 Cropping

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Methods | Hybrid spatial-contiguous | 3 methods (weighted random) | 3 methods (configurable weights) | 3 methods (weighted random) |
| Spatial | KD-tree by 3D distance | KD-tree by 3D distance | Distance-based spatial | KD-tree by 3D distance |
| Contiguous | Chain-walk with neighborhood window | AF-Multimer Algorithm 1 | AF-Multimer Algorithm 1 (with fixes) | Random chain, random start, walk forward |
| Interface-seeded | Via `interface_id` parameter | `SpatialInterfaceCropping` (40% weight) | Interface chain selection | Seed from tokens within 15 Å of another chain |
| Train weights | Unified (neighborhood_size controls blend) | contiguous=0.2, spatial=0.4, interface=0.4 | configurable `crop_weights` | contiguous=0.2, spatial=0.4, interface=0.4 |
| Max tokens | configurable | 384 (train) | configurable `token_budget` | 384 (configurable) |
| Chainwise pre-crop | No | No | Yes (selects N closest chains first) | No |
| Ligand handling | Not detailed | Complete-ligand enforcement | Proximity-based ligand inclusion | No special handling |

### 3.3 Featurization

Both Boltz and Protenix produce equivalent feature sets (res_type, MSA, templates, ref atoms, GT positions). DeepFold3 produces all features including:

- **`resolution`**: loaded from manifest metadata; gates confidence losses to [0.1, 4.0] Å
- **Paired MSA**: loaded via per-chain `paired_msa_id`; merged as paired rows + dense unpaired
- **Bond features**: `bond_atom_indices`, `bond_gt_distances`, `bond_loss_mask` (backbone + ligand bonds, observed GT distances)
- **Templates**: not loaded during training (lower priority; inference templates work)

## 4. Model Training

### 4.1 Recycling

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Max recycles (train) | 3 | 4 (N_cycle=4) | configurable | configurable (default 4) |
| Random recycles | `randint(0, max)` per step | fixed N_cycle | configurable | `randint(1, max)` per step |
| Gradient | Last recycle only | Last recycle only | Last recycle only | Last recycle only |
| Implementation | Correct (loop with `set_grad_enabled`) | Correct (loop with `no_grad` wrapper) | Lightning `training_step` | Correct (loop with `set_grad_enabled`; reads `_prev_pair`/`_prev_single` from features) |

### 4.2 Diffusion Training

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Noise distribution | `σ_data × exp(N(-1.2, 1.5²))` | `σ_data × exp(N(-1.2, 1.5²))` | Same | Same |
| σ_data | 16.0 | 16.0 | 16.0 | 16.0 |
| Diffusion batch size | 16 | 48 | configurable | 48 |
| Augmentation | `center_random_augmentation` per step | `centre_random_augmentation` per step | Same | `_random_augmentation` per step |
| Alignment | Kabsch (SVD, fp32) | Weighted rigid alignment (fp32) | Weighted rigid alignment (Algorithm 28) | Kabsch alignment |
| Preconditioning | EDM (c_skip, c_out, c_in) | EDM (c_skip, c_out, c_in) | EDM | EDM |
| Chunking | Not mentioned | `diffusion_chunk_size=4` | chunk_size=4 (configurable) | Configurable (`diffusion_chunk_size`, default 0 = no chunking) |

### 4.3 Mini-Rollout (for Confidence)

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Steps | 20 | 20 | 20 | 20 |
| Samples | 1 | 1 | 1 | 1 |
| Gradient | No | No | No | No |
| Purpose | Predicted positions for confidence head | Same | Same | Same |

### 4.4 Loss Functions

| Loss | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|------|-------|----------|------------|-----------|
| Diffusion MSE | ✓ (EDM-weighted) | ✓ (EDM-weighted) | ✓ (EDM-weighted, Eq. 3) | ✓ (EDM-weighted) |
| Smooth LDDT | ✓ (optional, default on) | ✓ (on in stage 1) | ✓ | ✓ |
| Bond violation | ✗ | ✓ (stage 2, α=1.0) | ✓ (Eq. 5, for bonded ligands) | ✓ (stage 2, α configurable; uses observed GT distances) |
| Angle loss | ✗ | ✗ | ✓ | ✗ |
| Distogram | ✓ (CE, α=0.03) | ✓ (CE, α=0.03) | ✓ (all-atom distogram) | ✓ (CE, α=0.03) |
| pLDDT | ✓ (CE, 50 bins) | ✓ (CE, 50 bins) | ✓ | ✓ (CE, 50 bins) |
| PAE | ✓ (CE, gated by α_pae) | ✓ (off stage 1, on stage 3) | ✓ | ✓ (α_pae=0 default) |
| PDE | ✓ | ✓ | ✓ | ✓ |
| Exp resolved | ✓ (BCE) | ✓ (BCE) | ✓ | ✓ (BCE) |
| Resolution gate | Not mentioned | ✓ (0.1–4.0 Å) | ✓ (configurable per-dataset) | ✓ (0.1–4.0 Å, configurable) |

**Molecule-type loss reweighting:**

| Mol type | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|----------|-------|----------|------------|-----------|
| Protein | 1× | 1× | 1× | 1× |
| DNA | Not detailed | 5× | configurable | 5× (configurable) |
| RNA | Not detailed | 5× | configurable | 5× (configurable) |
| Ligand | 10× | 10× | configurable | 10× (configurable) |
| Nucleotide (Boltz) | 5× | — | — | — |

### 4.5 Loss Weights (Combined)

```
total_loss = α_diffusion × (MSE + smooth_LDDT [+ bond])
           + α_distogram × distogram_CE
           + α_confidence × (pLDDT + PDE + exp_resolved + α_pae × PAE)
```

| Weight | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| α_diffusion | 4.0 | 4.0 | configurable | 4.0 |
| α_distogram | 0.03 | 0.03 | configurable | 0.03 |
| α_confidence | 3e-3 (full) / 1e-4 (struct) | 1e-4 | configurable per-dataset | 1e-4 |
| α_pae | 0 (stage 1) → 1 | 0 (stage 1) → 1 (stage 3) | configurable | 0 (default) |
| α_bond | N/A | 0 (stage 1) → 1 (stage 2) | included in diffusion loss | 0 (stage 1) → configurable (stage 2) |

## 5. Optimizer & Schedule

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Optimizer | Adam | Adam (AdamW optional) | Adam | Adam (AdamW-style weight decay on ≥2D) |
| Learning rate | 0.0018 | 0.0018 | configurable `max_lr` | 0.0018 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| ε | 1e-8 | 1e-8 | 1e-8 | 1e-8 |
| Weight decay | Not mentioned | 1e-8 (optional) | Not detailed | 0.0 (default) |
| Warmup steps | 1000 | 2000 (demo) / 10 (config) | configurable `warmup_no_steps` | 1000 |
| Decay start | 50,000 | After warmup | configurable `start_decay_after_n_steps` | 50,000 |
| Decay schedule | `0.95^(step/50000)` | `0.95^(step/50000)` | `decay_factor^(steps_since/decay_every+1)` | `0.95^(step/50000)` |
| Finetune schedule | Not detailed | Cosine annealing + warmup | Same scheduler | AF3 or Cosine annealing (selectable) |

## 6. EMA

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Decay | configurable | 0.999 | configurable | 0.999 |
| Warm-start | `min(decay, (1+n)/(10+n))` | Not mentioned | Not detailed | `min(decay, (1+n)/(10+n))` |
| Apply for eval | Swap weights | Swap weights | Clone → swap → restore | Swap weights |
| Checkpoint | Saved in main checkpoint | Separate `{step}_ema_{decay}.pt` | Saved in `checkpoint['ema']` | Saved in main checkpoint |
| Selective update | No | Yes (`mutable_param_keywords`) | Yes (`submodules_to_update`) | No |
| Update timing | Every step | Every step | After optimizer step (accumulation-aware) | Every step |

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

### OpenFold 3

Config-driven via YAML dataset specifications:

- Loss weights are **per-dataset** (e.g., `confidence_loss_weight`, `diffusion_loss_weight`, `distogram_loss_weight`), enabling multi-dataset training with different loss configurations in a single run.
- `train_confidence_only` flag controls trunk freezing.
- `low_mem_validation` flag for memory-efficient evaluation.

### DeepFold3

Config-override-driven stages (no hardcoded presets — use `--config` JSON or `--override key=value`):

1. **Stage 1 — Initial**: crop 384, diffusion batch 48. `alpha_diffusion=4.0`, `alpha_pae=0.0`, `alpha_bond=0.0`. AF3 LR schedule.
2. **Fine-tune 1 — Structure (640)**: crop 640, diffusion batch 32. `alpha_bond=1.0`, `alpha_smooth_lddt=null`. Load Stage 1 with `--params`.
3. **Fine-tune 2 — Structure (768)**: crop 768, diffusion batch 32. `alpha_bond=1.0`, smooth LDDT back on. TF distillation sets on. Load Fine-tune 1 with `--params`.
4. **Fine-tune 3 — Confidence**: `train_confidence_only=True` (freezes trunk). `alpha_pae=1.0`. `alpha_diffusion=0`, `alpha_distogram=0`. Max chains 50. Load Fine-tune 2 with `--params`.

## 8. Checkpointing

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Format | Lightning `.ckpt` | `{step}.pt` | Lightning `.ckpt` | `checkpoint_{step}.pt` |
| Contents | model + optimizer + scheduler + EMA | model + optimizer + scheduler + step | model + optimizer + scheduler + EMA | model + optimizer + scheduler + EMA + step |
| Rotation | `save_top_k` (configurable) | All kept | Lightning callbacks | `max_checkpoints` (default 5, excludes final) |
| Resume | `ckpt_path` to `trainer.fit()` | `load_checkpoint_path` + fine-grained flags | Lightning `ckpt_path` | `--resume` flag |
| Partial load | via `strict_loading` flag | `load_strict`, `skip_load_optimizer/scheduler` | Lightning `strict_loading` | `load_strict`, `skip_load_optimizer/scheduler/step` |
| Confidence from trunk | Special trunk→confidence weight copy | `load_strict=False` + new params | Not detailed | `load_strict=False` + `skip_load_*` flags |

## 9. Evaluation

| Aspect | Boltz | Protenix | OpenFold 3 | DeepFold3 |
|--------|-------|----------|------------|-----------|
| Metrics | Per-type lDDT, RMSD, confidence MAE | lDDT, TM-score, DockQ (via external) | All losses + custom scoring | All training losses (with EMA weights) |
| EMA for eval | Yes (weight swap) | Yes (weight swap) | Yes (clone → swap → restore) | Yes (weight swap) |
| Eval frequency | Every epoch | Every `eval_interval` steps | Lightning callbacks | Every `eval_interval` steps |
| Eval data | Separate validation set | Separate validation set | Separate `ValidationDataset` | 5% random split (seeded, reproducible) |
| Output writer | Not detailed | Not detailed | `OF3OutputWriter` (mmCIF + confidences JSON/NPZ) | Not implemented |

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

### OpenFold 3 Style

- **Lightning-based**: `ModelRunner(LightningModule)` with `training_step()`, `validation_step()`, `configure_optimizers()`
- **YAML config**: Dataset specifications with per-dataset loss weights, MSA/template config, crop config
- **Deep hierarchy**: `SingleDataset` → `BaseOF3Dataset` → `WeightedPDBDataset` etc.
- **Multi-dataset**: `SamplerDataset` draws from multiple datasets with configurable probabilities
- **On-the-fly featurization**: No pre-computed features — structures parsed and featurized during training
- **Preprocessing pipeline**: 7-step pipeline from raw data to batched features
- **Biotite-native**: Uses biotite `AtomArray` throughout the data pipeline

### DeepFold3 Style (current)

- **Native DDP**: Manual `Trainer` class, no framework dependency
- **Dataclass config**: `TrainingConfig`, `ModelConfig` — type-safe, validated in `__post_init__`
- **Clean separation**: `trainer.py`, `loss.py`, `ema.py`, `scheduler.py` are independent modules
- **Config-driven stages**: No hardcoded stage presets — stages via JSON config files + CLI `--override`
- **Feature-complete**: Cluster-weighted sampling, 3-strategy cropping, paired MSA, bond loss, mol-type reweighting, NaN detection, WandB, checkpoint rotation, partial checkpoint loading, confidence-only freezing, dual LR schedulers
- **Data interop**: `import-of3` command imports OpenFold 3 preprocessed structures

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
| No multi-dataset training | `extra_datasets` config + `MultiDatasetSampler` — N datasets with independent data_dir, weight, strategy, is_distillation |

### Remaining

| Gap | Impact | Reference |
|-----|--------|-----------|
| **No template loading (training)** | No structural homolog signal during training | Protenix loads + dropout 0.0–0.25; OF3 loads CIF/BCIF + pre-parsed NPZ with per-dataset date cutoffs |
| **No nucleic backbone bonds** | O3'→P inter-residue bonds missing from bond loss (protein N-CA-C + peptide C-N are covered) | Protenix covers all backbone |
| **No per-dataset loss weights** | All datasets use global loss weights | OF3 supports per-dataset `confidence_loss_weight`, `diffusion_loss_weight`, etc. |
| **No bucket padding (training)** | `torch.compile` recompiles per sequence length | Low priority |
