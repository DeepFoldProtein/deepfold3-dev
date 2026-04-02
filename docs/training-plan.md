# DeepFold3 Training Implementation Plan

> **Approach**: Follow Protenix's training procedure (stages, sampling, loss design) with Boltz-like coding style (clean abstractions, composition, minimal framework deps).

## Phase 0: Critical Bug Fixes

### 0.1 Fix recycling warm-start in `forward_training`

**Problem**: `forward_training()` writes `_prev_pair`/`_prev_single` to the features dict, but `run_evoformer()` always initializes from zeros — multi-recycle training is effectively 1 recycle.

**Fix**: Pass `prev_pair`/`prev_single` as explicit arguments to `run_evoformer()`, or have `run_evoformer()` read them from features when present.

**File**: [model.py](src/deepfold3/model/model.py)

```python
# In run_evoformer(), after initializing prev_pair/prev_single from zeros:
if "_prev_pair" in features:
    prev_pair = features["_prev_pair"]
    prev_single = features["_prev_single"]
```

**Validation**: Run `forward_training` with `num_recycles=4`, verify that recycle 2-4 receive non-zero `prev_pair`/`prev_single`.

---

## Phase 1: Data Pipeline Completeness

### 1.1 Sequence clustering

**What**: Download PDB pre-computed 40% identity clusters (Protenix approach — simpler than running MMseqs2 ourselves).

**Design** (Boltz style — clean data type + loader):

```
src/deepfold3/data/prepare/clustering.py
    load_pdb_clusters(path: str) -> dict[str, int]
    # Parses wwPDB clusters-by-entity-40.txt
    # Returns: sequence_hash -> cluster_id mapping
```

**Integration**: `deepfold3 prepare --cluster-file clusters-by-entity-40.txt` writes `cluster_id` into each `ChainInfo` in the manifest.

### 1.2 Cluster-weighted sampling

**What**: Replace uniform `DistributedSampler` with cluster-balanced weighted sampling (AF3 §2.5.6).

**Design** (Boltz-style `Sampler` abstraction):

```
src/deepfold3/data/sample/
    __init__.py
    weighted.py
        class WeightedClusterSampler(DistributedSampler):
            # Per-record weight = β[type] × Σ(α[mol_type] × count) / cluster_size
            # α: prot=3, nucl=3, lig=1
            # β: chain=0.5, interface=1.0
            # Synthetic epoch_size (default 10,000)
            # Uses torch.multinomial with replacement
```

**Integration**: `Trainer` uses `WeightedClusterSampler` when `manifest.json` has cluster IDs, falls back to `DistributedSampler` otherwise.

### 1.3 Interface-seeded cropping

**What**: Add a third crop strategy that seeds spatial crop from interface tokens (within 15 Å of another chain).

**Design**: Extend existing `cropper.py`:

```python
def crop(self, tokenized, max_tokens, max_atoms, method=None):
    if method is None:
        # Protenix weights: contiguous=0.2, spatial=0.4, interface=0.4
        method = random.choices(
            ["contiguous", "spatial", "interface"],
            weights=[0.2, 0.4, 0.4],
        )[0]
    if method == "interface":
        # Pick seed token from interface region
        # Fall back to spatial if no interfaces
```

### 1.4 Paired MSA loading

**What**: Load paired MSAs during training for cross-chain coevolution signal.

**Requires**: Pre-computed paired MSAs (taxonomy-based pairing). The `deepfold3 msa --use-pairing` command already supports this for inference. For training, the `prepare` pipeline needs to:

1. Run MSA search per unique chain sequence
2. Pair MSAs across chains sharing taxonomy
3. Store paired + unpaired MSAs as separate `.npz` files

**Design**: Add `paired_msa_id` to `ChainInfo`. `TrainingDataset.__getitem__` loads both `msa/{msa_id}.npz` (unpaired) and `msa/{paired_msa_id}.npz` (paired) when available.

### 1.5 Resolution in features

**What**: Include `resolution` scalar in the feature dict so confidence loss gating works.

**File**: [featurizer.py](src/deepfold3/data/feature/featurizer.py) — add `resolution` from `Structure.resolution` (already stored in NPZ metadata).

### 1.6 Template loading for training

**What**: Load per-chain template features during training with date-cutoff filtering.

**Design**: Extend `TrainingDataset.__getitem__` to optionally load pre-computed template hits (HHR files) per chain. Apply release-date cutoff (from Record metadata). Template dropout = 0.25 (randomly zero out templates during training).

**Priority**: Lower — can train without templates initially (Protenix v0.5.0 had `n_blocks=0`).

---

## Phase 2: Loss Improvements

### 2.1 Molecule-type MSE reweighting

**What**: Apply per-atom-type weights to diffusion MSE loss (Protenix: DNA/RNA 5×, ligand 10×).

**File**: [loss.py](src/deepfold3/training/loss.py) — in `diffusion_mse_loss()`:

```python
# Build mol_type_weight from features["mol_type"]
# mol_type: 0=protein, 1=RNA, 2=DNA, 3=ligand
type_weights = {0: 1.0, 1: 5.0, 2: 5.0, 3: 10.0}
atom_weight = type_weights[mol_type].unsqueeze(-1)  # [N, 1]
```

Add corresponding config fields: `nucleotide_loss_weight=5.0`, `ligand_loss_weight=10.0`.

### 2.2 Bond violation loss

**What**: MSE on bonded atom pair distances vs observed GT distances (Protenix approach — uses experimentally resolved distances, not CCD ideal).

**Design**:

```
src/deepfold3/training/loss.py
    def bond_loss(
        pred_pos,           # [S, N, M, 3]
        bond_atom_indices,  # [B, 4] — (token_1, atom_1, token_2, atom_2)
        ideal_distances,    # [B] — observed GT distances from structure
        bond_mask,          # [B]
        sigma_data,         # 16.0
        noise_levels,       # [S] optional EDM weighting
    ) -> Tensor
```

**Integration**: `training_loss()` includes `alpha_bond × bond_loss` when `alpha_bond > 0`.

**Data**: Bond indices + GT distances produced by `featurizer._build_bond_features()` — backbone (intra: N-CA, CA-C; inter: C[i]-N[i+1]) + ligand-ligand bonds.

### 2.3 Diffusion chunk size

**What**: Process diffusion batch in chunks to reduce peak memory (Protenix uses `chunk_size=4`).

**File**: [diffusion.py](src/deepfold3/model/diffusion.py) — in `sample_training()`, loop over chunks of the diffusion batch dimension instead of processing all `S=48` samples at once.

**Config**: `TrainingConfig.diffusion_chunk_size: int = 4`

---

## Phase 3: Training Stages

### 3.1 Multi-stage training support

**Design** (Boltz-style — config-driven, no code changes per stage):

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # Stage control
    stage: int = 1  # 1=initial, 2=finetune-structure, 3=finetune-confidence
    train_confidence_only: bool = False  # Stage 3: freeze trunk
    finetune_lr: float | None = None  # Separate LR for new/finetune params
    finetune_scheduler: str = "af3"  # "af3" or "cosine"
    cosine_min_lr_ratio: float = 0.1
    cosine_max_steps: int = 20000
```

**Stage presets** (applied via JSON config or CLI):

| Config field | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| `alpha_pae` | 0.0 | 0.0 | 1.0 |
| `alpha_bond` | 0.0 | 1.0 | 0.0 |
| `smooth_lddt_weight` | 1.0 | 0.0 | 0.0 |
| `alpha_diffusion` | 4.0 | 4.0 | 0.0 |
| `alpha_distogram` | 0.03 | 0.03 | 0.0 |
| `train_confidence_only` | False | False | True |

### 3.2 Confidence-only training (Stage 3)

**What**: Freeze everything except confidence head + `out_token_feat_update`. Only compute confidence losses.

**Implementation**: In `Trainer._train_step()`:

```python
if self.train_cfg.train_confidence_only:
    for name, param in model.named_parameters():
        if "confidence" not in name and "out_token_feat" not in name:
            param.requires_grad_(False)
```

### 3.3 Cosine annealing scheduler

**What**: Add cosine annealing with warmup for fine-tuning stages.

**File**: [scheduler.py](src/deepfold3/training/scheduler.py)

```python
class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
        ...
```

### 3.4 Partial checkpoint loading

**What**: Support loading a checkpoint with `strict=False` + options to skip optimizer/scheduler/step.

**File**: [trainer.py](src/deepfold3/training/trainer.py) — add `TrainingConfig` fields:

- `load_strict: bool = True`
- `skip_load_optimizer: bool = False`
- `skip_load_scheduler: bool = False`
- `skip_load_step: bool = False`

---

## Phase 4: Training Ergonomics

### 4.1 NaN detection and skip

**What**: Detect NaN/Inf loss, skip iteration, log warning. Distributed all-reduce ensures all ranks agree.

**File**: [trainer.py](src/deepfold3/training/trainer.py)

```python
if torch.isnan(loss) or torch.isinf(loss):
    if self.distributed:
        nan_flag = torch.ones(1, device=self.device)
        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
    logger.warning(f"Step {step}: NaN/Inf loss, skipping")
    optimizer.zero_grad()
    continue
```

### 4.2 Experiment tracking (WandB)

**What**: Optional WandB integration for loss curves, LR, grad norms.

**Design** (Boltz style — minimal, optional):

```python
# In Trainer.__init__:
if train_cfg.wandb_project:
    import wandb
    wandb.init(project=train_cfg.wandb_project, config=asdict(train_cfg))

# In _train_step:
if self.wandb:
    wandb.log({"loss": loss.item(), "lr": lr, "step": step})
```

### 4.3 Evaluation metrics

**What**: Compute meaningful eval metrics (lDDT, distogram accuracy) during evaluation.

**File**: [trainer.py](src/deepfold3/training/trainer.py) — rewrite `_evaluate()` to:

1. Run full `forward_training` (with EMA weights)
2. Compute `training_loss` (for loss comparison)
3. Compute lDDT from distogram predictions
4. Log all metrics

### 4.4 Checkpoint rotation

**What**: Keep only the N most recent checkpoints + the best by eval metric.

**Config**: `TrainingConfig.max_checkpoints: int = 5`

### 4.5 Bucket padding in training

**What**: Use bucket padding during training to reduce `torch.compile` recompilation.

Currently `bucket.py` exists but is only used for inference. Enable it in the training DataLoader collation path.

---

## Implementation Status

```
Priority  Phase  Task                         Status
────────  ─────  ──────────────────────────   ──────
   P0     0.1    Fix recycling bug            DONE ✓
   P1     2.1    Mol-type MSE reweighting     DONE ✓
   P1     1.5    Resolution in features       DONE ✓
   P1     1.3    Interface-seeded cropping     DONE ✓
   P1     4.1    NaN detection (distributed)  DONE ✓
   P2     1.1    Sequence clustering           DONE ✓
   P2     1.2    Cluster-weighted sampling     DONE ✓
   P2     2.2    Bond violation loss           DONE ✓
   P2     2.3    Diffusion chunk config        DONE ✓
   P2     3.1    Multi-stage config            DONE ✓ (config override approach)
   P2     3.2    Confidence-only training      DONE ✓
   P2     3.3    Cosine annealing scheduler    DONE ✓
   P2     3.4    Partial checkpoint loading    DONE ✓
   P3     4.2    WandB integration             DONE ✓
   P3     4.3    Evaluation metrics            DONE ✓
   P3     4.4    Checkpoint rotation           DONE ✓
   P3     1.4    Paired MSA loading            DONE ✓ (ChainInfo.paired_msa_id + TrainingDataset)
   P3     2.2b   Bond feature pipeline         DONE ✓ (backbone + ligand bonds, GT distances)
   ──     ──     ─────────────────────────     ────
   P3     1.6    Template loading (training)   TODO (needs HHR search pipeline)
   P3     4.5    Bucket padding in training    TODO
```

## Remaining Work

- **Template loading (training)**: Requires HMM-based template search with date cutoff. Inference template pipeline works; training needs: `hmmsearch` against PDB HMM DB, date-cutoff filtering, template dropout (0.25).
- **Nucleic backbone bonds**: O3'→P inter-residue bonds need atom name lookup in featurizer (protein backbone N-CA, CA-C, C[i]-N[i+1] are implemented).
- **Bucket padding in training**: Low priority, only matters for `torch.compile`.
