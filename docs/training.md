# Training

DeepFold3 uses native PyTorch DDP for distributed training. The training procedure follows AF3 §3.7 with Protenix-style implementation choices.

## Quick Start

```bash
# Single GPU
deepfold3 train --data-dir /data/training --output runs/exp1

# Multi-GPU (DDP via torchrun)
torchrun --nproc_per_node=4 -m deepfold3.scripts.train \
    --data-dir /data/training --output runs/exp1

# Resume from checkpoint
deepfold3 train --data-dir /data/training --output runs/exp1 \
    --resume runs/exp1/checkpoint_1000.pt

# With config overrides
deepfold3 train --data-dir /data/training --output runs/exp1 \
    --config config.json --lr 0.001 --max-tokens 512
```

## Training Config

All training hyperparameters are in `TrainingConfig` (`src/deepfold3/model/config.py`). Nothing is hardcoded — override via JSON config or CLI flags.

```json
{
  "training": {
    "lr": 0.0018,
    "weight_decay": 0.0,
    "betas": [0.9, 0.95],
    "warmup_steps": 1000,
    "start_decay_after": 50000,
    "decay_every": 50000,
    "decay_factor": 0.95,
    "grad_clip_norm": 10.0,
    "accumulate_grad_batches": 1,
    "ema_decay": 0.999,
    "max_tokens": 384,
    "max_atoms": 4096,
    "max_recycles": 4,
    "diffusion_batch_size": 48,
    "diffusion_chunk_size": 0,
    "mini_rollout_steps": 20,
    "alpha_diffusion": 4.0,
    "alpha_distogram": 0.03,
    "alpha_confidence": 1e-4,
    "alpha_pae": 0.0,
    "alpha_bond": 0.0,
    "nucleotide_loss_weight": 5.0,
    "ligand_loss_weight": 10.0,
    "min_resolution": 0.1,
    "max_resolution": 4.0,
    "train_confidence_only": false,
    "finetune_scheduler": "af3",
    "max_checkpoints": 5,
    "wandb_project": "",
    "dtype": "bf16",
    "max_steps": 200000,
    "seed": 42
  },
  "model": {
    "num_recycles": 10,
    "msa_crop_size": 1024
  }
}
```

## Training Procedure

### Forward Pass (per batch)

```
1. Evoformer trunk (random 1–4 recycles, gradient on last only)
     → single [N, 384] + pair [N, N, 128]

2. Diffusion training step (gradient flows)
     - Augment GT positions: random rotation + translation per sample
     - Sample noise: log-normal σ ~ σ_data × exp(p_mean + p_std × z)
     - Add noise: x_noisy = x_gt_aug + σ × ε
     - Denoise once → x_denoised
     - Losses: weighted MSE + smooth LDDT

3. Mini-rollout (no gradient, eval mode)
     - 20-step denoising → predicted positions
     - Used as input to confidence head

4. Confidence head (gradient flows through confidence PairFormer)
     - Runs on mini-rollout positions
     - pLDDT, PAE, PDE, exp_resolved losses against GT

5. Distogram head
     - Cross-entropy on distance bin predictions
```

### Loss Components

| Loss | Weight | Description |
|------|--------|-------------|
| Diffusion MSE | `alpha_diffusion` (4.0) | EDM-weighted MSE: w(σ) = (σ²+σ_d²)/(σ·σ_d)²; mol-type reweight: DNA/RNA 5×, ligand 10× |
| Smooth LDDT | `alpha_diffusion` (4.0) | Distance-based accuracy (cutoffs: 0.5, 1.0, 2.0, 4.0 Å) |
| Bond violation | `alpha_diffusion × alpha_bond` (0) | MSE on bonded atom distances vs observed GT; enable for stage 2 |
| Distogram | `alpha_distogram` (0.03) | Cross-entropy on 64-bin distance predictions |
| pLDDT | `alpha_confidence` (1e-4) | Per-atom confidence cross-entropy (50 bins); gated on resolution ∈ [0.1, 4.0] Å |
| PDE | `alpha_confidence` (1e-4) | Predicted distance error cross-entropy |
| exp_resolved | `alpha_confidence` (1e-4) | Binary resolved/unresolved classification |
| PAE | `alpha_confidence × alpha_pae` (0) | Predicted aligned error (disabled in stage 1) |

### LR Schedule

Two schedulers available, selected by `finetune_scheduler`:

**AF3 schedule** (default, `finetune_scheduler="af3"`):

```
1. Warmup (0 → warmup_steps): linear 0 → lr
2. Plateau (warmup_steps → start_decay_after): constant lr
3. Step decay: lr × decay_factor^n every decay_every steps
```

Default: lr=0.0018, warmup=1000, decay starts at 50k, factor=0.95 every 50k.

**Cosine annealing** (`finetune_scheduler="cosine"`): warmup then cosine decay to `lr × cosine_min_lr_ratio` over `cosine_max_steps`. Useful for fine-tuning stages.

### EMA

Exponential Moving Average with warm-start ramp:

```
effective_decay = min(decay, (1 + n_updates) / (10 + n_updates))
shadow = (1 - effective_decay) × param + effective_decay × shadow
```

EMA weights are swapped in for evaluation and saved in checkpoints. Fused kernel parameters are excluded from EMA shadow copies (they are derived from base weights). After `apply()`/`restore()`, fused kernels are re-synced automatically.

### Recycling

During training, the number of recycles is randomly sampled per batch: `n ∈ [1, max_recycles]`. Only the final recycle propagates gradients; earlier recycles are `.detach()`ed.

### DDP Configuration

- Backend: NCCL
- `static_graph=True` (required for gradient checkpointing compatibility)
- `find_unused_parameters=False` (default; set True if freezing submodules)
- Launch with `torchrun` for multi-GPU

### Gradient Management

- Gradient clipping: L2 norm (default 10.0)
- Gradient accumulation: configurable via `accumulate_grad_batches`
- Loss scaled by `1/accumulate_grad_batches` before backward
- Optimizer steps only on accumulation boundaries
- **NaN detection**: distributed all-reduce ensures all ranks agree on NaN/Inf; affected steps are skipped with a warning

### Experiment Tracking

Optional WandB integration: `--wandb <project>`. Logs train/eval losses, learning rate, epoch, and gradient norms. Only rank 0 logs.

## Data Requirements

Training data directory structure:

```
data_dir/
├── manifest.json             # Record index (optional — auto-scanned if missing)
├── structures/
│   ├── {id}.npz              # Structure files (atoms, residues, chains, bonds)
│   └── ...
└── msa/
    ├── {msa_id}.npz          # MSA files (sequences, mask, deletion_matrix)
    └── ...
```

Records are loaded via `discover_records()` from `data/prepare/manifest.py`: reads `manifest.json` if present, otherwise scans `structures/` for NPZ files.

**Train/eval split**: records are shuffled with a fixed seed (42) then split 95/5 (at least 1 eval record). The seeded shuffle ensures reproducible splits regardless of manifest ordering.

**Sampling**: If records have integer cluster IDs (from wwPDB `clusters-by-entity-40.txt`), `WeightedClusterSampler` applies AF3-style cluster-balanced weights: `w = β[type] × Σ(α[mol] × count) / cluster_size` with α(protein)=3, α(nucleic)=3, α(ligand)=1, β(chain)=0.5, β(interface)=1.0. Synthetic epoch size of 10,000 samples. Falls back to uniform `DistributedSampler` if no cluster data.

**Cropping**: Three strategies with weighted random selection — contiguous (20%), spatial KD-tree (40%), interface-seeded (40%). Interface cropping seeds from tokens within 15 Å of another chain, then expands spatially.

**Preprocessing**: use `deepfold3 prepare` to generate the data directory from a PDB mmCIF mirror. See `docs/data-pipeline.md` § Data Preprocessing for details.

### Sequence Clustering

Cluster-balanced sampling requires integer cluster IDs in the manifest. Two options:

#### **Option A: Download pre-computed wwPDB clusters**

```bash
wget https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
```

Apply during prepare or to an existing manifest:

```bash
# During prepare
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --cluster-file clusters-by-entity-40.txt

# Or apply to existing manifest (writes manifest_clustered.json by default)
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json
```

#### **Option B: Compute locally with MMseqs2**

Use `deepfold3 cluster` to extract sequences from processed structures and cluster with MMseqs2. Requires the `mmseqs` binary on PATH.

```bash
# Default: 40% identity, 80% coverage
deepfold3 cluster --data-dir /data/training --output clusters.txt

# Custom thresholds
deepfold3 cluster --data-dir /data/training --output clusters.txt \
    --min-seq-id 0.3 --coverage 0.8 --cov-mode 0

# Then apply to manifest
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters.txt --output /data/training/manifest.json
```

The `cluster` command extracts all unique polymer entity sequences from the `structures/` directory, runs `mmseqs cluster`, and writes a wwPDB-format cluster file (one cluster per line, space-separated `pdb_chain` identifiers).

Chains not matched to any cluster keep their SHA256 hash fallback IDs. The trainer detects integer cluster IDs and automatically activates `WeightedClusterSampler`; otherwise it falls back to uniform sampling.

The featurizer produces these training-specific features:

- `atom_gt_positions [N, M, 3]`: experimentally observed atom coordinates.
- `atom_gt_mask [N, M]`: which atoms have resolved positions.

These are separate from `ref_pos` (CCD ideal coordinates used for atom cross-attention conditioning).

## Checkpointing

Checkpoints are saved at configurable intervals (default: every 1000 steps) and contain:

- Model state dict (**excludes** fused kernel derived params — see below)
- Optimizer state dict
- LR scheduler state dict
- EMA state (if enabled, also excludes fused params)
- Step counters
- Config (model + training)

**Checkpoint rotation**: keeps the `max_checkpoints` (default 5) most recent checkpoints by modification time. `checkpoint_final.pt` is always preserved.

Resume training by passing `--resume path/to/checkpoint.pt`.

**Partial loading** for stage transitions: use `--params` with a previous stage's checkpoint plus config overrides. Control what gets loaded via `load_strict` (default true), `skip_load_optimizer`, `skip_load_scheduler`, `skip_load_step`.

**Fused kernel portability**: `DeepFold3.state_dict()` automatically filters out derived fused kernel parameters (identified by `OptimConfig.FUSED_STATE_DICT_MARKERS`). On load, `load_state_dict()` re-syncs fused params from base weights. This means checkpoints saved with fused kernels enabled can be loaded without them, and vice versa.

## Training Stages

Stages are controlled by config overrides — no hardcoded presets. Use `--config` with stage JSON files or `--override key=value`.

### Stage 1: Initial Training

```json
{"alpha_diffusion": 4.0, "alpha_pae": 0.0, "alpha_bond": 0.0}
```

### Stage 2: Fine-tune Structure

```json
{"alpha_diffusion": 4.0, "alpha_pae": 0.0, "alpha_bond": 1.0, "finetune_scheduler": "cosine"}
```

Bond loss replaces smooth LDDT as the geometry regularizer. Load stage 1 checkpoint with `--params`.

### Stage 3: Fine-tune Confidence

```json
{"train_confidence_only": true, "alpha_diffusion": 0.0, "alpha_distogram": 0.0, "alpha_pae": 1.0}
```

Freezes all parameters except confidence head. Only confidence losses (pLDDT, PDE, PAE, exp_resolved) are active.

## Source Layout

```
src/deepfold3/training/
├── __init__.py       # Package exports
├── loss.py           # Loss functions (smooth LDDT, distogram, confidence)
├── ema.py            # Exponential Moving Average
├── scheduler.py      # AF3 LR scheduler (warmup → plateau → decay)
└── trainer.py        # DDP trainer (training loop, checkpointing, eval)
```

## Module Dependencies

| Module | Depends On |
|--------|-----------|
| `loss.py` | `model.config.TrainingConfig` |
| `ema.py` | `torch.nn`, `model.config.OptimConfig` |
| `scheduler.py` | `torch.optim`, `model.config.TrainingConfig` |
| `trainer.py` | All above + `model.model.DeepFold3`, `data.module.*` |
