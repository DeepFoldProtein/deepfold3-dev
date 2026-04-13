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

# Multi-dataset training (primary PDB + distillation AFDB)
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.5:uniform:true

# Multi-dataset via JSON config
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --config configs/multi_dataset.json
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
| PAE | `alpha_confidence × alpha_pae` (0) | Frame-aligned predicted aligned error (AF3 §3.6.2 / §4.3.2 Algorithm 29 — disabled in stage 1) |

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

During training, the number of recycles is randomly sampled per microstep: `n ∈ [0, max_recycles]` (AF3 convention — total trunk passes = n + 1). With `uniform_recycle=true` (default), all DDP ranks use the same n per microstep (seeded on `seed + global_step`) to avoid straggler divergence. Only the final trunk pass propagates gradients; earlier passes run under `torch.no_grad()`.

### DDP Configuration

- Backend: NCCL
- `static_graph=True` when unchunked; auto-disabled when `diffusion_chunk_size > 0` (chunked path uses `model.no_sync()` which conflicts with static_graph)
- `find_unused_parameters=False` (default)
- `use_reentrant=False` for all gradient checkpoints (required for DDP compatibility)
- `destroy_process_group()` called at end of training loop with a `dist.barrier()` first
- `faulthandler` registered on SIGUSR1 for DDP deadlock diagnosis
- Bucket-padded collation (`bucket_pad=True`) for deterministic kernel shapes across microsteps
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

### Multi-Dataset Training

Train with multiple data sources (e.g. PDB experimental + AFDB distillation) by specifying extra datasets. Each dataset has its own `data_dir`, sampling weight, strategy, and distillation flag.

**Via CLI** (`--dataset name:path[:weight[:strategy[:distill]]]`):

```bash
# PDB (primary, cluster-weighted) + AFDB (distillation, uniform, 25% weight)
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.25:uniform:true

# Three datasets
deepfold3 train --data-dir /data/pdb --output runs/exp1 \
    --dataset afdb:/data/afdb:0.25:uniform:true \
    --dataset esmfold:/data/esmfold:0.1:uniform:true
```

**Via JSON config** (`extra_datasets` field):

```json
{
  "training": {
    "extra_datasets": [
      {
        "name": "afdb",
        "data_dir": "/data/afdb",
        "weight": 0.5,
        "strategy": "uniform",
        "is_distillation": true
      },
      {
        "name": "pdb_highres",
        "data_dir": "/data/pdb_highres",
        "weight": 2.0,
        "strategy": "weighted",
        "is_distillation": false
      }
    ]
  }
}
```

**Fields per dataset:**

| Field | Default | Description |
|-------|---------|-------------|
| `name` | (required) | Identifier for logging |
| `data_dir` | (required) | Path to `structures/` + `msa/` |
| `weight` | 1.0 | Relative sampling probability (primary dataset always 1.0) |
| `strategy` | `"uniform"` | `"weighted"` (cluster-balanced) or `"uniform"` |
| `is_distillation` | `false` | If true, disables confidence/distogram losses for this dataset |

The primary dataset (`--data-dir`) always uses cluster-weighted sampling when cluster IDs are available. Extra datasets are combined via `ConcatDataset` + `MultiDatasetSampler`, which draws samples from each dataset proportionally to weights.

**Legacy compatibility**: `distillation_data_dir` and `distillation_weight` in `TrainingConfig` still work — they're internally converted to an `extra_datasets` entry.

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

### MSA Generation

Generate MSAs for all unique protein sequences in the training data:

```bash
# Search all unique sequences via local MMseqs2 (GPU-accelerated)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu

# Resume-safe: re-running skips sequences that already have NPZ files.
# Sequences with A3M but no NPZ are converted without re-searching.

# Convert-only mode (A3M→NPZ, no search)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --convert-only
```

The command extracts unique protein sequences from the manifest, deduplicates by `msa_id` (SHA256 of sequence), searches against UniRef30 + ColabFoldDB, and writes per-sequence A3M + NPZ files to `{data_dir}/msa/`. Requires local MMseqs2 databases (see `scripts/setup_databases.sh`).

#### Paired MSAs for Multimers

For multimer complexes (>1 protein chain), paired MSAs provide cross-chain co-evolutionary signal. Add `--use-pairing` to the batch command:

```bash
# Generate unpaired + paired MSAs
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ \
    --use-pairing --gpu
```

This runs a two-phase pipeline: (1) unpaired search for all unique sequences, then (2) paired search per multimer record using MMseqs2's `pairaln` taxonomy-based pairing. Paired MSAs are written as `{record_id}_paired_{chain_idx}.npz` and referenced via `ChainInfo.paired_msa_id` in the manifest.

The featurizer merges paired and unpaired MSAs into a single full-width `[S, N]` tensor: paired rows first (cross-chain aligned), then dense unpaired rows (all chains' row *i* on the same full-width row). See `docs/data-pipeline.md` § Paired MSA for the full row layout and comparison with OpenFold 3 and Protenix.

### Distillation Datasets (AFDB, ESMFold, etc.)

Distillation databases like AFDB provide predicted structures (CIF) alongside pre-computed MSAs (A3M). The pipeline handles these without re-searching:

```bash
# Option 1: Import during prepare (CIF + A3M together)
deepfold3 prepare --mmcif-dir /data/afdb/cifs --output /data/distill \
    --msa-dir /data/afdb/a3ms

# Option 2: Import A3Ms into an existing dataset
deepfold3 import-msa --data-dir /data/distill --msa-dir /data/afdb/a3ms

# Then search only the missing sequences (if any)
deepfold3 msa-batch --data-dir /data/distill --db-dir /data/mmseqs_dbs/ --gpu
```

Both `--msa-dir` and `import-msa` match A3M files to chains by reading the query sequence from each A3M file, computing its SHA256 hash, and copying to `{data_dir}/msa/{msa_id}.a3m`. This works regardless of file naming convention (AFDB `AF-{uniprot}-F1-model_v4.a3m`, PDB-style, or custom).

### Importing from OpenFold 3

OpenFold 3 (Apache 2.0, AlQuraishi Laboratory) stores preprocessed structures as biotite AtomArray NPZ files. These can be converted to DeepFold3 format:

```bash
# Convert all OF3 preprocessed structures
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training

# With OF3 dataset cache (provides resolution, cluster IDs, metadata)
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training \
    --cache /data/of3/dataset_cache.json

# Then generate MSAs for the imported data
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
```

The converter handles mol-type mapping (OF3 uses `PROTEIN=0, RNA=1, DNA=2, LIGAND=3` while DeepFold3 uses `PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3`), bond remapping, and entity/chain grouping.

Supported OF3 directory layouts:
- Nested: `{pdb_id}/{pdb_id}.npz`
- RODA monomer: `{id}/structure.npz`
- Flat: `{pdb_id}.npz`

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

## AF3 Data Pipeline Conformance

Systematic comparison of DeepFold3's data pipeline against the AF3 supplementary material (§2 + §5). See `docs/af3-paper-reference.md` for the full organized reference.

### ✅ Matching

| Paper section | Item | Implementation |
|---|---|---|
| §2.5.1 | Weighted PDB sampling (β/α weights, chain+interface) | `data/sample/weighted.py` |
| §2.5.3 | Protein 40% clustering | wwPDB clusters via `data/prepare/clustering.py` |
| §2.5.3 | 100% clustering for nucleic acids + peptides | Hash-based cluster_id = 100% identity; `scripts/prepare.py:_apply_clusters` |
| §2.5.3 | CCD identity clustering for ligands | `ChainInfo.res_name` + `_apply_clusters` maps `ccd_{res_name}` |
| §2.5.4 | Resolution ≤ 9Å, max 300 chains, min 4 residues | `data/prepare/filters.py` |
| §2.5.4 | Clashing chains (30%/1.7Å, iterative 3-level tie-break) | `data/prepare/filters.py:filter_clashing_chains` |
| §2.5.4 | Consecutive Cα > 10Å | `data/prepare/filters.py` |
| §2.6 | Standard AA/nucleotide = 1 token, ligand = per-atom | `data/tokenize/tokenizer.py` |
| §2.7 | Cropping weights (0.2/0.4/0.4), interface 15Å | `data/crop/cropper.py` |
| §2.8 | ref_pos (random rot+trans), ref_element (one-hot 128) | `data/feature/featurizer.py` |
| §2.8 | token_bonds filtered to < 2.4Å during training | `data/feature/featurizer.py:_build_bond_features` |
| §3.7.1 | Diffusion batch 48, MSE mol-type weight (5×/10×) | `model/config.py`, `training/loss.py` |
| §3.7.1 | EDM weighting, smooth LDDT (15/30Å radii) | `training/loss.py` |
| §3.7.1 | Bond loss (Eq. 5 — GT distances) | `training/loss.py:bond_loss` |
| §3.7.1 | Cross-entity bonds (disulfide, metal, covalent) | `data/prepare/structure.py` extracts from `_struct_conn`; `tokenizer.py` propagates with atom indices |
| §5.3 | Loss weights (α_diff=4, α_disto=0.03, α_conf=1e-4) | `model/config.py` |
| §5.4 | Adam β₁=0.9 β₂=0.95 ε=1e-8, LR 1.8e-3, warmup 1000 | `model/config.py` |
| §5.4 | LR decay 0.95/50k, grad clip 10, batch 256 | `model/config.py`, `training/trainer.py` |

### ⚠️ Divergences

| Paper section | Item | Our approach | Impact |
|---|---|---|---|
| §3.7.1 | Smooth LDDT weight | Now configurable via `alpha_smooth_lddt` (None=off) | AF3 turns it off in fine-tune 1 (Table 6) |

### ❌ Missing (known gaps)

| Paper section | Item | Priority | Notes |
|---|---|---|---|
| §2.5.3 | SCOP modified residue normalization | Low | Modified residues not normalized before clustering |
| §2.5.4 | >20 chains → select 20 closest to interface | Low | Rare for training data (most <20 chains) |
| §2.5.4 | CCD atom name validation | Low | Atoms outside CCD atom set not filtered |
| §2.5.4 | Leaving atom removal | Low | Mostly H atoms (already skipped) + water (separate entity) |
| §2.6 | Modified residue per-atom tokenization | Medium | All residues use 1-token (structural simplification) |

### Stage Configuration Quick Reference

From AF3 Table 6. Use `--config` with stage JSON files or `--override`:

```bash
# Stage 1: Initial training
deepfold3 train --data-dir /data --output runs/s1

# Fine-tune 1: crop 640, bond loss on, smooth LDDT off
deepfold3 train --data-dir /data --output runs/ft1 --params runs/s1/checkpoint_final.pt \
    --override max_tokens=640 --override alpha_bond=1.0 --override alpha_smooth_lddt=null

# Fine-tune 2: crop 768 (+ TF distillation sets)
deepfold3 train --data-dir /data --output runs/ft2 --params runs/ft1/checkpoint_final.pt \
    --override max_tokens=768 --override alpha_bond=1.0

# Fine-tune 3: PAE only
deepfold3 train --data-dir /data --output runs/ft3 --params runs/ft2/checkpoint_final.pt \
    --override train_confidence_only=true --override alpha_pae=1.0
```

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
