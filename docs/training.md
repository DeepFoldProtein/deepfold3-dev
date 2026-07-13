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
    "weight_decay": 0.05,
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
    "max_recycles": 3,
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
    "max_checkpoints": 0,
    "wandb_project": "",
    "dtype": "bf16",
    "max_steps": 200000,
    "seed": 42
  },
  "model": {
    "num_recycles": 10,
    "msa_crop_size": 512
  }
}
```

## Training Procedure

### Forward Pass (per batch)

```
1. Evoformer trunk (random 0–3 recycles = 1–4 trunk passes, gradient on last only)
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

### Chain permutation & the stability gate

The chain permutation (AF3 §4.3) is resolved **once** from the mini-rollout and shared with every
GT-based loss (diffusion MSE, smooth-lDDT, distogram, FAPE, pLDDT/PAE/PDE, tm_true) — `loss.py`
`chain_perm_from_mini_rollout`. For a symmetric multimer whose rollout is still collapsed, the
greedy centroid assignment is a **coin-flip**; across steps the relabelling oscillates and the
symmetry-breaking gradient self-cancels, latching the assembly near the collapsed
(symmetric-average) point. See `export/chain-perm-latch/` for the mechanism, the offline probe, and
the simulation.

The **stability gate** (`chain_perm_stability_gate`, default off; `chain_perm_gate_k`, default 5)
mitigates this: the mini-rollout is run with `K = chain_perm_gate_k` independent samples and the
permutation is applied **only when all K agree** on the whole-assembly assignment, else it falls
back to identity (un-permuted GT). While copies overlap the K noisy rollouts disagree → a
deterministic separating target; once they separate the K agree → the AF3 permutation fires. The
simulation shows K=3 recovers the latch zone and K=5 is safe to extreme rollout noise; **unanimity,
not majority, is required**. The K samples live under `pred["chain_perm_gate_pos"]`; the confidence
/ tm_true / atom-symmetry consumers still see only `mini_rollout_samples` of them via
`pred["mini_rollout_pos"]`.

### Loss Components

| Loss            | Weight                             | Description                                                                                                                                                                    |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Diffusion MSE   | `alpha_diffusion` (4.0)            | EDM-weighted MSE per AF3 §5.3 eq. 3: `(1/3) · Σ‖Δ‖²·w·mask / Σ mask` (per-coordinate average; denominator is atom count, not sum-of-weights). w(σ) = (σ²+σ_d²)/(σ·σ_d)²; mol-type reweight: DNA/RNA 5×, ligand 10×. |
| Smooth LDDT     | `alpha_diffusion` (4.0)            | Distance-based accuracy (cutoffs: 0.5, 1.0, 2.0, 4.0 Å)                                                                                                                        |
| Bond violation  | `alpha_diffusion × alpha_bond` (0) | MSE on bonded atom distances vs observed GT; enable for stage 2                                                                                                                |
| Distogram       | `alpha_distogram` (0.03)           | Cross-entropy on 64-bin distance predictions                                                                                                                                   |
| Masked MSA      | `alpha_masked_msa` (0)             | BERT-style MSA reconstruction CE over corrupted entries. Requires `ModelConfig.masked_msa.enabled`. Single-query MSA cross-attention head at the end of the MSA module; trains the MSA stack + single track. All samples incl. distillation. |
| pLDDT           | `alpha_confidence` (1e-4)          | Per-atom confidence cross-entropy (50 bins); gated on resolution ∈ [0.1, 4.0] Å. AF3 §4.3.1: j-axis restricted to representative atoms (CA / C1') of standard polymer tokens, with inclusion radius 15 Å for protein-rep j and 30 Å for nucleic-rep j; ligand j atoms drop. |
| PDE             | `alpha_confidence` (1e-4)          | Predicted distance error cross-entropy                                                                                                                                         |
| exp_resolved    | `alpha_confidence` (1e-4)          | Binary resolved/unresolved classification                                                                                                                                      |
| PAE             | `alpha_confidence × alpha_pae` (0) | Frame-aligned predicted aligned error (AF3 §4.3.2, Algorithm 29 — disabled in stage 1). Frames whose ∠(a, b, c) lies within 25° of co-linearity are dropped per-sample on both pred and GT. |
| Single-repr reg | `alpha_single_reg` (0)             | Magnitude regularizer on trunk single output: mean per-token Σ_c &#124;s&#124;^p / C. Observation-only at α=0.                                                                 |
| Pair-repr reg   | `alpha_pair_reg` (1e-4)            | Same form on pair output. Controls DistogramHead logit magnitude; prevents late-trajectory distogram destabilization (see [regularization-story.md](regularization-story.md)). |

#### Diagnostic metrics (logged, not in total)

`training_loss` also emits scalars that are logged via `losses[...]` but never added to `total`:

| Key        | Source                              | Description                                                                                                                                                                                                          |
| ---------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tm_true`  | mini-rollout positions + GT frames  | TM-like global accuracy in [0, 1] computed from the *true* per-frame distance error `‖pred_local − gt_local‖` via the same `1/(1+(d/d0)²)` kernel used for inference pTM. Uses N (effective token count) for `d0`.   |
| `itm_true` | as above, restricted to cross-chain | Cross-chain TM-like accuracy (`asym_id_i ≠ asym_id_j`). Skipped on single-chain examples (no cross-chain pairs to score).                                                                                            |

Both fire from the mini-rollout (S=1) and are computed under `torch.no_grad()`, so they're available **independent of `alpha_confidence`** — they show up in stage 0/1/2 (when the confidence head forward is gated off) and can be compared directly to inference-time pTM/ipTM. PAE-derived `tm_pred` / `itm_pred` are not currently emitted; `scripts/run_status.py` parses both pairs of keys when present.

These diagnostics are O(N²) and run **every step**, so they're individually gateable for throughput:

- `log_tm_true` (default `True`) — `tm_true` / `itm_true`. Suppresses the frame expansion in stages 1–3; **computed regardless when `alpha_pae > 0`** (FT3), since the PAE loss already builds the same `express_coordinates_in_frame` error from the same mini-rollout frames, making `tm_true` ~free on the margin there.
- `log_distogram_entropy` (default `True`) — `distogram_entropy` (`H(p_θ)` over the `[·, N, N, 64]` distogram softmax). The distogram *loss* is unaffected; only the entropy readout is gated.

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

Constant-decay exponential moving average (matches OF3 `ExponentialMovingAverage`):

```
shadow = (1 - decay) × param + decay × shadow
```

EMA weights are swapped in for evaluation (zero-copy pointer swap on `param.data`) and saved in checkpoints. Buffers are not tracked — DF3 buffers are constants set once at init (Fourier weights, ref-atom tables) or shadows derived from base params (fused/tiled kernels), so EMAing them would either be a no-op or actively wrong. Fused-kernel parameters are excluded from EMA shadow copies for the same reason — they are re-derived from base weights at every step.

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

**Global vs. per-rank clipping.** By default the clip runs on the DDP-averaged gradient (all-reduce fires, overlapped with backward, then `clip_grad_norm_` clips the mean). `per_rank_grad_clip: true` (OF3-style) instead runs the **entire backward under `no_sync`** so `.grad` stays each rank's local contribution, clips that local gradient to `grad_clip_norm`, and only then averages the clipped per-rank grads with a single **manual all-reduce** (`SUM ÷ world_size`). This bounds any one rank before it is mixed into the mean, so a pathological sample cannot dominate the batch-mean gradient. Trade-offs: the manual all-reduce runs after the full backward (no backward/all-reduce overlap), the `reduce_dtype` comm hook is bypassed (fp32 reduction), and `static_graph` is forced off. With `batch_size_per_rank > 1` the clip is per-rank over the within-rank sample mean, not literally per-sample. The finite/outlier skip consensus uses the per-rank pre-clip norm (catches a blow-up before the reduce spreads it). Two norms are logged: `grad_norm` (per-rank pre-clip) and `grad_norm_global` (`train/grad_norm_global`, post-reduce, ≤ clip).

### Mixed Precision

Training runs under `bf16` autocast by default (`TrainingConfig.dtype`). A few numerically sensitive regions are pinned to `fp32` via `ModelConfig.precision` (`PrecisionConfig`):

| Flag                             | Default | Region                                                                       |
| -------------------------------- | :-----: | ---------------------------------------------------------------------------- |
| `fp32_loss`                      | `true`  | `training_loss()` — softmax / CE see fp32 logits                             |
| `fp32_confidence_head`           | `true`  | ConfidenceHead final LN+Linear + distance embedding                          |
| `fp32_distogram_head`            | `false` | DistogramHead logit projection                                               |
| `fp32_diffusion_sample`          | `false` | Inference Euler-Maruyama sampler (mirrors `DiffusionConfig.skip_amp_sample`) |
| `fp32_diffusion_sample_training` | `false` | Training-time rollout (mirrors `DiffusionConfig.skip_amp_sample_training`)   |

Each flag disables `torch.autocast("cuda")` over its scope and upcasts inputs to fp32. Rigid alignment (`weighted_rigid_align` in `training/loss.py`) and ligand frame construction (`express_coordinates_in_frame`) are always fp32 — they hold their own `autocast(enabled=False)` contexts regardless of these flags.

Flip `fp32_distogram_head: true` if the per-pair distogram loss exceeds `ln(num_bins) ≈ 4.16`, which signals bf16 overflow in the final projection. Expected runtime cost is <5% — the guarded regions are tiny head-level matmuls.

YAML:

```yaml
model:
  precision:
    fp32_loss: true
    fp32_confidence_head: true
    fp32_distogram_head: false
```

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

**Sampling**: If records have cluster IDs (from wwPDB `clusters-by-entity-40.txt`), `WeightedClusterSampler` applies AF3 §2.5.1 cluster-balanced weights with per-chain and per-interface rows: `w = β[type] × Σ(α[mol] × count) / N_clust` with α(protein)=3, α(nucleic)=3, α(ligand)=1, β(chain)=0.5, β(interface)=1.0. For chains, `N_clust` is the chain cluster size. For interfaces, `N_clust` is the **interface cluster** size per AF3 §2.5.3 — interfaces share a cluster iff their constituent chain pairs map to the same pair of chain clusters. Synthetic epoch size of 10,000 samples. Falls back to uniform `DistributedSampler` if no cluster data.

**Cropping**: Three strategies with weighted random selection — contiguous (20%), spatial KD-tree (40%), interface-seeded (40%). Interface cropping seeds from tokens within 15 Å of another chain, then expands spatially.

**Preprocessing**: End-to-end recipe from a PDB mmCIF mirror to training-ready data:

```bash
# 1. Process structures: mmCIF → Structure NPZ + manifest
#    Filters: resolution ≤ 9 Å, max 300 chains, min 4 residues per chain
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --max-resolution 9.0 --workers 16

# 2. Enrich manifest with release dates (needed for template date cutoff).
#    ``scan-mmcif`` defaults to ``--output data/mmcif_meta.json``, which is
#    also the default ``--mmcif-cache`` for ``prepare-templates`` and
#    ``enrich-manifest`` — just run them from the repo root and wiring
#    happens automatically.
deepfold3 scan-mmcif --mmcif-dir /scratch/pdb/mmCIF --data-dir /data/training --workers 32
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx

# 3. Apply sequence clustering (40% identity, wwPDB pre-computed)
wget -q https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
deepfold3 apply-clusters --manifest /data/training/manifest.json \
    --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json

# 4. Generate unpaired MSAs (local MMseqs2, GPU-accelerated)
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu

# 5. Generate paired MSAs for multimer complexes
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ \
    --gpu --use-pairing

# 6. Discover monomer templates (per-chain m8 hits). Auto-uses
#    ``data/mmcif_meta.json`` from step 2 as the release-date source; without
#    it every hit gets ``release_date=None`` and is dropped at training time.
deepfold3 prepare-templates --data-dir /data/training \
    --m8-dir /data/msa/pdb/merged/m8 --max-templates 20 --workers 16

# 7. Drop obsolete PDB entries (optional, if using a stale mirror)
deepfold3 drop-obsolete --data-dir /data/training
```

See `docs/data-pipeline.md` for details on each step.

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

| Field             | Default     | Description                                                    |
| ----------------- | ----------- | -------------------------------------------------------------- |
| `name`            | (required)  | Identifier for logging                                         |
| `data_dir`        | (required)  | Path to `structures/` + `msa/`                                 |
| `weight`          | 1.0         | Relative sampling probability (primary dataset always 1.0)     |
| `strategy`        | `"uniform"` | `"weighted"` (cluster-balanced) or `"uniform"`                 |
| `is_distillation` | `false`     | If true, disables confidence/distogram losses for this dataset |

The primary dataset (`--data-dir`) always uses cluster-weighted sampling when cluster IDs are available. Extra datasets are combined via `ConcatDataset` + `MultiDatasetSampler`, which draws samples from each dataset proportionally to weights.

**Resume-aware sampler state**: both `WeightedClusterSampler` (single-dataset path) and `MultiDatasetSampler` implement `state_dict` / `load_state_dict`, exposing `epoch + sample_idx + seed`. `CheckpointManager.save` snapshots the sampler under `blob["sampler"]`; on resume the iterator fast-forwards past indices already trained on, so within-epoch position is exact (no record duplication). `set_epoch` only resets `sample_idx` when the epoch number actually changes — the trainer's `set_epoch(state.epoch)` at the top of the loop does not clobber a freshly-loaded position. A seed mismatch between yaml and saved state emits a runtime warning (the per-epoch RNG derives from `seed + epoch`).

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

The featurizer merges paired and unpaired MSAs into a single full-width `[S, N]` tensor: paired rows first (cross-chain aligned), then dense unpaired rows (all chains' row _i_ on the same full-width row). See `docs/data-pipeline.md` § Paired MSA for the full row layout and comparison with OpenFold 3 and Protenix.

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

## Optimizer

AdamW (or Schedule-Free AdamW when `optimizer_type="schedule_free"`) with **decoupled weight decay split into two groups** (`build_optimizer` in `training/optim.py`):

- **No-decay** group (`weight_decay=0`):
  - 1D params (LayerNorm γ/β, biases) and 2D weights with `min(shape)==1` (per-channel scale-like projections).
  - **Gates**: `to_g`, `out_gate`, `.gate.`, AdaLN-zero `cond_scale` / `cond_bias` / `zero_cond` (incl. `attn_cond_scale.{i}`, `k_cond_scale.{i}`, `ffn_cond_scale.{i}`, `*_cond_bias.{i}` `ModuleList` variants in the atom transformer). Decay would pull sigmoids toward 0.5 (always-on), erasing the conditional behaviour.
  - **Input embeddings**: `single_activations`, `msa_activations`, `extra_msa_target_feat`, `{left,right}_single`, `position_activations`, `bond_embedding`, `prev_embedding`, `prev_single_embedding`, plus per-atom and template `embed_*` linears. No preceding LN, so decay shrinks the input representation directly.
- **Decay** group (`weight_decay=cfg.weight_decay`): all remaining 2D+ matmul weights — attention QKV / output projections, FFN linears, triangle multiplication output projections, OPM `out_w`, etc.

See `_NO_DECAY_NAME_SUBSTRINGS` in `training/optim.py` for the exact pattern list.

## Checkpointing

Checkpoints are saved at configurable intervals (default: every 1000 steps) and contain:

- Model state dict (**excludes** fused kernel derived params — see below)
- Optimizer state dict
- LR scheduler state dict
- EMA state (if enabled, also excludes fused params)
- Step counters
- Config (model + training) — `model_cfg` and `train_cfg` are pickled into the blob and used at inference time to rebuild the model with the exact architecture flags (e.g. `pairformer.residual_output_scaling`) that were active at training. See _Configuration persistence_ below.

**Checkpoint rotation**: keeps the `max_checkpoints` most recent checkpoints by modification time. Default is `0`, which disables rotation (all checkpoints are kept). `checkpoint_final.pt` and `checkpoint_best.pt` are always preserved.

Resume training by passing `--resume path/to/checkpoint.pt`.

**LR override on resume**: `optimizer.load_state_dict` and `LambdaLR.load_state_dict` both restore the saved `lr` / `initial_lr` / `base_lrs` verbatim, which silently shadows any `--config` change. The trainer's `[resume]` block detects this and force-overrides `param_groups['lr','initial_lr']` + `scheduler.base_lrs` to the configured value, so `--config new.yaml --resume ...` honors the new lr. Schedule-Free AdamW's accumulator state (`z` buffer, `lr_max`, `weight_sum`) is preserved across the override. A `[resume] LR override: <saved> → <target>` line is logged when this fires.

**Resume audit (warn-only)**: every other yaml-controlled field that is silently shadowed by saved state — `weight_decay` (per param-group), Adam `betas` / `eps`, `ema_decay`, scheduler instance attributes (`warmup_steps`, `start_decay_after`, `decay_every`, `decay_factor`, `cosine_max_steps`, `cosine_min_lr_ratio`, `wsds_decay_windows`) — is compared against the live config and logged as `[resume] yaml fields silently shadowed by checkpoint state` when divergent. The audit is read-only; only `lr` is force-overridden. Fix the yaml or accept the saved value (most fields can also be re-injected via `--override`).

**Partial loading** for stage transitions: use `--params` with a previous stage's checkpoint plus config overrides. Control what gets loaded via `load_strict` (default true), `skip_load_optimizer`, `skip_load_scheduler`, `skip_load_step`.

**Fused kernel portability**: `DeepFold3.state_dict()` automatically filters out derived fused kernel parameters (identified by `OptimConfig.FUSED_STATE_DICT_MARKERS`). On load, `load_state_dict()` re-syncs fused params from base weights. This means checkpoints saved with fused kernels enabled can be loaded without them, and vice versa.

### Configuration persistence

A fresh `ModelConfig()` at inference does **not** match a checkpoint trained with non-default flags (`pairformer.residual_output_scaling`, `pairformer.{single,pair}_transition_residual_gain`, layer counts, …). These flags often only change a forward-pass scalar, so `load_state_dict(strict=False)` reports `missing=0, unexpected=0` and the silent failure is a wrong prediction.

The checkpoint pipeline embeds the architectural `ModelConfig` on both sides of the format:

- **`.pt` training checkpoints**: `CheckpointManager.save` already pickles `model_cfg`. `deepfold3 inference --params <ckpt>.pt` deep-copies it out of the blob and uses it as the base.
- **`.safetensors` files**: `convert_weights.py` and any future writer embed a JSON-serialised `ModelConfig` under the `model_cfg_json` metadata key (`SAFETENSORS_MODEL_CFG_KEY`). `deepfold3 inference --params <weights>.safetensors` reads the metadata via `safe_open(...).metadata()` (header-only, cheap) and reconstructs the dataclass via `model_cfg_from_metadata`. Forward-compatible: unknown future fields are silently dropped.

The `--optim` flag always overrides only `OptimConfig`, never the architectural fields. Legacy `.safetensors` (no metadata) and raw AF3 `.bin` / `.bin.zst` paths fall back to `ModelConfig()` defaults with a one-line warning.

## Training Stages

Stages are controlled by config overrides — no hardcoded presets. Use `--config` with stage JSON files or `--override key=value`.

AF3 trains in 4 stages (Table 6 of supplementary). Each loads weights from the previous stage.

### Stage 1: Initial Training

```json
{
  "alpha_diffusion": 4.0,
  "alpha_pae": 0.0,
  "alpha_bond": 0.0,
  "diffusion_batch_size": 48,
  "max_tokens": 384
}
```

### Fine-tune 1: Structure (crop 640)

```json
{
  "alpha_bond": 1.0,
  "alpha_smooth_lddt": null,
  "diffusion_batch_size": 32,
  "max_tokens": 640
}
```

Bond loss on, smooth LDDT off. Disorder PDB distillation weight reduced to 0.01; masked diffusion for non-protein disabled. Load Stage 1 checkpoint with `--params`.

### Fine-tune 2: Structure (crop 768)

```json
{ "alpha_bond": 1.0, "diffusion_batch_size": 32, "max_tokens": 768 }
```

Smooth LDDT back on (default). Transcription factor distillation sets enabled. Disorder PDB weight back to 0.02. Load Fine-tune 1 checkpoint with `--params`.

### Fine-tune 3: Confidence

```json
{
  "train_confidence_only": true,
  "alpha_diffusion": 0.0,
  "alpha_distogram": 0.0,
  "alpha_pae": 1.0,
  "diffusion_batch_size": 32,
  "max_tokens": 768
}
```

Freezes all parameters except confidence head. Only confidence losses (pLDDT, PDE, PAE, exp_resolved) are active. Max chains increased to 50. Load Fine-tune 2 checkpoint with `--params`.

## AF3 Data Pipeline Conformance

Systematic comparison of DeepFold3's data pipeline against the AF3 supplementary material (§2 + §5). See `docs/af3-paper-reference.md` for the full organized reference.

### ✅ Matching

| Paper section | Item                                                        | Implementation                                                                                        |
| ------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| §2.5.1        | Weighted PDB sampling (β/α weights, chain+interface)        | `data/sample/weighted.py`                                                                             |
| §2.5.3        | Protein 40% clustering                                      | wwPDB clusters via `data/prepare/clustering.py`                                                       |
| §2.5.3        | 100% clustering for nucleic acids + peptides                | Hash-based cluster_id = 100% identity; `scripts/prepare.py:_apply_clusters`                           |
| §2.5.3        | CCD identity clustering for ligands                         | `ChainInfo.res_name` + `_apply_clusters` maps `ccd_{res_name}`                                        |
| §2.5.4        | Resolution ≤ 9Å, max 300 chains, min 4 residues             | `data/prepare/filters.py`                                                                             |
| §2.5.4        | Clashing chains (30%/1.7Å, iterative 3-level tie-break)     | `data/prepare/filters.py:filter_clashing_chains`                                                      |
| §2.5.4        | Consecutive Cα > 10Å                                        | `data/prepare/filters.py`                                                                             |
| §2.6          | Standard AA/nucleotide = 1 token, ligand = per-atom         | `data/tokenize/tokenizer.py`                                                                          |
| §2.7          | Cropping weights (0.2/0.4/0.4), interface 15Å               | `data/crop/cropper.py`                                                                                |
| §2.8          | ref_pos (random rot+trans), ref_element (one-hot 128)       | `data/feature/featurizer.py`                                                                          |
| §2.8          | token_bonds filtered to < 2.4Å during training              | `data/feature/featurizer.py:_build_bond_features`                                                     |
| §3.7.1        | Diffusion batch 48, MSE mol-type weight (5×/10×)            | `model/config.py`, `training/loss.py`                                                                 |
| §3.7.1        | EDM weighting, smooth LDDT (15/30Å radii)                   | `training/loss.py`                                                                                    |
| §3.7.1        | Bond loss (Eq. 5 — GT distances)                            | `training/loss.py:bond_loss`                                                                          |
| §3.7.1        | Cross-entity bonds (disulfide, metal, covalent)             | `data/prepare/structure.py` extracts from `_struct_conn`; `tokenizer.py` propagates with atom indices |
| §2.5.4        | >20 polymer chains → greedy select 20 maximising interfaces | `data/prepare/filters.py:select_interface_chains`                                                     |
| §2.5.3        | Modified residue normalization for clustering               | `data/prepare/structure.py:_chain_sequence` uses `_NONSTANDARD_TO_STANDARD`                           |
| §2.5.4        | CCD atom name validation                                    | `data/ccd.py:build_atom_name_cache` + `process_mmcif(ccd_atom_names=...)`                             |
| §2.5.4        | Leaving atom removal                                        | `process_mmcif(ccd_leaving_atoms=...)` parses `pdbx_leaving_atom_flag` from `_struct_conn`            |
| §2.6          | Modified residue per-atom tokenization                      | `data/tokenize/tokenizer.py` checks `is_standard`; non-standard → per-atom                            |
| §5.3          | Loss weights (α_diff=4, α_disto=0.03, α_conf=1e-4)          | `model/config.py`                                                                                     |
| §5.4          | Adam β₁=0.9 β₂=0.95 ε=1e-8, LR 1.8e-3, warmup 1000          | `model/config.py`                                                                                     |
| §5.4          | LR decay 0.95/50k, grad clip 10, batch 256                  | `model/config.py`, `training/trainer.py`                                                              |

### ⚠️ Divergences

| Paper section | Item               | Our approach                                        | Impact                                    |
| ------------- | ------------------ | --------------------------------------------------- | ----------------------------------------- |
| §3.7.1        | Smooth LDDT weight | Now configurable via `alpha_smooth_lddt` (None=off) | AF3 turns it off in fine-tune 1 (Table 6) |
| §4.2          | Symmetric (ambiguous) atom resolution | Hardcoded per-residue swap table (`data/symmetry/atom_perms.py`: ASP/GLU/ARG/PHE/TYR/LEU/VAL + nucleic OP1/OP2). **TODO: replace with RDKit conformer automorphism per chemical component** — `mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=1000)` → permute GT to min-RMSD, exactly AF3 §4.2 / OF3 `get_cropped_permutations` ([conformer.py:384](../references/openfold-3/openfold3/core/data/primitives/structure/conformer.py#L384)). | Table ≡ RDKit for the 7 standard residues; RDKit additionally covers modified/non-standard residues and 3-fold+ symmetries (e.g. 5′-terminal phosphate OP1/OP2/OP3) generically, and drops the manual HIS pseudo-symmetry that was removed to match AF3. |

### ❌ Missing (known gaps)

| Paper section | Item                                                               | Notes                                                  |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------ |
| §5.2, Table 6 | Masked diffusion loss for non-protein in disorder PDB distillation | Not applicable — diffusion without masking is intended |

### Stage Configuration Quick Reference

From AF3 Table 6. Use `--config` with stage JSON files or `--override`:

```bash
# Stage 1: Initial training (crop 384, diffusion batch 48, max_chains 20)
deepfold3 train --data-dir /data --output runs/s1 --config configs/initial.json

# Fine-tune 1: crop 640, bond loss on, smooth LDDT off, diffusion batch 32
deepfold3 train --data-dir /data --output runs/ft1 --params runs/s1/checkpoint_final.pt \
    --config configs/finetune1_structure_640.json

# Fine-tune 2: crop 768, smooth LDDT back on, TF distillation sets on
deepfold3 train --data-dir /data --output runs/ft2 --params runs/ft1/checkpoint_final.pt \
    --config configs/finetune2_structure_768.json

# Fine-tune 3: PAE only, max chains 50
deepfold3 train --data-dir /data --output runs/ft3 --params runs/ft2/checkpoint_final.pt \
    --config configs/finetune3_confidence.json
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

| Module         | Depends On                                           |
| -------------- | ---------------------------------------------------- |
| `loss.py`      | `model.config.TrainingConfig`                        |
| `ema.py`       | `torch.nn`, `model.config.OptimConfig`               |
| `scheduler.py` | `torch.optim`, `model.config.TrainingConfig`         |
| `trainer.py`   | All above + `model.model.DeepFold3`, `data.module.*` |
