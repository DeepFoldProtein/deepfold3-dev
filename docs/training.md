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
    "mini_rollout_steps": 20,
    "alpha_diffusion": 4.0,
    "alpha_distogram": 0.03,
    "alpha_confidence": 1e-4,
    "alpha_pae": 0.0,
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
| Diffusion MSE | `alpha_diffusion` (4.0) | EDM-weighted MSE: w(σ) = (σ²+σ_d²)/(σ·σ_d)² |
| Smooth LDDT | `alpha_diffusion` (4.0) | Distance-based accuracy (cutoffs: 0.5, 1.0, 2.0, 4.0 Å) |
| Distogram | `alpha_distogram` (0.03) | Cross-entropy on 64-bin distance predictions |
| pLDDT | `alpha_confidence` (1e-4) | Per-atom confidence cross-entropy (50 bins) |
| PDE | `alpha_confidence` (1e-4) | Predicted distance error cross-entropy |
| exp_resolved | `alpha_confidence` (1e-4) | Binary resolved/unresolved classification |
| PAE | `alpha_confidence × alpha_pae` (0) | Predicted aligned error (disabled in stage 1) |

### LR Schedule

AF3-style three-phase schedule:

```
1. Warmup (0 → warmup_steps): linear 0 → lr
2. Plateau (warmup_steps → start_decay_after): constant lr
3. Step decay: lr × decay_factor^n every decay_every steps
```

Default: lr=0.0018, warmup=1000, decay starts at 50k, factor=0.95 every 50k.

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

## Data Requirements

Training data directory structure:

```
data_dir/
├── structures/
│   ├── {id}.npz          # Structure files (atoms, residues, chains, bonds)
│   └── ...
└── msa/
    ├── {msa_id}.npz      # MSA files (sequences, mask, deletion_matrix)
    └── ...
```

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

Resume training by passing `--resume path/to/checkpoint.pt`.

**Fused kernel portability**: `DeepFold3.state_dict()` automatically filters out derived fused kernel parameters (identified by `OptimConfig.FUSED_STATE_DICT_MARKERS`). On load, `load_state_dict()` re-syncs fused params from base weights. This means checkpoints saved with fused kernels enabled can be loaded without them, and vice versa.

## Training Stages

### Stage 1: Full Training
```json
{"alpha_diffusion": 4.0, "alpha_pae": 0.0, "alpha_bond": 0.0}
```

### Stage 2: Finetuning
```json
{"alpha_diffusion": 4.0, "alpha_pae": 1.0, "alpha_bond": 1.0}
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
