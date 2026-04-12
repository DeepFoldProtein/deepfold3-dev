# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFold3 is a PyTorch biomolecular complex structure prediction system, architecturally equivalent to AlphaFold 3.

The project has a working inference pipeline and a core training pipeline — model, data, I/O, training, and scripts are implemented in `src/`.

**Status:** Training-time template date-cutoff filter is implemented (`dataset.py:_load_monomer_templates` applies `min(max_template_date, record.released - cutoff_days)` per record). Symmetry alignment for AA / chain / ligand permutation is implemented (`data/symmetry/`). Sampler currently aggregates per-record, not per-(chain|interface) — see `docs/training.md` for the divergence list.

- **Package manager**: `uv` (lockfile present as `uv.lock`)
- **Python version**: 3.11+ (pinned in `.python-version`)
- **Source layout**: `src/deepfold3/` (packages found under `src/` via setuptools)

## Commands

```bash
# Install dependencies (including dev)
uv sync --dev

# Run tests
uv run pytest

# Run a single test file or test
uv run pytest tests/path/to/test_file.py
uv run pytest tests/path/to/test_file.py::test_name

# Lint / format
uv run ruff check src/
uv run ruff format src/

# CLI (unified entry point via click — registered in pyproject.toml as deepfold3)
deepfold3 msa       --sequence MKTII --output msa_out/ --json
deepfold3 msa       --sequence SEQ1:SEQ2:SEQ1 --output msa_out/ --use-pairing --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --gpu-server
deepfold3 msa       --sequence MKTII --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --use-templates --json
deepfold3 msa       --sequence MKTII --output msa_out/ --link-templates --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 inference --msa query.a3m --output out.cif --params weights.safetensors
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --template-cache /tmp/cache
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 convert   --params af3.bin.zst --af3-ref references/alphafold3 --output weights.safetensors
deepfold3 plot      --input out_confidences.json --output confidences.png
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-ids 1pfe,8qy0
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-id-file pdb_ids.txt --workers 16
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-id-file pdb_ids.txt --no-assembly --workers 16
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --max-resolution 4.0 --workers 8
deepfold3 prepare   --mmcif-dir /scratch/pdb/mmCIF --output /data/training --cluster-file clusters-by-entity-40.txt
deepfold3 cluster   --data-dir /data/training --output clusters.txt
deepfold3 cluster   --data-dir /data/training --output clusters.txt --min-seq-id 0.3
deepfold3 apply-clusters --manifest /data/training/manifest.json --cluster-file clusters-by-entity-40.txt --output /data/training/manifest.json
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu --use-pairing
deepfold3 msa-batch --data-dir /data/training --convert-only
deepfold3 import-msa --data-dir /data/training --msa-dir /data/afdb/a3ms --workers 32
deepfold3 import-paired-msa --data-dir /data/training --paired-dir /data/msa/paired --workers 32
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training
deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training --cache /data/of3/dataset_cache.json
deepfold3 prepare-templates --data-dir /data/training --m8-dir /data/msa/pdb/merged/m8
deepfold3 prepare-templates --data-dir /data/training --m8-dir /data/msa/pdb/merged/m8 --max-templates 4 --workers 16
deepfold3 scan-mmcif --mmcif-dir ~/deepfold3/pdb/divided --data-dir /data/training --output mmcif_meta.json --workers 32
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx
deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx --mmcif-cache mmcif_meta.json
deepfold3 drop-obsolete --data-dir /data/training
deepfold3 drop-obsolete --data-dir /data/training --delete-files
deepfold3 train     --data-dir /data/training --output runs/exp1 --params weights.safetensors
deepfold3 train     --data-dir /data/training --output runs/exp1 --params weights.safetensors --mmcif-dir /data/pdb/mmcif
deepfold3 train     --data-dir /data/training --output runs/exp1 --resume runs/exp1/checkpoint_1000.pt
deepfold3 train     --data-dir /data/training --output runs/exp1 --config configs/stage2_finetune_structure.json --params runs/exp1/checkpoint_final.pt
deepfold3 train     --data-dir /data/training --output runs/exp1 --override alpha_pae=1.0 --override train_confidence_only=true
deepfold3 train     --data-dir /data/training --output runs/exp1 --wandb deepfold3
deepfold3 train     --data-dir /data/pdb --output runs/exp1 --dataset afdb:/data/afdb:0.25:uniform:true
deepfold3 train     --data-dir /data/pdb --output runs/exp1 --dataset afdb:/data/afdb:0.5:uniform:true --dataset pdb2:/data/pdb2:1.0:weighted
# Multi-GPU training via torchrun
torchrun --nproc_per_node=4 -m deepfold3.scripts.train --data-dir /data/training --output runs/exp1
# Pre-warm the persistent Triton autotune cache (run once per GPU model on a single rank
# before launching DDP training; B200 ships with hardcoded configs and skips this step).
deepfold3 warmup-triton --config configs/initial.json
```

## Ruff Configuration

Line length 120; rules: E, F, I (isort), B (bugbear), UP (pyupgrade). `E501` is ignored (handled by `ruff format`). Double quotes, space indent. Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Attributes:`).

## Documentation Maintenance

When modifying source code, update the corresponding developer documentation in `docs/`:

- **Architecture changes** (new modules, changed forward pass, new heads) → update `docs/architecture.md`
- **Data pipeline changes** (new features, changed shapes, new parsing formats) → update `docs/data-pipeline.md`
- **New patterns or conventions** (testing, code style, extension points) → update `docs/contributing.md`
- **Feature table** in `docs/data-pipeline.md` must stay in sync with `data/feature/featurizer.py` output

Keep the docs concise — update existing sections rather than appending.

## Architecture

The model is a **diffusion-based generative model** with three major heads and a recycled trunk:

```
Input features (tokens + atoms)
    → Per-atom conditioning (AtomCrossAttEncoder)
    → Evoformer trunk  [10 recycles]
        ├─ Template embedding (optional, T templates → pair)
        ├─ MSA processing (EvoformerBlock × 4)
        └─ PairFormer stack (48 layers)
               → single [N, 384]  +  pair [N, N, 128]
    → Diffusion Head  (200-step denoising)
        └─ atom_positions [S, N, M, 3]
    → Confidence Head (pLDDT, PAE, PDE, exp_resolved, pTM, ipTM)
    → Distogram Head
```

**Key dimensions**: `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `num_recycles=10`, `diffusion_steps=200`, `num_samples=5`.

**Atom representation**: Two layouts — *all-tokens* (one representative atom per token) and *all-token-atoms* `[N_tokens, max_atoms_per_token]` (all heavy atoms). Proteins/RNA/DNA use one token per residue; ligands use one token per atom.

## Source Layout

```
src/deepfold3/
├── model/           # Neural network modules
│   ├── primitives.py     # LinearNoBias, LayerNorm, SwiGLU, TransitionBlock, AdaLN
│   ├── config.py         # EvoformerConfig, DiffusionConfig, ConfidenceConfig, ModelConfig, TrainingConfig
│   ├── embedders.py      # create_relative_encoding
│   ├── attention.py      # GridSelfAttention, MSAAttention, SingleAttention
│   ├── ops.py            # TriangleMul, OuterProductMean
│   ├── evoformer.py      # EvoformerBlock, PairFormerBlock, Evoformer
│   ├── template_embedding.py  # TemplateEmbedding, SingleTemplateEmbedding, TemplatePairFormerBlock
│   ├── atom_cross_attn.py # AtomCrossAttEncoder/Decoder, EvoformerConditioningEncoder
│   ├── diffusion.py      # DiffusionTransformer, DiffusionHead (sampling + training noise/denoise)
│   ├── confidence.py     # ConfidenceHead (pLDDT, PAE, PDE, exp_resolved)
│   ├── bucket.py         # Bucket-based sequence length padding for training
│   ├── model.py          # DeepFold3 (top-level, forward + forward_training), DeepFold3Output, DistogramHead
│   └── triton/           # Triton fused kernels (optional GPU acceleration)
│       ├── __init__.py            # Precision enum, cvt_tf32_rn, precision_dot
│       ├── evo_attention.py       # Fused evoformer attention with pair bias
│       ├── fused_pair_bias.py     # Fused LN + tl.dot projection + permute + mask
│       ├── fused_gated_dual_gemm.py # Dual GEMM + sigmoid (TWO_INPUTS, bias, mask)
│       ├── layernorm_linear.py    # Fused LayerNorm + Linear
│       ├── ada_layernorm_linear.py # Fused AdaLN + Linear
│       ├── layernorm_transpose.py # Fused LayerNorm + dim transpose
│       ├── triangle_mul_gate.py   # Fused TriangleMul gate (LN + packed GEMM)
│       ├── fused_ada_transition.py # Fused AdaTransition (diffusion FFN)
│       ├── fused_transition.py    # Fused TransitionBlock
│       ├── swiglu.py              # Fused SwiGLU
│       ├── tuning.py              # Pre-tuned block size configs (CUDA/HIP)
│       └── cache.py               # Persistent autotuning cache (GPU fingerprint)
├── data/            # Data pipeline
│   ├── const.py          # RES_TYPE_ORDER, ELEMENT_TO_INT, REF_ATOMS, element_name_to_int(), get_res_type()
│   ├── types.py          # Structure, MSA, Input, Tokenized, Record dataclasses
│   ├── masking.py        # TokenMask (per-token mask channels for loss control: fixed, region)
│   ├── ccd.py            # CcdMolecule, load_ccd_molecule(), smiles_to_molecule()
│   ├── paths.py          # shard(), structure_path, mask_path, template_path — sharded layout helpers
│   ├── msa/             # mmseqs2.py (search) + paths.py (unpaired_msa_path, paired_msa_path)
│   ├── parse/            # mmcif.py (mol_type_from_gemmi), a3m.py (parse_a3m + convert_a3m_file_to_npz), templates.py (find_pdb_mmcif_path), template_realign.py, hhr.py, json_input.py (4 dialects)
│   ├── tokenize/         # tokenizer.py (Input → Tokenized)
│   ├── crop/             # cropper.py (contiguous + spatial + interface-seeded)
│   ├── sample/           # weighted.py (WeightedClusterSampler — AF3-style cluster-balanced)
│   ├── feature/          # ref_atoms.py (CCD), featurizer.py (Tokenized → Tensor dict)
│   ├── prepare/          # structure.py (mmCIF→NPZ + AF3-aligned metadata extract), filters.py, manifest.py (atomic save), clustering.py, template_search.py, materialize.py (parallel A3M→NPZ runner), entries_idx.py (wwPDB TSV parser)
│   └── module/           # dataset.py (TrainingDataset, InferenceInput), collate.py (batch)
├── convert/         # Weight conversion
│   └── af3.py            # Haiku → PyTorch state_dict converter
├── io/              # Output serialization
│   ├── mmcif.py          # save_mmcif() (gemmi-based, pLDDT in B-factors)
│   └── confidences.py    # Confidence metric computation + save_confidences_json()
├── training/        # Training pipeline
│   ├── loss.py           # smooth_lddt, diffusion_mse, bond, distogram, plddt, pae, pde, exp_resolved, training_loss
│   ├── ema.py            # EMA with warm-start decay + fused kernel re-sync
│   ├── scheduler.py      # AF3LRScheduler + CosineAnnealingScheduler
│   └── trainer.py        # DDP Trainer (training loop, checkpointing, eval, WandB)
├── scripts/         # CLI (click group: `deepfold3 <command>`)
│   ├── __init__.py       # click.Group `cli` — unified entry point
│   ├── _common.py        # setup_cli_logging(), require_manifest()
│   ├── inference.py      # `deepfold3 inference`
│   ├── msa_search.py     # `deepfold3 msa`
│   ├── msa_batch.py      # `deepfold3 msa-batch`
│   ├── convert_weights.py  # `deepfold3 convert`
│   ├── plot_confidences.py # `deepfold3 plot`
│   ├── prepare.py        # `deepfold3 prepare`, `import-msa`, `import-paired-msa`, `import-of3`, `enrich-manifest`, `drop-obsolete`, etc.
│   ├── prepare_templates.py # `deepfold3 prepare-templates`
│   ├── scan_mmcif.py     # `deepfold3 scan-mmcif` (parallel mmCIF metadata cache)
│   └── train.py          # `deepfold3 train`
└── params.py        # read_af3_params() (zstd decompression)

scripts/                     # Shell scripts for database setup and GPU server
├── setup_databases.sh       # Download and index UniRef30, ColabFoldDB, PDB100
├── mmseqs_gpuserver.sh      # Start persistent MMseqs2 GPU server processes
├── mmseqs_msa.sh            # Verbatim ColabFold server-side MSA search
└── mmseqs_pair.sh           # Verbatim ColabFold server-side paired MSA search
```

## Key Dependencies

| Package | Purpose |
| ------- | ------- |
| `torch` | Core deep learning framework |
| `gemmi` | mmCIF/PDB structure file parsing |
| `rdkit` | Small molecule chemistry (ligand handling, symmetry) |
| `pdbeccdutils` | CCD (Chemical Component Dictionary) utilities |
| `safetensors` | Model weight serialization |
| `zstandard` | Compression (model weights are `.zst`) |
| `click` | CLI interface |
| `numpy` | Numerical arrays / structured array storage |

## Training Pipeline

### Overview

Training uses native PyTorch DDP. The core training loop, losses, EMA, and LR scheduler are implemented in `src/deepfold3/training/`. See `docs/training.md` for full details.

### Forward Pass (per batch)

```
1. Evoformer trunk (random 1–max_recycles recycles, gradient on last only)
2. Diffusion training step (noise sampling + single denoise, with gradient)
3. Mini-rollout (20-step denoise, no gradient → predicted positions for confidence)
4. Confidence head (on mini-rollout positions, gradient flows)
5. Distogram head (cross-entropy on distance bins)
```

### Loss Weights (AF3 §3.7)

| Loss | Weight | Notes |
|------|--------|-------|
| Diffusion MSE + smooth LDDT | `alpha_diffusion` (4.0) | EDM-weighted; mol-type reweight: DNA/RNA 5×, ligand 10× |
| Bond violation | `alpha_diffusion × alpha_bond` (0) | MSE on bond distances vs CCD ideal; enable for stage 2 |
| Distogram | `alpha_distogram` (0.03) | 64-bin cross-entropy |
| pLDDT, PDE, exp_resolved | `alpha_confidence` (1e-4) | Per-atom/pair confidence; gated on resolution ∈ [0.1, 4.0] |
| PAE | `alpha_confidence × alpha_pae` (0) | Disabled in stage 1; set alpha_pae=1.0 for finetuning |

### Token Masks (Loss Control)

Per-token mask channels stored in `data_dir/masks/{pdb_id}.npz`, loaded automatically by `TrainingDataset`. Managed via `TokenMask` (`data/masking.py`). Two reserved channels:

| Channel | Dtype | Effect |
|---------|-------|--------|
| `"fixed"` | float32 `[N]` | 1.0 = experimental, 0.0 = model-generated. Scales `exp_resolved` loss target (soft CE). For PATCHR ATLAS distillation. |
| `"region"` | int16 `[N]` | Region label (0=unset, 1+=CDR-H1/H2/H3 etc.). Applies per-token weight via `region_loss_weights` config to diffusion MSE loss. |

Generate PATCHR ATLAS masks from inpainting metadata JSON:

```python
from deepfold3.data.masking import TokenMask, MASK_FIXED
fixed = TokenMask.fixed_from_patchr(json_path, tokens, chains, chain_infos=record.chains)
np.savez(masks_dir / f"{pdb_id}.npz", **{MASK_FIXED: fixed})
```

### Multi-Stage Training

Stages are controlled by config overrides, not hardcoded presets. Use `--config` with stage JSON files or `--override key=value`:

```bash
# Stage 1: Initial training (default config)
deepfold3 train --data-dir /data/training --output runs/exp1

# Stage 2: Fine-tune structure (bond loss on, cosine LR)
deepfold3 train --data-dir /data/training --output runs/exp2 \
    --config configs/stage2_finetune_structure.json --params runs/exp1/checkpoint_final.pt

# Stage 3: Fine-tune confidence (freeze trunk, enable PAE)
deepfold3 train --data-dir /data/training --output runs/exp3 \
    --config configs/stage3_finetune_confidence.json --params runs/exp2/checkpoint_final.pt
```

### Training Data Format

Every persistent file is **sharded** by the first two characters of its id (256 buckets) so listing, rsync, and inode walks stay fast at 100k+ files. Path construction goes through the helpers in `data/paths.py` (`structure_path`, `mask_path`, `template_path`) and `data/msa/paths.py` (`unpaired_msa_path`, `paired_msa_path`) — never inline.

```
data_dir/
├── manifest.json                                # list of Record objects (atomic save with .old backup)
├── structures/{ab}/{pdb_id}.npz                 # Structure (atoms, bonds, residues, chains)
├── masks/{ab}/{pdb_id}.npz                      # TokenMask channels (optional)
├── templates/{ab}/{pdb_id}.json                 # Multimer template hits (optional)
└── msa/
    ├── unpaired/{ab}/{msa_id}.npz               # per-sequence (SHA256 hash)
    └── paired/{ab}/{record_id}_paired_{chain_hash}.npz  # per-complex
```

Records are loaded via `discover_records()` from `data/prepare/manifest.py` (reads `manifest.json` or recursively scans `structures/`). Each `ChainInfo` has a `msa_id` that maps to the unpaired MSA file name (SHA256 of the chain sequence). `ChainInfo` also stores `auth_chain_name` (original chain name from input CIF, before assembly renaming) for accurate chain mapping in mask generation.

`Record.structure_info` (`resolution`, `method`, `deposited`, `released`) can be populated either at prepare time (from mmCIF metadata) or after the fact via `deepfold3 enrich-manifest --idx entries.idx [--mmcif-cache mmcif_meta.json]`. Use `deepfold3 scan-mmcif` first to generate the cache from a local PDB mirror — that's the only canonical source for `released` (= earliest `_pdbx_audit_revision_history.revision_date`, matching AlphaFold 3's template date-cutoff convention).

### Data Preprocessing Pipeline

Implemented: `deepfold3 prepare` (steps 1 + 2 + 7), `deepfold3 msa` (step 3 for inference), `deepfold3 cluster` (step 2 via MMseqs2), `deepfold3 apply-clusters` (apply pre-computed clusters). Remaining steps (3 batch, 4, 5) need implementation for training.

```bash
# Process specific PDB IDs
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --pdb-ids 1pfe,8qy0,4hhb

# Batch process with filters
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \
    --max-resolution 9.0 --methods "X-RAY DIFFRACTION,ELECTRON MICROSCOPY" --workers 8

# With date cutoff for train/val split
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --max-date 2023-09-30

# Sample N structures
deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training --limit 100
```

Steps:

1. **Structure processing** (implemented): PDB mmCIF → `gemmi.make_assembly()` (assembly 1) → structure-level filters (resolution, method, date, chain count, residue count, polymer count) → chain-level filters (min/max residue count, all-unknown, consecutive CA distance, chain clashing) → Structure NPZ.
2. **Sequence clustering** (implemented): `deepfold3 cluster` (MMseqs2 at 40% identity) or download pre-computed wwPDB `clusters-by-entity-40.txt`. Apply via `deepfold3 apply-clusters` or `--cluster-file` during prepare.
3. **MSA generation** (implemented): `deepfold3 msa` for inference, `deepfold3 msa-batch` for training. Local MMseqs2 (GPU optional) against UniRef30/ColabFoldDB → A3M → NPZ per unique sequence. Resume support, batch splitting.
4. **Template discovery** (implemented): `deepfold3 prepare-templates` discovers multimer templates from per-chain m8 files (MMseqs2 SEQRES hits). Uses Hungarian algorithm for optimal chain pairing. Writes lightweight JSON index to `{data_dir}/templates/`.
5. **ID filtering** (implemented): `--include-list` / `--exclude-list` text files for `deepfold3 train`
6. **Manifest creation** (implemented): collect records/ → `manifest.json`

Source: `src/deepfold3/data/prepare/` (structure.py, filters.py, manifest.py, clustering.py, template_search.py).

PDB mmCIF mirror: `/scratch/pdb/mmCIF/` (layout: `{mid2}/{pdb_id}.cif.gz`, ~250k structures).

### Template Pipeline

Training-time template support uses pre-computed per-chain m8 files (MMseqs2 hits against PDB SEQRES):

1. **Offline discovery**: `deepfold3 prepare-templates` reads m8 files, groups hits by template PDB ID, finds PDB entries covering ≥2 query chains, uses bipartite matching (Hungarian algorithm) for chain pairing, writes `{data_dir}/templates/{pdb_id}.json`.
2. **Training-time loading**: `TrainingDataset` loads template JSON index, reads mmCIF from PDB mirror (`--mmcif-dir`), extracts `TemplateFeatures` via `extract_from_seqres_alignment()` (with seqres→structure realignment), passes `template_feats_per_chain` to featurizer.
3. **Seqres realignment**: `template_realign.py` handles the mismatch between PDB SEQRES (used by search tools) and the actual mmCIF structure (which may have missing/different residues). Applied automatically in both the HHR inference path (`extract_template_features_from_hhr`) and the training CIGAR path (`extract_from_seqres_alignment`).
4. **Template augmentation**: each sample randomly uses 0..max_templates from available hits (uniform). Per-chain dropout randomly drops chain alignments from multimer templates (50% per chain, always keeping ≥1), so the model sees full, partial, or no cross-chain template context.
5. **Per-template pair mask**: Template embedding uses per-template coverage mask (`pb_mask[i] × pb_mask[j]`) instead of the query's `multichain_mask`. This allows multimer templates to provide cross-chain structural information (interface distances, relative orientations). Monomer templates naturally mask to intra-chain only since they cover a single chain.
6. **Not yet implemented**: Date cutoff filtering to avoid data leakage.

The inference template pipeline (`data/parse/templates.py`) supports per-chain HHR templates and whole-complex JSON templates.

## Confidence Metrics

The confidence head produces four raw outputs per diffusion sample:

- **pLDDT** `[S, N, M*50]` — per-atom confidence (50 bins → expected value in [0, 100])
- **PAE** `[S, N, N, 64]` — predicted aligned error (64 bins, max 31 Å)
- **PDE** `[S, N, N, 64]` — predicted distance error (symmetrised, same bins as PAE)
- **exp_resolved** `[S, N, M*2]` — per-atom 2-class (not-resolved, resolved)

Derived metrics computed in `io/confidences.py`:

- **pTM**: max_i Σ_j (tm_adjusted_pae[i,j] × normed_mask[i,j]), d_0 = 1.24×(max(N,19)-15)^(1/3)−1.8
- **ipTM**: same as pTM but restricted to cross-chain pairs
- **Ranking score**: 0.8×ipTM + 0.2×pTM (single-chain: just pTM)
- **Contact probs**: from distogram logits, P(d < 8 Å)
- **gPDE**: global average PDE over all token pairs
