# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFold 3 is a PyTorch biomolecular complex structure prediction system. The design intent is:

- **Model**: architecturally equivalent to AlphaFold3
- **Data pipeline**: structured after Boltz
- **Raw data processing**: structured after Protenix

The project has a working inference pipeline — model, data, I/O, and scripts are implemented in `src/`.

- **Package manager**: `uv` (lockfile present as `uv.lock`)
- **Python version**: 3.12 (pinned in `.python-version`)
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
deepfold3 inference --msa query.a3m --output out.cif --params weights.safetensors
deepfold3 convert   --params af3.bin.zst --af3-ref references/alphafold3 --output weights.safetensors
deepfold3 plot      --input out_confidences.json --output confidences.png
```

## Ruff Configuration

Line length 120; rules: E, F, I (isort), B (bugbear), UP (pyupgrade). `E501` is ignored (handled by `ruff format`). Double quotes, space indent.

## Intended Architecture

### Model (≡ AlphaFold3, `references/alphafold3/`)

The model is a **diffusion-based generative model** with three major heads and a recycled trunk:

```
Input features (tokens + atoms)
    → Per-atom conditioning (atom_cross_att_encoder)
    → Evoformer trunk  [10 recycles]
        ├─ MSA processing (EvoformerIteration × 4)
        ├─ Template embedding
        └─ PairFormer stack (48 layers)
               → single [N_tokens, 384]  +  pair [N_tokens, N_tokens, 128]
    → Diffusion Head  (200-step denoising)
        └─ atom_positions [N_samples, N_atoms, 3]
    → Confidence Head (pLDDT, PAE, PDE, exp_resolved, pTM, ipTM)
    → Distogram Head
```

**Key dimensions**: `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `num_msa=1024`, `num_recycles=10`, `diffusion_steps=200`, `num_samples=5`.

**Atom representation**: Two layouts used in parallel — *all-tokens* (one representative atom per token) and *all-token-atoms* `[N_tokens, max_atoms_per_token]` (all heavy atoms). Proteins/RNA/DNA use one token per residue; ligands use one token per atom.

**Core network modules** (`references/alphafold3/src/alphafold3/model/network/`):

- `evoformer.py` — Evoformer + PairFormer stacks
- `atom_cross_attention.py` — encoder/decoder between atom-level ↔ token-level
- `diffusion_head.py` / `diffusion_transformer.py` — AdaLN-conditioned diffusion transformer
- `confidence_head.py` — shallow 4-layer PairFormer → pLDDT/PAE/PDE/exp_resolved
- `modules.py` — TriangleMultiplication, OuterProductMean, MSAAttention, TransitionBlock

---

### Data Pipeline (≡ Boltz, `references/boltz/`)

Seven-stage pipeline: **Parse → Input → Sample → Tokenize → Crop → Featurize → Batch**

```
Raw mmCIF / YAML
    → [Parse]     →  Structure (NPZ: atoms, bonds, residues, chains)
    → [Input]     →  Input(structure, msa_dict)       loaded from NPZ files
    → [Sample]    →  Sample  (cluster / random / distillation sampler)
    → [Tokenize]  →  Tokenized(tokens[], bonds[], structure, msa)
    → [Crop]      →  Tokenized (≤ max_tokens, ≤ max_atoms)
    → [Featurize] →  Dict[str, Tensor]  (token, atom, MSA, symmetry features)
    → [Batch]     →  padded batch tensors
```

**Storage layout**:

```
dataset_root/
├── structures/{id}.npz   # Structure numpy structured arrays
├── msa/{msa_id}.npz      # MSA numpy structured arrays
└── manifest.json         # Record list (id, chains, interfaces, cluster_ids)
```

**Feature categories produced by featurizer**:

- *Token*: `residue_index`, `asym_id`, `entity_id`, `mol_type`, `res_type`, `token_bonds`, distogram center coords
- *Atom*: `ref_pos`, `ref_element`, `ref_charge`, `ref_atom_name_chars`, `atom_to_token`, `coords`, `disto_target`
- *MSA*: `msa` (one-hot), `profile`, `deletion_value`, `msa_paired`
- *Symmetry*: chain/residue/ligand atom swaps, covalent bond graph

**Cropping** uses two strategies (randomly weighted): *contiguous* (sequential chain sampling) and *spatial* (KD-tree distance, interface-centred).

**MSA pairing**: sequences paired across chains by shared taxonomy IDs; up to 8,192 paired + 16,384 unpaired sequences per sample.

---

### Raw Data Processing (≡ Protenix, `references/Protenix/`)

Pre-training data preparation script produces cleaned bioassembly files and an index CSV.

```
PDB mmCIF files
    → [Parse]    gemmi / Biotite → AtomArray
    → [Filter]   remove: water, H/D, UNX/UNL, crystallisation aids (SO4, GOL…),
                         clashing chains (1.6 Å threshold, cell-list method)
                 remap: MSE→MET, ASX→ASP, GLX→GLU
                 add:   missing heavy atoms/residues (mark is_resolved=False)
    → [Tokenize] protein/RNA/DNA: 1 token/residue; ligands: 1 token/atom
    → [Save]     {pdb_id}.pkl.gz  (bioassembly dict with atom_array, token_array)
    → [Index]    manifest CSV  (pdb_id, resolution, chain/interface rows, cluster_ids)
```

**Index rows** carry: `pdb_id`, `assembly_id`, `resolution`, `chain_{1,2}_id`, `mol_{1,2}_type`, `entity_{1,2}_id`, `cluster_{1,2}_id`, `type` (chain|interface), `eval_type`.

**Ligand chemistry** handled via CCD cache (pre-built from `pdbeccdutils` / `rdkit`); atom symmetries (swappable oxygens, chirality, stereo bonds) extracted from RDKit for symmetry-aware training loss.

---

## Reference Implementations

| Path | Framework | Primary use |
|------|-----------|-------------|
| `references/alphafold3/` | JAX / Haiku | Canonical model architecture |
| `references/boltz/` | PyTorch | Data pipeline patterns (most relevant) |
| `references/Protenix/` | PyTorch | Raw preprocessing scripts |

`resources/` contains the AF3 paper PDFs and model weights (`af3.bin.zst`, zstandard-compressed).

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

## Implemented Source Layout

```
src/deepfold3/
├── model/           # Neural network modules
│   ├── primitives.py     # LinearNoBias, LayerNorm, SwiGLU, TransitionBlock, AdaLN
│   ├── config.py         # EvoformerConfig, DiffusionConfig, ConfidenceConfig, ModelConfig
│   ├── embedders.py      # create_relative_encoding
│   ├── attention.py      # GridSelfAttention, MSAAttention, SingleAttention
│   ├── ops.py            # TriangleMul, OuterProductMean
│   ├── evoformer.py      # EvoformerBlock, PairFormerBlock, Evoformer
│   ├── atom_cross_attn.py # AtomCrossAttEncoder/Decoder, EvoformerConditioningEncoder
│   ├── diffusion.py      # DiffusionTransformer, DiffusionHead (sampling + training)
│   ├── confidence.py     # ConfidenceHead (pLDDT, PAE, PDE, exp_resolved)
│   └── model.py          # DeepFold3 (top-level), DeepFold3Output, DistogramHead
├── data/            # Data pipeline
│   ├── const.py          # RES_TYPE_ORDER, ELEMENT_TO_INT, REF_ATOMS
│   ├── types.py          # Structure, MSA, Input, Tokenized, Record dataclasses
│   ├── parse/            # mmcif.py (gemmi), a3m.py (MSA)
│   ├── tokenize/         # tokenizer.py (Input → Tokenized)
│   ├── crop/             # cropper.py (contiguous + spatial)
│   ├── feature/          # ref_atoms.py (CCD), featurizer.py (Tokenized → Tensor dict)
│   └── module/           # dataset.py (InferenceInput), collate.py (batch)
├── convert/         # Weight conversion
│   └── af3.py            # Haiku → PyTorch state_dict converter
├── io/              # Output serialization
│   ├── mmcif.py          # save_mmcif() (gemmi-based, pLDDT in B-factors)
│   └── confidences.py    # Confidence metric computation + save_confidences_json()
├── scripts/         # CLI (click group: `deepfold3 <command>`)
│   ├── __init__.py       # click.Group `cli` — unified entry point
│   ├── inference.py      # `deepfold3 inference` — evoformer + diffusion + confidence → mmCIF + JSON
│   ├── convert_weights.py  # `deepfold3 convert` — AF3 params → safetensors
│   └── plot_confidences.py # `deepfold3 plot` — PAE/PDE/contacts from confidences JSON
└── params.py        # read_af3_params() (zstd decompression)
```

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
