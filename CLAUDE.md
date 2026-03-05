# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFold3 is a PyTorch biomolecular complex structure prediction system, architecturally equivalent to AlphaFold 3.

The project has a working inference pipeline — model, data, I/O, and scripts are implemented in `src/`.

**Not yet implemented:** multi-chain complexes, ligands, full data pipeline (PDB preprocessing, clustering, dataset generation), training pipeline.

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
deepfold3 msa       --sequence MKTII --output msa_out/ --json
deepfold3 msa       --sequence SEQ1:SEQ2:SEQ1 --output msa_out/ --use-pairing --json
deepfold3 msa       --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --json
deepfold3 msa       --sequence MKTII --output msa_out/ --link-templates --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 inference --msa query.a3m --output out.cif --params weights.safetensors
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --template-cache /tmp/cache
deepfold3 inference --json input.json --output output_dir/ --params weights.safetensors --pdb-mmcif-dir /data/pdb/mmcif/
deepfold3 convert   --params af3.bin.zst --af3-ref references/alphafold3 --output weights.safetensors
deepfold3 plot      --input out_confidences.json --output confidences.png
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
│   ├── config.py         # EvoformerConfig, DiffusionConfig, ConfidenceConfig, ModelConfig
│   ├── embedders.py      # create_relative_encoding
│   ├── attention.py      # GridSelfAttention, MSAAttention, SingleAttention
│   ├── ops.py            # TriangleMul, OuterProductMean
│   ├── evoformer.py      # EvoformerBlock, PairFormerBlock, Evoformer
│   ├── template_embedding.py  # TemplateEmbedding, SingleTemplateEmbedding, TemplatePairFormerBlock
│   ├── atom_cross_attn.py # AtomCrossAttEncoder/Decoder, EvoformerConditioningEncoder
│   ├── diffusion.py      # DiffusionTransformer, DiffusionHead (sampling + training)
│   ├── confidence.py     # ConfidenceHead (pLDDT, PAE, PDE, exp_resolved)
│   └── model.py          # DeepFold3 (top-level), DeepFold3Output, DistogramHead
├── data/            # Data pipeline
│   ├── const.py          # RES_TYPE_ORDER, ELEMENT_TO_INT, REF_ATOMS
│   ├── types.py          # Structure, MSA, Input, Tokenized, Record dataclasses
│   ├── ccd.py            # CcdMolecule, load_ccd_molecule(), smiles_to_molecule()
│   ├── msa/             # mmseqs2.py (server API + local binary MSA search + template m8 parsing)
│   ├── parse/            # mmcif.py, a3m.py, templates.py, hhr.py, json_input.py
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
│   ├── inference.py      # `deepfold3 inference`
│   ├── msa_search.py     # `deepfold3 msa`
│   ├── convert_weights.py  # `deepfold3 convert`
│   └── plot_confidences.py # `deepfold3 plot`
└── params.py        # read_af3_params() (zstd decompression)
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
