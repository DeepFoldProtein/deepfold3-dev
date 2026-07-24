# DeepFold3

PyTorch implementation of biomolecular complex structure prediction, architecturally equivalent to [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w).

## Features

- **Diffusion-based structure prediction** -- Evoformer trunk (MSA + PairFormer) with 200-step denoising diffusion head
- **Built-in MSA search** -- `deepfold3 msa` via ColabFold MMseqs2 server or local binary, with template search and multimer pairing
- **Template embedding** -- structural templates from PDB via mmCIF directory, HHsearch `.hhr` files, or automatic server-side search
- **Full confidence metrics** -- pLDDT, PAE, PDE, pTM, ipTM, experimentally resolved, ranking score
- **AF3 weight compatibility** -- direct conversion from official AlphaFold 3 parameters
- **Flexible input** -- A3M MSA files, four JSON dialects (AlphaFold 3, AlphaFold Server, Protenix, DeepFold3)
- **mmCIF + JSON output** -- standard structure files with per-atom pLDDT and comprehensive confidence JSON
- **Training pipeline** -- native PyTorch DDP with EMA, LR scheduling, gradient accumulation, and checkpointing
- **Data preprocessing** -- `deepfold3 prepare` converts PDB mmCIF mirror to training data with structure-level and chain-level quality filtering
- **Sequence clustering** -- `deepfold3 cluster` via MMseqs2, or use pre-computed wwPDB clusters with `deepfold3 apply-clusters`

## Installation

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and install
git clone https://github.com/vv137/deepfold3-dev.git && cd deepfold3-dev
uv sync

# For development (includes ruff, pytest, jax for comparison tests)
uv sync --dev
```

## Quick Start

### 1. Convert weights

Convert official AF3 parameters to safetensors format. `--af3-ref` should point to a local clone of [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) (used to read Haiku scope names):

```bash
deepfold3 convert \
    --params resources/af3.bin.zst \
    --af3-ref references/alphafold3 \
    --output resources/deepfold3.safetensors
```

The output `.safetensors` embeds the architectural `ModelConfig` as JSON metadata so `deepfold3 inference` reconstructs the model with the exact flags it was trained on (default `ModelConfig()` for AF3-converted weights). Same goes for training-produced `.pt` checkpoints, which carry the dataclass directly. `--optim` at inference still overrides only the optimization sub-config (fused kernels, SDPA, …).

### 2. Search for MSAs

Search using the ColabFold MMseqs2 server (no local databases required):

```bash
# Monomer
deepfold3 msa --sequence MKTIIALSYIFCLVFA --output msa_out/ --json

# Multimer (colon-separated chains, duplicates auto-deduplicated)
deepfold3 msa --sequence AAAA:BBBB:AAAA --output msa_out/ --json
```

This produces A3M files, template hits (`pdb70.m8`), downloaded template structures, and a DeepFold3-format JSON ready for inference.

For local search with your own databases:

```bash
deepfold3 msa --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --json
```

### 3. Run inference

From the skeleton JSON produced by `deepfold3 msa`:

```bash
deepfold3 inference \
    --json msa_out/query.json \
    --output output/ \
    --params resources/deepfold3.safetensors \
    --num-samples 5
```

Or directly from an A3M MSA file:

```bash
deepfold3 inference \
    --msa data/query.a3m \
    --output output/prediction.cif \
    --params resources/deepfold3.safetensors
```

This produces:

- `prediction.cif` -- mmCIF with MODEL blocks per diffusion sample, pLDDT as B-factors
- `prediction_confidences.json` -- full confidence metrics (PAE, PDE, pTM, ipTM, contacts, etc.)

### 4. Plot confidence maps

```bash
deepfold3 plot --input output/prediction_confidences.json --output output/confidences.png
```

`--input` also takes the raw logit `.npz` files directly, or the directory holding them —
the companions (confidences JSON, per-seed distogram) are discovered around whatever is passed:

```bash
deepfold3 plot --input output/seed-42_sample-0/            # whole sample directory
deepfold3 plot --input output/seed-42_sample-0/logits.npz  # PAE + PDE (+ distogram beside it)
deepfold3 plot --input output/seed-42_distogram.npz        # trunk distogram only
deepfold3 plot --input output/prediction_logits.npz --panels pae,contact --contact-threshold 12
```

## CLI Reference

```
deepfold3 --help
deepfold3 msa --help
deepfold3 inference --help
deepfold3 convert --help
deepfold3 plot --help
deepfold3 prepare --help
deepfold3 cluster --help
deepfold3 apply-clusters --help
deepfold3 train --help
```

| Command                    | Description                                                                       |
| -------------------------- | --------------------------------------------------------------------------------- |
| `deepfold3 msa`            | Search MSAs via MMseqs2 (server or local), with template hits                     |
| `deepfold3 inference`      | Run structure prediction from A3M or JSON input                                   |
| `deepfold3 convert`        | Convert AF3 Haiku parameters to safetensors                                       |
| `deepfold3 plot`           | Plot PAE/PDE/distogram/contact maps from a confidences JSON, logit `.npz`, or dir |
| `deepfold3 prepare`        | Preprocess PDB mmCIF files into training data                                     |
| `deepfold3 cluster`        | Cluster sequences with MMseqs2 (builds cluster file from training data)           |
| `deepfold3 apply-clusters` | Apply a cluster file to a manifest for cluster-balanced sampling                  |
| `deepfold3 msa-batch`      | Generate MSAs for all unique sequences in training data (local MMseqs2)           |
| `deepfold3 import-msa`     | Import pre-existing A3M files (e.g. AFDB) into training data by sequence matching |
| `deepfold3 import-paired-msa`   | Import pre-existing paired MSA `.npz` files into training data                          |
| `deepfold3 import-of3`     | Import OpenFold-3 distillation structures into training data                      |
| `deepfold3 prepare-templates` | Discover multimer templates from per-chain MMseqs2 m8 hits (Hungarian pairing) |
| `deepfold3 scan-mmcif`     | Build a parallel mmCIF metadata cache (resolution / method / release date)        |
| `deepfold3 enrich-manifest` | Backfill structure-info fields on an existing manifest from idx/mmcif cache       |
| `deepfold3 drop-obsolete`  | Drop manifest entries flagged obsolete in the wwPDB idx                           |
| `deepfold3 warmup-triton`  | Pre-warm the persistent Triton autotune cache for the configured GPU              |
| `deepfold3 train`          | Train the model (single GPU or multi-GPU via torchrun)                            |

## JSON Input Dialects

DeepFold3 auto-detects and supports four JSON input formats:

| Dialect               | Key style                                        | MSA                               | Templates                    | Auto-search   |
| --------------------- | ------------------------------------------------ | --------------------------------- | ---------------------------- | ------------- |
| **`alphafold3`**      | `protein`/`dna`/`rna`/`ligand` with `id`         | inline or path                    | inline `templates` list      | No            |
| **`alphafoldserver`** | `proteinChain`/`rnaSequence`/`dnaSequence`/`ion` | --                                | --                           | Yes (MMseqs2) |
| **`protenix`**        | `proteinChain`/`dnaSequence`/`rnaSequence`       | `msa.precomputed_msa_dir` or path | `templatesPath` (.json/.hhr) | No            |
| **`deepfold3`**       | Both AF3 and Protenix keys                       | All formats                       | All formats                  | No            |

- **alphafold3** -- Faithful to `references/alphafold3`. Set `"dialect": "alphafold3"`.
- **alphafoldserver** -- Faithful to AF3 server format. Set `"dialect": "alphafoldserver"`. MSAs and templates are searched automatically via ColabFold MMseqs2 server.
- **protenix** -- Faithful to `references/Protenix`. Auto-detected from `proteinChain` keys with MSA fields.
- **deepfold3** -- Our superset. Set `"dialect": "deepfold3"`. Accepts all key styles and MSA/template formats from both AF3 and Protenix.

## Architecture

```text
Input features (tokens + atoms)
    -> Per-atom conditioning (AtomCrossAttEncoder)
    -> Evoformer trunk  [10 recycles]
        |-- Template embedding (optional, T templates -> pair)
        |-- MSA processing (EvoformerBlock x 4)
        +-- PairFormer stack (48 layers)
               -> single [N, 384]  +  pair [N, N, 128]
    -> Diffusion Head  (200-step denoising)
        +-- atom_positions [S, N, M, 3]
    -> Confidence Head (pLDDT, PAE, PDE, exp_resolved)
    -> Distogram Head
```

Key dimensions: `seq_channel=384`, `pair_channel=128`, `msa_channel=64`, `diffusion_steps=200`.

## TODO

- [x] Sequence clustering (`deepfold3 cluster` via MMseqs2, or pre-computed wwPDB clusters via `deepfold3 apply-clusters`)
- [x] MSA generation for training (`deepfold3 msa-batch` -- local MMseqs2 with GPU support, resume, batch splitting)
- [x] Training-time template pipeline (`deepfold3 prepare-templates` -- multimer template discovery from per-chain MMseqs2 hits with Hungarian chain pairing; date-cutoff filter applied at sample time)

## Confidence Metrics

| Metric   | Shape    | Description                                      |
| -------- | -------- | ------------------------------------------------ |
| pLDDT    | per-atom | Predicted local distance difference test (0-100) |
| PAE      | NxN      | Predicted aligned error (Å)                      |
| PDE      | NxN      | Predicted distance error (Å, symmetrised)        |
| pTM      | scalar   | Predicted TM-score                               |
| ipTM     | scalar   | Interface predicted TM-score (multi-chain only)  |
| Ranking  | scalar   | 0.8 x ipTM + 0.2 x pTM                           |
| gPDE     | scalar   | Global average PDE                               |
| Contacts | NxN      | P(distance < 8 Å) from distogram                 |

## Development

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Test
uv run pytest

# Single test
uv run pytest tests/test_weight_loading.py::TestWeightLoadingIntegrity::test_no_unexpected_keys
```

## Project Structure

```
src/deepfold3/
|-- model/              # Neural network modules
|   |-- primitives.py   # LinearNoBias, LayerNorm, SwiGLU, TransitionBlock
|   |-- config.py       # All model configuration dataclasses
|   |-- attention.py    # GridSelfAttention, MSAAttention, SingleAttention
|   |-- ops.py          # TriangleMul, OuterProductMean
|   |-- evoformer.py    # EvoformerBlock, PairFormerBlock, Evoformer
|   |-- template_embedding.py  # TemplateEmbedding, TemplatePairFormerBlock
|   |-- atom_cross_attn.py     # AtomCrossAtt encoder/decoder
|   |-- diffusion.py    # DiffusionTransformer, DiffusionHead
|   |-- confidence.py   # ConfidenceHead (pLDDT, PAE, PDE, exp_resolved)
|   +-- model.py        # DeepFold3 (top-level), DistogramHead
|-- data/               # Data pipeline
|   |-- msa/            # MMseqs2 MSA search (server API + local binary)
|   |-- parse/          # mmcif.py, a3m.py, templates.py, template_realign.py, hhr.py, json_input.py
|   |-- tokenize/       # tokenizer.py (Input -> Tokenized)
|   |-- crop/           # cropper.py (contiguous + spatial + interface-seeded)
|   |-- sample/         # weighted.py (per-(chain|interface) cluster sampler)
|   |-- feature/        # featurizer.py, ref_atoms.py (CCD)
|   |-- symmetry/       # chain / atom / ligand permutation alignment (AF3 §4.2 / §4.3)
|   |-- prepare/        # structure.py, filters.py, manifest.py, clustering.py, template_search.py, materialize.py, entries_idx.py
|   +-- module/         # dataset.py (TrainingDataset, InferenceInput), collate.py
|-- training/           # Training pipeline
|   |-- loss.py         # All loss functions (diffusion MSE, smooth LDDT, distogram, confidence)
|   |-- fused_loss.py   # Chunked-linear fused CE for distogram / PAE / PDE
|   |-- ema.py          # Exponential Moving Average (constant decay, params-only)
|   |-- scheduler.py    # AF3 LR scheduler (warmup -> plateau -> decay) + cosine
|   |-- optim.py        # AdamW + WrappedOptimizer (NaN-skip + fp64 grad-norm)
|   |-- state.py        # TrainState + seed_everything (deterministic mode)
|   |-- data.py         # DataLoader assembly + worker RNG seeding
|   +-- trainer.py      # DDP trainer (training loop, checkpointing, eval)
|-- convert/af3.py      # Haiku -> PyTorch weight converter
|-- io/                 # Output: mmCIF + confidences JSON
+-- scripts/            # CLI: msa, inference, convert, plot, prepare, cluster, apply-clusters, msa-batch, import-msa, prepare-templates, scan-mmcif, enrich-manifest, drop-obsolete, warmup-triton, train
```

## References

- Abramson, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." _Nature_ (2024).
- Data pipeline patterns adapted from [Boltz](https://github.com/jwohlwend/boltz) (MIT License).
- Raw data processing patterns adapted from [Protenix](https://github.com/bytedance/Protenix) (Apache 2.0 License).

## Disclaimer

This is an independent reimplementation based on the published AlphaFold 3 paper.
It is not affiliated with or endorsed by Google DeepMind.
Model weights are not included in this repository and must be obtained separately.

## License

[MIT](LICENSE)
