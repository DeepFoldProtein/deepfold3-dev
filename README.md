# DeepFold3

PyTorch implementation of biomolecular complex structure prediction, architecturally equivalent to [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w).

## Features

- **Diffusion-based structure prediction** -- Evoformer trunk (MSA + PairFormer) with 200-step denoising diffusion head
- **Template embedding** -- structural templates from PDB via mmCIF directory or HHsearch `.hhr` files
- **Full confidence metrics** -- pLDDT, PAE, PDE, pTM, ipTM, experimentally resolved, ranking score
- **AF3 weight compatibility** -- direct conversion from official AlphaFold 3 parameters
- **Flexible input** -- A3M MSA files, Protenix-format JSON with paired/unpaired MSAs and HHR templates
- **mmCIF + JSON output** -- standard structure files with per-atom pLDDT and comprehensive confidence JSON

## Installation

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

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

### 2. Run inference

From an A3M MSA file:

```bash
deepfold3 inference \
    --msa data/query.a3m \
    --output output/prediction.cif \
    --params resources/deepfold3.safetensors \
    --num-samples 5
```

Or from a Protenix-format JSON (with MSA paths and HHR templates):

```bash
deepfold3 inference \
    --json input.json \
    --output output/prediction.cif \
    --params resources/deepfold3.safetensors \
    --template-cache /tmp/template_cache
```

This produces:

- `prediction.cif` -- mmCIF with MODEL blocks per diffusion sample, pLDDT as B-factors
- `prediction_confidences.json` -- full confidence metrics (PAE, PDE, pTM, ipTM, contacts, etc.)

### 3. Plot confidence maps

```bash
deepfold3 plot \
    --input output/prediction_confidences.json \
    --output output/confidences.png
```

## CLI Reference

```
deepfold3 --help
deepfold3 inference --help
deepfold3 convert --help
deepfold3 plot --help
```

| Command | Description |
| ------- | ----------- |
| `deepfold3 inference` | Run structure prediction from A3M or JSON input |
| `deepfold3 convert` | Convert AF3 Haiku parameters to safetensors |
| `deepfold3 plot` | Plot PAE/PDE/contact maps from confidences JSON |

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

- [ ] Multi-chain complexes and ligand support
- [ ] Full data pipeline (PDB preprocessing, clustering, dataset generation)
- [ ] Training pipeline

## Confidence Metrics

| Metric | Shape | Description |
| ------ | ----- | ----------- |
| pLDDT | per-atom | Predicted local distance difference test (0-100) |
| PAE | NxN | Predicted aligned error (Å) |
| PDE | NxN | Predicted distance error (Å, symmetrised) |
| pTM | scalar | Predicted TM-score |
| ipTM | scalar | Interface predicted TM-score (multi-chain only) |
| Ranking | scalar | 0.8 x ipTM + 0.2 x pTM |
| gPDE | scalar | Global average PDE |
| Contacts | NxN | P(distance < 8 Å) from distogram |

## Development

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Test (121 tests)
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
|   |-- parse/          # mmcif.py, a3m.py, templates.py, hhr.py, json_input.py
|   |-- tokenize/       # tokenizer.py (Input -> Tokenized)
|   |-- crop/           # cropper.py (contiguous + spatial)
|   |-- feature/        # featurizer.py, ref_atoms.py (CCD)
|   +-- module/         # dataset.py (InferenceInput), collate.py
|-- convert/af3.py      # Haiku -> PyTorch weight converter
|-- io/                 # Output: mmCIF + confidences JSON
+-- scripts/            # CLI: inference, convert, plot
```

## References

- Abramson, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* (2024).
- Data pipeline patterns adapted from [Boltz](https://github.com/jwohlwend/boltz) (MIT License).
- Raw data processing patterns adapted from [Protenix](https://github.com/bytedance/Protenix) (Apache 2.0 License).

## Disclaimer

This is an independent reimplementation based on the published AlphaFold 3 paper.
It is not affiliated with or endorsed by Google DeepMind.
Model weights are not included in this repository and must be obtained separately.

## License

[MIT](LICENSE)
