# Data Pipeline Walkthrough

This document traces a biomolecular input through every stage of the DeepFold3 data pipeline, from raw files to batched tensors ready for the model.

## Pipeline Overview

```text
Raw input (mmCIF / A3M / sequence string / JSON)
  -> Parse     ->  Structure + MSA objects
  -> Tokenize  ->  Tokenized (tokens[], bonds[], structure, msa)
  -> Crop      ->  Tokenized (<= max_tokens, <= max_atoms)
  -> Featurize ->  dict[str, Tensor]
  -> Collate   ->  padded batch tensors [B, ...]
```

During **inference**, cropping and random rotation are skipped. The `InferenceInput` static methods (`from_sequence`, `from_a3m`, `from_json`) handle the full pipeline in one call.

During **training**, all stages run: `TrainingDataset.__getitem__()` loads a Structure + MSA from disk, tokenizes, crops, and featurizes with random augmentation.

## Data Types (`data/types.py`)

All intermediate representations use numpy structured arrays with explicit dtypes. Access fields with dict syntax: `atoms["x"]`, not `atoms.x`.

### `ATOM_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `x` | float32 | x coordinate (Å) |
| `y` | float32 | y coordinate |
| `z` | float32 | z coordinate |
| `element` | int8 | Atomic number, clipped to [0, 127] |
| `charge` | float16 | Formal charge |
| `name` | S4 | Atom name (e.g. "CA", "C1'"), max 4 chars |
| `is_present` | bool | Resolved in experimental structure |

### `BOND_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `atom_1` | int32 | Global atom index |
| `atom_2` | int32 | Global atom index |
| `type` | int8 | Bond order: 1=single, 2=double, 3=triple, 4=aromatic |

### `RESIDUE_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name` | S5 | Residue name (3-letter code, e.g. "ALA", "DA") |
| `res_type` | int16 | Index in `RES_TYPE_ORDER` (0-30) |
| `mol_type` | int8 | `MOL_PROTEIN=0`, `MOL_DNA=1`, `MOL_RNA=2`, `MOL_NONPOLYMER=3` |
| `res_id` | int32 | PDB sequence number |
| `atom_idx` | int32 | Index of first atom in atoms array |
| `atom_num` | int16 | Number of atoms |
| `is_standard` | bool | Is a standard residue |
| `is_present` | bool | Has resolved coordinates |

### `CHAIN_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name` | S4 | PDB chain ID |
| `mol_type` | int8 | Molecule type |
| `entity_id` | int32 | Entity group (same sequence -> same entity) |
| `sym_id` | int32 | 0-based copy index within entity |
| `asym_id` | int32 | Unique sequential chain index |
| `res_idx` | int32 | Index of first residue |
| `res_num` | int32 | Number of residues |
| `atom_idx` | int32 | Index of first atom |
| `atom_num` | int32 | Total atoms in chain |

### `TOKEN_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `token_idx` | int32 | Sequential token index |
| `residue_index` | int32 | PDB residue number (polymer) or atom index (ligand) |
| `res_type` | int16 | Index in `RES_TYPE_ORDER` |
| `mol_type` | int8 | Molecule type |
| `asym_id` | int32 | Chain ID |
| `entity_id` | int32 | Entity ID |
| `sym_id` | int32 | Symmetry ID |
| `atom_idx` | int32 | Index in atoms array for this token's atoms |
| `atom_num` | int16 | Number of atoms belonging to this token |
| `center_x/y/z` | float32 | Representative atom coordinates (for spatial cropping) |

### `TOKEN_BOND_DTYPE`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `token_1` | int32 | Source token index |
| `token_2` | int32 | Target token index |

### Dataclasses

**`Structure`**: `atoms` (ATOM_DTYPE), `bonds` (BOND_DTYPE), `residues` (RESIDUE_DTYPE), `chains` (CHAIN_DTYPE). Has `save(path)` / `load(path)` for npz serialization.

**`MSA`**: `sequences` (int8 `[S, N]`), `mask` (bool `[S, N]`), `deletion_matrix` (int8 `[S, N]`). Has `save(path)` / `load(path)`.

**`Input`**: `structure: Structure`, `msa: dict[int, MSA]` (keyed by asym_id).

**`Tokenized`** (frozen): `tokens` (TOKEN_DTYPE `[N]`), `bonds` (TOKEN_BOND_DTYPE), `structure: Structure`, `msa: dict[int, MSA]`.

**`Record`**: `id: str`, `chains: list[ChainInfo]`, `interfaces: list[InterfaceInfo]` -- metadata for training dataset indexing.

## Parsing Stage

### mmCIF (`data/parse/mmcif.py`)

```python
def parse_mmcif(path: Path) -> Structure:
```

- Uses `gemmi.read_structure()` on the first model
- Maps entity types to `MOL_PROTEIN` / `MOL_DNA` / `MOL_RNA` / `MOL_NONPOLYMER`; skips water and unknown polymers
- Iterates subchains, collecting atoms (skipping H/D), residues, and chains
- Returns a `Structure` with four numpy arrays

### A3M (`data/parse/a3m.py`)

```python
def parse_a3m(text: str, mol_type: int = const.MOL_PROTEIN) -> MSA:
```

- Parses A3M text format: uppercase = match states, lowercase = insertions
- Lowercase characters increment a deletion counter but are not added as match states
- Returns:
  - `sequences [S, N]`: int8, using `PROTEIN_MSA_ID` / `RNA_MSA_ID` / `DNA_MSA_ID` mapping
  - `mask [S, N]`: bool, **always True at valid positions** -- gaps (idx 21) are valid aligned positions, only padding beyond sequence length is False
  - `deletion_matrix [S, N]`: int8, raw insertion counts per position

### Templates (`data/parse/templates.py`)

```python
def extract_template_features(mmcif_text, query_residue_index, ...) -> TemplateFeatures:
def extract_template_features_from_hhr(mmcif_text, hit, query_length, ...) -> TemplateFeatures:
def parse_template_mmcifs(template_dir, query_residue_index, max_templates=4) -> list[TemplateFeatures]:
def parse_hhr_templates(hhr_path, query_length, max_templates=4, cache_dir=None) -> list[TemplateFeatures]:
def fetch_pdb_mmcif(pdb_id, cache_dir=None) -> str:
```

- `TemplateFeatures` dataclass: `aatype [N]` int64, `atom_positions [N, 24, 3]` float32, `atom_mask [N, 24]` bool
- Two alignment modes:
  - **Residue-index matching**: `extract_template_features()` aligns template to query by residue_index values (for mmCIF directory input)
  - **HHR alignment**: `extract_template_features_from_hhr()` uses HHR query-to-hit mapping for alignment (for HHsearch output)
- `parse_hhr_templates()` reads an HHR file, selects top hits by `sum_probs`, downloads structures from RCSB, and extracts features
- Unaligned positions are filled with zeros and aatype=20 (UNK)
- `fetch_pdb_mmcif()` downloads from RCSB (`https://files.rcsb.org/download/{pdb_id}.cif`) with optional caching
- Template features are converted to pseudo-beta positions, backbone frames, and masks in the featurizer

### HHR (`data/parse/hhr.py`)

```python
def parse_hhr(hhr_text: str) -> list[TemplateHit]:
```

- `TemplateHit` frozen dataclass: `pdb_id`, `chain_id`, `aligned_cols`, `sum_probs`, `e_value`, `query_sequence`, `hit_sequence`, `indices_query`, `indices_hit`
- `query_to_hit_mapping` property: `{query_0based: hit_0based}` for non-gap aligned columns
- Parses HHsearch `.hhr` output: splits on `"No "` blocks, extracts Q/T alignment lines, builds 0-based index arrays

### JSON Input (`data/parse/json_input.py`)

```python
def parse_json_input(json_path: Path) -> list[JobInput]:
```

- Parses Protenix-format JSON specifying protein chains with MSA and template paths
- `ProteinChainInput`: `sequence`, `count`, `paired_msa_path`, `unpaired_msa_path`, `templates_path`
- `JobInput`: `name`, `protein_chains`, `model_seeds`
- Resolves relative paths against the JSON file's directory
- Currently supports single-chain protein only (`count=1`)

## Tokenization (`data/tokenize/tokenizer.py`)

```python
class Tokenizer:
    def tokenize(self, inp: Input) -> Tokenized:
```

**Polymer tokens** (protein, RNA, DNA): One token per residue.

- `residue_index` = PDB sequence number (`res_id`)
- `atom_num` = number of heavy atoms in that residue
- Center coordinates from representative atom: `CA` for protein, `C1'` for nucleic acids
- Consecutive residues in the same chain get a backbone bond

**Ligand tokens**: One token per heavy atom.

- `residue_index` = atom index within the ligand (0-based)
- `res_type` = 20 (UNK)
- `atom_num` = 1
- Center coordinates = atom's own x/y/z
- Intra-ligand bonds from `structure.bonds` mapped to token-level bonds

## Cropping (`data/crop/cropper.py`)

```python
class Cropper:
    def __init__(self, max_neighborhood: int = 40, spatial_prob: float = 0.5):
    def crop(self, data: Tokenized, max_tokens: int, max_atoms: int, ...) -> Tokenized:
```

Two strategies (50/50 random selection by default):

**Contiguous**: Pick a random chain, pick a random start position, accumulate tokens forward until `max_tokens` or `max_atoms` is reached.

**Spatial** (requires `scipy.spatial.cKDTree`): Build 3D point cloud from token center coordinates, pick a random seed token, query nearest neighbors, accumulate in distance order. Falls back to contiguous if scipy is unavailable.

After cropping, `_reindex()` rebuilds all token indices, filters bonds to those with both endpoints in the crop, slices atoms, and trims MSA columns per chain.

## Featurization (`data/feature/featurizer.py`)

```python
class Featurizer:
    def __init__(self, max_atoms: int = 24, max_msa_seqs: int = 2048):
    def process(self, data: Tokenized, training: bool = True, random_rot: bool = True, ...) -> dict[str, Tensor]:
```

### Complete Feature Table

N = number of tokens, M = MAX_ATOMS_PER_TOKEN (24), S = number of MSA sequences.

| Feature | Shape | Dtype | Description |
| ------- | ----- | ----- | ----------- |
| `res_type` | `[N]` | int64 | Token type (0-19=AA, 20=UNK, 21=gap, 22-25=RNA, 26-29=DNA, 30=N) |
| `residue_index` | `[N]` | int64 | Residue number (1-based for polymer) |
| `token_index` | `[N]` | int64 | Sequential token index (0-based) |
| `asym_id` | `[N]` | int64 | Chain ID |
| `entity_id` | `[N]` | int64 | Entity ID |
| `sym_id` | `[N]` | int64 | Symmetry ID |
| `seq_mask` | `[N]` | bool | Valid token mask |
| `pair_mask` | `[N, N]` | bool | `seq_mask[:, None] & seq_mask[None, :]` |
| `ref_pos` | `[N, M, 3]` | float32 | Reference atom positions (CCD ideal coords) |
| `ref_mask` | `[N, M]` | bool | Valid atom mask |
| `ref_element` | `[N, M]` | int64 | Atomic number (0-127 via `ELEMENT_TO_INT`) |
| `ref_charge` | `[N, M]` | float32 | Formal charge |
| `ref_atom_name_chars` | `[N, M, 4]` | int64 | Atom name encoding: `ord(c)-32` for each char of `name.ljust(4)[:4]` |
| `msa` | `[S, N]` | int64 | MSA sequences (same vocab as `res_type` for overlap) |
| `msa_mask` | `[S, N]` | bool | MSA validity mask (gaps are valid; only padding is False) |
| `deletion_matrix` | `[S, N]` | int64 | Raw deletion counts (number of insertions before each match state) |
| `profile` | `[N, 31]` | float32 | Per-position residue frequency (includes gap class at idx 21) |
| `deletion_mean` | `[N]` | float32 | `mean(deletion_matrix, dim=0)` -- raw mean, arctan applied later in model |
| `template_aatype` | `[T, N]` | int64 | Template residue types (same vocab as `res_type`) |
| `template_pseudo_beta` | `[T, N, 3]` | float32 | Template CB positions (CA for GLY) |
| `template_pseudo_beta_mask` | `[T, N]` | bool | Valid pseudo-beta mask |
| `template_backbone_frame` | `[T, N, 3, 3]` | float32 | Template backbone frames (Gram-Schmidt from N/CA/C) |
| `template_backbone_mask` | `[T, N]` | bool | Valid backbone frame mask |

### Reference Atom Augmentation

When `random_rot=True` (training), each token's `ref_pos` is:

1. Centered on the mean of valid atoms
2. Rotated by a uniformly random SO(3) rotation (QR decomposition of N(0,1) matrix)
3. Translated by a random N(0,1) vector

This matches AF3's `random_augmentation()` and is skipped during inference.

## Reference Atoms (`data/feature/ref_atoms.py`)

```python
def get_ref_atoms(res_name: str, mol_type: int, max_atoms: int = 24, ...) -> RefAtomData:
```

CCD lookup flow:

1. Try `_load_ccd_entry_gemmi(res_name)` -- downloads from RCSB, parses ideal coordinates, skips H atoms and OXT
2. If unavailable, fall back to hardcoded `const.REF_ATOMS[res_name]` + `const.REF_ATOM_POSITIONS[res_name]`
3. Pad/truncate to `max_atoms`, encode atom names as `[ord(c)-32 for c in name.ljust(4)[:4]]`

Returns `RefAtomData`: `positions [M, 3]`, `mask [M]`, `element [M]`, `charge [M]`, `atom_name_chars [M, 4]`.

## Inference Entry Points (`data/module/dataset.py`)

### `InferenceInput.from_sequence(sequence, mol_type="protein")`

- Builds a `Structure` from a 1-letter sequence string using CCD ideal coordinates
- Creates a single-row MSA `[1, N]` with all-True mask
- Tokenizes and featurizes with `training=False, random_rot=False`

### `InferenceInput.from_a3m(a3m_text, mol_type="protein", max_msa_seqs=2048, templates=None)`

- Parses full A3M text via `parse_a3m()`
- Reconstructs query sequence from MSA row 0 (skipping gap indices)
- Builds a `Structure` from the query using CCD ideal coordinates
- Tokenizes and featurizes with `training=False, random_rot=False`
- Optional `templates: list[TemplateFeatures]` -- if provided, template features (aatype, pseudo-beta, backbone frames) are computed and included in output

### `InferenceInput.from_json(json_path, job_index=0, max_msa_seqs=2048, max_templates=4, template_cache_dir=None)`

- Parses Protenix-format JSON via `parse_json_input()`
- Loads unpaired MSA (preferred) or paired MSA via `parse_a3m()`; falls back to single-sequence MSA
- If templates path is `.hhr`: calls `parse_hhr_templates()` to download and extract template features
- Builds Structure from query sequence, tokenizes and featurizes
- Currently supports single-chain protein only

All methods accept `mol_type` as `"protein"`, `"rna"`, or `"dna"` (case-insensitive).

### `TrainingDataset`

```python
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, records, data_dir, max_tokens=384, max_atoms=2048, max_msa_seqs=2048):
```

- Loads `{data_dir}/structures/{record.id}.npz` and `{data_dir}/msa/{chain.msa_id}.npz`
- Runs the full pipeline: tokenize -> crop -> featurize (with random rotation)

## Batch Collation (`data/module/collate.py`)

```python
def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
```

- For each feature key, computes the per-dimension max shape across the batch
- Pads each tensor to the max shape with zeros (right-padding)
- Stacks along a new batch dimension 0
- Returns `dict[str, Tensor]` with shapes `[B, ...]`

## Key Constants (`data/const.py`)

**Residue type vocabulary** (`RES_TYPE_ORDER`, 31 classes):

| Index | Type |
| ----- | ---- |
| 0-19 | 20 standard amino acids (ALA, ARG, ASN, ..., VAL) |
| 20 | UNK (unknown protein) |
| 21 | `-` (gap) |
| 22-25 | RNA (A, G, C, U) |
| 26-29 | DNA (DA, DG, DC, DT) |
| 30 | N (unknown nucleic) |

**MSA vocabulary** (`MSA_VOCAB_SIZE = 32`): Same indices as `RES_TYPE_ORDER` for overlapping types. Ambiguous amino acid codes mapped to nearest standard (e.g., B->ASP, Z->GLU, U->CYS).

**Other constants**: `MAX_ATOMS_PER_TOKEN = 24`, `NUM_ELEMENTS = 128`, `MOL_PROTEIN = 0`, `MOL_DNA = 1`, `MOL_RNA = 2`, `MOL_NONPOLYMER = 3`.
