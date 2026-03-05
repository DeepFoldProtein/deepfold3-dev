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

### Structure as Central Data Type

Structure is the canonical representation throughout the assembly lifecycle:

```text
mmCIF file  ──→  parse_mmcif()       ──→  Structure
JSON input  ──→  _build_complex_structure() ──→  Structure
Prediction  ──→  prediction_to_structure()  ──→  Structure  ──→  structure_to_mmcif()  ──→  mmCIF string
                                                            ──→  save_split()          ──→  coords.npz + meta.json
Templates   ──→  extract_template_features_from_structure() ──→  TemplateFeatures
```

**Split serialization** (`save_split` / `load_split`): separates coordinates (coords.npz: x, y, z, b_factor) from metadata (meta.json: name, chains, residues, bonds, atom metadata). Useful for large datasets where coordinate updates are frequent but metadata is static.

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
| `b_factor` | float32 | B-factor (experimental) or pLDDT (predicted) |

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

**`Structure`**: `atoms` (ATOM_DTYPE), `bonds` (BOND_DTYPE), `residues` (RESIDUE_DTYPE), `chains` (CHAIN_DTYPE), `name` (str). Has `save(path)` / `load(path)` for NPZ serialization, `save_split(directory)` / `load_split(directory)` for split serialization (coords.npz + meta.json). Structure is the central data type for the full assembly lifecycle: mmCIF → Structure, JSON → Structure, prediction → Structure, Structure → mmCIF.

**`MSA`**: `sequences` (int8 `[S, N]`), `mask` (bool `[S, N]`), `deletion_matrix` (int8 `[S, N]`). Has `save(path)` / `load(path)`.

**`Input`**: `structure: Structure`, `paired_msa: dict[int, MSA]`, `unpaired_msa: dict[int, MSA]` (both keyed by asym_id). Paired MSAs carry cross-chain co-evolutionary signal (same row = same species across chains). Use `Input.from_msa(structure, msa)` to treat all MSAs as unpaired (single-chain compat).

**`Tokenized`** (frozen): `tokens` (TOKEN_DTYPE `[N]`), `bonds` (TOKEN_BOND_DTYPE), `structure: Structure`, `paired_msa: dict[int, MSA]`, `unpaired_msa: dict[int, MSA]`.

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
def extract_template_features_from_structure(structure, query_to_template, query_length, ...) -> TemplateFeatures:
def parse_template_mmcifs(template_dir, query_residue_index, max_templates=4) -> list[TemplateFeatures]:
def parse_hhr_templates(hhr_path, query_length, max_templates=4, cache_dir=None) -> list[TemplateFeatures]:
def fetch_pdb_mmcif(pdb_id, cache_dir=None) -> str:
```

- `TemplateFeatures` dataclass: `aatype [N]` int64, `atom_positions [N, 24, 3]` float32, `atom_mask [N, 24]` bool
- Four alignment modes:
  - **Residue-index matching**: `extract_template_features()` aligns template to query by residue_index values (for mmCIF directory input)
  - **HHR alignment**: `extract_template_features_from_hhr()` uses HHR query-to-hit mapping for alignment (for HHsearch output)
  - **JSON template hits**: `parse_json_templates()` uses pre-computed query-to-template index mappings with local CIF files
  - **Structure-based**: `extract_template_features_from_structure()` extracts features directly from a Structure object with a query-to-template residue mapping
- `parse_hhr_templates()` reads an HHR file, selects top hits by `sum_probs`, downloads structures from RCSB, and extracts features
- `parse_json_templates()` reads a JSON file containing alignment mappings and local CIF template structures (see format below)
- Unaligned positions are filled with zeros and aatype=20 (UNK)
- `fetch_pdb_mmcif()` downloads from RCSB (`https://files.rcsb.org/download/{pdb_id}.cif`) with optional caching
- Template features are converted to pseudo-beta positions, backbone frames, and masks in the featurizer

**JSON template hits format** (produced by tools like Protenix/Boltz):

```json
[
  {
    "name": "template_hit_0.cif",
    "queryIndices": [0, 5, 10, ...],
    "templateIndices": [1, 6, 11, ...]
  }
]
```

- `name`: CIF filename in the same directory as the JSON file
- `queryIndices`: 0-based token positions in the full query complex
- `templateIndices`: residue label_seq_id numbers (mmCIF canonical numbering) in the template CIF
- JSON templates are whole-complex (aligned to all chains at once), unlike HHR which is per-chain
- Set `templatesPath` in the JSON input to point to the `*_query_to_hit.json` file

### HHR (`data/parse/hhr.py`)

```python
def parse_hhr(hhr_text: str) -> list[TemplateHit]:
```

- `TemplateHit` frozen dataclass: `pdb_id`, `chain_id`, `aligned_cols`, `sum_probs`, `e_value`, `query_sequence`, `hit_sequence`, `indices_query`, `indices_hit`
- `query_to_hit_mapping` property: `{query_0based: hit_0based}` for non-gap aligned columns
- Parses HHsearch `.hhr` output: splits on `"No "` blocks, extracts Q/T alignment lines, builds 0-based index arrays

### JSON Input (`data/parse/json_input.py`)

```python
def parse_json_input(json_path: Path, max_templates=4, template_cache_dir=None) -> list[JobInput]:
```

- Parses Protenix-format or AF3-format JSON specifying chains with MSA and template data
- All file I/O (reading A3M files, downloading template CIFs) happens at parse time
- `ProteinChainInput`: `sequence`, `count`, `paired_msa: MSA | None`, `unpaired_msa: MSA | None`, `template_hits: list[TemplateHitInput]`
- `NucleotideChainInput`: `sequence`, `count`, `mol_type: int`, `paired_msa: MSA | None`, `unpaired_msa: MSA | None`
- `TemplateHitInput` (in `templates.py`): `mmcif_text`, `query_indices | hhr_hit` — holds parsed template data ready for feature extraction
- `LigandChainInput`: `ccd_ids: list[str] | None`, `smiles: str | None`, `count: int` — exactly one of ccd_ids or smiles
- `JobInput`: `name`, `protein_chains`, `dna_chains`, `rna_chains`, `ligand_chains`, `model_seeds`
- Resolves relative paths against the JSON file's directory
- Supports multi-chain protein/DNA/RNA/ligand complexes (`count=1`)
- Protenix ligand format: `{"ligand": {"ligand": "CCD_ATP", "count": 1}}` (strip `CCD_` prefix)
- AF3 ligand format: `{"ligand": {"id": "B", "ccdCodes": ["ATP"]}}` or `{"ligand": {"id": "B", "smiles": "CCO"}}`

### CCD Molecule Builder (`data/ccd.py`)

```python
def load_ccd_molecule(ccd_code: str) -> CcdMolecule:
def smiles_to_molecule(smiles: str, res_name: str = "LIG") -> CcdMolecule:
```

- `CcdMolecule` dataclass: `atom_names`, `positions [n_atoms, 3]`, `elements`, `charges`, `bonds: list[(idx1, idx2, order)]`, `res_name`
- `load_ccd_molecule()` downloads CIF from RCSB, parses with gemmi, skips H atoms, extracts bonds from `_chem_comp_bond`
- `smiles_to_molecule()` uses RDKit (optional): `MolFromSmiles` -> AddHs -> ETKDG conformer -> MMFF optimize -> RemoveHs
- CCD files cached in `/tmp/deepfold3_ccd_cache/`; molecules cached in-memory per process
- Used by `InferenceInput.from_json()` for ligand chain assembly

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
| `mol_type` | `[N]` | int64 | Molecule type (0=protein, 1=DNA, 2=RNA, 3=non-polymer) |
| `token_res_name` | `[N, 5]` | int8 | Residue name as ASCII bytes (e.g. "ALA", "PCG"), 0-padded |
| `seq_mask` | `[N]` | bool | Valid token mask |
| `pair_mask` | `[N, N]` | bool | `seq_mask[:, None] & seq_mask[None, :]` |
| `frames_mask` | `[N]` | bool | True for polymer tokens with backbone frames (protein/RNA/DNA) |
| `token_to_rep_atom` | `[N]` | int64 | Representative atom index: CB for protein (CA for GLY), C1' for nucleic, 0 for ligand |
| `ref_pos` | `[N, M, 3]` | float32 | Reference atom positions (CCD ideal coords) |
| `ref_mask` | `[N, M]` | bool | Valid atom mask |
| `ref_element` | `[N, M]` | int64 | Atomic number (0-127 via `ELEMENT_TO_INT`) |
| `ref_charge` | `[N, M]` | float32 | Formal charge |
| `ref_atom_name_chars` | `[N, M, 4]` | int64 | Atom name encoding: `ord(c)-32` for each char of `name.ljust(4)[:4]` |
| `ref_space_uid` | `[N, M]` | int64 | Reference space group ID: token_idx for polymer, first token of ligand residue for ligand atoms |
| `msa` | `[S, N]` | int64 | MSA sequences: query row + paired rows + unpaired block-diagonal rows |
| `msa_mask` | `[S, N]` | bool | MSA validity mask (cross-chain gaps are valid; only padding is False) |
| `deletion_matrix` | `[S, N]` | int64 | Raw deletion counts (number of insertions before each match state) |
| `profile` | `[N, 31]` | float32 | Per-position residue frequency (includes gap class at idx 21) |
| `deletion_mean` | `[N]` | float32 | `mean(deletion_matrix, dim=0)` -- raw mean, arctan applied later in model |
| `template_aatype` | `[T, N]` | int64 | Template residue types (same vocab as `res_type`) |
| `template_pseudo_beta` | `[T, N, 3]` | float32 | Template CB positions (CA for GLY) |
| `template_pseudo_beta_mask` | `[T, N]` | bool | Valid pseudo-beta mask |
| `template_backbone_frame` | `[T, N, 3, 3]` | float32 | Template backbone frames (Gram-Schmidt from N/CA/C) |
| `template_backbone_mask` | `[T, N]` | bool | Valid backbone frame mask |

### Reference Atom Augmentation

When `random_rot=True` (training), `ref_pos` is augmented per `ref_space_uid` group:

1. All tokens sharing the same `ref_space_uid` are grouped (polymer: each token independent; ligand: all tokens of one residue share a UID)
2. Centered on the group mean of valid atoms
3. Rotated by a uniformly random SO(3) rotation (QR decomposition of N(0,1) matrix)
4. Translated by a random N(0,1) vector

This ensures ligand atoms within the same residue maintain their relative geometry while getting a shared random augmentation. Skipped during inference.

## Reference Atoms (`data/feature/ref_atoms.py`)

```python
def get_ref_atoms(res_name: str, mol_type: int, max_atoms: int = 24, ...) -> RefAtomData:
```

CCD lookup flow:

1. Try `_load_ccd_entry_gemmi(res_name)` -- downloads from RCSB, parses ideal coordinates, skips H atoms and OXT
2. If unavailable, fall back to hardcoded `const.REF_ATOMS[res_name]` + `const.REF_ATOM_POSITIONS[res_name]`
3. Pad/truncate to `max_atoms`, encode atom names as `[ord(c)-32 for c in name.ljust(4)[:4]]`

Returns `RefAtomData`: `positions [M, 3]`, `mask [M]`, `element [M]`, `charge [M]`, `atom_name_chars [M, 4]`.

**Ligand ref_pos**: For `MOL_NONPOLYMER` tokens, the featurizer looks up the CCD residue name from the structure and retrieves the atom's ideal position from the CCD entry. If CCD lookup fails (e.g. custom ligand from SMILES), it falls back to positions stored in the Structure atoms.

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

- Parses Protenix-format or AF3-format JSON via `parse_json_input()` — all file I/O (A3M reading, CIF downloading) happens in `parse_json_input()`
- MSA objects and template hits come pre-parsed from `JobInput` chain dataclasses — `from_json()` does only computation
- Falls back to single-sequence MSA when no MSA data is provided for a chain
- Converts `TemplateHitInput` objects to `TemplateFeatures` via `template_hit_input_to_features()` (no I/O)
- JSON-mode templates (query_indices populated): whole-complex alignment
- HHR-mode templates (hhr_hit populated): per-chain alignment
- Builds multi-chain Structure from query sequences, tokenizes and featurizes
- Supports multi-chain protein/DNA/RNA/ligand complexes
- Ligand chains: built from CCD codes via `load_ccd_molecule()` or SMILES via `smiles_to_molecule()`; no MSA or templates for ligands

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
