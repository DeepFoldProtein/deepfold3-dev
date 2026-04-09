# Data Pipeline Walkthrough

This document traces a biomolecular input through every stage of the DeepFold3 data pipeline, from raw files to batched tensors ready for the model.

## Pipeline Overview

```text
Sequence(s)  ──→  MSA Search  ──→  A3M files
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

**`Tokenized`** (frozen): `tokens` (TOKEN_DTYPE `[N]`), `bonds` (TOKEN_BOND_DTYPE), `structure: Structure`, `paired_msa: dict[int, MSA]`, `unpaired_msa: dict[int, MSA]`, `token_mask: TokenMask | None`. Use `with_mask(mask)` to attach a mask (returns a new instance).

**`Record`**: `id: str`, `chains: list[ChainInfo]`, `interfaces: list[InterfaceInfo]`, `structure_info: StructureInfo | None` -- metadata for training dataset indexing. Has `to_dict()` / `from_dict()` for JSON serialization.

**`ChainInfo`**: `chain_id`, `chain_name`, `mol_type`, `cluster_id`, `msa_id`, `num_residues`, `entity_id`, `valid: bool = True`, `auth_chain_name: str | None`. Chains failing quality filters are marked `valid=False`. `auth_chain_name` stores the original chain name from the input CIF (before assembly renaming), used for mask generation.

**`TokenMask`** (`data/masking.py`): `channels: dict[str, np.ndarray]` — per-token mask channels for loss control. Reserved channels: `"fixed"` (float32, 1.0=experimental / 0.0=generated) and `"region"` (int16, semantic labels). Has `select(indices)` for cropping, `load(path)` for NPZ, and `fixed_from_patchr(json_path, tokens, chains, chain_infos)` for PATCHR ATLAS mask generation.

**`InterfaceInfo`**: `chain_1`, `chain_2`, `valid: bool = True`.

**`StructureInfo`**: `resolution`, `method`, `deposited`, `released`, `num_chains`, `num_interfaces` -- experimental metadata extracted during preprocessing.

## MSA Search (`data/msa/mmseqs2.py`)

The `deepfold3 msa` command searches for MSAs using MMseqs2, producing A3M files that feed into `deepfold3 inference --msa`.

### Server Mode (default)

```bash
# Monomer — searches ColabFold public API
deepfold3 msa --sequence MKTIIALSYIFCLVFA --output msa_out/

# Multimer — colon-separated chains (duplicates searched only once)
deepfold3 msa --sequence AAAA:BBBB:AAAA --output msa_out/

# Multimer from FASTA with pairing
deepfold3 msa --fasta complex.fasta --output msa_out/ --use-pairing

# With caching
deepfold3 msa --fasta query.fasta --output msa_out/ --cache-dir /tmp/msa_cache/

# With templates (default on) + skeleton JSON
deepfold3 msa --sequence MKTII... --output msa_out/ --json

# Templates: save pdb70.m8 but skip mmCIF download
deepfold3 msa --sequence MKTII... --output msa_out/ --no-fetch-templates

# Disable template search entirely
deepfold3 msa --sequence MKTII... --output msa_out/ --no-templates
```

Uses the ColabFold MMseqs2 server API (`https://api.colabfold.com`). Submits sequences, polls for completion, downloads result tar.gz, and extracts per-chain A3M files. Supports environmental DB (`--use-env`, default on) and cross-chain pairing (`--use-pairing`).

**Templates**: The server returns `pdb70.m8` with template hits. By default, `--use-templates` extracts these hits and downloads mmCIF structures from RCSB (or a local PDB mirror via `--pdb-mmcif-dir`). Template options:

- `--fetch-templates` (default) — download mmCIF files, write per-chain JSON template hits with `chainId`
- `--link-templates` — create symlinks to files in `--pdb-mmcif-dir` instead of copying (requires `--pdb-mmcif-dir`)
- `--no-fetch-templates` — save only `pdb70.m8`, skip downloading structures
- `--no-templates` — disable template extraction entirely

**Colon-separated multimer**: `--sequence SEQ1:SEQ2:SEQ1` splits on `:` and deduplicates — only unique sequences are searched, results are copied to all matching chains.

### Local Mode

```bash
# CPU search
deepfold3 msa --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/

# GPU-accelerated search
deepfold3 msa --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu

# GPU with dedicated server (lower per-request overhead)
deepfold3 msa --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --gpu-server

# With template search against PDB100
deepfold3 msa --fasta query.fasta --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --use-templates --json

# GPU + templates + env pairing
deepfold3 msa --sequence SEQ1:SEQ2 --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ \
    --gpu --use-templates --use-env-pairing --json
```

Requires a local `mmseqs` binary and pre-built databases:

- `{db_dir}/uniref30_2302_db` (required)
- `{db_dir}/colabfold_envdb_202108_db` (optional, for `--use-env`)
- `{db_dir}/pdb100_230517` (optional, for `--use-templates`)
- `{db_dir}/spire_ctg10_2401_db` (optional, for `--use-env-pairing`)

#### Database Setup

Download and prepare databases using `scripts/setup_databases.sh`:

```bash
# CPU-only databases
./scripts/setup_databases.sh /data/mmseqs_dbs/

# GPU+CPU compatible databases (recommended if you have a GPU)
GPU=1 ./scripts/setup_databases.sh /data/mmseqs_dbs/
```

The script downloads and indexes:

1. **UniRef30** (`uniref30_2302_db`) — primary sequence database (~100 GB)
2. **ColabFold EnvDB** (`colabfold_envdb_202108_db`) — environmental sequences (~200 GB)
3. **PDB100** (`pdb100_230517`) — PDB sequences for template search
4. **PDB100 FoldSeek** (`pdb100_foldseek_230517`) — structural template profiles
5. **SPIRE** (`spire_ctg10_2401_db`) — environmental pairing database

GPU databases use pre-built format (`FAST_PREBUILT_DATABASES=1`) that supports both GPU and CPU. GPU indexing uses `--split 1 --index-subset 2`.

**Manual database preparation** (without setup script):

```bash
# Download pre-built GPU+CPU databases
wget https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz
tar xzf uniref30_2302.db.tar.gz
mmseqs createindex uniref30_2302_db tmp --remove-tmp-files 1 --split 1 --index-subset 2  # GPU
# For CPU-only, omit --split and --index-subset

# PDB100 for templates (GPU-compatible)
wget https://opendata.mmseqs.org/colabfold/pdb100_230517.fasta.gz
mmseqs createdb pdb100_230517.fasta.gz pdb100_230517_tmp
mmseqs makepaddedseqdb pdb100_230517_tmp pdb100_230517  # GPU-compatible
mmseqs rmdb pdb100_230517_tmp
mmseqs createindex pdb100_230517 tmp --remove-tmp-files 1 --split 1 --index-subset 2
```

#### GPU Search

GPU acceleration requires MMseqs2 release 16+ with GPU support. Key constraints:

- GPU uses ungapped prefilter mode (`--prefilter-mode 1`) exclusively
- Sensitivity parameter (`-s`) cannot be set — GPU runs at maximum sensitivity automatically
- Control which GPUs via `CUDA_VISIBLE_DEVICES` environment variable

**GPU server mode** (`--gpu-server`) reduces per-request overhead by ~1.5s. Requires pre-starting `mmseqs gpuserver` processes:

```bash
# Start GPU servers for each database (run in background)
mmseqs gpuserver /data/mmseqs_dbs/uniref30_2302_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
mmseqs gpuserver /data/mmseqs_dbs/colabfold_envdb_202108_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &

# Then run searches with --gpu --gpu-server
deepfold3 msa --sequence MKTII... --output msa_out/ --mode local --db-dir /data/mmseqs_dbs/ --gpu --gpu-server
```

#### Search Pipeline

Pipeline follows the [ColabFold search](https://github.com/soedinglab/MMseqs2/blob/master/data/workflow/colabfold_search.sh) workflow:

```
createdb → search (UniRef30, 3 iterations, -s 8.0 or GPU max sensitivity)
         → expandaln (-e inf, --expand-filter-clusters 1, --max-seq-id 0.95)
         → mvdb (extract profile from tmp/latest/profile_1)
         → lndb (link query headers to profile)
         → align (profile vs expanded hits, -e 10, --max-accept 100000, --alt-ali 10)
         → filterresult (--qsc 0.8, --filter-min-enable 100)
         → result2msa (--filter-msa 1, --diff 3000, --filter-min-enable 1000, --max-seq-id 0.95)
         → [env] search (profile vs envdb, same params)
         → [env] expandaln → align → filterresult → result2msa
         → [templates] search (profile vs pdb100, -s 7.5 or GPU)
         → [templates] convertalis → m8 format
         → mergedbs (uniref + env) → unpackdb
         → [pairing] search → expandaln → align → pairaln → align → pairaln → result2msa
```

Search parameters (filter=True): `SEARCH_PARAM="--num-iterations 3 --db-load-mode 2 -a -s 8.0 -e 0.1 --max-seqs 10000"`. With GPU: `--gpu 1 --prefilter-mode 1` replaces `-s 8.0`. Template search uses `-s 7.5` (CPU) or `--gpu 1` (GPU). For pairing: additional `pairaln` + `result2msa` with `msa-format-mode 5`.

**ColabFold DB structure**: the database has four components per DB — `_db` (cluster reps), `_db_seq` (member sequences), `_db_aln` (rep→member alignment), `_db.idx` (precomputed index). `search` targets cluster reps; `expandaln` uses `_db_seq` + `_db_aln` to expand cluster rep hits to member sequences; `align`/`result2msa` use `_db_seq` (or `_db.idx` for indexed mode) for final member-level alignment.

### Output

- `{output}/unpaired_0.a3m`, `unpaired_1.a3m`, ... (one per chain)
- `{output}/paired_0.a3m`, `paired_1.a3m`, ... (if pairing enabled)
- `{output}/pdb70.m8` — template hits in BLAST tabular format (if `--use-templates`)
- `{output}/templates/` — downloaded mmCIF files + per-chain JSON hits (if `--fetch-templates`)
- `{output}/{name}.json` — AF3-format skeleton JSON for `deepfold3 inference --json` (if `--json`)

The skeleton JSON uses AlphaFold 3 dialect (`"dialect": "alphafold3"`, `"version": 2`) with chain IDs (A, B, ...), relative paths to A3M files, and `templatesPath` references to per-chain JSON template hits (with `chainId` for selecting the correct chain from each mmCIF).

### API

```python
from deepfold3.data.msa import search_server, search_local, MsaResult, parse_m8

result = search_server(["MKTII..."], use_env=True)
# result.unpaired[0] → A3M text for chain 0
# result.m8_text → raw pdb70.m8 content (template hits)

result = search_local(["SEQ1", "SEQ2"], db_dir="/data/dbs/", use_pairing=True)
# result.paired[0], result.paired[1] → paired A3M text

# GPU-accelerated local search with templates
result = search_local(["MKTII..."], db_dir="/data/dbs/", gpu=True, use_templates=True)
# result.m8_text → template hits in BLAST tabular format

# Parse template hits from m8
hits = parse_m8(result.m8_text)
# hits[0] → list of TemplateM8Hit for chain 0
```

### Paired MSA (Cross-Chain Co-Evolution)

For multimer complexes, paired MSAs capture **cross-chain co-evolutionary signal**: if chain A and chain B interact, homologs of A and B from the same species are placed on the same MSA row. This allows the Evoformer to learn inter-chain contacts from evolutionary covariance.

#### What Paired MSA Is

A standard (unpaired) MSA is per-chain: each chain has its own `[S, L]` alignment. In a multimer, these are independent — row 3 in chain A's MSA has no relationship to row 3 in chain B's MSA.

A **paired** MSA aligns sequences **across chains by species**: if organism X has homologs of both chain A and chain B, those homologs share the same MSA row. This creates cross-chain covariance signal that the model uses to predict inter-chain contacts.

#### How Pairing Works

The pairing algorithm relies on **MMseqs2's `pairaln` command**, which matches sequences across chains by their taxonomic origin:

```
1. SEARCH:     Each chain searched against UniRef30 (3-iteration profile search)
2. EXPAND:     Cluster reps expanded to member sequences
3. REALIGN:    Full profile alignment on expanded hits
4. PAIRALN:    Match hits across chains by species/taxon ID
               - pairing-mode 0 (greedy): faster, partial coverage
               - pairing-mode 1 (complete): thorough, all pairings
5. BACKTRACE:  Re-align paired hits
6. PAIRALN:    Second pass with dummy mode (fills missing chains)
7. RESULT2MSA: Convert to per-chain A3M files (msa-format-mode 5)
```

Each chain gets its own paired A3M file. Row *i* in chain A's paired A3M and row *i* in chain B's paired A3M correspond to the same species. This is the cross-chain alignment.

For server mode (ColabFold API), the server runs `pairgreedy` or `paircomplete` modes internally and returns paired tar.gz alongside unpaired results.

#### Storage

Paired and unpaired MSAs are stored as **separate files**:

```
data_dir/msa/
├── {msa_id}.npz              # Unpaired MSA (keyed by SHA256 of chain sequence)
├── {msa_id}.a3m              # Unpaired A3M source
├── {record_id}_paired_0.npz  # Paired MSA for chain 0 of this complex
├── {record_id}_paired_0.a3m  # Paired A3M source
├── {record_id}_paired_1.npz  # Paired MSA for chain 1
└── ...
```

The manifest tracks both via `ChainInfo`:

```python
ChainInfo(
    msa_id="abc123...",            # SHA256 hash → unpaired MSA file
    paired_msa_id="1pfe_paired_0", # Complex-specific → paired MSA file
)
```

Unpaired MSAs are deduplicated by sequence (shared across records with identical chain sequences). Paired MSAs are per-record because the pairing depends on which chains co-occur in the same complex.

#### Full-Width MSA Construction (Featurizer)

The featurizer merges per-chain paired and unpaired MSAs into a single **full-width** `[S, N]` tensor where `N` spans all tokens across all chains:

```
Row 0:         Query (from paired row 0 if available, else unpaired row 0 per chain)
Rows 1..P:     Paired rows — cross-chain co-evolutionary signal
               For each row i: chain A's paired row i fills columns [A_start:A_end],
                               chain B's paired row i fills columns [B_start:B_end],
                               other columns = gap (21)
Rows P+1..S:   Dense unpaired rows — per-chain independent signal
               For each row i: ALL chains' unpaired row i placed simultaneously
               (NOT block-diagonal — same row has each chain's i-th sequence)
               Chains with fewer unpaired rows have gap-filled trailing rows.
```

Cross-chain gap positions have `msa_mask=True` (AF3 treats alignment gaps as valid). Total rows capped at `MAX_MSA_ROWS` (16,384). Paired rows capped at `MAX_PAIRED_MSA_ROWS + 1` (8,192 including query).

#### Generating Paired MSAs

**Inference** (single complex):

```bash
# Auto-enables pairing for multimers
deepfold3 msa --sequence SEQ1:SEQ2 --output msa_out/ --json

# Explicit pairing strategy
deepfold3 msa --sequence SEQ1:SEQ2 --output msa_out/ --pairing-strategy complete

# Local mode with environmental pairing
deepfold3 msa --sequence SEQ1:SEQ2 --output msa_out/ \
    --mode local --db-dir /data/mmseqs_dbs/ --use-env-pairing
```

**Training** (batch, all multimer records):

```bash
deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ \
    --use-pairing --gpu

# Resume-safe: skips records where all paired NPZ files already exist
# Updates manifest.json with paired_msa_id references
```

The batch command identifies all multimer records (>1 valid protein chain), runs the paired search for each, and writes per-chain paired A3M + NPZ files to `{data_dir}/msa/`.

#### Comparison with Other Implementations

| Aspect | DeepFold3 | OpenFold 3 | Protenix |
|--------|-----------|------------|----------|
| Pairing source | MMseqs2 `pairaln` (species-based) | Species-based alignment (custom) | Species-based alignment (custom) |
| Storage | Separate paired/unpaired NPZ per chain | Single `MsaArrayCollection` (parsed/processed dicts) | Feature dict with `_all_seq` suffix |
| Row layout | Paired (full-width) + unpaired (dense) | Query + paired + main (per-chain vstacked) | Paired (`_all_seq`) + unpaired (concatenated) |
| Paired row count | `min(counts)` across chains, cap 8192 | `n_rows_paired_subsampled + 1` (includes query) | `prot_paired_num_alignments` (excludes query) |
| Unpaired layout | Dense: row *i* has ALL chains' row *i* | Per-chain independent depth, padded | Padded to equal depth, then concat |
| Partial pairing | Supported (missing chain = full gap row) | Configurable `min_chains_paired_partial` | All ≥2 chain species paired |
| Subsampling | Global cap (`max_msa_seqs`) | Global `max_rows` split between paired/main | `max_size // 2` for paired, rest for unpaired |
| Deduplication | Not explicit | Removes main rows present in paired | Removes unpaired rows present in paired |
| Species sorting | MMseqs2 internal | Row-product ranking | Log-product ranking |

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
def parse_hhr_templates(hhr_path, query_length, max_templates=4, cache_dir=None, pdb_mmcif_dir=None) -> list[TemplateFeatures]:
def fetch_pdb_mmcif(pdb_id, cache_dir=None, pdb_mmcif_dir=None) -> str:
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
- `fetch_pdb_mmcif()` resolves template structures in order: (1) local PDB mirror `pdb_mmcif_dir` (layout: `{dir}/{mid2}/{id}.cif.gz`, supports gzip), (2) `cache_dir`, (3) RCSB download. CLI: `--pdb-mmcif-dir`
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

- Auto-detects and parses four JSON dialects: `alphafold3`, `alphafoldserver`, `protenix`, `deepfold3`
- **alphafold3**: faithful to AF3 — `protein`/`dna`/`rna`/`ligand` keys with `id`, inline `templates` or paths
- **alphafoldserver**: faithful to AF3 server — auto runs MMseqs2 MSA search + template lookup
- **protenix**: faithful to Protenix — `msa.precomputed_msa_dir` or `pairedMsaPath`/`unpairedMsaPath`
- **deepfold3**: our superset accepting both AF3 and Protenix key styles
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

**Spatial** (requires `scipy.spatial.KDTree`): Build 3D point cloud from token center coordinates, pick a random seed token, query nearest neighbors, accumulate in distance order. Falls back to contiguous if scipy is unavailable.

After cropping, `_reindex()` rebuilds all token indices, filters bonds to those with both endpoints in the crop, slices atoms, and crops MSA columns per chain using vectorized column indexing (`np.isin` to map selected tokens back to per-chain MSA positions).

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
| `mask_fixed` | `[N]` | float32 | *(optional)* Token-level fixed mask: 1.0=experimental, 0.0=model-generated (e.g. PATCHR ATLAS). Scales `exp_resolved` loss target. |
| `mask_region` | `[N]` | int16 | *(optional)* Region label (0=unset, 1+=app-defined, e.g. CDR-H1/H2/H3). Drives per-token `region_loss_weights` in diffusion MSE. |

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

- Parses JSON input via `parse_json_input()` (supports all four dialects) — all file I/O happens in `parse_json_input()`
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

**Shared helpers** (public API in `data/const.py`):

- `element_name_to_int(element_name)` — element symbol → atomic number, clamped to [0, 127]
- `get_res_type(res_name, mol_type)` — residue name → type index (returns 20/UNK for non-polymer)

**Shared helpers** (public API in `data/parse/mmcif.py`):

- `mol_type_from_gemmi(entity)` — maps `gemmi.Entity` type to `MOL_PROTEIN`/`MOL_DNA`/`MOL_RNA`/`MOL_NONPOLYMER`

**Shared helpers** (public API in `data/parse/templates.py`):

- `find_pdb_mmcif_path(pdb_id, pdb_mmcif_dir)` — resolves PDB ID to file path in a local PDB mirror (checks `{mid2}/{id}.cif.gz`, `.cif`, flat layout)

## Data Preprocessing (`data/prepare/`)

Converts raw PDB mmCIF files into training-ready Structure NPZ files with quality filtering and manifest generation.

### Structure Processing (`data/prepare/structure.py`)

```python
def process_mmcif(path, use_assembly=True, assembly_id="1") -> tuple[Structure, Record]:
```

- Reads mmCIF via gemmi, expands biological assembly 1 via `gemmi.make_assembly()`
- Iterates subchains, maps entities to mol types, builds atoms/residues/chains arrays
- Computes entity dedup (same sequence → same entity_id) and MSA ID (SHA256 of sequence)
- Detects chain-chain interfaces via KDTree proximity (5 Å cutoff)
- Extracts metadata (resolution, method, deposition/release dates)
- Returns `(Structure, Record)` with full `ChainInfo`, `InterfaceInfo`, `StructureInfo`

### Quality Filters (`data/prepare/filters.py`)

Two-level filtering controlled by `StructureFilterConfig`:

**Structure-level** (`filter_structure_metadata`) — reject entire structures (cheap, metadata only):

| Filter | CLI flag | Default |
|--------|----------|---------|
| max_resolution | `--max-resolution` | disabled |
| min_resolution | `--min-resolution` | disabled |
| allowed_methods | `--methods` | disabled |
| max_release_date | `--max-date` | disabled |
| min_release_date | `--min-date` | disabled |
| max_chains | `--max-chains` | 300 |
| max_total_residues | `--max-total-residues` | 16384 |
| min_polymer_chains | `--min-polymer-chains` | 1 |

**Chain-level** (`filter_structure`) — mark individual chains as `valid=False`:

- **min_length**: reject polymer chains with < 4 or > 5000 resolved residues
- **all_unknown**: reject chains where all residues are UNK/N
- **consecutive_ca**: reject protein chains with consecutive CA atoms > 10 Å apart
- **clashing_chains**: reject chains with > 30% of atoms clashing (< 1.7 Å) with other chains

`filter_structure` returns `(record, reasons)` where reasons maps chain_name to the filter that rejected it. Interfaces with any invalid chain endpoint are also marked invalid.

### Sequence Clustering (`data/prepare/clustering.py`)

```python
def load_pdb_clusters(path) -> dict[str, int]: ...  # wwPDB format -> pdb_chain -> cluster_id
def sequence_to_cluster_key(pdb_id, chain_name) -> str: ...  # "1abc_a"
```

Two workflows:

1. **Pre-computed**: Download `clusters-by-entity-40.txt` from wwPDB, apply via `deepfold3 apply-clusters`
2. **In-house**: `deepfold3 cluster` extracts sequences from Structure NPZs, runs `mmseqs cluster`, writes wwPDB-format output

### Batch MSA Generation (`scripts/msa_batch.py`)

`deepfold3 msa-batch` extracts unique protein sequences from the manifest, searches via local MMseqs2 (GPU optional) against UniRef30 + ColabFoldDB, and writes per-sequence A3M + NPZ to `{data_dir}/msa/`. Resume-safe: skips sequences with existing NPZ; converts A3M-only without re-searching.

### MSA Import for Distillation (`scripts/prepare.py`)

For distillation databases (AFDB, ESMFold, etc.) that provide pre-computed A3Ms:

- `deepfold3 prepare --msa-dir <dir>` — import during structure preprocessing
- `deepfold3 import-msa --data-dir <dir> --msa-dir <dir>` — import into existing dataset

Matching is naming-convention-agnostic: reads the query sequence from each A3M file, computes SHA256, and copies to `{data_dir}/msa/{msa_id}.a3m`.

### Manifest (`data/prepare/manifest.py`)

```python
def save_manifest(records, path): ...
def load_manifest(path) -> list[Record]: ...
def discover_records(data_dir) -> list[Record]: ...
def load_id_list(path) -> set[str]: ...  # text file of PDB IDs
```

- `discover_records()` loads `manifest.json` if present, otherwise scans `structures/` for NPZ files and builds minimal records
- `load_id_list()` loads a text file of PDB IDs (one per line, `#` comments) for `--include-list`/`--exclude-list` filtering
- Used by `deepfold3 train` to find training data
