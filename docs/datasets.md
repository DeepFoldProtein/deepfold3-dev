# Dataset Directory Layout

On-disk inventory of the data DeepFold3 trains and evaluates on. Everything lives on the shared GPFS volume `/data` (200 T, 58 T used). Trainjobs mount the same volume at `/data` (PVC `deepfold3-sync`, see `kakao/trainjob-*.yaml`), so a path is the same in the workspace and in a pod.

Snapshot of 2026-09-10. File counts are exact (`os.scandir`). Sizes marked `≈` are extrapolated from 400 random entries per directory; unmarked sizes are `ls` sizes. The layout is the stable part — re-count before quoting a number.

`/data/distillation` is a symlink to `/data/data`. Configs spell every training set `/data/distillation/...`, and so does this page.

## Overview

```
/data/
├── pdb_training_set/          of3pdb  OF3 weighted-PDB set, read in place   train 0.50 + eval
├── distillation -> data/
│   ├── long_monomers/         of3     AF2-predicted MGnify monomers         train 0.20
│   ├── short_monomers/        of3     AF2-predicted MGnify monomers         train 0.05
│   ├── afdb_nvda/             df3     AFDB homodimers                       train 0.15
│   ├── teddymer/              df3     TED domain pairs from AFDB            train 0.10
│   ├── patchr_training/       df3     PATCHR ATLAS PDB assemblies           off (was the PDB set in r11–r13)
│   ├── patchr/                        raw PATCHR archives + inference scratch
│   └── long_monomer/                  partial copy of long_monomers, unreferenced
├── pdb/                               wwPDB mmCIF mirror + per-chain template cache
├── db/                                MMseqs2 search databases
├── msa/  msa_search/                  MSA search runs and their query inputs
├── plm/                               complex-LM stores, raw pairs, PLM weights
├── ccd_cache.json  components.cif.gz  CCD
├── FoldBench/  foldbench/  benchmarks/    evaluation targets and baseline predictions
└── weights/  boltz_cache/  bin/           third-party weights, mmseqs binary
```

The weights are the `datasets:` mixture in `configs/initial.yaml`, which `df3-stage0-r15`…`r19` and the finetune configs inherit unchanged. `patchr_training` is still named by the r11–r13 configs, `scripts/triage_inf.py`, `scripts/measure_grad_variance.py` and `/data/plm/build_stores.sh`.

Not datasets (run outputs and scratch): `runs/`, `outputs/`, `predict/`, `work/`, `queries/`, `kfp_artifacts/`, `sequences/`, and the loose FASTA/JSON files at the top level.

## Shard rules

Three rules are in use. A path built with the wrong one looks valid and points at nothing.

| Tree | Shard key | Example |
| --- | --- | --- |
| df3 sets, `/data/pdb/cache` | first two characters of the file id, lowercased (`data/paths.py:shard`) | `structures/1b/1b55.npz` |
| `/data/pdb/{divided,assemblies,obsolete}` | middle two characters of the PDB id (wwPDB convention) | `divided/4g/14gs.cif.gz` |
| of3, of3pdb | none — one flat directory | `preprocessed/MGYP000062676673/` |

The df3 key comes from whichever id names the file, so the fan-out follows that id's alphabet: 256 shards for SHA-256 ids (`msa/unpaired/`, `monomer_templates/`), up to 324 for PDB ids (247 in `patchr_training`), and only 90 (`10`–`99`) for teddymer's numeric record ids.

## `format: df3`

Written by `deepfold3 prepare` / `import-*` / `msa-batch` / `prepare-templates`, read by `TrainingDataset`:

```
{data_dir}/
├── manifest.json                              list[Record]; atomic save, previous copy kept as manifest.json.old
├── structures/{ab}/{record_id}.npz            Structure
├── msa/
│   ├── unpaired/{ab}/{msa_id}.npz             msa_id = SHA-256 of the chain sequence
│   └── paired/{ab}/{record_id}_paired_{chain_hash}.npz
├── monomer_templates/{ab}/{msa_id}.json       template hits per sequence (shared across records)
├── templates/{ab}/{record_id}.json            multimer template hits per record
├── masks/{ab}/{record_id}.npz                 TokenMask channels
└── records/{ab}/{record_id}.json              per-record Record from prepare; the loader reads manifest.json
```

Every template hit in the three sets records `index_space: struct` (back-filled 2026-08-22 by `stamp-template-index-space`).

| | `afdb_nvda` | `teddymer` | `patchr_training` |
| --- | --- | --- | --- |
| Content | AFDB homodimers | TED domain pairs (AFDB) | PATCHR ATLAS PDB multimer assemblies |
| Record id | `8d_AF-0000000079072088` | `7424DI_AF-A0A009H2I0-F1-model_v4_TED01_TED02` | `1b55` |
| `manifest.json` | 1,588,368 records · 1.7 G | 450,949 · 514 M | 149,398 · 325 M |
| `structures/` | 1,588,368 · ≈121 G | 450,949 · ≈13 G | 159,380 · ≈12 G |
| `msa/unpaired/` | 1,502,220 · ≈1.0 T | 883,421 · ≈218 G | 112,776 · ≈84 G |
| `msa/paired/` | — | — | 87,918 · ≈57 G |
| `monomer_templates/` | 1,496,539 · ≈7 G | 650,545 · ≈2 G | — |
| `templates/` | — | — | 157,618 · ≈1 G |
| `masks/` · `records/` | — | — | 159,380 · ≈0.5 G · 159,380 · ≈0.3 G |

- **`afdb_nvda`** — each record id carries its own shard as a prefix (`8d_…` lives in `structures/8d/`). Every shard directory under `structures/` and `msa/unpaired/` has a sibling `{ab}.tar.gz` archive of that shard, which is why both list 512 entries; the loader reads only the directories. `templates.tar` (1.3 G) is an archive of `monomer_templates/`.
- **`teddymer`** — numeric ids put every record in 90 shards, unevenly (2,085–21,524 files in the shards sampled). `chunks/` is a full backup in 13 independently extractable parts (`teddymer_df3_chunk_{000..012}.tar.zst`, ≈235 G; `RESTORE.md`, `SHA256SUMS.txt`).
- **`patchr_training`** — `manifest.json` is the post-`refilter` manifest; the 159,380-record original is `manifest.json.pre_filter_backup.20260419_035359`, and the files of the dropped records are still on disk. 205 of the 411 `msa/paired/` shard directories are non-empty. `masks/` is ignored by the loader while `dataset.py:_DISK_TOKEN_MASKS_ENABLED = False`. Also here: `clusters.txt`, `sequences.fasta`, `msa_sequences.fasta`.

## `format: of3` — monomer distillation

OpenFold 3's preprocessed monomer distillation sets, read by `OF3Dataset`. One flat directory per MGnify sequence and no manifest:

```
{data_dir}/                      /data/distillation/{long,short}_monomers/preprocessed
├── ids.txt                      id list OF3Dataset caches from its directory scan
└── {MGYP id}/
    ├── structure.npz            AF2-predicted structure (OF3 AtomArray)
    ├── alignment.npz            MSA
    └── template.npz             template hits; idx_map in 1-based SEQRES
```

| Set | Records | Size | `ids.txt` |
| --- | --- | --- | --- |
| `long_monomers` | 16,099,473 | ≈36 T | 261 M |
| `short_monomers` | 430,418 | ≈235 G | absent |
| `long_monomer` | 253,500 | ≈564 G | stale |

`long_monomer` (singular) is not referenced by any config or script, and all 20 ids sampled from it also exist in `long_monomers`.

## `format: of3pdb` — `/data/pdb_training_set`

A mirror of `s3://openfold3-data/pdb_training_set/`, read in place (see [training.md](training.md#training-on-openfold-3s-pdb_training_set-in-place-format-of3pdb)). Only `df3_cache*/` is produced locally, by `deepfold3 build-of3-cache`.

```
/data/pdb_training_set/
├── preprocessed_pdb_data/
│   ├── standard/
│   │   ├── structure_files/{pdb}/{pdb}.npz        244,272 entries · ≈242 G  (+ {pdb}.cif, {pdb}.fasta)
│   │   ├── reference_mols/{ccd}.sdf, {pdb}_{n}.sdf 68,146 · ≈205 M
│   │   ├── metadata.json                          1.3 G
│   │   └── all-unique-seqs.fasta                  65 M
│   └── disordered/structure_files/{pdb}/          61,227 · ≈41 G   not read by any config
├── alignment_arrays/{rep}.npz                     208,745 · ≈789 G  one MSA per alignment representative (e.g. 1o7f_A)
├── templates/
│   ├── template_structure_arrays/{pdb}/{pdb}_{chain}.npz   234,879 entries · ≈31 G  (+ chain_id_to_moltype.npz)
│   ├── train_template_cache/{rep}.npz             123,282 · ≈19 G   hit lists, training split
│   └── val_template_cache/{rep}.npz               1,840 · ≈1.5 G    hit lists, validation split
├── dataset_caches/
│   ├── training_cache_with_templates.json         1.6 G
│   ├── validation_cache_with_templates.json       35 M
│   └── disordered_cache_with_templates.json       148 M
├── df3_cache/                                     mmap'd .npy columns + meta.json (284 M), from the training JSON
└── df3_cache_val/                                 same, from the validation JSON
```

| Cache | Structures | Chains | Interfaces | Template hits | Clusters |
| --- | --- | --- | --- | --- | --- |
| `df3_cache` | 180,975 | 1,807,923 | 3,330,818 | 10,947,729 | 87,748 |
| `df3_cache_val` | 1,719 | 10,507 | 14,310 | 67,947 | 3,088 |

Columns: `struct_{id,release,resolution,chain_off,iface_off}`, `chain_{key,label,entity,moltype,cluster,aln_rep,tmpl_off}`, `iface_{a,b}`, `tmpl_id`.

The two splits share structures, alignments and template geometry; only the cache and the hit lists are per split. `of3_pdb` points at `df3_cache` + `templates/train_template_cache`, and `of3_pdb_eval` at `df3_cache_val` + `templates/val_template_cache`. Hits address `template_structure_arrays` by SEQRES `res_id`, so no mmCIF mirror is involved.

## `/data/pdb` — PDB mirror and template chain cache

```
/data/pdb/
├── divided/{mid2}/{pdb}.cif.gz                    216,627 · ≈59 G    wwPDB mmCIF mirror
├── assemblies/{mid2}/{pdb}-assembly{N}.cif.gz     353,425 · ≈108 G   biological assemblies
├── obsolete/{mid2}/{pdb}.cif.gz                   4,481 · ≈2.6 G     obsoleted entries
├── cache/{ab}/{pdb}[-assembly{N}]/                570,052 entry dirs · ≈217 G
│   ├── {chain}.npz
│   └── _done                                      marks a completed entry
└── PDB_MMCIF_READY
```

- `divided/` is the `--mmcif-dir` for `scan-mmcif` and `build-pdb-cache`, and the df3 template loader's cold-parse fallback.
- `cache/` is the disk tier of `PdbChainCache` (`training.pdb_chain_cache_dir`; `build-pdb-cache` writes here by default): each template chain's `seqres`, `struct_seq` and residue→atom lookup, one directory per entry or assembly.

## MSA search — `/data/msa`, `/data/msa_search`

Each `/data/msa/{run}/` is the working tree of a chunked ColabFold MMseqs2 search. The only driver kept in place is `/data/msa/tcr_pmhc/run_local.sh`.

```
/data/msa/{run}/
├── chunks/              query DB and its splits (qdb, qdb_split_{i}_{n}*)
├── logs/chunk_{i}.log
├── pair/{i}/            paired-search shards
├── merged/
│   ├── a3m/             unpaired A3M per query sequence
│   ├── m8/              PDB SEQRES hits per query (input to prepare-templates --m8-dir)
│   ├── paired/          paired A3M per complex
│   └── .done_chunk_{i}, .done_pair_{i}
└── output/  tmp/
```

| Run | `a3m/` | `m8/` | `paired/` |
| --- | --- | --- | --- |
| `msa_search` | 160,554 | 160,554 | — |
| `patchr_rep` | 121,518 | 121,518 | 32,130 |
| `patchr` | 112,491 | 112,420 | 33,210 |
| `msa_search_pdb_pair` | — | — | 29,730 |
| `msa_search_patchr_pair` | — | — | 24 |
| `foldbench` | 3,004 | 2,955 | 288 |
| `dockgen_multi` | 134 | 134 | 61 |
| `dockgen_single` | 269 | 269 | — |
| `tcr_pmhc` | 195 | 195 | — |

`msa_search/merged/a3m/` alone is ≈431 G. `msa_search` (`a3m/`) and `msa_search_pdb_pair` (`paired/`) are the MSA sources that `scripts/generate_msa_inputs.py` imports into the workspace `data/pdb_training/`. `/data/msa/foldbench.tar.zst` (2.1 G) archives the FoldBench run, and `/data/msa/mgy_clu/` is empty.

`/data/msa_search/` holds query inputs, not results: `combined.fasta` and the `{pdb,patchr}_complexes.json` / `complexes.json` pairing inputs written by `scripts/generate_msa_inputs.py`, the FoldBench inputs, and `mgy_clu.fasta` (19 G).

## Sequence databases — `/data/db` (≈1.0 T)

The MMseqs2 databases that `scripts/setup_databases.sh` downloads and indexes, for `deepfold3 msa --mode local --db-dir /data/db` and `msa-batch`. `*_READY` files are completion sentinels.

| Database | `_seq` | `_aln` | `.idx` | GPU |
| --- | --- | --- | --- | --- |
| `uniref30_2302_db` | 125 G | 26 G | 216 G | `.GPU_READY` |
| `colabfold_envdb_202108_db` | 122 G | 45 G | 268 G | `.GPU_READY` |
| `pdb100_230517` | 62 M | — | 168 M | — |

## Complex-LM — `/data/plm`

```
/data/plm/
├── raw/                 PPLM paired-sequence release: {PDB,STRING}_seqs.fasta, {train,valid}set.*.txt, README.txt
├── store/{source}/      ComplexStore — ComplexLMConfig.data_root defaults to /data/plm/store
│   ├── seq.npy  chain_off.npy  cx_off.npy  slot_chain.npy  slot_iface.npy (structure stores only)
│   ├── chain_cluster.npy  cx_cluster.npy  cx_date.npy
│   └── ids.txt  meta.json  stats.json
├── clust/               plm-prepare cluster: chains.fasta, clust_{all_seqs,rep_seq}.fasta, clust_cluster.tsv, clustering.json
├── build_stores.sh      rebuilds every store, sequentially
├── esm2_t33_650M_UR50D/  SaESM2_650M/  esmc-600m/  PPLM/    baseline PLM weights
└── probe*/  sweeps/  logs/                                   experiment outputs
```

| Store | Built from | Complexes | Unique chains | Unique residues |
| --- | --- | --- | --- | --- |
| `pdb` | `patchr_training` | 84,119 | 70,241 | 18.4 M |
| `afdb` | `afdb_nvda` | 1,588,368 | 1,502,237 | 615 M |
| `teddymer` | `teddymer` | 450,949 | 883,421 | 115 M |
| `pplm_pdb` / `pplm_pdb_valid` | `raw/` PDB pairs | 357,611 / 2,207 | 86,319 / 3,160 | 26.6 M / 1.0 M |
| `string` / `string_valid` | `raw/` STRING pairs | 2,825,843 / 2,471 | 668,714 / 4,713 | 264 M / 2.1 M |

## Benchmarks

```
/data/FoldBench/       FoldBench checkout: targets/{category}.csv (10 categories), examples/ground_truths/ (1,903 CIF), evaluation/, results/
/data/foldbench/       baseline-run inputs: af3_inputs/, boltz_inputs/, json_inputs/, yaml_inputs/, per-tool *_shards*/, tool envs and weights
/data/benchmarks/
├── {AF3,Boltz-1,Boltz-2,Protenix,Protenix-v1,Protenix-v2}/{category}/   baseline predictions
└── eval/{tool}/                                                          their evaluations
```

## Other top-level entries

| Path | What |
| --- | --- |
| `/data/ccd_cache.json` (106 M), `/data/components.cif.gz` (110 M) | CCD and its parsed cache (`fetch-ccd`, `build-ccd-cache`); `kakao/trainjob-multinode.yaml` passes `--ccd-cache /data/ccd_cache.json` |
| `/data/distillation/patchr/` | raw PATCHR ATLAS archives (`pdb_atlas_multimer_{cif,json,yaml}.tar.zst`, 20 G), `msas/` (sequence clustering + merged `a3m/` / `m8/`, 121,518 each), Boltz and antibody inference scratch |
| `/data/weights/alphafold2/params/` | AF2 parameters |
| `/data/boltz_cache/` | Boltz-1/2 checkpoints, CCD pickle, `mols/` |
| `/data/bin/mmseqs/` | MMseqs2 build used by the search drivers |

## Workspace-local `data/`

`data/` at the repo root is on the workspace volume, not `/data`, and is gitignored.

| Path | What |
| --- | --- |
| `mmcif_meta.json` | `scan-mmcif` cache (`--mmcif-meta` / `--mmcif-cache`) |
| `entries.idx`, `clusters-by-entity-40.txt`, `all_pdb_ids.txt` | wwPDB entry index, 40 % entity clusters, id list |
| `ccd_cache.json`, `components.cif.gz` | older (2026-04-13) CCD copies |
| `pdb_training/` | df3-format PDB set from 2026-04-29: 204,231 structures, 44,164 template indexes; MSAs from the `msa_search` / `msa_search_pdb_pair` runs |
| `patchr_training/` | partial copy of the `/data` set (manifest, records, masks, msa; no structures or templates) |
| `patchr/` | `cifs/`, `json/` |
