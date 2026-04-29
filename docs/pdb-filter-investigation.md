# PDB training dataset quality investigation

**Scope.** Full audit of `data/pdb_training` (204,231 records in
`manifest.json`) covering (a) sub-angstrom heavy-atom clashes and
(b) UNK-residue content. Parallel to the PATCHR investigation in
[patchr-filter-investigation.md](patchr-filter-investigation.md); this
report only covers population analysis — no inf-grad explosion events
were logged against this dataset in `runs/test/*` (the runs there used
PATCHR + OF3 distillation only).

**Artifacts.**

- Manifest backup:
  `data/pdb_training/manifest.json.pre_filter_backup.20260419_063710`
  (same dir, byte-identical copy from before any filter is applied).
- Per-record audit CSV:
  [docs/filter_audits/pdb_audit.csv](filter_audits/pdb_audit.csv)
  — 204,231 rows, columns:
  `pdb_id, n_atoms, n_tight_clash, max_unk_frac, n_chains,
n_lig_chains, n_prot_tokens`.
- Companion PATCHR audit CSV:
  [docs/filter_audits/patchr_audit.csv](filter_audits/patchr_audit.csv)
  — 159,380 rows, same schema.

## 1. Headline contrast with PATCHR

| metric                  |  PATCHR |     **PDB** |
| ----------------------- | ------: | ----------: |
| records                 | 159,380 | **204,231** |
| zero tight clashes      |  82.81% |  **57.55%** |
| ≥ 1 tight clash         |  17.19% |  **42.45%** |
| p95 tight clash count   |      10 |      **96** |
| p99 tight clash count   |     157 |     **278** |
| p99.9 tight clash count |     751 |   **2,479** |
| max tight clash count   |  59,673 |  **62,047** |
| UNK frac p99            |    1.00 |    **0.33** |
| records at 100% UNK     |    ~850 |     **668** |

**PDB has substantially more clashing than PATCHR.** Nearly half of
PDB entries have some sub-0.8 Å non-bonded heavy-atom pair.
Presumably this reflects the unfiltered mix of experimental sources
(cryo-EM placement errors, low-resolution X-ray assemblies, fragment
screening campaigns that share systematic artifacts). PATCHR is a
distillation dataset with cleaner geometry. **The filter thresholds
that worked for PATCHR (`tc > 5`) would drop ~35% of PDB, which is
too aggressive.** Different thresholds are appropriate here.

## 2. Full PDB distribution (n = 204,231)

![tight-clash histogram](filter_audits/hist_tight_clash.png)
![UNK fraction histogram](filter_audits/hist_unk_frac.png)

### Heavy-atom tight-clash count (pairs closer than 0.8 Å, bonded excluded)

| tight-clash bucket | records |  share |
| ------------------ | ------: | -----: |
| 0                  | 117,525 | 57.55% |
| 1                  |   1,488 |  0.73% |
| 2–5                |  15,403 |  7.54% |
| 6–10               |  12,620 |  6.18% |
| 11–50              |  36,514 | 17.88% |
| 51–100             |  10,948 |  5.36% |
| 101–500            |   8,901 |  4.36% |
| > 500              |     832 |  0.41% |

Non-zero tail is thicker here than in PATCHR (42% vs 17%) and the
mass is shifted toward the 11–50 bucket — the typical PDB entry with
artifacts has a few dozen tight contacts, not just one.

### UNK fraction

```
p50 = 0.000    p90 = 0.000    p95 = 0.004    p99 = 0.333    p99.9 = 1.000
```

UNK is much rarer in PDB than in PATCHR. Only 2.6% of PDB records
exceed 5% UNK. The 100%-UNK spike (668 records) is the usual
low-resolution density-fit cluster.

### Size / ligand distribution

```
p50 n_chains       =  5      p90 = 18     p99 = 48     max =  64
p50 ligand_chains  =  3      p90 = 13     p99 = 37     max =  61
p50 protein_tokens ~ 300     p90 ~ 1400   p99 ~ 4800
```

PDB entries are materially larger than PATCHR (median chain count
5 vs 2; median ligand count 3 vs 0). Large multi-chain assemblies
dominate the high-clash tail.

## 3. Filter drop rate on PDB

![drop-rate curves](filter_audits/dropcurve_single.png)
![combined 2D heatmap](filter_audits/heatmap_combined.png)
![clash vs. protein size](filter_audits/scatter_clash_vs_tokens.png)

| criterion         | full drop % | count dropped |
| ----------------- | ----------: | ------------: |
| tight_clash > 0   |       42.45 |        86,706 |
| tight_clash > 1   |       41.73 |        85,218 |
| tight_clash > 5   |       34.18 |        69,815 |
| tight_clash > 10  |       28.01 |        57,195 |
| tight_clash > 20  |       21.27 |        43,432 |
| tight_clash > 50  |       10.13 |        20,681 |
| tight_clash > 100 |        4.77 |         9,733 |
| tight_clash > 200 |        1.66 |         3,383 |
| tight_clash > 500 |        0.41 |           832 |
| unk_frac > 0.05   |        2.64 |         5,393 |
| unk_frac > 0.10   |        1.98 |         4,048 |
| unk_frac > 0.20   |        1.37 |         2,804 |
| unk_frac > 0.50   |        0.63 |         1,296 |

### Combined (tight_clash > TC) OR (unk_frac > U)

| (TC, U)        |    drop % |    dropped |
| -------------- | --------: | ---------: |
| (10, 0.10)     |     29.54 |     60,328 |
| (20, 0.10)     |     21.97 |     44,868 |
| **(50, 0.10)** | **11.97** | **24,443** |
| (50, 0.20)     |     11.40 |     23,285 |
| (75, 0.20)     |      7.99 |     16,326 |
| (100, 0.20)    |      6.11 |     12,470 |
| (100, 0.50)    |      5.39 |     10,998 |
| (200, 0.50)    |      2.29 |      4,670 |
| (500, 0.50)    |      1.04 |      2,127 |

### Threshold sweep (unk fixed at 0.10)

```
tc> 30 OR unk>0.10: 17.26%
tc> 40 OR unk>0.10: 14.14%
tc> 50 OR unk>0.10: 11.97%     ← matches ~12% drop rate
tc> 60 OR unk>0.10: 10.37%
tc> 70 OR unk>0.10:  9.06%     ← matches PATCHR's 9.4%
tc> 80 OR unk>0.10:  8.11%
tc> 90 OR unk>0.10:  7.32%
tc>100 OR unk>0.10:  6.70%
```

## 4. Top-15 worst offenders

Records by `n_tight_clash`, descending. These set the absolute scale
of corruption in PDB and should drop under any sensible threshold.

| pid  | tc (< 0.8 Å) | UNK frac | chains | ligands |  prot_toks |
| ---- | -----------: | -------: | -----: | ------: | ---------: |
| 8qph |   **62,047** |     0.00 |     39 |      26 |      3,159 |
| 8qfe |       38,362 |     0.00 |     18 |      12 |      2,100 |
| 8x1h |       23,544 |     0.00 |     24 |       8 |      2,028 |
| 2vtu |       22,224 |     0.00 |     48 |       0 | **12,336** |
| 7lb6 |       20,520 |     0.00 |     24 |       0 |      6,276 |
| 6een |       15,638 |     0.00 |      8 |       0 |      1,260 |
| 5qe2 |       12,759 |     0.00 |      3 |       2 |        284 |
| 5qeu |       12,655 |     0.00 |      3 |       2 |        282 |
| 5qfp |       12,481 |     0.00 |      3 |       2 |        283 |
| 5qfs |       12,444 |     0.00 |      3 |       2 |        282 |
| 5qdw |       12,340 |     0.00 |      3 |       2 |        282 |
| 5qea |       12,241 |     0.00 |      3 |       2 |        284 |
| 5qf1 |       12,186 |     0.00 |      3 |       2 |        282 |
| 8sog |       12,159 |     0.00 |      5 |       4 |        279 |
| 5qg4 |       12,153 |     0.00 |      3 |       2 |        284 |

**Two patterns are visible:**

1. **Large assemblies with placement errors** — `8qph`, `8qfe`,
   `8x1h`, `2vtu` (48-chain / 12k-token), `7lb6` (24 chains), `6een`.
   These are typically cryo-EM or crystal-form assembly expansions
   with systematic clash between symmetry-related copies.
2. **XChem fragment-screening series** — every `5q*` entry in the top 15. These are Diamond XChem campaigns: hundreds of closely-related
   crystal structures of the same protein with different fragment
   ligands soaked in. All share a ~12k-clash artifact
   (probably symmetric lattice contact). As a group they represent
   ~hundreds of records in the PDB manifest that should be evaluated
   collectively, not one-by-one.

The 668 100%-UNK records have more moderate clash counts (top entry
`4v8t` has 1,493 clashes) and are a separate failure mode — they are
low-resolution density fits where sequence could not be assigned.

## 5. Recommended thresholds

Different regimes make sense depending on training priority.

| mode                     | `--max-tight-clashes` | `--max-unk-fraction` |     drops | rationale                                                                                                                      |
| ------------------------ | --------------------: | -------------------: | --------: | ------------------------------------------------------------------------------------------------------------------------------ |
| permissive               |                   200 |                 0.50 |     ~2.3% | Drops only the long tail of severe corruption (>200 clashes) and fully unresolved density fits. Safest — preserves 99% of PDB. |
| conservative             |                   100 |                 0.50 |     ~5.4% | Cuts 8qph / 8qfe / XChem series and other severely clashy records. Keeps ~95% of PDB.                                          |
| **recommended**          |               **100** |             **0.20** | **~6.1%** | Standard middle ground. Similar effective drop rate to the conservative choice with slightly stricter UNK.                     |
| moderate                 |                    50 |                 0.20 |    ~11.4% | Drops the full 11–50 clash band along with the severe tail. Kept fraction (~180k) is still much larger than PATCHR.            |
| PATCHR-equivalent        |                    70 |                 0.10 |     ~9.1% | Matches the drop rate applied to PATCHR (9.4%). Different threshold because PDB's cleanliness distribution is different.       |
| strict (not recommended) |                    10 |                 0.10 |    ~29.5% | Matches PATCHR's `(tc>5 OR unk>0.10)` in spirit but is too aggressive — drops ~30% of PDB.                                     |

**Suggested starting point:** `--max-tight-clashes 100 --max-unk-fraction 0.20`. This drops about 6% (the clearly-pathological tail) while preserving enough coverage that PDB remains the dominant diversity source for early training. If early-training grad skips remain after PATCHR was filtered, tighten to `(50, 0.20)`.

## 6. Applying the filter

```bash
# Dry-run first:
deepfold3 refilter \
    --data-dir data/pdb_training \
    --max-tight-clashes 100 \
    --max-unk-fraction 0.20 \
    --workers 64 \
    --dry-run

# Commit:
deepfold3 refilter \
    --data-dir data/pdb_training \
    --max-tight-clashes 100 \
    --max-unk-fraction 0.20 \
    --workers 64
```

The refilter command preserves prepare-time `valid=False` state
(unchanged by [recent patch](../src/deepfold3/scripts/refilter.py)).
Before applying, the backup
`manifest.json.pre_filter_backup.20260419_063710` is already in place
— for iteration on thresholds, restore from that file before each
re-run so `refilter` starts from the known-good state.

## 7. Open items

- **No training-explosion cross-reference.** The runs in `runs/test/*`
  used PATCHR + OF3 distillation only, not PDB. Once a training run
  with PDB as a dataset logs an inf-grad event, we can cross-reference
  the record_ids against this filter's drop set (same methodology as
  PATCHR §5.2) to validate threshold choice empirically.
- **XChem fragment-screening cluster deserves dataset-level decision.**
  The `5q*` series is hundreds of near-duplicate structures — the
  filter drops each individually, but if the cluster shares a single
  systematic clash artifact, treatment should arguably be "keep one,
  drop the rest" rather than "drop all". Worth a separate conversation
  with anyone curating PDB.
- **PATCHR & PDB share outliers.** `8qph`, `8qfe`, `6een` appear at
  the top of both audits. Both datasets deduplicate by PDB id;
  neither drops these severe cases. A cross-dataset blacklist for the
  clearly-corrupted handful (e.g. > 10k heavy-atom tight clashes)
  would save preparing them again in future pipelines.
