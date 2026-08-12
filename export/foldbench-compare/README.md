# af3-vs-r14-foldbench — FoldBench-small comparison report

Research note comparing DeepFold3 **r14** against published FoldBench results (AF3, Boltz-1/2,
Protenix v1/v2, OpenFold-3 preview-2, ESMFold2). See `report.md`.

> **`af3_ref` is retired as a displayed baseline** (2026-07-28). It was AlphaFold3 folded through
> our own harness, which attached **no templates** — a handicapped AF3, 3.8 points under the
> published AF3 on protein–protein and 9.9 on antibody–antigen on identical targets. Its two
> figures are no longer regenerated and their report sections are frozen at step 23k; headline
> claims use the FoldBench official rows. Revive by re-folding `af3_ref` with templates and
> re-adding the two lines to `update_foldbench.sh`.

## Per-checkpoint update (whole report)

The external models are fixed; only r14 advances. One command refreshes **all four live figures**
once the checkpoint's bench eval lands:

```bash
export/af3-vs-r14-foldbench/update_foldbench.sh          # latest step, auto-detected
export/af3-vs-r14-foldbench/update_foldbench.sh --step N # or pin a step
```

Every plot script auto-detects the latest step whose `summary.csv` exists (written last, so it is
the completion marker) and accepts `--step N`. All of them read r14's already-benched raw CSVs
(`bench/runs/df3v4r14/eval/<cat>/step_<N>/raw/`) — no OST, no re-parsing of external JSONs — so the
whole refresh is near-instant.

| figure | what it shows |
|---|---|
| `confidence_calibration.png` | r14 mean / rank-1 / oracle + r(ranking_score, quality) per category |
| `foldbench_models_ppi_agab.png` | 8-model **mean → oracle** (+`_post2023-06-01` cutoff control) |
| `foldbench_conf_models.png` | 6-model **rank-1 → oracle** (from the tarball parse) |
| `foldbench_ranked_ppi_agab.png` | 8-model **rank-1 ‖ oracle** twin bars, mean as a line (+`_post2023-06-01`) |

## Two external data sources — know which one a figure uses

| source | file | models with rank-1 | multi-interface targets |
|---|---|---|---|
| our tarball parse | `foldbench_ext_cache.json` | 6 (OF3p2/EF2 all-zero) | **one designated interface only** |
| FoldBench per-target summary | `foldbench_all_models_per_target.csv` | **7** (only EF2 lacks one) | all annotated interfaces pooled |

`plot_foldbench_ranked.py` uses the **summary** (authoritative, and the only source with an
OpenFold3-p2 ranking); `plot_foldbench_models.py` / `plot_foldbench_conf.py` still use the tarball
cache. The two agree exactly on the median single-interface target but diverge on multi-interface
ones (25/239 ppi, 55/113 agab) and on a ~10% tail — so **do not mix numbers across the two**.
DF3r14 is recomputed from its own benched raw CSVs in every case, aggregated to match whichever
convention that script uses.

### Training-cutoff control

```bash
.venv/bin/python export/af3-vs-r14-foldbench/plot_foldbench_models.py --after 2023-06-01
```

Restricts to targets **released after** a date (Boltz-2's cutoff), writing
`foldbench_models_ppi_agab_post2023-06-01.png` + `..._data_post2023-06-01.csv` (default output
unchanged). Per-target release dates come from `foldbench_target_dates.csv` (fetched from RCSB
`initial_release_date`). This exposes train-set overlap: Boltz-2 drops from 81%→28% on pre→post
antibody, while DF3r14/PXv2 are stable.

### Files

| file | what |
|---|---|
| `report.md` | the write-up (overview, confidence, multimer split, external comparison, cutoff control) |
| `precompute_external.py` | **one-time**: parse external tarball → `foldbench_ext_cache.json` (per-target `[mean, oracle]` DockQ). Re-run only if `foldbench_results.tar.gz` changes. |
| `foldbench_ext_cache.json` | cached external per-target `[mean, oracle]` DockQ (fixed reference) |
| `plot_foldbench_models.py` | **per-ckpt**: cache + latest r14 → `foldbench_models_ppi_agab.png` (mean→oracle, all 8 models); `--after DATE` for the cutoff control |
| `plot_foldbench_conf.py` | confidence-only (rank-1→oracle), 6 models with ranking_score |
| `foldbench_target_dates.csv` | per-target RCSB release dates (for `--after` cutoff filtering) |
| `foldbench_all_models_per_target.csv` | **cached**: FoldBench's own per-target `ranked/mean/oracle` DockQ for the 7 external models (ppi 239 + agab 113 targets, n_samples=20). Fixed reference — replace only if FoldBench reissues it. |
| `plot_foldbench_ranked.py` | **per-ckpt**: rank-1 ‖ oracle twin bars + mean line, from the summary above; `--after DATE` for the cutoff control |
| `update_foldbench.sh` | wrapper for the per-ckpt refresh — runs the four live plot scripts |
| `plot_confidence.py` | confidence calibration (within-target r + rank-1→oracle band), auto-detects the latest step; `--step N` to pin |
| `plot_overview.py` / `plot_multimer_breakdown.py` | **RETIRED, not run**: af3_ref-vs-r14 overview and homo/hetero×dimer/N-mer split. Kept runnable; still draw the template-free `af3_ref` |

Metric: per-target DockQ over all seed×sample predictions on the FoldBench-designated interface
(from `bench/data/targets/`), CAPRI-tiered. **mean** (N-unbiased; external 4 seeds×5, r14 2×5) and
**oracle** (best sample) shown for all 8 models; **rank-1** only where `ranking_score` exists.
Common target set = FoldBench-small ∩ successfully-scored across all models.
