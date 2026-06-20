# bench/ — FoldBench evaluation of df3 checkpoint series

Tracks DeepFold3 structure quality on **FoldBench-small** (`num_tokens <= 384`,
a subset of [FoldBench](https://github.com/BEAM-Labs/FoldBench)) across a
training trajectory, scored with OpenStructure (OST) and stratified by DSSP
secondary-structure fold class.

## Model series

A *series* is one snapshot lineage. The current one is **`df3v1`** (the
FAPE/`fapeK5` run). Future retrains are `df3v2`, `df3v3`, … — each gets its
own `ckpts/<series>/` and `runs/<series>/`; nothing else changes.

`df3v1` is sampled on a **5k-step grid, 10k–90k** (17 checkpoints). The
checkpoints live in `runs/` and are symlinked here, latest first in the sweep
so good results surface early.

> **df3v1 has no trained confidence head** (`alpha_confidence = 0`). pTM /
> pLDDT / `ranking_score` are therefore meaningless and must NOT be used to
> rank or judge samples. Eval reports confidence-free aggregates:
> `mean` (over samples) and `oracle` (best-of-N by the metric itself).
> It is a mid-training model — modest LDDT is expected; the trajectory is
> the signal.

## Layout

```
bench/
├── ckpts/<series>/step_<N>.pt        # symlinks -> real runs/.../checkpoint_<N>.pt
├── data/                             # symlinks to read-only sources
│   ├── targets        -> /data/FoldBench/targets
│   ├── ground_truths  -> /data/FoldBench/examples/ground_truths
│   └── msa            -> /data/msa/foldbench/merged   ({pdb}__{chain}.a3m, m8, paired/)
├── inputs/
│   ├── <category>/<pdb>.json         # generated deepfold3 inference inputs
│   ├── <category>_manifest.csv       # pdb_id, num_tokens, in_small (<=384)
│   └── <category>_dssp.csv           # pdb_id, pct_H/E/C, fold_class
├── runs/<series>/
│   ├── pred/step_<N>/<pdb>/seed-<s>_sample-<i>/model.cif
│   ├── eval/step_<N>/{prediction_reference.csv, raw/, detail/, summary.csv}
│   ├── trajectory_<category>.csv     # step x mean LDDT/TM/GDT
│   ├── byclass_<category>.csv        # step x fold-class LDDT
│   └── trajectory_<category>.png
└── scripts/
```

## Pipeline (monomer_protein implemented; interfaces are TODO)

```bash
cd /home/jovyan/work/deepfold3-dev
PY=.venv/bin/python

# 1. inputs + FoldBench-small filter  (once per category)
$PY bench/scripts/build_inputs.py  --category monomer_protein --max-tokens 384
# 2. DSSP fold classes                (once per category)
$PY bench/scripts/dssp_classify.py --category monomer_protein
# 3. inference — 4 GPUs, latest first ('all' = full 5k grid)
$PY bench/scripts/run_sweep.py --series df3v1 --steps 90000 --category monomer_protein
# 4. score with OST (per step)
$PY bench/scripts/eval_ost.py  --series df3v1 --step 90000 --category monomer_protein
# 5. trajectory table + plot (after several steps scored)
$PY bench/scripts/summarize.py --series df3v1 --category monomer_protein
```

## Scripts

| script | role |
|--------|------|
| `build_inputs.py`  | targets CSV + MSA → per-target deepfold3 JSON + token-count manifest |
| `dssp_classify.py` | mkdssp on GT CIF → `fold_class` ∈ {alpha, beta, alpha_beta, coil} |
| `run_sweep.py`     | length-balanced target shards × N GPUs, one `inference-batch` per GPU (loads weights once, `--snapshot average`); resumable |
| `make_pred_ref.py` | prediction tree → FoldBench `prediction_reference.csv` |
| `eval_ost.py`      | `ost compare-structures` per sample (FoldBench-faithful flags) → mean/oracle LDDT, TM, GDT-TS |
| `summarize.py`     | per-step + per-fold-class trajectory CSVs and plot |

## Conventions / gotchas

- **OST binary**: `/home/jovyan/.conda/envs/foldbench/bin/ost` (2.8.0). The
  conda env's bundled `python` can't import its own pip deps, so `eval_ost.py`
  calls the `ost` CLI directly from our venv and parses the JSON.
- **Metric**: monomer = LDDT (primary), TM-score, GDT-TS. FoldBench's
  ranking-score "rank" metric is unavailable here (no confidence head).
- **FoldBench-small**: `num_tokens <= 384`. monomer_protein = 261 / 334 targets.
- **Resumable**: inference skips targets that already have `seed-*` dirs;
  OST skips comparisons whose detail JSON exists. Re-run to fill gaps.
- **Sources are read-only** (`/data/...`); everything written stays under `bench/`.
```
