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
| `disto_probe.py`   | per-(target,step) **trunk** distogram recall / distoMed vs **structure** realization — separates "trunk knows the contact" from "sampler builds it" |
| `disto_traj.py`    | **distoMed trajectory** over checkpoints (2 arms side-by-side) + plot — the trunk β-distance *consolidation gauge* |

## Distogram / trunk-distance probe (`distoMed`)

These two scripts read the **distogram head straight off `confidences.json`** —
i.e. the trunk (PairFormer) pair-representation's belief about pairwise
distances, **bypassing the diffusion sampler**. Use them to ask *"does the
trunk encode the right inter-residue distances?"* independently of whether the
diffusion head realizes them in 3D.

**Data each sample writes** (`.../seed-<s>_sample-<i>/confidences.json`):
`distogram` `[N,N]` (expected distance, 64 bins 2.31–21.69 Å → caps ~21.7 Å) and
`contact_probs` `[N,N]` (P(d < 8 Å)). Both are trunk readouts (one per
forward, identical across diffusion samples). GT comes from
`tmp/bench/<target>/_ref.cif`.

**`distoMed`** = median, over a target's true long-range contacts
(GT **CB–CB** < 8 Å, |seq i−j| ≥ 6 — dominated by β-sheet register / tertiary
contacts), of the trunk's predicted distance.
`distoMed ≈ 6–8 Å` → trunk *knows* the sheet; `≈ 13–18 Å` → trunk thinks they're
far (no opinion). It is **deterministic** (no diffusion noise) so far less
noisy than TM/GDT; it is a per-target **consolidation gauge**.

> **Rep atom = CB (Gly → CA).** The distogram is trained on CB–CB distances
> (`featurizer.py` `token_to_rep_atom`, `loss.py` `gt_rep`). Both scripts use CB;
> **do not use CA** (it silently shifts every number).

```bash
PY=.venv/bin/python
# per-checkpoint trunk-vs-sampler breakdown (by seq-sep band lives in disto_probe internals)
$PY bench/scripts/disto_probe.py <arm> 3000            # ARM, comma-steps optional
$PY bench/scripts/disto_probe.py df3-stage1-crop640-average 2500,2750,3000

# distoMed trajectory across all step_* checkpoints of two arms → PNG
$PY bench/scripts/disto_traj.py                         # default: base df3v1 vs crop640
$PY bench/scripts/disto_traj.py <arm_left> <arm_right>  # → tmp/bench/ss_analysis_full/distoMed_trajectory.png
```

**Reading it.** `disto_probe` prints `TRUNK recall` (contact_probs > 0.5 on GT
contacts) and `distoMed` next to `STRUCT recall` (fraction realized in the
generated CIF): trunk-high / struct-low ⇒ sampler bottleneck; both low ⇒ the
**trunk** lost the distances. `disto_traj` is the **plateau watch** — while
distoMed is still descending the trunk β representation is still consolidating;
flat slope = converged. (This probe is how the stage1-crop640 β-regression was
diagnosed as a *trunk* distance-rep collapse, band-localized to seq-sep 6–23;
see `memory/project_stage1_crop640`.)

> Targets are hard-coded to three mainly-β cases (`9EDF`, `9OT9`, `29MW`);
> edit the `targets=[…]` list in-script to probe others. Arms/steps are CLI args.

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
