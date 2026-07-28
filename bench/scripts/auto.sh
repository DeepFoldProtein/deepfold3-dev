#!/usr/bin/env bash
# Poll the live training dir for new 1k-grid checkpoints and benchmark each on
# the FoldBench-small pipeline across ALL configured categories — monomer
# (protein / RNA / DNA) and interface (protein-protein, antibody-antigen,
# protein-DNA / -RNA / -peptide). Per category: symlink -> sweep -> OST eval
# -> summarize. monomer_protein additionally gets the DSSP secondary-structure
# analysis (per-class F1 + confusion + precision/recall + fold-class + per-target
# beta trajectories, into runs/$SERIES/ss_analysis/).
#
# FoldBench-small = num_tokens <= MAX_TOKENS (default 1024). Each target is
# folded with 2 seeds (SEEDS) x --num-samples samples. GPU-gated: never launches
# while another sweep (orchestrate.sh / run_sweep.py) is active, so it coexists
# with a separate orchestrator. Resumable (per checkpoint x category).
#
#   nohup bash bench/scripts/auto.sh [TRAIN_DIR] [POLL_SECONDS] \
#         > bench/runs/df3v1/logs/auto.log 2>&1 &
#
# Env overrides:
#   SERIES        series name              (default df3v1)
#   CATEGORIES    space list of categories (default: the 8 below)
#   SEEDS         comma list of seeds      (default 42,43 — lighter than FoldBench's 4)
#   SEEDS_<CAT>   per-category seed override, CAT upper-cased with '-'/' ' -> '_'
#                 (e.g. SEEDS_INTERFACE_ANTIBODY_ANTIGEN=42,43,44,45). Defaults
#                 below give antibody-antigen FoldBench's full 4 seeds.
#   NUM_SAMPLES   diffusion samples / seed (default 5)
#   MAX_TEMPLATES structural templates per protein chain (default 4; 0 = off)
#   REF_SERIES    measured baseline overlay (default: none — see note below)
#   MAX_TOKENS    FoldBench-small cap      (default 1024)
#   SNAPSHOT      .pt weight snapshot      (default auto: shadow if present else average)
#   STEP_OFFSET   series-global step shift (default 0)
set -u
cd "$(dirname "$0")/../.."
SERIES="${SERIES:-df3v4r14}"
TRAIN_DIR="${1:-runs/df3-stage0-r14}"
POLL="${2:-3600}"            # ~1 h
# STEP_OFFSET shifts the benchmarked step number (raw checkpoint step N is
# logged as N+OFFSET) so a fork that restarts the step counter (e.g. stage1
# resumed with skip_load_step) appends to the SAME series trajectory instead
# of colliding with the earlier phase's points. e.g. STEP_OFFSET=110000:
# stage1 checkpoint_5000 -> df3v1 step_115000 (continues after stage0's 110k).
OFFSET="${STEP_OFFSET:-0}"
SEEDS="${SEEDS:-42,43}"
# Per-category seed count. Sample count is a real axis of the comparison, not a
# free knob: `oracle` is monotone in N, so an N mismatch biases the model with
# fewer draws downward. BOTH reference sets use 4 seeds x 5 = 20 samples —
# FoldBench's external models, and the `af3_ref` fold of AlphaFold3 (verified:
# seeds 42,43,44,45 in every category's raw CSV). Running r14 at 2 seeds was
# therefore not "lighter", it was self-handicapping.
#
# Measured, not extrapolated (r14 @ step 20k, 10 -> 20 samples):
#   protein-protein   rank-1 65.0 -> 66.7   oracle 70.8 -> 73.8
#   antibody-antigen  rank-1 27.5 -> 26.8   oracle 40.6 -> 47.8   (@19k)
# Extrapolating oracle from the 10-sample data badly understates it (predicted
# +1.3 for agab, actual +7.2): targets with 0/10 successes get p_hat = 0 under an
# iid model and are frozen as unsolvable, which is exactly the population extra
# seeds crack. Measure, do not extrapolate.
#
# run_sweep --skip-existing is keyed on (target, seed), so raising these tops up
# the new seeds without re-folding 42/43. Cost is ~2x the category's sweep time;
# drop a category back to "$SEEDS" here if bench throughput becomes the blocker.
SEEDS_INTERFACE_ANTIBODY_ANTIGEN="${SEEDS_INTERFACE_ANTIBODY_ANTIGEN:-42,43,44,45}"
SEEDS_INTERFACE_PROTEIN_PROTEIN="${SEEDS_INTERFACE_PROTEIN_PROTEIN:-42,43,44,45}"
# Resolve the seed list for a category: SEEDS_<CAT_UPPER> if set, else SEEDS.
cat_seeds() {
  local var="SEEDS_$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')"
  printf '%s' "${!var:-$SEEDS}"
}
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
# Structural templates per protein chain, attached to the bench inputs from the
# pre-computed m8 hits in bench/data/msa/m8 and filtered to release dates at
# least `template_cutoff_days` before the query (11.4% of bench chains list
# their own PDB entry as a hit, so the filter is not optional).
#
# The bench folded template-free until now, which is in-distribution for the
# model (training draws k = min(Uniform[0, n_avail], 4), so k = 0 happens) but
# is its weakest supported mode, and it made `af3_ref` a handicapped AlphaFold3:
# on identical targets our template-free fold trails the published AF3 by 3.8
# points on protein-protein and 9.9 on antibody-antigen — which is why af3_ref is
# no longer drawn as a baseline (see REF_SERIES).
#
# CAUTION: turning this on changes the bench inputs, so numbers before and after
# are not comparable. Do not flip it in the same window as a training-side
# change unless you are willing to lose the attribution — run one checkpoint
# both ways (MAX_TEMPLATES=0 into a separate SERIES) if you need the split.
MAX_TEMPLATES="${MAX_TEMPLATES:-4}"
# Measured baseline overlaid on the per-category figures. Empty = none, and the
# figures fall back to the published AF3 numbers (FoldBench Supp. Table 3, drawn
# dotted and labelled "paper, full set" — a target marker, not like-for-like).
#
# `af3_ref` used to be the default and is now RETIRED: it was folded before the
# bench attached templates, so it is a handicapped AlphaFold3 (3.8 points below
# the published AF3 on protein-protein, 9.9 on antibody-antigen, same targets).
# summarize.py / plot_lddt_grid.py refuse it by name, so setting it here has no
# effect until af3_ref is re-folded WITH templates and dropped from their
# RETIRED_REF_SERIES.
REF_SERIES="${REF_SERIES:-}"
# GPUs for run_sweep sharding — default to every visible GPU (auto-adapts to the
# node: 8-GPU box -> 0..7, 4-GPU box -> 0..3). Override with GPUS=0,1 to reserve.
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)}"
GPUS="${GPUS:-0}"
# Note: interface_protein_ligand is omitted by default — eval_ost has no
# ligand-aware scorer, so it would be mis-scored as a monomer. Add it to
# CATEGORIES only alongside a proper ligand metric.
# Category priority (highest first): the three we watch most closely lead, so a
# fresh checkpoint scores monomer_protein / antibody-antigen / protein-protein
# before the rest. Order here is the per-checkpoint eval order.
# antibody-antigen runs BEFORE protein-protein: it is the smaller sweep (95
# in-cap targets vs 212) and it is the category whose frontier we are tracking,
# so the metric we most want lands first. protein-protein is the long pole —
# especially at 4 seeds — and putting it last means a slow ppi sweep no longer
# delays agab.
CATEGORIES="${CATEGORIES:-monomer_protein interface_antibody_antigen interface_protein_protein \
monomer_rna monomer_dna \
interface_protein_dna interface_protein_rna interface_protein_peptide}"
PY=.venv/bin/python

# --- fast lscpu shim (perf) -------------------------------------------------
# numpy.testing forks `lscpu` at import (numpy gh-22982, ARM-SVE probe). Every
# OST `compare-structures` imports numpy, so with --workers 48 up to 48 `lscpu`
# run at once and serialize in the kernel walking 288 CPUs' sysfs (~24s each in
# sys time), stalling the whole eval. Shim: bare `lscpu` -> cached output
# (instant); any flagged call -> real /usr/bin/lscpu. Built once, then reused.
_FASTBIN="$HOME/.local/fastbin"
if [ ! -x "$_FASTBIN/lscpu" ]; then
  mkdir -p "$_FASTBIN"
  /usr/bin/lscpu > "$_FASTBIN/lscpu.cache" 2>/dev/null || true
  cat > "$_FASTBIN/lscpu" <<'SH'
#!/bin/sh
# fast drop-in for lscpu; see bench/scripts/auto.sh for rationale
if [ "$#" -eq 0 ] && [ -s "$HOME/.local/fastbin/lscpu.cache" ]; then
  cat "$HOME/.local/fastbin/lscpu.cache"
else
  exec /usr/bin/lscpu "$@"
fi
SH
  chmod +x "$_FASTBIN/lscpu"
fi
export PATH="$_FASTBIN:$PATH"
# --- OST binary ------------------------------------------------------------
# eval_ost.py defaults to `apptainer run <sif> compare-structures` (the cluster
# image). This dev host has no apptainer but ships OpenStructure in the
# foldbench conda env, so point OST_CMD there when that binary exists; leave it
# unset otherwise so the apptainer default still works on the cluster. Override
# with OST_CMD=... .
if [ -z "${OST_CMD:-}" ] && [ -x "$HOME/.conda/envs/foldbench/bin/ost" ]; then
  export OST_CMD="$HOME/.conda/envs/foldbench/bin/ost compare-structures"
fi
# ---------------------------------------------------------------------------

echo "[auto] start $(date '+%m-%d %H:%M')  series=$SERIES train_dir=$TRAIN_DIR offset=$OFFSET poll=${POLL}s"
echo "[auto] categories: $CATEGORIES"
echo "[auto] seeds=$SEEDS samples=$NUM_SAMPLES max_tokens=$MAX_TOKENS templates=$MAX_TEMPLATES ref=${REF_SERIES:-none (paper AF3)}"
[ "$MAX_TEMPLATES" -gt 0 ] && echo "[auto]   NOTE templates are ON — numbers are not comparable to template-free steps"
for _c in $CATEGORIES; do
  _s="$(cat_seeds "$_c")"
  [ "$_s" != "$SEEDS" ] && echo "[auto]   seed override: $_c = $_s"
done

sweep_active() {
  pgrep -f "orchestrate.sh" >/dev/null || pgrep -f "run_sweep.py" >/dev/null \
    || pgrep -f "inference-batch" >/dev/null
}

# Build inputs once at startup (idempotent; rebuilds JSONs + manifest at the
# current cap). New categories / a raised cap slot in without re-benchmarking
# existing predictions (sweep is skip-existing).
for CAT in $CATEGORIES; do
  echo "[auto] $(date '+%m-%d %H:%M') building inputs for $CAT (<=$MAX_TOKENS tokens, templates=$MAX_TEMPLATES) ..."
  $PY bench/scripts/build_inputs.py --category "$CAT" --max-tokens "$MAX_TOKENS" \
      --max-templates "$MAX_TEMPLATES" || \
    echo "[auto] WARN build_inputs failed for $CAT"
done

run_ss_analysis() {  # monomer_protein only — DSSP secondary-structure suite
  local CAT="$1" G="$2"
  $PY bench/scripts/ss_f1.py           --series "$SERIES" --category "$CAT" --steps all --workers 48
  $PY bench/scripts/ss_confusion.py    --series "$SERIES" --category "$CAT" --step "$G"  --workers 48
  $PY bench/scripts/ss_pr_traj.py      --series "$SERIES" --category "$CAT" --steps all --workers 48
  $PY bench/scripts/ss_byclass_traj.py --series "$SERIES" --category "$CAT" --steps all --workers 48
  $PY bench/scripts/ss_target_traj.py  --series "$SERIES" --category "$CAT" --group beta --steps all --workers 48
  $PY bench/scripts/ss_struct_corr.py  --series "$SERIES" --category "$CAT" --step "$G"
  $PY bench/scripts/ss_coherence.py    --series "$SERIES" --category "$CAT" --step "$G"  --workers 48
}

# Process ONE (checkpoint, category) unit per iteration, re-scanning every time,
# so priority is re-evaluated at each unit boundary: newest checkpoint first,
# and within it the CATEGORIES order (monomer_protein / antibody-antigen /
# protein-protein lead). A fresh checkpoint landing mid-run preempts the older
# backlog at the next boundary instead of waiting a whole pass.
while true; do
  # Highest-priority pending unit = newest 1k-grid checkpoint x its first
  # still-missing category (per-category summary.csv marker).
  sel_ckpt=""; sel_step=""; sel_cat=""
  for ckpt in $(ls "$TRAIN_DIR"/checkpoint_*.pt 2>/dev/null | sort -t_ -k2 -nr); do
    N=$(basename "$ckpt" | sed 's/checkpoint_//;s/\.pt//')
    case "$N" in (*[!0-9]*) continue ;; esac          # skip _checkpoint_* etc.
    (( N % 1000 == 0 )) || continue                    # 1k grid only (raw step)
    G=$(( N + OFFSET ))                                 # series-global step
    for CAT in $CATEGORIES; do
      if [ ! -e "bench/runs/$SERIES/eval/$CAT/step_$G/summary.csv" ]; then
        sel_ckpt="$ckpt"; sel_step="$G"; sel_cat="$CAT"; break
      fi
    done
    [ -n "$sel_cat" ] && break                          # newest checkpoint wins
  done

  if [ -z "$sel_cat" ]; then
    sleep "$POLL"; continue                             # all done -> wait for new checkpoints
  fi
  if sweep_active; then
    echo "[auto] $(date '+%m-%d %H:%M') external sweep active -- defer"; sleep 60; continue
  fi

  G="$sel_step"; CAT="$sel_cat"
  echo "[auto] $(date '+%m-%d %H:%M') === $CAT step $G (newest-pending) ==="
  mkdir -p "bench/ckpts/$SERIES"                        # self-heal for new series
  ln -sfn "$(realpath "$sel_ckpt")" "bench/ckpts/$SERIES/step_$G.pt"

  # --snapshot auto: shadow when the checkpoint carries an EMA shadow (AdamW+EMA
  # runs, e.g. df3v4/r12 — the deployment target), else ckpt['model'] (SF's x /
  # live AdamW). Matches `deepfold3 export`/`inference`/eval; average would score
  # the raw live params (NOT deployed) for an AdamW+EMA run. Env-overridable.
  CAT_SEEDS="$(cat_seeds "$CAT")"
  [ "$CAT_SEEDS" != "$SEEDS" ] && echo "[auto] $CAT uses seeds=$CAT_SEEDS (global $SEEDS)"
  $PY bench/scripts/run_sweep.py --series "$SERIES" --steps "$G" --category "$CAT" \
      --gpus "$GPUS" --num-samples "$NUM_SAMPLES" --num-recycles 10 \
      --seed "$CAT_SEEDS" --snapshot "${SNAPSHOT:-auto}"
  $PY bench/scripts/eval_ost.py  --series "$SERIES" --step "$G" --category "$CAT" --workers 48
  if [ "$CAT" = "monomer_protein" ]; then
    run_ss_analysis "$CAT" "$G"
  fi
  $PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT" --ref-series "$REF_SERIES"
  # cross-category lDDT overview (mean + median per step, all categories)
  $PY bench/scripts/plot_lddt_grid.py --series "$SERIES" --ref "$REF_SERIES" || \
    echo "[auto] WARN plot_lddt_grid failed"
  # Refresh the whole af3-vs-r14 FoldBench report after an interface eval:
  # overview, confidence calibration, multimer decomposition, the 8-model
  # mean->oracle comparison and its post-cutoff control, and the rank-1||oracle
  # twin-bar figure. External model rows are cached; only r14 is recomputed, and
  # every script auto-detects the latest benched step per category, so running
  # this after either interface category is safe and cheap.
  case "$CAT" in
    interface_protein_protein|interface_antibody_antigen)
      if [ -f export/af3-vs-r14-foldbench/foldbench_ext_cache.json ]; then
        export/af3-vs-r14-foldbench/update_foldbench.sh \
          && echo "[auto] foldbench report refreshed" \
          || echo "[auto] WARN foldbench report update failed"
      fi ;;
  esac
  echo "[auto] $(date '+%m-%d %H:%M') done $CAT step $G"
done
