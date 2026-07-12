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
# folded with 4 seeds (SEEDS) x --num-samples samples. GPU-gated: never launches
# while another sweep (orchestrate.sh / run_sweep.py) is active, so it coexists
# with a separate orchestrator. Resumable (per checkpoint x category).
#
#   nohup bash bench/scripts/auto.sh [TRAIN_DIR] [POLL_SECONDS] \
#         > bench/runs/df3v1/logs/auto.log 2>&1 &
#
# Env overrides:
#   SERIES        series name              (default df3v1)
#   CATEGORIES    space list of categories (default: the 8 below)
#   SEEDS         comma list of seeds      (default 42,43,44,45 — FoldBench uses 4)
#   NUM_SAMPLES   diffusion samples / seed (default 5)
#   MAX_TOKENS    FoldBench-small cap      (default 1024)
#   STEP_OFFSET   series-global step shift (default 0)
set -u
cd "$(dirname "$0")/../.."
SERIES="${SERIES:-df3v2}"
TRAIN_DIR="${1:-runs/df3-stage0-r6}"
POLL="${2:-3600}"            # ~1 h
# STEP_OFFSET shifts the benchmarked step number (raw checkpoint step N is
# logged as N+OFFSET) so a fork that restarts the step counter (e.g. stage1
# resumed with skip_load_step) appends to the SAME series trajectory instead
# of colliding with the earlier phase's points. e.g. STEP_OFFSET=110000:
# stage1 checkpoint_5000 -> df3v1 step_115000 (continues after stage0's 110k).
OFFSET="${STEP_OFFSET:-0}"
SEEDS="${SEEDS:-42,43,44,45}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
# Note: interface_protein_ligand is omitted by default — eval_ost has no
# ligand-aware scorer, so it would be mis-scored as a monomer. Add it to
# CATEGORIES only alongside a proper ligand metric.
# Category priority (highest first): the three we watch most closely lead, so a
# fresh checkpoint scores monomer_protein / protein-protein / antibody-antigen
# before the rest. Order here is the per-checkpoint eval order.
CATEGORIES="${CATEGORIES:-monomer_protein interface_protein_protein interface_antibody_antigen \
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
# ---------------------------------------------------------------------------

echo "[auto] start $(date '+%m-%d %H:%M')  series=$SERIES train_dir=$TRAIN_DIR offset=$OFFSET poll=${POLL}s"
echo "[auto] categories: $CATEGORIES"
echo "[auto] seeds=$SEEDS samples=$NUM_SAMPLES max_tokens=$MAX_TOKENS"

sweep_active() {
  pgrep -f "orchestrate.sh" >/dev/null || pgrep -f "run_sweep.py" >/dev/null \
    || pgrep -f "inference-batch" >/dev/null
}

# Build inputs once at startup (idempotent; rebuilds JSONs + manifest at the
# current cap). New categories / a raised cap slot in without re-benchmarking
# existing predictions (sweep is skip-existing).
for CAT in $CATEGORIES; do
  echo "[auto] $(date '+%m-%d %H:%M') building inputs for $CAT (<=$MAX_TOKENS tokens) ..."
  $PY bench/scripts/build_inputs.py --category "$CAT" --max-tokens "$MAX_TOKENS" || \
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
# and within it the CATEGORIES order (monomer_protein / protein-protein /
# antibody-antigen lead). A fresh checkpoint landing mid-run preempts the older
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

  $PY bench/scripts/run_sweep.py --series "$SERIES" --steps "$G" --category "$CAT" \
      --gpus 0,1,2,3 --num-samples "$NUM_SAMPLES" --num-recycles 10 \
      --seed "$SEEDS" --snapshot average
  $PY bench/scripts/eval_ost.py  --series "$SERIES" --step "$G" --category "$CAT" --workers 48
  if [ "$CAT" = "monomer_protein" ]; then
    run_ss_analysis "$CAT" "$G"
  fi
  $PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT"
  # cross-category lDDT overview (mean + median per step, all categories)
  $PY bench/scripts/plot_lddt_grid.py --series "$SERIES" || \
    echo "[auto] WARN plot_lddt_grid failed"
  echo "[auto] $(date '+%m-%d %H:%M') done $CAT step $G"
done
