#!/usr/bin/env bash
# Poll the live training dir for new 5k-grid checkpoints and benchmark each on
# the monomer FoldBench-small pipeline (symlink -> sweep -> OST -> SS analysis
# -> summarize). SS analysis = per-class F1 + confusion matrix + precision/recall,
# fold-class, and per-target-β trajectories (DSSP, into runs/$SERIES/ss_analysis/).
# GPU-gated: never launches while another sweep is active, so it coexists with
# the interface orchestrator. Monomer only. Resumable.
#
#   nohup bash bench/scripts/auto_5k.sh [TRAIN_DIR] [POLL_SECONDS] \
#         > bench/runs/df3v1/logs/auto_5k.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
SERIES="${SERIES:-df3v1}"
CATEGORY=monomer_protein
TRAIN_DIR="${1:-runs/df3-25e5-c10-g35}"
POLL="${2:-7200}"            # ~2 h; a new 5k checkpoint lands ~every 34 h
# STEP_OFFSET shifts the benchmarked step number (raw checkpoint step N is
# logged as N+OFFSET) so a fork that restarts the step counter (e.g. stage1
# resumed with skip_load_step) appends to the SAME series trajectory instead
# of colliding with the earlier phase's points. e.g. STEP_OFFSET=110000:
# stage1 checkpoint_5000 -> df3v1 step_115000 (continues after stage0's 110k).
OFFSET="${STEP_OFFSET:-0}"
PY=.venv/bin/python
echo "[auto5k] start $(date '+%m-%d %H:%M')  series=$SERIES train_dir=$TRAIN_DIR offset=$OFFSET poll=${POLL}s"

sweep_active() {
  pgrep -f "orchestrate.sh" >/dev/null || pgrep -f "run_sweep.py" >/dev/null \
    || pgrep -f "inference-batch" >/dev/null
}

while true; do
  for ckpt in $(ls "$TRAIN_DIR"/checkpoint_*.pt 2>/dev/null | sort -t_ -k2 -n); do
    N=$(basename "$ckpt" | sed 's/checkpoint_//;s/\.pt//')
    case "$N" in (*[!0-9]*) continue ;; esac          # skip _checkpoint_* etc.
    (( N % 5000 == 0 )) || continue                    # 5k grid only (raw step)
    G=$(( N + OFFSET ))                                 # series-global step
    [ -e "bench/ckpts/$SERIES/step_$G.pt" ] && continue # already benchmarked
    if sweep_active; then
      echo "[auto5k] $(date '+%m-%d %H:%M') sweep active -- defer step $G"; break
    fi
    echo "[auto5k] $(date '+%m-%d %H:%M') benchmarking checkpoint $N -> step $G"
    ln -sfn "$(realpath "$ckpt")" "bench/ckpts/$SERIES/step_$G.pt"
    $PY bench/scripts/run_sweep.py --series $SERIES --steps $G --category $CATEGORY \
        --gpus 0,1,2,3 --num-samples 5 --num-recycles 10 --seed 42 --snapshot average
    $PY bench/scripts/eval_ost.py  --series $SERIES --step $G --category $CATEGORY --workers 48
    # --- SS analysis (DSSP) -> runs/$SERIES/ss_analysis/ ---
    # ss_confusion warms the per-sample ss_strings.json cache for step $G (also
    # writes the per-step matrix under ss_analysis/confusion/); the --steps all
    # trajectory scripts then reuse that cache, so they only DSSP uncached steps.
    $PY bench/scripts/ss_f1.py         --series $SERIES --category $CATEGORY --steps all --workers 48
    $PY bench/scripts/ss_confusion.py  --series $SERIES --category $CATEGORY --step $G   --workers 48
    $PY bench/scripts/ss_pr_traj.py    --series $SERIES --category $CATEGORY --steps all --workers 48
    $PY bench/scripts/ss_byclass_traj.py --series $SERIES --category $CATEGORY --steps all --workers 48
    $PY bench/scripts/ss_target_traj.py  --series $SERIES --category $CATEGORY --group beta --steps all --workers 48
    $PY bench/scripts/ss_struct_corr.py  --series $SERIES --category $CATEGORY --step $G   # SS vs lDDT/TM/GDT correlation
    $PY bench/scripts/ss_coherence.py    --series $SERIES --category $CATEGORY --step $G --workers 48  # dRMSD coherence length (size-binned)
    $PY bench/scripts/summarize.py --series $SERIES --category $CATEGORY
    echo "[auto5k] $(date '+%m-%d %H:%M') done step $G"
  done
  sleep "$POLL"
done
