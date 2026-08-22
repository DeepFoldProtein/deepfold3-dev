#!/usr/bin/env bash
# Orchestrator: watch the in-flight ln driver, plot intermediate, then run
# the single-repr arms, then plot combined.
#
# Expects scripts/run_round1_ln.sh to already be running (writing to
# runs/round1_ln/driver.log). Exits when both waves and their plots are done.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LN_DRIVER_LOG="runs/round1_ln/driver.log"
SINGLE_DRIVER_LOG="runs/round1_ln/driver_single.log"
WRAP_LOG="runs/round1_ln/driver_wrap.log"
DONE_MARKER="=== wave 5s done"

log() { echo "[wrap $(date +%H:%M:%S)] $*"; }

log "watching $LN_DRIVER_LOG for completion marker"
while true; do
    if [[ -f "$LN_DRIVER_LOG" ]] && grep -q "$DONE_MARKER" "$LN_DRIVER_LOG"; then
        break
    fi
    sleep 20
done
log "ln driver finished"

log "plotting intermediate (ln arms)"
bash scripts/plot_round1_ln.sh || log "warn: intermediate plot failed"

log "launching single-reg arms"
bash scripts/run_round1_single.sh both > "$SINGLE_DRIVER_LOG" 2>&1
log "single-reg arms done"

log "plotting final (all arms)"
bash scripts/plot_round1_ln.sh || log "warn: final plot failed"

log "orchestrator done"
