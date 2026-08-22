#!/usr/bin/env bash
# Round 1 / single-repr regularizer ablation.
#
# Layout:
#   configs/round1_ln/{single,single_pair}.yaml
#   runs/round1_ln/{1s,5s}_{single,single_pair}/<timestamp>/
#   runs/round1_ln/{1s,5s}_{single,single_pair}.launch.log
#
# Usage:
#   scripts/run_round1_single.sh              # run both waves (1s then 5s)
#   scripts/run_round1_single.sh 1s           # only 1-sample wave
#   scripts/run_round1_single.sh 5s           # only 5-sample wave
#
# Assumes 4 B200 GPUs (visible as 0..3). Runs 2 arms in parallel per wave
# (GPU 0 and 1 only; GPU 2 and 3 idle for this ablation).

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_DIR="/data/distillation/patchr_training"
OUT_ROOT="runs/round1_ln"
ARMS=(single single_pair)

mkdir -p "$OUT_ROOT"

run_wave() {
    local wave="$1"            # "1s" or "5s"
    local include_list
    case "$wave" in
        1s) include_list="runs/overfit_1.txt" ;;
        5s) include_list="runs/overfit_5.txt" ;;
        *)  echo "[error] unknown wave: $wave" >&2; return 1 ;;
    esac

    if [[ ! -f "$include_list" ]]; then
        echo "[error] include list not found: $include_list" >&2
        return 1
    fi

    echo "=== wave: $wave  (include=$include_list) ==="
    local pids=()
    for i in "${!ARMS[@]}"; do
        local arm="${ARMS[$i]}"
        local cfg="configs/round1_ln/${arm}.yaml"
        local run_name="${wave}_${arm}"
        local out_dir="$OUT_ROOT/${run_name}"
        local launch_log="$OUT_ROOT/${run_name}.launch.log"
        mkdir -p "$out_dir"
        echo "  [gpu $i] $run_name  →  $out_dir"
        CUDA_VISIBLE_DEVICES="$i" \
            deepfold3 train \
                --data-dir "$DATA_DIR" \
                --output "$out_dir" \
                --config "$cfg" \
                --include-list "$include_list" \
                >"$launch_log" 2>&1 &
        pids+=($!)
    done

    local failed=0
    for p in "${pids[@]}"; do
        if ! wait "$p"; then
            echo "  [warn] pid $p exited non-zero" >&2
            failed=$((failed + 1))
        fi
    done
    echo "=== wave $wave done (failures: $failed) ==="
    return 0
}

case "${1:-both}" in
    1s)   run_wave 1s ;;
    5s)   run_wave 5s ;;
    both) run_wave 1s && run_wave 5s ;;
    *)    echo "usage: $0 [1s|5s|both]" >&2; exit 2 ;;
esac
