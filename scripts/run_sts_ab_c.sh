#!/usr/bin/env bash
# Launch the 3-arm single-track stability ablation (A, B, C) on 3 GPUs.
# Each run is single-rank, no DDP. Logs and probe JSONL go under
# runs/sts_<DATE>/<arm>/.
#
# A: control (per-layer ROS, single_transition uses inherited 1/sqrt(ℓ+1))
# B: A + single_transition × 0.35 (constant gain)
# C: A + single_transition × extra 1/sqrt(ℓ+1) → s_single_ffn ∝ 1/(ℓ+1)
#
# Usage:
#   bash scripts/run_sts_ab_c.sh [DATA_DIR] [PARAMS_PATH]
#
# DATA_DIR defaults to runs/sts_minidata (built by scripts/build_sts_minidata.py).
# PARAMS_PATH is optional (initial weights). Skip to train from scratch.
set -euo pipefail

DATA_DIR="${1:-runs/sts_minidata}"
PARAMS_PATH="${2:-}"
DATE_TAG="$(date +%Y%m%d-%H%M%S)"
ROOT="runs/sts_${DATE_TAG}"
mkdir -p "${ROOT}"

if [[ ! -d "${DATA_DIR}/structures" ]]; then
    echo "ERROR: ${DATA_DIR} does not look like a deepfold3 data-dir (missing structures/)." >&2
    echo "Run: uv run python scripts/build_sts_minidata.py --src /data/data/patchr_training --dst ${DATA_DIR}" >&2
    exit 1
fi

PARAMS_FLAG=()
if [[ -n "${PARAMS_PATH}" ]]; then
    PARAMS_FLAG=("--params" "${PARAMS_PATH}")
fi

_arm_description() {
    # s_ros = 1/√(ℓ+1) under inherited per-layer ROS. The gain factor below
    # is what's multiplied on top of s_ros for the single FFN branch only.
    case "$1" in
        A) echo "s_ros = 1/√(ℓ+1) (control, no extra gain)" ;;
        B) echo "0.35 · s_ros = 0.35/√(ℓ+1) (constant 0.35× damp)" ;;
        C) echo "(1/√(ℓ+1)) · s_ros = 1/(ℓ+1) (steeper depth schedule)" ;;
        *) echo "(unknown arm)" ;;
    esac
}

_write_readme() {
    local arm="$1"
    local out="$2"
    cat > "${out}/README.md" <<EOF
# ${ROOT##*/} / arm ${arm}

**Single-track stability ablation** — 32-record PATCHR overfit set.
Goal: separate transient single-track magnitude growth from a
load-bearing aligned-gain branch. See
\`docs/single-track-stability-experiments.md\` §11.

## Arms

| Arm | s_single_ffn (trunk single FFN) |
| --- | --- |
$( [[ "$arm" == "A" ]] && echo "| **A (this run)** | $(_arm_description A) |" || echo "| A | $(_arm_description A) |" )
$( [[ "$arm" == "B" ]] && echo "| **B (this run)** | $(_arm_description B) |" || echo "| B | $(_arm_description B) |" )
$( [[ "$arm" == "C" ]] && echo "| **C (this run)** | $(_arm_description C) |" || echo "| C | $(_arm_description C) |" )

For all three, every other branch (single_attn / tri_mul / pair_attn /
pair_transition) keeps the inherited \`1/√(ℓ+1)\` ROS schedule — only
the **single FFN** update is altered.

## Files in this directory

- \`config.yaml\` — fully resolved config (all inheritance + CLI overrides applied).
- \`provenance.yaml\` — git SHA, hostname, argv, env.
- \`stdout.log\` / \`train.log\` — trainer log (every 50 effective steps).
- \`single_track_probe.jsonl\` — per-step (every 5) JSONL with all loss
  components + per-block RMS(s) / RMS(γ·u) / cos(s,u) / α=(s·u)/‖s‖² /
  Σ γ_k·α_k + ‖W₂(single_trans)‖ / ‖Wo(single_attn)‖.
- \`checkpoint_*.pt\` — every 1000 effective steps, kept all.

## Plots (run-root level)

- \`../plots/overview.png\` — losses + 3 scatter decision plots.
- \`../plots/depth.png\` — RMS(s) / RMS(γ·u) / cos(s,u) at L=0 / mid / last.
- \`../plots/wakeup.png\` — wake-up dynamics + Σ γ_k·α_k aligned-gain budget.

Render / live-monitor:
\`\`\`bash
uv run python scripts/plot_sts_probe.py ${ROOT}              # one-shot
uv run python scripts/plot_sts_probe.py ${ROOT} --watch 60   # live
\`\`\`

## Decision rule (docs §11.6)

| Pattern | Read |
| --- | --- |
| L↓, RMS(s_l_last)↑ (lockstep) | aligned gain is load-bearing → B/C should bound RMS at matched loss |
| L↓, RMS(s_l_last) plateau | transient only — A is fine |
| L plateau, RMS(s_l_last)↑ | pathological drift |
EOF
}

run_arm() {
    local arm="$1"
    local gpu="$2"
    local out="${ROOT}/${arm}"
    mkdir -p "${out}"
    _write_readme "${arm}" "${out}"
    local log="${out}/stdout.log"
    echo "[${arm}] launching on GPU ${gpu} → ${out}"
    # 3 arms share one host's CPU. Cap per-rank thread pools (OMP / MKL /
    # OpenBLAS / NumExpr) so model construction + cuDNN heuristics don't
    # spawn ~70 threads/arm = 200+ threads contending for 32-64 cores.
    # Pair with ``num_workers=2`` in sts_A.yaml.
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
        uv run deepfold3 train \
            --config "configs/sts_${arm}.yaml" \
            --data-dir "${DATA_DIR}" \
            --output "${out}" \
            "${PARAMS_FLAG[@]}" \
            >"${log}" 2>&1 &
    echo "[${arm}] pid=$!"
}

run_arm A 0
run_arm B 1
run_arm C 2

echo
echo "All 3 arms launched. Tail logs:"
echo "  tail -F ${ROOT}/{A,B,C}/stdout.log"
echo "Probe JSONL files:"
echo "  ${ROOT}/{A,B,C}/single_track_probe.jsonl"
echo
echo "Waiting for all arms to complete..."
wait
echo "All arms finished."
