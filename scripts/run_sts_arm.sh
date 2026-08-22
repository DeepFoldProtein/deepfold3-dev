#!/usr/bin/env bash
# Launch a single sts arm on a specific GPU under an existing run root.
# Used both by run_sts_ab_c.sh and as a standalone for adding arms (e.g.
# arm D = η=0.5) into a mid-flight run dir.
#
# Usage:
#   bash scripts/run_sts_arm.sh <arm> <gpu> <run_root> [data_dir] [params_path]
#
# Example — add arm D to the run started at sts_20260503-190359, on GPU 3:
#   bash scripts/run_sts_arm.sh D 3 runs/sts_20260503-190359
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <arm> <gpu> <run_root> [data_dir] [params_path]" >&2
    exit 1
fi

ARM="$1"
GPU="$2"
ROOT="$3"
DATA_DIR="${4:-runs/sts_minidata}"
PARAMS_PATH="${5:-}"

PARAMS_FLAG=()
if [[ -n "${PARAMS_PATH}" ]]; then
    PARAMS_FLAG=("--params" "${PARAMS_PATH}")
fi

if [[ ! -d "${DATA_DIR}/structures" ]]; then
    echo "ERROR: ${DATA_DIR} is not a deepfold3 data-dir (no structures/)." >&2
    exit 1
fi
if [[ ! -f "configs/sts_${ARM}.yaml" ]]; then
    echo "ERROR: configs/sts_${ARM}.yaml not found." >&2
    exit 1
fi

out="${ROOT}/${ARM}"
mkdir -p "${out}"

_arm_description() {
    case "$1" in
        A) echo "s_ros = 1/√(ℓ+1) (control, no extra gain)" ;;
        B) echo "0.35 · s_ros = 0.35/√(ℓ+1) (constant 0.35× damp)" ;;
        C) echo "(1/√(ℓ+1)) · s_ros = 1/(ℓ+1) (steeper depth schedule)" ;;
        D) echo "0.5 · s_ros = 0.5/√(ℓ+1) (constant 0.5× damp)" ;;
        *) echo "(unknown arm)" ;;
    esac
}

cat > "${out}/README.md" <<EOF
# ${ROOT##*/} / arm ${ARM}

**Single-track stability ablation** — 32-record PATCHR overfit set.
Goal: separate transient single-track magnitude growth from a
load-bearing aligned-gain branch. See
\`docs/single-track-stability-experiments.md\` §11.

## Arms

| Arm | s_single_ffn (trunk single FFN) |
| --- | --- |
$( [[ "$ARM" == "A" ]] && echo "| **A (this run)** | $(_arm_description A) |" || echo "| A | $(_arm_description A) |" )
$( [[ "$ARM" == "B" ]] && echo "| **B (this run)** | $(_arm_description B) |" || echo "| B | $(_arm_description B) |" )
$( [[ "$ARM" == "C" ]] && echo "| **C (this run)** | $(_arm_description C) |" || echo "| C | $(_arm_description C) |" )
$( [[ "$ARM" == "D" ]] && echo "| **D (this run)** | $(_arm_description D) |" || echo "| D | $(_arm_description D) |" )

For all arms, every other branch (single_attn / tri_mul / pair_attn /
pair_transition) keeps the inherited \`1/√(ℓ+1)\` ROS schedule — only
the **single FFN** update is altered.

## Files

- \`config.yaml\` — fully resolved config.
- \`provenance.yaml\` — git SHA, hostname, argv, env.
- \`stdout.log\` / \`train.log\` — trainer log (every 50 effective steps).
- \`single_track_probe.jsonl\` — per-step (every 5) JSONL with all loss
  components + per-block RMS(s) / RMS(γ·u) / cos(s,u) / α + Σγα.
- \`checkpoint_*.pt\` — every 1000 steps, kept all.

## Plots (run-root level)

\`\`\`bash
uv run python scripts/plot_sts_probe.py ${ROOT} --arms A,B,C,D --watch 60
\`\`\`
EOF

echo "[${ARM}] launching on GPU ${GPU} → ${out}"
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
    uv run deepfold3 train \
        --config "configs/sts_${ARM}.yaml" \
        --data-dir "${DATA_DIR}" \
        --output "${out}" \
        "${PARAMS_FLAG[@]}" \
        >"${out}/stdout.log" 2>&1 &
echo "[${ARM}] pid=$!"
