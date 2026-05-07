#!/usr/bin/env bash
# DeepFold3 multi-node training submitter (8 nodes × 8 B200 = 64 ranks).
#
# Usage:
#   bash kakao/submit.sh                                  # auto: df3-YYYYMMDD-HHMMSS
#   bash kakao/submit.sh my-run
#   bash kakao/submit.sh my-run "--override alpha_pae=1.0 --override max_steps=50000"
#   bash kakao/submit.sh my-run --resume runs/my-run/checkpoint_5000.pt
#   WANDB_PROJECT=deepfold3-debug bash kakao/submit.sh my-run
#
# One-time prerequisites:
#   1. TrainingRuntime (reuse existing one shared across kakao examples):
#        kc apply -f ~/work/nipa-kakao/examples/esm2/trainruntime-deepspeed.yaml
#   2. WandB API key as k8s secret (DO NOT hard-code in YAML; rotate
#      `runs/notes` key after committing):
#        kubectl -n kbm-g-np-kias create secret generic wandb-api-key \
#            --from-literal=api-key="$WANDB_API_KEY"
#   3. Repo synced to /home/jovyan/work/deepfold3-dev on vv137-workspace PVC.
#   4. Training data on deepfold3-sync PVC (see configs/initial.yaml paths).
#   5. First submission creates /home/jovyan/work/deepfold3-dev/.venv via
#      `uv sync --frozen`; subsequent submissions reuse it.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
YAML="${HERE}/trainjob-multinode.yaml"

# Running from the in-cluster gateway notebook — plain `kubectl` picks up
# the SA token automatically. Override $KUBECTL / $NAMESPACE for other contexts.
NAMESPACE="${NAMESPACE:-kbm-g-np-kias}"
KUBECTL="${KUBECTL:-kubectl -n ${NAMESPACE}}"

RUN_NAME="${1:-df3-$(date "+%Y%m%d-%H%M%S")}"
shift || true
EXTRA_ARGS="${*:-}"

WANDB_PROJECT="${WANDB_PROJECT:-deepfold3-stage0}"
WANDB_ENTITY="${WANDB_ENTITY:-DeepFoldProtein}"
N_NODES="${N_NODES:-8}"

# k8s names must be DNS-1123 (lowercase alphanumerics + hyphens).
# Cap at 45 so JobSet-generated pod names fit the 63-char DNS-1123 label
# limit. Pods are ``<jobset>-launcher-0-<jobIdx>-<podIdx>``: the
# ``-launcher-0-X-YYY`` suffix can reach ~18 chars, leaving 63 - 18 = 45
# for the run name. JobSet's admission webhook rejects anything longer
# (``JobSet name is too long, pod names ... will exceed 63 characters``).
JOB_NAME="$(printf '%s' "${RUN_NAME}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-45)"
if [[ -z "${JOB_NAME}" ]]; then
    echo "run name '${RUN_NAME}' sanitises to empty job name" >&2
    exit 1
fi

cat <<EOF
╭────────────────────────────────────────────────────────────╮
│ DeepFold3 multi-node submission
│   run name      : ${RUN_NAME}
│   trainjob name : ${JOB_NAME}
│   wandb         : ${WANDB_ENTITY}/${WANDB_PROJECT}
│   extra args    : ${EXTRA_ARGS:-<none>}
│   nodes × gpus  : ${N_NODES} × 8 = $((N_NODES * 8)) ranks
╰────────────────────────────────────────────────────────────╯
EOF

# sed: use a delimiter that will not appear in substituted values. '|' is fine
# for names; EXTRA_ARGS is supplied as a single env var (not substituted in the
# command string) so embedded quotes do not corrupt the YAML.
sed \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    -e "s|__RUN_NAME__|${RUN_NAME}|g" \
    -e "s|__WANDB_PROJECT__|${WANDB_PROJECT}|g" \
    -e "s|__WANDB_ENTITY__|${WANDB_ENTITY}|g" \
    -e "s|__EXTRA_ARGS__|${EXTRA_ARGS}|g" \
    -e "s|__N_NODES__|${N_NODES}|g" \
    "${YAML}" \
    | ${KUBECTL} apply -f -

cat <<EOF

Watch  : kc get trainjob ${JOB_NAME}
Logs   : kc logs -f \$(kc get pods -l job-name=${JOB_NAME}-launcher-0 -o name | head -1)
Ranks  : tail -f /home/jovyan/work/deepfold3-dev/runs/${RUN_NAME}/logs/rank_*.log
Cancel : kc delete trainjob ${JOB_NAME}
(kc is the interactive kubectl alias from the nipa-kakao trainer guide.)
EOF
