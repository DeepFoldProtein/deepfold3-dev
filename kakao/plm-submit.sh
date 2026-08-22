#!/usr/bin/env bash
# Complex-LM training submitter (1 node × 8 B200 by default).
#
# Usage:
#   bash kakao/plm-submit.sh                                   # auto: plm-YYYYMMDD-HHMMSS
#   bash kakao/plm-submit.sh plm-r1
#   bash kakao/plm-submit.sh plm-ccg "--override model.layer_pattern=ccg"
#   bash kakao/plm-submit.sh plm-r1 --resume runs/plm-r1/checkpoint_20000.pt
#   PLM_CONFIG=configs/plm-smoke.yaml bash kakao/plm-submit.sh plm-smoke
#   N_NODES=2 bash kakao/plm-submit.sh plm-big
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
# Chosen after N_NODES is known, below.

# Running from the in-cluster gateway notebook — plain `kubectl` picks up
# the SA token automatically. Override $KUBECTL / $NAMESPACE for other contexts.
NAMESPACE="${NAMESPACE:-kbm-g-np-kias}"
KUBECTL="${KUBECTL:-kubectl -n ${NAMESPACE}}"

RUN_NAME="${1:-plm-$(date "+%Y%m%d-%H%M%S")}"
shift || true
EXTRA_ARGS="${*:-}"

WANDB_PROJECT="${WANDB_PROJECT:-deepfold3-complexlm}"
WANDB_ENTITY="${WANDB_ENTITY:-DeepFoldProtein}"
N_NODES="${N_NODES:-1}"
PLM_CONFIG="${PLM_CONFIG:-configs/plm-base.yaml}"

# One node launches torchrun directly; more than one needs mpirun to fan it
# out. They are separate templates because numNodes=1 creates no worker pods
# for mpirun's hostfile to name.
# One node is a plain Job: a TrainJob with numNodes=1 still materialises a
# launcher *and* a worker pod, each asking for 8 GPUs, so it deadlocks against
# itself on a 8-GPU allocation. Above one node the TrainJob/mpirun path is what
# fans torchrun out.
if [[ "${N_NODES}" == "1" ]]; then
    YAML="${HERE}/plm-job.yaml"
    KIND="job"
else
    YAML="${HERE}/plm-trainjob-multinode.yaml"
    KIND="trainjob"
fi

# --- NCCL / collective debug logging (opt-in via NCCL_LOG=1) ------------------
# Confirms the ~80-min periodic collective stall (see export/burst-periodic-stall/).
# NCCL_LOG=1: verbose per-collective + net/proxy timing (NCCL_DEBUG=INFO, SUBSYS
# INIT,NET,COLL,PROFILE) written to a NODE-LOCAL file /tmp/nccl.<host>.<pid>.log
# (not the shared PVC, so it doesn't add fabric/PVC I/O and confound the
# measurement) + flight-recorder collective timing. Collect after a burst with
#   bash monitor/collect_nccl_logs.sh <job>
# RAS straggler detection is ALWAYS on regardless of this toggle — query it live:
#   kubectl exec <pod> -- ncclras
# Default (NCCL_LOG unset) = WARN to stdout, run unperturbed.
if [[ "${NCCL_LOG:-0}" == "1" ]]; then
    NCCL_DEBUG_VAL="INFO"
    NCCL_DEBUG_SUBSYS_VAL="INIT,NET,COLL,PROFILE"
    NCCL_DEBUG_FILE_VAL="/tmp/nccl.%h.%p.log"
    TORCH_TIMING_VAL="1"
    NCCL_LOG_BANNER="ON -> /tmp/nccl.<host>.<pid>.log (node-local; monitor/collect_nccl_logs.sh)"
else
    NCCL_DEBUG_VAL="WARN"
    NCCL_DEBUG_SUBSYS_VAL="INIT,TUNING"
    NCCL_DEBUG_FILE_VAL=""
    TORCH_TIMING_VAL="0"
    NCCL_LOG_BANNER="off (NCCL_LOG=1 to capture collective timing)"
fi

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

LOG_SUFFIX=""
[[ "${KIND}" == "trainjob" ]] && LOG_SUFFIX="-launcher-0"

cat <<EOF
╭────────────────────────────────────────────────────────────╮
│ DeepFold3 multi-node submission
│   run name      : ${RUN_NAME}
│   trainjob name : ${JOB_NAME}
│   wandb         : ${WANDB_ENTITY}/${WANDB_PROJECT}
│   config        : ${PLM_CONFIG}
│   extra args    : ${EXTRA_ARGS:-<none>}
│   nodes × gpus  : ${N_NODES} × 8 = $((N_NODES * 8)) ranks (${KIND})
│   nccl debug    : ${NCCL_LOG_BANNER}
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
    -e "s|__PLM_CONFIG__|${PLM_CONFIG}|g" \
    -e "s|__N_NODES__|${N_NODES}|g" \
    -e "s|__NCCL_DEBUG__|${NCCL_DEBUG_VAL}|g" \
    -e "s|__NCCL_DEBUG_SUBSYS__|${NCCL_DEBUG_SUBSYS_VAL}|g" \
    -e "s|__NCCL_DEBUG_FILE__|${NCCL_DEBUG_FILE_VAL}|g" \
    -e "s|__TORCH_NCCL_ENABLE_TIMING__|${TORCH_TIMING_VAL}|g" \
    "${YAML}" \
    | ${KUBECTL} apply -f -

cat <<EOF

Watch  : kc get ${KIND} ${JOB_NAME}
Logs   : kc logs -f \$(kc get pods -l job-name=${JOB_NAME}${LOG_SUFFIX} -o name | head -1)
Ranks  : tail -f /home/jovyan/work/deepfold3-dev/runs/${RUN_NAME}/logs/*/*/stdout.log
Cancel : kc delete ${KIND} ${JOB_NAME}
(kc is the interactive kubectl alias from the nipa-kakao trainer guide.)
EOF
