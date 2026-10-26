#!/usr/bin/env bash
# One-time environment setup on Hangang. Run on the GPU login node
# (hangang.ksc.re.kr) so the venv is built for aarch64 (Grace / GH200):
#
#   ssh <user>@hangang.ksc.re.kr
#   git clone <repo> /scratch/$USER/deepfold3/deepfold3-dev     # or copy it over
#   bash /scratch/$USER/deepfold3/deepfold3-dev/hangang/setup_env.sh
#
# The lockfile resolves torch's CUDA 13 wheels for linux-aarch64 too, so a
# plain `uv sync --frozen` gives a GPU build. If the login node cannot reach
# PyPI, run this on the data mover only if it is also aarch64 (`uname -m`);
# otherwise use the KISTI PyTorch container (see hangang/README.md).

set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "arch: $(uname -m)  host: $(hostname)"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "WARNING: not aarch64 — a venv built here will not run on the GH200 nodes." >&2
fi

mkdir -p "${DF3_ROOT}/bin" "${RUNS}"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${DF3_ROOT}/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi

cd "${REPO}"
uv sync --frozen --python 3.11
uv run python - <<'EOF'
import platform, torch
print("python", platform.python_version(), platform.machine())
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", ".".join(map(str, torch.cuda.nccl.version())))
EOF
n=$(find "${REPO}/.venv" -type f | wc -l)
echo "venv: ${n} files (counts toward the 5M /scratch group quota)"
echo "Next: sbatch hangang/check.sbatch"
