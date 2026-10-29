#!/usr/bin/env bash
# One-time environment setup on Hangang. Run on the GPU login node
# (hangang.ksc.re.kr) so the venv is built for aarch64 (Grace / GH200):
#
#   ssh <user>@hangang.ksc.re.kr
#   git clone <repo> /scratch/$USER/deepfold3/deepfold3-dev     # or copy it over
#   bash /scratch/$USER/deepfold3/deepfold3-dev/hangang/setup_env.sh
#
# `uv sync --frozen` installs the lockfile's torch, which is the CUDA 13 build;
# Hangang's driver only supports CUDA 12.8, so torch is then reinstalled at the
# same version from PyTorch's ${DF3_TORCH_CUDA} index (see env.sh). Afterwards do
# NOT run `uv run` / `uv sync` in this repo: both re-sync to the lockfile and
# put the CUDA 13 build back. The hangang/ scripts call .venv/bin/python
# directly; re-run this script after pulling a lockfile change. If the login node cannot reach
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
PY="${REPO}/.venv/bin/python"
if [[ -n "${DF3_TORCH_CUDA}" ]]; then
    ver=$("${PY}" -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
    echo "Reinstalling torch ${ver} as the ${DF3_TORCH_CUDA} build (driver supports CUDA 12.8)"
    uv pip install --python "${PY}" --reinstall-package torch \
        --extra-index-url "https://download.pytorch.org/whl/${DF3_TORCH_CUDA}" \
        "torch==${ver}+${DF3_TORCH_CUDA}"
fi
"${PY}" - <<'EOF'
import os, platform, torch
print("python", platform.python_version(), platform.machine())
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", ".".join(map(str, torch.cuda.nccl.version())))
want = os.environ.get("DF3_TORCH_CUDA", "")
if want and not torch.__version__.endswith("+" + want):
    raise SystemExit(f"expected a +{want} torch build, got {torch.__version__}")
EOF
n=$(find "${REPO}/.venv" -type f | wc -l)
echo "venv: ${n} files (counts toward the 5M /scratch group quota)"
echo "Next: sbatch hangang/check.sbatch"
