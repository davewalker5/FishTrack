#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$project_root"

# Reuse a compatible environment; never delete an existing environment.
if [[ ! -x venv/bin/python ]]; then
    if [[ -n ${FISHTRACK_PYTHON:-${SAM2_PYTHON:-}} ]]; then
        "${FISHTRACK_PYTHON:-$SAM2_PYTHON}" -m venv venv
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv venv
    elif command -v python3.13 >/dev/null 2>&1; then
        python3.13 -m venv venv
    elif command -v uv >/dev/null 2>&1; then
        uv python install 3.12 --install-dir "$project_root/.runtime/python" \
            --cache-dir "$project_root/.cache/uv"
        UV_PYTHON_INSTALL_DIR="$project_root/.runtime/python" \
            uv venv --python 3.12 --seed venv --cache-dir "$project_root/.cache/uv"
    else
        echo "Install Python 3.12/3.13 or uv, or set FISHTRACK_PYTHON to a compatible executable." >&2
        exit 1
    fi
fi

if ! venv/bin/python -c 'import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] < (3, 14)))'; then
    echo "FishTrack setup needs Python 3.12/3.13. The existing venv was preserved." >&2
    echo "Move it aside and rerun with FISHTRACK_PYTHON pointing to a compatible Python." >&2
    exit 1
fi

echo "Installing FishTrack, desktop controls and tracking dependencies ..."
venv/bin/python -m pip install --upgrade pip

# SAM 2's optional CUDA extension is unnecessary for CPU/MPS inference.
SAM2_BUILD_CUDA=0 venv/bin/python -m pip install -e .
echo "Preparing SAM 2 Tiny and Grounding DINO Tiny models ..."
venv/bin/python -m fishtrack.model_setup --model tiny --directory "$project_root/models"
venv/bin/python -m fishtrack.grounding_setup --directory "$project_root/models/grounding-dino-tiny"
venv/bin/python -m pip check

echo "FishTrack is ready. Launch with ./scripts/run.sh"
