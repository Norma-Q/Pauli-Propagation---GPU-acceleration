#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_name="${1:-pps-tutorial}"
python_ver="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

echo "[1/3] Creating conda env: $env_name (python=$python_ver)"
conda create -y -n "$env_name" "python=$python_ver" pip

echo "[2/3] Installing pip dependencies from requirements-tutorial.txt"
conda run -n "$env_name" python -m pip install --upgrade pip
conda run -n "$env_name" python -m pip install -r requirements-tutorial.txt

echo "[3/3] Registering Jupyter kernel: $env_name"
conda run -n "$env_name" python -m ipykernel install --user --name "$env_name" --display-name "PPS Tutorial ($env_name)"

cat <<EOF

Done.
- Activate: conda activate $env_name
- Launch Jupyter: jupyter lab
- In notebooks: Kernel -> Change kernel -> PPS Tutorial ($env_name)

Note:
- The compiled backend lives at src_tensor/_pps_tensor_backend_local.so.
  It expects a compatible Python (3.11) + PyTorch (2.2.0+cu121).
EOF
