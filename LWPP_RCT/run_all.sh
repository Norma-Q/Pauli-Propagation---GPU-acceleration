#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/pps-tutorial/bin/python}"
CONFIG_PATH="${1:-${SCRIPT_DIR}/q25_experiment_config.json}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_q25_experiment.py" --config "${CONFIG_PATH}"
