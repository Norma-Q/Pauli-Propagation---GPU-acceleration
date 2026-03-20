#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/pps-tutorial/bin/python}"
CONFIG_PATH="${SCRIPT_DIR}/q10_example_config.json"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_q10_example}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_q25_experiment.py" \
  --config "${CONFIG_PATH}"
