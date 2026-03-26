#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/pps-tutorial/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
CONFIG_PATH="${SCRIPT_DIR}/q22_experiment_config.json"
RUNNER="${SCRIPT_DIR}/run_q22_experiment.py"

echo "Starting Q22 batch"
echo "  python=${PYTHON_BIN}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "  config=${CONFIG_PATH}"

# depth 6
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 0 --depth 6 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 1 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 2 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 3 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 4 --depth 6 --skip-aggregate

"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 0 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 1 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 2 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 3 --depth 6 --skip-aggregate
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 4 --depth 6 --skip-aggregate

"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy tqa --seed 0 --depth 6 --skip-aggregate

# depth 9
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 0 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 1 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 2 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 3 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy random --seed 4 --depth 9 --skip-aggregate

# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 0 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 1 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 2 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 3 --depth 9 --skip-aggregate
# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy near_zero --seed 4 --depth 9 --skip-aggregate

# "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG_PATH}" --init-strategy tqa --seed 0 --depth 9 --skip-aggregate

# Uncomment when you want one final aggregate pass after the batch.
# python "${SCRIPT_DIR}/aggregate_q25_results.py" --config "${CONFIG_PATH}"

echo "Q22 batch complete."
