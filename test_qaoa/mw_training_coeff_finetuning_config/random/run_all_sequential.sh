#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT="/home/ubuntu/PPS-lab"
RUNNER="$ROOT/test_qaoa/mw_training_coeff_finetuning.py"
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_STRATEGY="random"

mapfile -t CONFIG_PATHS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)

echo "Starting MW + coefficient finetuning batch"
echo "  init_strategy=${INIT_STRATEGY}"
echo "  python=${PYTHON_BIN}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "  config_dir=${CONFIG_DIR}"

total=${#CONFIG_PATHS[@]}
for i in "${!CONFIG_PATHS[@]}"; do
  idx=$((i + 1))
  cfg_path="${CONFIG_PATHS[$i]}"
  cfg_name="$(basename "$cfg_path")"

  echo
  echo "[$idx/$total] Running ${cfg_name}"
  "${PYTHON_BIN}" "$RUNNER" --config "$cfg_path" --init-strategy "${INIT_STRATEGY}"
done

echo
echo "MW + coefficient finetuning batch complete."
