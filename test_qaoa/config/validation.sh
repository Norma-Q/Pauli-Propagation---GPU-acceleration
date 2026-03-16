#!/bin/bash

set -euo pipefail

DEVICE="${1:-cuda}"
ROOT="/home/ubuntu/PPS-lab"
SCRIPT="$ROOT/test_qaoa/entire_qaoa_validation.py"
CFG_DIR="$ROOT/test_qaoa/config"

CONFIGS=(
  # "Q40_L3.yaml"
  # "Q40_L5.yaml"
  # "Q40_L7.yaml"
  # "Q40_L9.yaml" # 얘는 왜인지 안됨
  # "Q45_L3.yaml"
  # "Q45_L5.yaml"
  # "Q45_L7.yaml"
  # "Q45_L9.yaml"
  # "Q50_L3.yaml"
  # "Q50_L5.yaml"
  # "Q50_L7.yaml"
  # "Q50_L9.yaml"
  # "Q55_L3.yaml"
  # "Q55_L5.yaml"
  # "Q55_L7.yaml"
  # "Q55_L9.yaml",
)

echo "Starting validation batch for 40Q/45Q (device=$DEVICE)"

total=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
  idx=$((i + 1))
  cfg_name="${CONFIGS[$i]}"
  cfg_path="$CFG_DIR/$cfg_name"

  echo "[$idx/$total] Running validation: $cfg_name"
  python "$SCRIPT" -c "$cfg_path" --device "$DEVICE"
done

echo "All 40Q/45Q validations completed."
