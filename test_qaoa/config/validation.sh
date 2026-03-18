#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-auto}"
THRESHOLDS="${2:-1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4}"
STOP_AFTER_FAILS="${3:-1}"

ROOT="/home/ubuntu/PPS-lab"
SCRIPT="$ROOT/test_qaoa/entire_qaoa_validation.py"
CFG_DIR="$ROOT/test_qaoa/config"

CONFIGS=(
  "Q40_L3.yaml"
  "Q40_L4.yaml"
  "Q40_L5.yaml"
  "Q40_L7.yaml"
  "Q40_L9.yaml"
  "Q45_L3.yaml"
  "Q45_L5.yaml"
  "Q45_L7.yaml"
  "Q45_L9.yaml"
  "Q50_L3.yaml"
  "Q50_L5.yaml"
  "Q50_L7.yaml"
  "Q50_L9.yaml"
  "Q55_L3.yaml"
  "Q55_L5.yaml"
  "Q55_L7.yaml"
  "Q55_L9.yaml"
  "Q70_L3.yaml"
  "Q70_L5.yaml"
  "Q100_L3.yaml"
  "Q100_L5.yaml"
)

echo "Starting validation batch"
echo "  device=${DEVICE}"
echo "  thresholds=${THRESHOLDS}"
echo "  stop_after_consecutive_failures=${STOP_AFTER_FAILS}"

total=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
  idx=$((i + 1))
  cfg_name="${CONFIGS[$i]}"
  cfg_path="$CFG_DIR/$cfg_name"

  echo
  echo "[$idx/$total] Running validation: ${cfg_name}"
  "${PYTHON_BIN}" "$SCRIPT" \
    --config "$cfg_path" \
    --device "$DEVICE" \
    --thresholds "$THRESHOLDS" \
    --eval-mode auto \
    --stop-after-consecutive-failures "$STOP_AFTER_FAILS"
done

echo
echo "Validation batch complete."
