#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
CONFIG_DIR="/home/ubuntu/PPS-lab/test_qaoa/config"
RUNNER="/home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py"

echo "[gpu0] Starting balanced batch on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q100_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q50_L9.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q50_L7.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q50_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q100_L3.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q70_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q45_L7.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q40_L7.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q40_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q40_L4.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q40_L3.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q45_L9.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q45_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q70_L3.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q45_L3.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q55_L3.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q55_L9.yaml"
python "${RUNNER}" --config "${CONFIG_DIR}/Q40_L9.yaml"
python "${RUNNER}" --config "${CONFIG_DIR}/Q55_L7.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q55_L5.yaml"
# python "${RUNNER}" --config "${CONFIG_DIR}/Q50_L3.yaml"

echo "[gpu0] Batch complete."
