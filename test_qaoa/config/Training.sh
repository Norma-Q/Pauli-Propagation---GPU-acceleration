#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/ubuntu/PPS-lab/test_qaoa/logs"
mkdir -p "${LOG_DIR}"

echo "Starting parallel QAOA batch process across 4 GPUs..."

pids=()
names=()
for worker in 0 1 2 3; do
  log_path="${LOG_DIR}/training_gpu${worker}.log"
  echo "[launcher] Starting Training_gpu${worker}.sh -> ${log_path}"
  bash "${SCRIPT_DIR}/Training_gpu${worker}.sh" > "${log_path}" 2>&1 &
  pids+=($!)
  names+=("gpu${worker}")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[launcher] ${names[$i]} completed successfully."
  else
    echo "[launcher] ${names[$i]} failed. Check ${LOG_DIR}/training_${names[$i]}.log"
    status=1
  fi
done

exit ${status}
