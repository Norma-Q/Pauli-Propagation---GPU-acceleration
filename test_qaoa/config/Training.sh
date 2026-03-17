#!/bin/bash

# [핵심] 에러 발생 시 즉시 스크립트 종료
set -e

# 현재 어떤 명령어가 실행되는지 출력하고 싶다면 아래 주석을 해제하세요
# set -x

echo "Starting QAOA batch process..."

# echo "[1/17] Running Q40_L3..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q40_L3.yaml

# echo "[2/17] Running Q40_L4..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q40_L4.yaml

# echo "[3/17] Running Q40_L5..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q40_L5.yaml

# echo "[4/17] Running Q40_L7..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q40_L7.yaml

# echo "[5/17] Running Q40_L9..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q40_L9.yaml

# echo "[6/17] Running Q45_L3..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q45_L3.yaml

# echo "[7/17] Running Q45_L5..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q45_L5.yaml

# echo "[8/17] Running Q45_L7..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q45_L7.yaml

# echo "[9/17] Running Q45_L9..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q45_L9.yaml

# echo "[10/17] Running Q50_L3..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q50_L3.yaml

# echo "[11/17] Running Q50_L5..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q50_L5.yaml

# echo "[12/17] Running Q50_L7..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q50_L7.yaml

# echo "[13/17] Running Q50_L9..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q50_L9.yaml

# echo "[14/17] Running Q55_L3..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q55_L3.yaml

# echo "[15/17] Running Q55_L5..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q55_L5.yaml

# echo "[16/17] Running Q55_L7..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q55_L7.yaml

# echo "[17/17] Running Q55_L9..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q55_L9.yaml



########### 100Q 실험해보자
# echo "[1/2] Running Q100_L3..."
# python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q100_L3.yaml

echo "[2/2] Running Q100_L5..."
python /home/ubuntu/PPS-lab/test_qaoa/entire_qaoa_process.py --config /home/ubuntu/PPS-lab/test_qaoa/config/Q100_L5.yaml

echo "All tasks completed successfully!"
