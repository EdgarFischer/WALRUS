#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

mkdir -p logs/logs_training

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/logs_training/train_${timestamp}.log"

nohup env PYTHONPATH=src python3 -u scripts/trainingScripts/train.py \
  --config configs/Training/train_10p5T.yaml \
  > "$log_file" 2>&1 &

echo "Started training."
echo "PID: $!"
echo "Log: $log_file"