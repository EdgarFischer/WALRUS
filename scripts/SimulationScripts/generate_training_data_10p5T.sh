#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

mkdir -p logs/logs_training_data

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/logs_training_data/generate_training_data_${timestamp}.log"

nohup env PYTHONPATH=src python3 -u scripts/SimulationScripts/generate_training_data.py \
  --config configs/Simulation/generate_training_data_10p5T.yaml \
  > "$log_file" 2>&1 &

echo "Started training-data generation."
echo "PID: $!"
echo "Log: $log_file"
