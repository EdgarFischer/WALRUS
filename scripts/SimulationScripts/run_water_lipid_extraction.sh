#!/usr/bin/env bash
set -euo pipefail

# Wechsel vom Ordner scripts/SimulationScripts zum Projekt-Root.
cd "$(dirname "$0")/../.."

config_path="configs/Simulation/10p5T_water_lipid_v1.yaml"

log_dir="logs/logs_water_lipid_extraction"
mkdir -p "$log_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/water_lipid_extraction_${timestamp}.log"

python_executable="/home/hfischer/venvs/walinet/bin/python"

if [[ ! -x "$python_executable" ]]; then
    echo "Python executable not found:"
    echo "  $python_executable"
    exit 1
fi

if [[ ! -f "$config_path" ]]; then
    echo "Config file not found:"
    echo "  $config_path"
    exit 1
fi

nohup env PYTHONPATH="$PWD/src" \
    "$python_executable" -u \
    scripts/SimulationScripts/extract_water_lipid.py \
    "$config_path" \
    > "$log_file" 2>&1 &

pid=$!

echo "$pid" > "${log_dir}/latest.pid"

echo "Started water/lipid extraction."
echo "PID: $pid"
echo "Log: $log_file"
echo
echo "Follow progress with:"
echo "tail -f \"$log_file\""