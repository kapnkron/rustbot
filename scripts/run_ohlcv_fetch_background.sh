#!/bin/bash

# Activate the virtual environment
source ~/.pytorch_venv/bin/activate

# Create output directories
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run the fetcher in the background with nohup
# This allows the process to continue running even if you close the terminal
nohup python scripts/fetch_historical_ohlcv.py "$@" > logs/ohlcv_fetch_${TIMESTAMP}.log 2>&1 &

# Get the process ID
PID=$!

echo "Started OHLCV fetcher in background with PID: $PID"
echo "Log file: logs/ohlcv_fetch_${TIMESTAMP}.log"
echo "To check progress, use: tail -f logs/ohlcv_fetch_${TIMESTAMP}.log"
echo "To stop the process, use: kill $PID"

# Write the PID to a file for easy reference
echo $PID > logs/ohlcv_fetch_pid.txt 