#!/bin/bash

# Fixed ML Pipeline for Python 3.12
# This script runs the ML pipeline with fixes for signature conversion

# Activate virtual environment
source ~/.pytorch_venv/bin/activate
echo "Activated Python virtual environment: $VIRTUAL_ENV"

# Apply the signature conversion fix
python fix_signature_issue.py

# Set up compute limit
export COMPUTE_UNITS_PER_DAY=100000000

# Directory setup
mkdir -p data/processed data/raw

# Parse command-line arguments
INITIAL_RUN=false
DAYS=0
TOKEN=""
MIN_DATA=10000
RETRAIN_HOURS=6

while [[ $# -gt 0 ]]; do
  case $1 in
    --initial-run)
      INITIAL_RUN=true
      shift
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --min-data)
      MIN_DATA="$2"
      shift 2
      ;;
    --retrain-hours)
      RETRAIN_HOURS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build command
CMD="python scripts/run_pipeline.py --progressive"

if [ "$INITIAL_RUN" = true ]; then
  CMD="$CMD --initial-run"
fi

if [ "$DAYS" -gt 0 ]; then
  CMD="$CMD --days $DAYS"
fi

if [ -n "$TOKEN" ]; then
  CMD="$CMD --token $TOKEN"
fi

CMD="$CMD --min-data $MIN_DATA --retrain-hours $RETRAIN_HOURS"

echo "Running fixed ML pipeline with progressive training on Python 3.12..."
echo "Command: $CMD"
echo ""
echo "⚠️ IMPORTANT NOTES ABOUT FIXES:"
echo "1. Sentiment data collection has been removed from the pipeline"
echo "2. Fixed signature conversion in OHLCV fetcher (Signature object handling)"
echo ""

# Export variable to skip sentiment step
export SKIP_SENTIMENT=true

# Execute the pipeline
$CMD

# Check exit status
if [ $? -eq 0 ]; then
  echo "Pipeline completed successfully!"
else
  echo "Pipeline failed with exit code $?"
fi 