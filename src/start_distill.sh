#!/bin/bash
# start_distill.sh: Script to launch the TensorFlow distillation process

set -e

PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_PATH="./config/smartfallmm/distill.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/distillation_$TIMESTAMP.log"
WORK_DIR="experiments/distill_$TIMESTAMP"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR"

# Check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file $CONFIG_PATH not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# Check Python and dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not installed!" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -c "import tensorflow, yaml, tqdm, sklearn" &> /dev/null; then
    echo "Error: Missing required packages!" | tee -a "$LOG_FILE"
    exit 1
fi

# Set GPU configuration
export CUDA_VISIBLE_DEVICES="0"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run distillation
echo "Starting distillation at $(date)" | tee -a "$LOG_FILE"
python3 distillation.py --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Distillation completed successfully at $(date)" | tee -a "$LOG_FILE"
    echo "Results in: $WORK_DIR/scores.csv" | tee -a "$LOG_FILE"
else
    echo "Distillation failed with exit status $EXIT_STATUS! Check $LOG_FILE" | tee -a "$LOG_FILE"
    exit $EXIT_STATUS
fi
