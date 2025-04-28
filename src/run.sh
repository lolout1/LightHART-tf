#!/bin/bash
# Enhanced script for training and knowledge distillation

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
TEACHER_WORK_DIR="experiments/teacher_$TIMESTAMP"
DISTILL_WORK_DIR="experiments/distill_$TIMESTAMP"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$TEACHER_WORK_DIR"
mkdir -p "$DISTILL_WORK_DIR"
mkdir -p models

# Logging function
log() {
    echo "[$(date +%H:%M:%S)] $1"
    echo "[$(date +%H:%M:%S)] $1" >> "$LOG_DIR/run_$TIMESTAMP.log"
}

# Check for tensorflow and NumPy installation
if ! python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" &> /dev/null; then
    log "ERROR: TensorFlow not installed. Please install TensorFlow."
    exit 1
fi

log "Starting TensorFlow training pipeline..."

# First train the teacher model
log "Starting teacher model training..."
python train.py \
    --config config/smartfallmm/teacher.yaml \
    --work-dir "$TEACHER_WORK_DIR" \
    --model-saved-name "teacher_model" \
    --device 0 \
    --base-lr 1e-3 2>&1 | tee -a "$LOG_DIR/teacher_$TIMESTAMP.log"

# Create symlink to latest teacher
ln -sf "teacher_$TIMESTAMP" "experiments/latest_teacher"

# Then run knowledge distillation
log "Starting knowledge distillation"
python train.py \
    --config config/smartfallmm/distill.yaml \
    --work-dir "$DISTILL_WORK_DIR" \
    --model-saved-name "distilled_model" \
    --teacher-weight "$TEACHER_WORK_DIR/models/teacher_model" \
    --device 0 \
    --base-lr 5e-4 2>&1 | tee -a "$LOG_DIR/distill_$TIMESTAMP.log"

# Create symlink to latest distillation
ln -sf "distill_$TIMESTAMP" "experiments/latest_distill"

log "Training completed successfully."
