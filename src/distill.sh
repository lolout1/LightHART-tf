#!/bin/bash
# Robust knowledge distillation script for LightHART-TF

# Enable strict error handling
set -euo pipefail

# Default configuration
CONFIG_FILE="config/smartfallmm/distill.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/distilled_${TIMESTAMP}"
MODEL_NAME="distilled_model"
GPU_ID=0
LEARNING_RATE=0.001
DROPOUT=0.5
BATCH_SIZE=16
NUM_EPOCHS=80
NUM_WORKERS=40
TEMPERATURE=4.5
ALPHA=0.6
USE_SMV=false

# Override default teacher weights path
# Use the actual path from your directory listing
TEACHER_WEIGHTS="../experiments/teacher_2025-05-09_21-04-16/models/teacher_model_20250509_210419"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --dropout)
      DROPOUT="$2"
      shift 2
      ;;
    --batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --workdir)
      WORK_DIR="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --teacher)
      TEACHER_WEIGHTS="$2"
      shift 2
      ;;
    --temp)
      TEMPERATURE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --use-smv)
      USE_SMV=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Display job configuration
echo "====== LightHART-TF Knowledge Distillation ======"
echo "Configuration:"
echo "  Teacher weights: $TEACHER_WEIGHTS"
echo "  GPU: $GPU_ID"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Worker processes: $NUM_WORKERS"
echo "  Temperature: $TEMPERATURE"
echo "  Alpha: $ALPHA"
echo "  Using SMV: $USE_SMV"
echo "  Working directory: $WORK_DIR"
echo "==========================================\n"

# Verify teacher weights exist for at least one subject
TEACHER_WEIGHTS_FOUND=false
echo "Checking for teacher weights..."

# List all subjects we need to check
SUBJECTS=(32 39 30 31 33 34 35 37 43 44 29 36 45)

for SUBJECT in "${SUBJECTS[@]}"; do
    # Check for both weight file formats
    WEIGHT_PATH_H5="${TEACHER_WEIGHTS}_${SUBJECT}.weights.h5"
    WEIGHT_PATH_KERAS="${TEACHER_WEIGHTS}_${SUBJECT}.keras"
    
    if [ -f "$WEIGHT_PATH_H5" ]; then
        TEACHER_WEIGHTS_FOUND=true
        echo "✓ Verified teacher weights for subject ${SUBJECT}: ${WEIGHT_PATH_H5}"
    elif [ -f "$WEIGHT_PATH_KERAS" ]; then
        TEACHER_WEIGHTS_FOUND=true
        echo "✓ Verified teacher weights for subject ${SUBJECT}: ${WEIGHT_PATH_KERAS}"
    fi
done

if [ "$TEACHER_WEIGHTS_FOUND" = false ]; then
    echo "ERROR: No teacher weights found at path: ${TEACHER_WEIGHTS}_*.weights.h5 or ${TEACHER_WEIGHTS}_*.keras"
    echo "Please check the path and try again."
    exit 1
fi

# Verify data directory exists
echo "Checking for data directory..."
declare -a DATA_DIRS=(
  "../data/smartfallmm"
  "./data/smartfallmm"
  "$HOME/data/smartfallmm"
  "/mmfs1/home/sww35/data/smartfallmm"
)

DATA_DIR_FOUND=false
for dir in "${DATA_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Found data directory at: $dir"
    DATA_DIR_FOUND=true
    break
  fi
done

if [ "$DATA_DIR_FOUND" = false ]; then
  echo "WARNING: No data directory found! Please ensure SmartFall data is available."
  echo "Expected locations: ${DATA_DIRS[*]}"
fi

# Create required directories
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/code"

# Save code for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
cp "distiller.py" "${WORK_DIR}/code/"
cp "models/transformer_optimized.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: transformer_optimized.py not found"
cp "models/mm_transformer.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: mm_transformer.py not found"
cp "utils/dataset_tf.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: dataset_tf.py not found"
cp "utils/loss.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: loss.py not found"

# Create a custom config with our parameters
CUSTOM_CONFIG="${WORK_DIR}/distill_custom.yaml"
cp "${CONFIG_FILE}" "${CUSTOM_CONFIG}"

# Update parameters in the config
sed -i "s/base_lr:.*/base_lr: ${LEARNING_RATE}/" "${CUSTOM_CONFIG}"
sed -i "s/batch_size:.*/batch_size: ${BATCH_SIZE}/" "${CUSTOM_CONFIG}"
sed -i "s/num_epoch:.*/num_epoch: ${NUM_EPOCHS}/" "${CUSTOM_CONFIG}"
sed -i "s/temperature:.*/temperature: ${TEMPERATURE}/" "${CUSTOM_CONFIG}"
sed -i "s/alpha:.*/alpha: ${ALPHA}/" "${CUSTOM_CONFIG}"
sed -i "s|teacher_weight:.*|teacher_weight: ${TEACHER_WEIGHTS}|" "${CUSTOM_CONFIG}"
sed -i "s/num_worker:.*/num_worker: ${NUM_WORKERS}/" "${CUSTOM_CONFIG}"
sed -i "s/use_smv:.*/use_smv: ${USE_SMV}/" "${CUSTOM_CONFIG}"
sed -i "s/dropout:.*/dropout: ${DROPOUT}/" "${CUSTOM_CONFIG}"
sed -i "s/feeder:.*/feeder: utils.dataset_tf.UTD_MM_TF/" "${CUSTOM_CONFIG}"
sed -i "s/verbose:.*/verbose: true/" "${CUSTOM_CONFIG}"

# Ensure model and teacher model paths are correct
sed -i "s|model:.*|model: models.transformer_optimized.TransModel|" "${CUSTOM_CONFIG}"
sed -i "s|teacher_model:.*|teacher_model: models.mm_transformer.MMTransformer|" "${CUSTOM_CONFIG}"

# Update dataset mode to selective window
sed -i "s/mode:.*/mode: 'selective_window'/" "${CUSTOM_CONFIG}"

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Performance settings
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Add current directory to Python path
export PYTHONPATH=".:${PYTHONPATH:-}"

# Run distillation
echo "Starting knowledge distillation using teacher model: ${TEACHER_WEIGHTS}"
echo "Running with configuration from: ${CUSTOM_CONFIG}"

python distiller.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --teacher-weight "${TEACHER_WEIGHTS}" \
  --num-worker "${NUM_WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --alpha "${ALPHA}" \
  --use-smv "${USE_SMV}" \
  --phase "distill"

# Check distillation status
DISTILL_STATUS=$?

if [ ${DISTILL_STATUS} -eq 0 ]; then
  echo "Distillation completed successfully at $(date)"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_distilled 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_distilled
  cd - > /dev/null
  
  echo "Created latest_distilled symlink"
  echo "All distillation artifacts saved to: ${WORK_DIR}"
  
  # List the results
  echo "Generated models:"
  ls -la "${WORK_DIR}/models/" || echo "No models found"
  
  echo "Results:"
  if [ -f "${WORK_DIR}/distillation_scores.csv" ]; then
    cat "${WORK_DIR}/distillation_scores.csv"
  else
    echo "No scores file found"
  fi
  
else
  echo "⚠️ Distillation failed with status ${DISTILL_STATUS}"
  exit ${DISTILL_STATUS}
fi

echo "✅ Knowledge distillation workflow completed successfully!"
