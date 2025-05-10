#!/bin/bash
# Robust student model training script with error prevention

# Enable strict error handling but handle undefined variables safely
set -eo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/student.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/student_${TIMESTAMP}"
MODEL_NAME="student_model_${TIMESTAMP}"
GPU_ID=0
LEARNING_RATE=0.0005
DROPOUT=0.5
BATCH_SIZE=16
NUM_EPOCHS=80
USE_SMV=false
NUM_WORKERS=40

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
echo "====== LightHART-TF Student Model Training ======"
echo "Configuration:"
echo "  Model: Student (Transformer Optimized)"
echo "  GPU: $GPU_ID"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Worker processes: $NUM_WORKERS"
echo "  Using SMV: $USE_SMV"
echo "  Working directory: $WORK_DIR"
echo "==========================================\n"

# Create required directories
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/config"  # Separate config directory

# Save original config and code for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/config/original_config.yaml"
mkdir -p "${WORK_DIR}/code"
cp "models/transformer_optimized.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: models/transformer_optimized.py not found"
cp "utils/dataset_tf.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: utils/dataset_tf.py not found"
cp "utils/processor_tf.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: utils/processor_tf.py not found"
cp "train.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: train.py not found"

# Create a custom config with our parameters
CUSTOM_CONFIG="${WORK_DIR}/config/student_custom.yaml"  # Different path to avoid collision
cp "${CONFIG_FILE}" "${CUSTOM_CONFIG}"

# Update parameters in the config
sed -i "s|^model:.*|model: models.transformer_optimized.TransModel|" "${CUSTOM_CONFIG}"
sed -i "s/base_lr:.*/base_lr: ${LEARNING_RATE}/" "${CUSTOM_CONFIG}"
sed -i "s/batch_size:.*/batch_size: ${BATCH_SIZE}/" "${CUSTOM_CONFIG}"
sed -i "s/num_epoch:.*/num_epoch: ${NUM_EPOCHS}/" "${CUSTOM_CONFIG}"
sed -i "s/dropout:.*/dropout: ${DROPOUT}/" "${CUSTOM_CONFIG}"
sed -i "s/feeder:.*/feeder: utils.dataset_tf.UTD_MM_TF/" "${CUSTOM_CONFIG}"
sed -i "s/num_worker:.*/num_worker: ${NUM_WORKERS}/" "${CUSTOM_CONFIG}"
sed -i "s/use_smv:.*/use_smv: ${USE_SMV}/" "${CUSTOM_CONFIG}"

# Important change: When using only accelerometer, disable DTW
if grep -q "modalities: \['accelerometer'\]" "${CUSTOM_CONFIG}"; then
  echo "Accelerometer-only mode detected. Disabling DTW alignment."
  sed -i "s/use_dtw:.*/use_dtw: false/" "${CUSTOM_CONFIG}"
fi

# Set TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# For better multi-processing performance
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Add current directory to Python path (fix for undefined variable)
export PYTHONPATH=".:${PYTHONPATH:-}"

# Run training
echo "Starting student model training..."
python train.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --num-worker "${NUM_WORKERS}" \
  --use-smv "${USE_SMV}" \
  --phase "train"

# Check training status
TRAINING_STATUS=$?

if [ ${TRAINING_STATUS} -eq 0 ]; then
  echo "Student model training completed successfully at $(date)"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_student 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_student
  cd - > /dev/null
  
  echo "Created latest_student symlink"
  echo "All training artifacts saved to: ${WORK_DIR}"
else
  echo "⚠️ Student model training failed with status ${TRAINING_STATUS}"
  exit ${TRAINING_STATUS}
fi

echo "✅ Student model training workflow completed successfully!"
