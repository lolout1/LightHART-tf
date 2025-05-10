#!/bin/bash
# Modified train_teacher.sh with path resolution fix

# Enable strict error handling
set -euo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/teacher.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="modelsKD"
MODEL_NAME="teacher_model"
GPU_ID=0
LEARNING_RATE=0.0001
DROPOUT=0.2
BATCH_SIZE=16
NUM_EPOCHS=80
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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Display job configuration
echo "====== LightHART-TF Teacher Model Training ======"
echo "Configuration:"
echo "  Model: Teacher (Multi-Modal Transformer)"
echo "  GPU: $GPU_ID"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Worker processes: $NUM_WORKERS"
echo "  Working directory: $WORK_DIR"
echo "==========================================\n"

# Create required directories
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations" 
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/config" # Separate config directory

# Save original config for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/config/original_config.yaml"

# Save code for reference
mkdir -p "${WORK_DIR}/code"
cp "models/mm_transformer.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: models/mm_transformer.py not found"
cp "utils/dataset_tf.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: utils/dataset_tf.py not found"
cp "train.py" "${WORK_DIR}/code/" 2>/dev/null || echo "Warning: train.py not found"

# Create a custom config with our parameters
CUSTOM_CONFIG="${WORK_DIR}/config/teacher_custom.yaml" # Changed path to avoid collision
cp "${CONFIG_FILE}" "${CUSTOM_CONFIG}"

# Update parameters in the config
sed -i "s|^model:.*|model: models.mm_transformer.MMTransformer|" "${CUSTOM_CONFIG}"
sed -i "s/base_lr:.*/base_lr: ${LEARNING_RATE}/" "${CUSTOM_CONFIG}"
sed -i "s/batch_size:.*/batch_size: ${BATCH_SIZE}/" "${CUSTOM_CONFIG}"
sed -i "s/num_epoch:.*/num_epoch: ${NUM_EPOCHS}/" "${CUSTOM_CONFIG}"
sed -i "s/drop_rate:.*/drop_rate: ${DROPOUT}/" "${CUSTOM_CONFIG}"
sed -i "s/feeder:.*/feeder: utils.dataset_tf.UTD_MM_TF/" "${CUSTOM_CONFIG}"
sed -i "s/num_worker:.*/num_worker: ${NUM_WORKERS}/" "${CUSTOM_CONFIG}"

# Set TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TF logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Performance settings
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Add current directory to Python path (with proper handling of undefined variable)
export PYTHONPATH=".:${PYTHONPATH:-}"

# Run training
echo "Starting teacher model training..."
python train.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --num-worker "${NUM_WORKERS}" \
  --phase "train"

# Check training status
TRAINING_STATUS=$?

if [ ${TRAINING_STATUS} -eq 0 ]; then
  echo "Teacher model training completed successfully at $(date)"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_teacher 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_teacher
  cd - > /dev/null
  
  echo "Created latest_teacher symlink"
  echo "All training artifacts saved to: ${WORK_DIR}"
else
  echo "⚠️ Teacher model training failed with status ${TRAINING_STATUS}"
  exit ${TRAINING_STATUS}
fi

echo "✅ Teacher model training workflow completed successfully!"
