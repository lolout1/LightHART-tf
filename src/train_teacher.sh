#!/bin/bash
# train_teacher.sh - Train the teacher model with skeleton and accelerometer data

# Enable strict error handling
set -euo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/teacher.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/teacher_${TIMESTAMP}"
MODEL_NAME="teacher_model"
GPU_ID=0
LEARNING_RATE=0.001
DROPOUT=0.2
BATCH_SIZE=16
NUM_EPOCHS=80

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Make sure directories exist
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"

echo "Starting teacher model training at $(date)"
echo "Using config: ${CONFIG_FILE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Dropout: ${DROPOUT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Results will be in: ${WORK_DIR}"

# Save config for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
cp "models/mm_transformer.py" "${WORK_DIR}/models/"

# Run training
python train.py \
  --config "${CONFIG_FILE}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --base-lr "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epoch "${NUM_EPOCHS}" \
  --drop-rate "${DROPOUT}"

# Check training status
TRAINING_STATUS=$?

if [ ${TRAINING_STATUS} -eq 0 ]; then
  echo "Teacher model training completed successfully at $(date)"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_teacher
  ln -s "$(basename "${WORK_DIR}")" latest_teacher
  cd -
else
  echo "Teacher model training failed with status ${TRAINING_STATUS}"
  exit ${TRAINING_STATUS}
fi

echo "Teacher model training completed. Output directory: ${WORK_DIR}"
