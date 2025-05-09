#!/bin/bash
# train_teacher.sh - Train the skeleton-based teacher model
set -e

# Configuration
CONFIG_FILE="config/smartfallmm/teacher.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/teacher_${TIMESTAMP}"
MODEL_NAME="teacher_model"
GPU_ID=0

# Make sure directories exist
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"

echo "Starting teacher model training at $(date)"
echo "Using config: ${CONFIG_FILE}"
echo "Results will be in: ${WORK_DIR}"

# Save config for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
cp "models/mm_transformer.py" "${WORK_DIR}/models/"

# Run training with retries
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python train.py \
      --config "${CONFIG_FILE}" \
      --work-dir "${WORK_DIR}" \
      --model-saved-name "${MODEL_NAME}" \
      --device "${GPU_ID}"; then
        echo "Teacher model training completed successfully at $(date)"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Training attempt $RETRY_COUNT failed. Retrying..."
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Teacher model training failed after $MAX_RETRIES attempts"
    exit 1
fi

# Create symbolic link to latest experiment
cd "$(dirname "${WORK_DIR}")"
rm -f latest_teacher
ln -s "$(basename "${WORK_DIR}")" latest_teacher
cd -

echo "Teacher model training completed. Output directory: ${WORK_DIR}"
