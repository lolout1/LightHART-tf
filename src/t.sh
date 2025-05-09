#!/bin/bash
# train_mm_transformer.sh - Script to train the multi-modal transformer model from src directory

# Enable strict error handling
set -euo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/mm_transformer.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/mm_transformer_${TIMESTAMP}"
GPU_ID=0
LOG_DIR="../logs"

# Create directories
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/visualizations"

# Log file
LOG_FILE="${LOG_DIR}/mm_transformer_${TIMESTAMP}.log"

# Save the model file
cp models/mm_transformer.py "${WORK_DIR}/models/"
cp "${CONFIG_FILE}" "${WORK_DIR}/"

echo "Starting training at $(date)" | tee -a "${LOG_FILE}"
echo "Configuration: ${CONFIG_FILE}" | tee -a "${LOG_FILE}"
echo "Work directory: ${WORK_DIR}" | tee -a "${LOG_FILE}"

# Run training
python train.py \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR}" \
    --model-saved-name "mm_transformer" \
    --device "${GPU_ID}" 2>&1 | tee -a "${LOG_FILE}"

TRAINING_STATUS=$?

if [ ${TRAINING_STATUS} -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "${LOG_FILE}"
    
    # Export to TFLite
    echo "Exporting model to TFLite..." | tee -a "${LOG_FILE}"
    python -c "
from utils.tflite_converter import convert_to_tflite
import tensorflow as tf
import os

model_path = '${WORK_DIR}/models/mm_transformer.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    tflite_path = '${WORK_DIR}/models/mm_transformer.tflite'
    success = convert_to_tflite(
        model=model,
        save_path=tflite_path,
        input_shape=(1, 128, 3),
        quantize=False
    )
    print(f'TFLite export success: {success}')
else:
    print(f'Model not found at {model_path}')
" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "Training failed with status ${TRAINING_STATUS}" | tee -a "${LOG_FILE}"
fi

echo "Done!"
