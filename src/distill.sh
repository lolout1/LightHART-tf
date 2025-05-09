#!/bin/bash
# distill.sh - Script to run knowledge distillation from src directory

# Enable strict error handling
set -euo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/distill.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/distillation_${TIMESTAMP}"
GPU_ID=0
LOG_DIR="../logs"

# Create directories
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/visualizations"

# Log file
LOG_FILE="${LOG_DIR}/distillation_${TIMESTAMP}.log"

echo "Starting knowledge distillation at $(date)" | tee -a "${LOG_FILE}"
echo "Configuration: ${CONFIG_FILE}" | tee -a "${LOG_FILE}"
echo "Work directory: ${WORK_DIR}" | tee -a "${LOG_FILE}"

# Check for teacher model weights
TEACHER_WEIGHTS=$(grep 'teacher_weight:' "${CONFIG_FILE}" | awk '{print $2}')
if [ ! -f "${TEACHER_WEIGHTS}" ] && [ ! -d "${TEACHER_WEIGHTS}" ]; then
    echo "WARNING: Teacher weights not found at ${TEACHER_WEIGHTS}" | tee -a "${LOG_FILE}"
    echo "Checking for subject-specific weights..." | tee -a "${LOG_FILE}"
    FOUND=false
    for SUBJECT in {29..45}; do
        if [ -f "${TEACHER_WEIGHTS}_${SUBJECT}.weights.h5" ] || [ -f "${TEACHER_WEIGHTS}_${SUBJECT}.keras" ]; then
            FOUND=true
            echo "Found teacher weights for subject ${SUBJECT}" | tee -a "${LOG_FILE}"
            break
        fi
    done
    
    if [ "$FOUND" = false ]; then
        echo "ERROR: No teacher weights found. Cannot proceed with distillation." | tee -a "${LOG_FILE}"
        exit 1
    fi
fi

# Copy configuration files
cp "${CONFIG_FILE}" "${WORK_DIR}/"
cp "distiller.py" "${WORK_DIR}/"
cp "utils/loss_tf.py" "${WORK_DIR}/"

# Run distillation
python distiller.py \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR}" \
    --model-saved-name "student_distilled" \
    --device "${GPU_ID}" \
    --phase "distill" 2>&1 | tee -a "${LOG_FILE}"

DISTILL_STATUS=$?

if [ ${DISTILL_STATUS} -eq 0 ]; then
    echo "Distillation completed successfully at $(date)" | tee -a "${LOG_FILE}"
    
    # Export to TFLite
    echo "Exporting distilled student model to TFLite..." | tee -a "${LOG_FILE}"
    python -c "
from utils.tflite_converter import convert_to_tflite
import tensorflow as tf
import os
import glob

# Find the best model weights
model_dir = '${WORK_DIR}/models'
weight_files = glob.glob(f'{model_dir}/student_distilled_*.weights.h5')

if weight_files:
    from models.transformer_optimized import TransModel
    # Create model with same architecture
    model = TransModel(
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
    )
    
    # Load the best weights
    best_weight = weight_files[0]
    subject_id = best_weight.split('_')[-1].split('.')[0]
    print(f'Loading weights for subject {subject_id}: {best_weight}')
    model.load_weights(best_weight)
    
    # Export to TFLite
    tflite_path = f'{model_dir}/student_distilled_{subject_id}.tflite'
    success = convert_to_tflite(
        model=model,
        save_path=tflite_path,
        input_shape=(1, 128, 3),
        quantize=False
    )
    print(f'TFLite export success: {success}')
else:
    print('No model weights found for TFLite export')
" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "Distillation failed with status ${DISTILL_STATUS}" | tee -a "${LOG_FILE}"
fi

echo "Done!" | tee -a "${LOG_FILE}"
