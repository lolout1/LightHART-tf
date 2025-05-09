#!/bin/bash
# distill.sh - Perform knowledge distillation from teacher to student

# Enable strict error handling
set -euo pipefail

# Configuration
CONFIG_FILE="config/smartfallmm/distill.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/distillation_${TIMESTAMP}"
MODEL_NAME="student_distilled"
GPU_ID=0
LEARNING_RATE=0.001
DROPOUT=0.5
BATCH_SIZE=16
NUM_EPOCHS=80
TEMPERATURE=4.5
ALPHA=0.6

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
    --temp)
      TEMPERATURE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --teacher)
      TEACHER_WEIGHTS="$2"
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

# Check for teacher model weights
if [ -z "${TEACHER_WEIGHTS+x}" ]; then
    # No teacher provided, check latest
    if [ -L "../experiments/latest_teacher" ]; then
        LATEST_TEACHER="../experiments/latest_teacher"
        TEACHER_WEIGHTS=$(find "$LATEST_TEACHER/models" -name "teacher_model_*.weights.h5" | head -n 1)
        if [ -z "$TEACHER_WEIGHTS" ]; then
            echo "No teacher weights found in latest_teacher directory"
            exit 1
        fi
        echo "Using latest teacher weights: $TEACHER_WEIGHTS"
    else
        echo "No teacher weights specified and no latest_teacher symlink found"
        exit 1
    fi
fi

echo "Starting knowledge distillation at $(date)"
echo "Using config: ${CONFIG_FILE}"
echo "Teacher weights: ${TEACHER_WEIGHTS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Dropout: ${DROPOUT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Temperature: ${TEMPERATURE}"
echo "Alpha: ${ALPHA}"
echo "Results will be in: ${WORK_DIR}"

# Save config for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
cp "distiller.py" "${WORK_DIR}/"
cp "utils/loss_tf.py" "${WORK_DIR}/"

# Create custom config with updated parameters
CUSTOM_CONFIG="${WORK_DIR}/distill_custom.yaml"
cp "${CONFIG_FILE}" "${CUSTOM_CONFIG}"

# Update parameters in config
sed -i "s/base_lr: .*/base_lr: ${LEARNING_RATE}/" "${CUSTOM_CONFIG}"
sed -i "s/batch_size: .*/batch_size: ${BATCH_SIZE}/" "${CUSTOM_CONFIG}"
sed -i "s/num_epoch: .*/num_epoch: ${NUM_EPOCHS}/" "${CUSTOM_CONFIG}"
sed -i "s/temperature: .*/temperature: ${TEMPERATURE}/" "${CUSTOM_CONFIG}"
sed -i "s/alpha: .*/alpha: ${ALPHA}/" "${CUSTOM_CONFIG}"
sed -i "s|teacher_weight: .*|teacher_weight: ${TEACHER_WEIGHTS}|" "${CUSTOM_CONFIG}"

# Run distillation
python distiller.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --teacher-weight "${TEACHER_WEIGHTS}" \
  --base-lr "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epoch "${NUM_EPOCHS}" \
  --temperature "${TEMPERATURE}" \
  --alpha "${ALPHA}" \
  --phase "distill"

# Check training status
DISTILL_STATUS=$?

if [ ${DISTILL_STATUS} -eq 0 ]; then
  echo "Distillation completed successfully at $(date)"
  
  # Export to TFLite
  echo "Exporting distilled student model to TFLite..."
  python -c "
from utils.tflite_converter import convert_to_tflite
import tensorflow as tf
import os
import glob

model_dir = '${WORK_DIR}/models'
weight_files = glob.glob(f'{model_dir}/{MODEL_NAME}_*.weights.h5')

if weight_files:
    from models.transformer_optimized import TransModel
    
    model = TransModel(
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=${DROPOUT}
    )
    
    best_weight = weight_files[0]
    subject_id = best_weight.split('_')[-1].split('.')[0]
    print(f'Loading weights for subject {subject_id}: {best_weight}')
    model.load_weights(best_weight)
    
    tflite_path = f'{model_dir}/{MODEL_NAME}_{subject_id}.tflite'
    success = convert_to_tflite(
        model=model,
        save_path=tflite_path,
        input_shape=(1, 128, 3),
        quantize=False
    )
    print(f'TFLite export success: {success}')
else:
    print('No model weights found for TFLite export')
"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_distilled
  ln -s "$(basename "${WORK_DIR}")" latest_distilled
  cd -
else
  echo "Distillation failed with status ${DISTILL_STATUS}"
  exit ${DISTILL_STATUS}
fi

echo "Distillation completed. Output directory: ${WORK_DIR}"
