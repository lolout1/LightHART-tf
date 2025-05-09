#!/bin/bash
# train_teacher.sh - Robust training script for the teacher model in LightHART-TF

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
NUM_WORKERS=40  # Support for multi-worker processing

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

# Save original config and code for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
mkdir -p "${WORK_DIR}/code"
cp "models/mm_transformer.py" "${WORK_DIR}/code/"
cp "utils/dataset_tf.py" "${WORK_DIR}/code/"
cp "train.py" "${WORK_DIR}/code/"

# Create a custom config with our parameters
CUSTOM_CONFIG="${WORK_DIR}/teacher_custom.yaml"
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
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# For better multi-processing performance
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Run training
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
  
  # Export to TFLite (teacher model might be too complex for TFLite, but we'll try)
  echo "Attempting to export teacher model to TFLite..."
  python -c "
import os
import glob
import sys
import tensorflow as tf
import traceback
from utils.tflite_converter import convert_to_tflite

try:
    # Load the model architecture
    from models.mm_transformer import MMTransformer
    
    # Find the best weights
    model_dir = '${WORK_DIR}/models'
    model_name = '${MODEL_NAME}'
    weight_files = glob.glob(f'{model_dir}/{model_name}_*.weights.h5')
    
    if weight_files:
        # Initialize the model
        model = MMTransformer(
            mocap_frames=128,
            acc_frames=128,
            num_joints=32,
            in_chans=3,
            num_patch=4,
            acc_coords=3,
            spatial_embed=16,
            sdepth=4,
            adepth=4,
            tdepth=2,
            num_heads=2,
            mlp_ratio=2,
            qkv_bias=True,
            drop_rate=${DROPOUT},
            attn_drop_rate=${DROPOUT},
            drop_path_rate=${DROPOUT},
            num_classes=1
        )
        
        # Load weights
        best_weight = weight_files[0]
        subject_id = best_weight.split('_')[-1].split('.')[0]
        print(f'Loading weights for subject {subject_id}: {best_weight}')
        
        # Build the model with dummy inputs
        dummy_acc = tf.zeros((1, 128, 3), dtype=tf.float32)
        dummy_skl = tf.zeros((1, 128, 32, 3), dtype=tf.float32)
        _ = model({'accelerometer': dummy_acc, 'skeleton': dummy_skl})
        
        # Load weights
        model.load_weights(best_weight)
        
        # Export to TFLite
        tflite_path = f'{model_dir}/{model_name}_{subject_id}.tflite'
        print('NOTE: Teacher model export to TFLite may fail due to complexity')
        success = convert_to_tflite(
            model=model,
            save_path=tflite_path,
            input_shape=(1, 128, 3),
            quantize=False
        )
        print(f'TFLite export success: {success}')
    else:
        print('No model weights found for TFLite export')
except Exception as e:
    print(f'Error exporting to TFLite: {e}')
    traceback.print_exc()
"
  
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
