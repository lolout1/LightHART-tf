#!/bin/bash
# distill.sh - Robust knowledge distillation script for LightHART-TF

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

# Set the teacher model path directly to a .h5 weights file
TEACHER_WEIGHTS="../experiments/teacher_2025-05-09_15-04-49_2025-05-09_15-04-52/models/teacher_model_32.weights.h5"

# Check if teacher weights exist
if [ ! -f "$TEACHER_WEIGHTS" ]; then
    # Try to find .weights.h5 file
    if [ -f "${TEACHER_WEIGHTS%.keras}.weights.h5" ]; then
        TEACHER_WEIGHTS="${TEACHER_WEIGHTS%.keras}.weights.h5"
        echo "Using weights file: $TEACHER_WEIGHTS"
    else
        echo "ERROR: Teacher model weights not found at: $TEACHER_WEIGHTS"
        echo "Looking for other weights files..."
        
        # Search for any available weights files
        TEACHER_DIR=$(dirname "$TEACHER_WEIGHTS")
        if [ -d "$TEACHER_DIR" ]; then
            echo "Searching in directory: $TEACHER_DIR"
            FOUND_WEIGHTS=$(find "$TEACHER_DIR" -name "teacher_model_*.weights.h5" | head -n 1)
            if [ -n "$FOUND_WEIGHTS" ]; then
                TEACHER_WEIGHTS="$FOUND_WEIGHTS"
                echo "Found alternative teacher weights: $TEACHER_WEIGHTS"
            else
                echo "No teacher weight files found. Please train a teacher model first."
                exit 1
            fi
        else
            echo "Teacher model directory not found."
            exit 1
        fi
    fi
fi

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

# Create required directories
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"

# Save code for reference
cp "${CONFIG_FILE}" "${WORK_DIR}/"
mkdir -p "${WORK_DIR}/code"
cp "distiller.py" "${WORK_DIR}/code/"
cp "models/transformer_optimized.py" "${WORK_DIR}/code/"
cp "models/mm_transformer.py" "${WORK_DIR}/code/"
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

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Performance settings
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Run distillation
echo "Starting knowledge distillation using teacher model: ${TEACHER_WEIGHTS}"
PYTHONPATH=".:$PYTHONPATH" python distiller.py \
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

# Check training status
DISTILL_STATUS=$?

if [ ${DISTILL_STATUS} -eq 0 ]; then
  echo "Distillation completed successfully at $(date)"
  
  # Export to TFLite
  echo "Exporting distilled model to TFLite..."
  PYTHONPATH=".:$PYTHONPATH" python -c "
import os
import glob
import sys
import tensorflow as tf
import traceback

# Import TFLite converter
try:
    from utils.tflite_converter import convert_to_tflite
except ImportError:
    print('TFLite converter not found, using simplified converter')
    def convert_to_tflite(model, save_path, input_shape, quantize=False):
        try:
            # Build converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Set optimization options
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f'Model exported to {save_path}')
            return True
        except Exception as e:
            print(f'Error converting to TFLite: {e}')
            return False

# Find the best weights
model_dir = '${WORK_DIR}/models'
keras_files = glob.glob(f'{model_dir}/{MODEL_NAME}_*.keras')
weight_files = glob.glob(f'{model_dir}/{MODEL_NAME}_*.weights.h5')

try:
    # Import model
    from models.transformer_optimized import TransModel
    
    model = TransModel(
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=${DROPOUT},
        activation='relu'
    )
    
    # Build model with dummy input
    dummy_input = {'accelerometer': tf.zeros((1, 128, 3), dtype=tf.float32)}
    _ = model(dummy_input, training=False)
    
    # Try loading keras file first, then weights
    loaded = False
    
    if keras_files:
        model_file = keras_files[0]
        subject_id = model_file.split('_')[-1].split('.')[0]
        try:
            model = tf.keras.models.load_model(model_file)
            loaded = True
            print(f'Loaded full model for subject {subject_id}: {model_file}')
        except Exception as e:
            print(f'Error loading keras model: {e}')
    
    if not loaded and weight_files:
        weight_file = weight_files[0]
        subject_id = weight_file.split('_')[-1].split('.')[0]
        try:
            model.load_weights(weight_file)
            loaded = True
            print(f'Loaded weights for subject {subject_id}: {weight_file}')
        except Exception as e:
            print(f'Error loading weights: {e}')
    
    if loaded:
        # Export to TFLite
        tflite_path = f'{model_dir}/{MODEL_NAME}_{subject_id}.tflite'
        success = convert_to_tflite(
            model=model,
            save_path=tflite_path,
            input_shape=(1, 128, 3 if not ${USE_SMV} else 4),
            quantize=False
        )
        print(f'TFLite export success: {success}')
    else:
        print('No model weights loaded for TFLite export')
        for file in os.listdir(model_dir):
            print(f'Found file: {file}')
except Exception as e:
    print(f'Error exporting to TFLite: {traceback.format_exc()}')
"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_distilled 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_distilled
  cd - > /dev/null
  
  echo "Created latest_distilled symlink"
  echo "All distillation artifacts saved to: ${WORK_DIR}"
else
  echo "⚠️ Distillation failed with status ${DISTILL_STATUS}"
  exit ${DISTILL_STATUS}
fi

echo "✅ Knowledge distillation workflow completed successfully!"
