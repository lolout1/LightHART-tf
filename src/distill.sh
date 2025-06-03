#!/bin/bash
set -euo pipefail

CONFIG_FILE="config/smartfallmm/distill.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/distilled_${TIMESTAMP}"
MODEL_NAME="distilled_model"
GPU_ID=0
LEARNING_RATE=0.0005
DROPOUT=0.5
BATCH_SIZE=32
NUM_EPOCHS=100
NUM_WORKERS=40
TEMPERATURE=3.5
ALPHA=0.3
BETA=0.4
GAMMA=0.3
USE_SMV=false
USE_PROGRESSIVE=true
OLDER_RATIO=0.3
TEACHER_WEIGHTS="modelsKD_20250527_194605/models/teacher_model"

while [[ $# -gt 0 ]]; do
  case $1 in
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --batch) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --workdir) WORK_DIR="$2"; shift 2 ;;
    --workers) NUM_WORKERS="$2"; shift 2 ;;
    --teacher) TEACHER_WEIGHTS="$2"; shift 2 ;;
    --temp) TEMPERATURE="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --use-smv) USE_SMV=true; shift ;;
    --older-ratio) OLDER_RATIO="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "====== LightHART-TF Progressive Knowledge Distillation ======"
echo "Configuration:"
echo "  Teacher weights: $TEACHER_WEIGHTS"
echo "  GPU: $GPU_ID"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Worker processes: $NUM_WORKERS"
echo "  Temperature: $TEMPERATURE"
echo "  Alpha: $ALPHA, Beta: $BETA, Gamma: $GAMMA"
echo "  Using SMV: $USE_SMV"
echo "  Progressive distillation: $USE_PROGRESSIVE"
echo "  Older subject ratio: $OLDER_RATIO"
echo "  Working directory: $WORK_DIR"
echo "=============================================="

TEACHER_WEIGHTS_FOUND=false
SUBJECTS=(32 39 30 31 33 34 35 37 43 44 29 36 45)
echo "Checking for teacher weights..."
for SUBJECT in "${SUBJECTS[@]}"; do
    WEIGHT_PATH_H5="${TEACHER_WEIGHTS}_${SUBJECT}.weights.h5"
    WEIGHT_PATH_KERAS="${TEACHER_WEIGHTS}_${SUBJECT}.keras"
    if [ -f "$WEIGHT_PATH_H5" ] || [ -f "$WEIGHT_PATH_KERAS" ]; then
        TEACHER_WEIGHTS_FOUND=true
        [ -f "$WEIGHT_PATH_H5" ] && echo "✓ Found: ${WEIGHT_PATH_H5}"
        [ -f "$WEIGHT_PATH_KERAS" ] && echo "✓ Found: ${WEIGHT_PATH_KERAS}"
    fi
done

if [ "$TEACHER_WEIGHTS_FOUND" = false ]; then
    echo "ERROR: No teacher weights found at: ${TEACHER_WEIGHTS}_*.weights.h5 or *.keras"
    echo "Available teacher model directories:"
    ls -la modelsKD_*/models/ 2>/dev/null || echo "No modelsKD directories found"
    exit 1
fi

DATA_DIR_FOUND=false
for dir in "../data/smartfallmm" "./data/smartfallmm" "$HOME/data/smartfallmm"; do
  if [ -d "$dir" ]; then
    echo "✓ Found data directory: $dir"
    DATA_DIR_FOUND=true
    break
  fi
done
[ "$DATA_DIR_FOUND" = false ] && echo "WARNING: SmartFallMM data directory not found!"

mkdir -p "${WORK_DIR}/models" "${WORK_DIR}/logs" "${WORK_DIR}/visualizations" "${WORK_DIR}/results" "${WORK_DIR}/config"
cp "${CONFIG_FILE}" "${WORK_DIR}/config/original_config.yaml"

CUSTOM_CONFIG="${WORK_DIR}/config/distill_custom.yaml"
cat > "${CUSTOM_CONFIG}" << EOF
model: models.transformer_optimized.TransModel
teacher_model: models.mm_transformer.MMTransformer
dataset: smartfallmm
subjects: [32, 39, 30, 31, 33, 34, 35, 37, 43, 44, 45, 36, 29]
train_subjects_fixed: [45, 36, 29]
val_subjects_fixed: [38, 46]
test_eligible_subjects: [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
include_older_subjects: true
temperature: ${TEMPERATURE}
alpha: ${ALPHA}
beta: ${BETA}
gamma: ${GAMMA}
use_attention_transfer: true
use_hint_learning: true
use_cross_aligner: true
use_progressive_distillation: ${USE_PROGRESSIVE}
older_subject_sample_ratio: ${OLDER_RATIO}
batch_size: ${BATCH_SIZE}
test_batch_size: ${BATCH_SIZE}
val_batch_size: ${BATCH_SIZE}
num_epoch: ${NUM_EPOCHS}
base_lr: ${LEARNING_RATE}
weight_decay: 0.0004
optimizer: adamw
seed: 2
phase: distill
num_worker: ${NUM_WORKERS}
print_log: true
verbose: true
use_smv: ${USE_SMV}
teacher_weight: ${TEACHER_WEIGHTS}
model_saved_name: ${MODEL_NAME}
class_weights:
  0: 1.0
  1: 3.0
model_args:
  acc_frames: 128
  num_classes: 1
  num_heads: 4
  acc_coords: 3
  embed_dim: 32
  num_layers: 2
  dropout: ${DROPOUT}
  activation: relu
  norm_first: true
teacher_args:
  acc_frames: 128
  mocap_frames: 128
  num_joints: 32
  in_chans: 3
  spatial_embed: 32
  tdepth: 2
  num_heads: 2
  mlp_ratio: 2.0
  drop_rate: 0.2
  attn_drop_rate: 0.2
  drop_path_rate: 0.2
  num_classes: 1
dataset_args:
  mode: selective_window
  max_length: 128
  task: fd
  modalities: [accelerometer, skeleton]
  age_group: [young, old]
  sensors: [watch]
  use_dtw: true
  verbose: true
  fall_height: 1.4
  fall_distance: 50
  non_fall_height: 1.2
  non_fall_distance: 100
feeder: utils.dataset_tf.UTD_MM_TF
work_dir: ${WORK_DIR}
EOF

echo "Config created: ${CUSTOM_CONFIG}"

if [ -f "train.py" ]; then
    echo "✓ Found train.py"
else
    echo "ERROR: train.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la *.py 2>/dev/null || echo "No .py files found"
    exit 1
fi

if [ -f "distiller.py" ]; then
    echo "✓ Found distiller.py"
else
    echo "ERROR: distiller.py not found"
    exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}
export PYTHONPATH=".:${PYTHONPATH:-}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Starting distillation..."
python distiller.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --teacher-weight "${TEACHER_WEIGHTS}" \
  --num-worker "${NUM_WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --phase "distill" \
  --subjects 32 39 30 31 33 34 35 37 43 44 45 36 29 \
  --train-subjects-fixed 45 36 29 \
  --val-subjects-fixed 38 46 \
  --test-eligible-subjects 32 39 30 31 33 34 35 37 43 44 \
  --include-older-subjects true \
  --older-subject-sample-ratio "${OLDER_RATIO}" \
  --use-progressive-distillation "${USE_PROGRESSIVE}" \
  --use-cross-aligner true \
  --use-attention-transfer true \
  --use-hint-learning true \
  --use-smv "${USE_SMV}" \
  2>&1 | tee "${WORK_DIR}/distillation_output.log"

DISTILL_STATUS=${PIPESTATUS[0]}

if [ ${DISTILL_STATUS} -eq 0 ]; then
  echo "✅ Distillation completed successfully at $(date)"
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_distilled 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_distilled
  cd - > /dev/null
  echo "Created latest_distilled symlink"
  echo "All artifacts saved to: ${WORK_DIR}"
  echo "Generated models:"
  ls -la "${WORK_DIR}/models/" 2>/dev/null || echo "No models found yet"
  if [ -f "${WORK_DIR}/distillation_results.csv" ]; then
    echo -e "\n=== Distillation Results ==="
    cat "${WORK_DIR}/distillation_results.csv"
  fi
  if [ -f "${WORK_DIR}/distillation_statistics.csv" ]; then
    echo -e "\n=== Statistics ==="
    cat "${WORK_DIR}/distillation_statistics.csv"
  fi
  cat > "${WORK_DIR}/distillation_complete.json" << EOFJ
{
  "status": "completed",
  "timestamp": "$(date -Iseconds)",
  "duration_seconds": $SECONDS,
  "configuration": {
    "learning_rate": ${LEARNING_RATE},
    "batch_size": ${BATCH_SIZE},
    "epochs": ${NUM_EPOCHS},
    "dropout": ${DROPOUT},
    "temperature": ${TEMPERATURE},
    "alpha": ${ALPHA},
    "beta": ${BETA},
    "gamma": ${GAMMA},
    "use_progressive": ${USE_PROGRESSIVE},
    "older_ratio": ${OLDER_RATIO},
    "gpu_id": ${GPU_ID}
  },
  "environment": {
    "python_version": "$(python3 --version)",
    "tensorflow_version": "$(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'unknown')",
    "cuda_version": "$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//' 2>/dev/null || echo 'unknown')",
    "hostname": "$(hostname)"
  }
}
EOFJ
else
  echo "⚠️ Distillation failed with status ${DISTILL_STATUS}"
  echo "Last 50 lines of log:"
  tail -n 50 "${WORK_DIR}/distillation_output.log" 2>/dev/null || echo "No log found"
  exit ${DISTILL_STATUS}
fi

echo "✅ Progressive knowledge distillation completed successfully!"
echo "🕐 Total time: ${SECONDS} seconds"
