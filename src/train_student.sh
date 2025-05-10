#!/bin/bash
# train_student.sh - Robust student model training script for LightHART-TF
# Compatible with updated train.py that includes comprehensive debugging

# Enable strict error handling
set -euo pipefail

# Default configuration
CONFIG_FILE="config/smartfallmm/student.yaml"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
WORK_DIR="../experiments/student_${TIMESTAMP}"
MODEL_NAME="student_model"
GPU_ID=0
LEARNING_RATE=0.001
DROPOUT=0.5
BATCH_SIZE=16
NUM_EPOCHS=80
USE_SMV=false
NUM_WORKERS=40
WEIGHT_DECAY=0.0004
OPTIMIZER="adamw"
SEED=2
DEBUG_MODE=false
MODALITIES="accelerometer"  # Default to accelerometer only for student

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --lr|--learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --dropout)
      DROPOUT="$2"
      shift 2
      ;;
    --batch|--batch-size)
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
    --workdir|--work-dir)
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
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --modalities)
      MODALITIES="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --lr RATE             Learning rate (default: 0.001)"
      echo "  --dropout RATE        Dropout rate (default: 0.5)"
      echo "  --batch SIZE          Batch size (default: 16)"
      echo "  --epochs NUM          Number of epochs (default: 80)"
      echo "  --gpu ID              GPU device ID (default: 0)"
      echo "  --workdir DIR         Working directory (default: ../experiments/student_TIMESTAMP)"
      echo "  --workers NUM         Number of worker processes (default: 40)"
      echo "  --use-smv            Enable Signal Magnitude Vector calculation"
      echo "  --config FILE         Configuration file (default: config/smartfallmm/student.yaml)"
      echo "  --model-name NAME     Model name for saving (default: student_model)"
      echo "  --weight-decay DECAY  Weight decay (default: 0.0004)"
      echo "  --optimizer OPT       Optimizer (adam/adamw/sgd, default: adamw)"
      echo "  --seed SEED           Random seed (default: 2)"
      echo "  --debug              Enable debug mode"
      echo "  --modalities MODS     Modalities to use (default: accelerometer)"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      print_color "$RED" "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Display job configuration
print_color "$GREEN" "====== LightHART-TF Student Model Training ======"
echo "Configuration:"
echo "  Model: Student (Transformer Optimized)"
echo "  Config: ${CONFIG_FILE}"
echo "  GPU: ${GPU_ID}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Dropout: ${DROPOUT}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Optimizer: ${OPTIMIZER}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Worker processes: ${NUM_WORKERS}"
echo "  Using SMV: ${USE_SMV}"
echo "  Modalities: ${MODALITIES}"
echo "  Random seed: ${SEED}"
echo "  Working directory: ${WORK_DIR}"
echo "  Debug mode: ${DEBUG_MODE}"
print_color "$GREEN" "==========================================\n"

# Verify configuration file exists
if [ ! -f "${CONFIG_FILE}" ]; then
  print_color "$RED" "ERROR: Configuration file not found: ${CONFIG_FILE}"
  exit 1
fi

# Create working directory with required structure
print_color "$YELLOW" "Creating directory structure..."
mkdir -p "${WORK_DIR}/models"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/visualizations"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/config"

# Save original config
print_color "$YELLOW" "Saving configuration files..."
cp "${CONFIG_FILE}" "${WORK_DIR}/config/original_config.yaml"

# Save this script for reference
cp "$0" "${WORK_DIR}/train_student.sh"

# Save relevant code files
print_color "$YELLOW" "Archiving code files..."
mkdir -p "${WORK_DIR}/code"
for file in "models/transformer_optimized.py" "utils/dataset_tf.py" "utils/processor_tf.py" "train.py"; do
  if [ -f "$file" ]; then
    cp "$file" "${WORK_DIR}/code/" 2>/dev/null || print_color "$YELLOW" "Warning: $file not found"
  fi
done

# Create a custom config with our parameters
CUSTOM_CONFIG="${WORK_DIR}/config/student_custom.yaml"
cp "${CONFIG_FILE}" "${CUSTOM_CONFIG}"

# Convert bash boolean to Python boolean
if [ "$USE_SMV" = "true" ]; then
  PY_USE_SMV="True"
else
  PY_USE_SMV="False"
fi

# Update parameters in the config using Python for better YAML handling
print_color "$YELLOW" "Updating configuration parameters..."
python3 -c "
import yaml

# Load config
with open('${CUSTOM_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)

# Update parameters
config['base_lr'] = ${LEARNING_RATE}
config['batch_size'] = ${BATCH_SIZE}
config['test_batch_size'] = ${BATCH_SIZE}
config['val_batch_size'] = ${BATCH_SIZE}
config['num_epoch'] = ${NUM_EPOCHS}
config['optimizer'] = '${OPTIMIZER}'
config['weight_decay'] = ${WEIGHT_DECAY}
config['seed'] = ${SEED}

# Update model args
if 'model_args' in config:
    config['model_args']['dropout'] = ${DROPOUT}

# Update dataset args
if 'dataset_args' in config:
    # Handle modalities
    modalities = '${MODALITIES}'.split(',')
    config['dataset_args']['modalities'] = modalities
    
    # If using accelerometer only, disable DTW
    if modalities == ['accelerometer']:
        config['dataset_args']['use_dtw'] = False

# Update feeder
config['feeder'] = 'utils.dataset_tf.UTD_MM_TF'

# Add SMV configuration
config['use_smv'] = ${PY_USE_SMV}

# Save updated config
with open('${CUSTOM_CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

# Set TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# For better multi-processing performance
export OMP_NUM_THREADS=${NUM_WORKERS}
export TF_NUM_INTRAOP_THREADS=${NUM_WORKERS}
export TF_NUM_INTEROP_THREADS=${NUM_WORKERS}

# Add current directory to Python path
export PYTHONPATH=".:${PYTHONPATH:-}"

# Enable debug logging if requested
if [ "$DEBUG_MODE" = true ]; then
  export TF_CPP_MIN_LOG_LEVEL=0
  export TF_CPP_VLOG_LEVEL=1
fi

# Check for data directory
if [ ! -d "../data/smartfallmm" ] && [ ! -d "data/smartfallmm" ]; then
  print_color "$YELLOW" "WARNING: SmartFallMM data directory not found!"
  print_color "$YELLOW" "Expected locations: ../data/smartfallmm or data/smartfallmm"
  print_color "$YELLOW" "Please ensure data is available before training starts."
fi

# Function to monitor GPU usage in background
monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader >> "${WORK_DIR}/logs/gpu_usage.csv"
    sleep 60
  done
}

# Start GPU monitoring in background (optional)
if command -v nvidia-smi &> /dev/null; then
  print_color "$YELLOW" "Starting GPU monitoring..."
  monitor_gpu &
  GPU_MONITOR_PID=$!
  
  # Ensure we kill the monitoring process on exit
  trap "kill ${GPU_MONITOR_PID} 2>/dev/null || true" EXIT
fi

# Run training
print_color "$GREEN" "Starting student model training..."
python train.py \
  --config "${CUSTOM_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "${MODEL_NAME}" \
  --device "${GPU_ID}" \
  --num-worker "${NUM_WORKERS}" \
  --phase "train" \
  --seed "${SEED}" \
  2>&1 | tee "${WORK_DIR}/logs/training_output.log"

# Check training status
TRAINING_STATUS=${PIPESTATUS[0]}

if [ ${TRAINING_STATUS} -eq 0 ]; then
  print_color "$GREEN" "Student model training completed successfully at $(date)"
  
  # Create symbolic link to latest experiment
  cd "$(dirname "${WORK_DIR}")"
  rm -f latest_student 2>/dev/null || true
  ln -sf "$(basename "${WORK_DIR}")" latest_student
  cd - > /dev/null
  
  print_color "$GREEN" "Created latest_student symlink"
  
  # Generate summary of results
  if [ -f "${WORK_DIR}/final_results.csv" ]; then
    print_color "$YELLOW" "\nFinal Results Summary:"
    echo "----------------------------------------"
    python3 -c "
import pandas as pd
import os

results_file = '${WORK_DIR}/final_results.csv'
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    print(df.to_string(index=False))
    
    # Print average metrics
    avg_row = df[df['test_subject'] == 'Average']
    if not avg_row.empty:
        print('\nAverage Performance:')
        print(f\"  Accuracy: {avg_row['accuracy'].values[0]:.2f}%\")
        print(f\"  F1 Score: {avg_row['f1_score'].values[0]:.2f}%\")
        print(f\"  Precision: {avg_row['precision'].values[0]:.2f}%\")
        print(f\"  Recall: {avg_row['recall'].values[0]:.2f}%\")
        print(f\"  AUC: {avg_row['auc'].values[0]:.2f}%\")
"
    echo "----------------------------------------"
  fi
  
  # List generated artifacts
  print_color "$YELLOW" "\nGenerated artifacts:"
  echo "  Working directory: ${WORK_DIR}"
  echo "  Models: ${WORK_DIR}/models/"
  echo "  Visualizations: ${WORK_DIR}/visualizations/"
  echo "  Logs: ${WORK_DIR}/logs/"
  echo "  Results: ${WORK_DIR}/results/"
  
  # Check for main output files
  [ -f "${WORK_DIR}/final_results.csv" ] && echo "  ✓ final_results.csv"
  [ -f "${WORK_DIR}/summary_report.txt" ] && echo "  ✓ summary_report.txt"
  [ -f "${WORK_DIR}/training.log" ] && echo "  ✓ training.log"
  
else
  print_color "$RED" "⚠️ Student model training failed with status ${TRAINING_STATUS}"
  
  # Show last few lines of log for debugging
  if [ -f "${WORK_DIR}/logs/training_output.log" ]; then
    print_color "$YELLOW" "\nLast 20 lines of training log:"
    tail -n 20 "${WORK_DIR}/logs/training_output.log"
  fi
  
  exit ${TRAINING_STATUS}
fi

# Create a completion file with metadata
cat > "${WORK_DIR}/training_complete.json" << EOF
{
  "status": "completed",
  "timestamp": "$(date -Iseconds)",
  "duration_seconds": $SECONDS,
  "configuration": {
    "learning_rate": ${LEARNING_RATE},
    "batch_size": ${BATCH_SIZE},
    "epochs": ${NUM_EPOCHS},
    "dropout": ${DROPOUT},
    "optimizer": "${OPTIMIZER}",
    "weight_decay": ${WEIGHT_DECAY},
    "use_smv": ${USE_SMV},
    "modalities": "${MODALITIES}",
    "seed": ${SEED},
    "gpu_id": ${GPU_ID}
  },
  "environment": {
    "python_version": "$(python3 --version)",
    "tensorflow_version": "$(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'unknown')",
    "cuda_version": "$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//' 2>/dev/null || echo 'unknown')",
    "hostname": "$(hostname)"
  }
}
EOF

print_color "$GREEN" "✅ Student model training workflow completed successfully!"
print_color "$GREEN" "🕐 Total time: ${SECONDS} seconds"
