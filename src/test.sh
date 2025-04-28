#!/bin/bash
# Start script for LightHART-TF training
# Usage: ./start.sh [config_file] [gpu_id]

# Set default values
CONFIG_FILE=${1:-"config/smartfallmm/optimized.yaml"}
GPU_ID=${2:-0}
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
WORK_DIR="experiments/student_${TIMESTAMP}"
LOG_FILE="../logs/training_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p ../logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LightHART-TF training at $(date)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Config file: ${CONFIG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Working directory: ${WORK_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   GPU ID: ${GPU_ID}"

# Check Python dependencies
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking Python dependencies..."
MISSING_PACKAGES=()
for pkg in "tensorflow" "numpy" "pandas" "matplotlib" "scikit-learn" "pyyaml" "tqdm"; do
  if ! python -c "import $pkg" 2>/dev/null; then
    MISSING_PACKAGES+=($pkg)
  fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Missing Python packages: ${MISSING_PACKAGES[*]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing missing packages..."
  pip install ${MISSING_PACKAGES[*]}
fi

# Check GPU availability
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU availability..."
nvidia-smi || echo "WARNING: No GPU found or nvidia-smi not available"

# Verify required files exist
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verifying required files..."
REQUIRED_FILES=(
  "${CONFIG_FILE}"
  "train.py"
  "trainer/base_trainer.py"
  "utils/dataset_tf.py"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Required file not found: $file"
    MISSING_FILES=1
  fi
done

if [ $MISSING_FILES -eq 1 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Missing required files. Aborting."
  exit 1
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All required files verified."
fi

# Back up code files
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backing up code files..."
mkdir -p "${WORK_DIR}/code_backup"
cp "${CONFIG_FILE}" "${WORK_DIR}/code_backup/" 2>/dev/null || true
cp train.py "${WORK_DIR}/code_backup/"
cp -r trainer "${WORK_DIR}/code_backup/"
cp -r utils "${WORK_DIR}/code_backup/"
cp -r models "${WORK_DIR}/code_backup/" 2>/dev/null || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Code backup created at ${WORK_DIR}/code_backup"

# Check for range error fix in dataset_tf.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for range error fix in dataset_tf.py..."
if grep -q "if start_idx >= end_idx:" utils/dataset_tf.py; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Range error fix already present in dataset_tf.py"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Range error fix not found. This may cause errors."
fi

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=1     # Suppress info messages
export TF_DETERMINISTIC_OPS=0      # Disable determinism to allow timestamp
export PYTHONIOENCODING=utf-8      # Proper encoding for progress bar
export PYTHONUNBUFFERED=1          # Unbuffered output

# Start the training process
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training process..."
python train.py \
  --config "${CONFIG_FILE}" \
  --work-dir "${WORK_DIR}" \
  --model-saved-name "student_model" \
  --device "${GPU_ID}" \
  --phase "train" 2>&1 | tee -a "${LOG_FILE}"

TRAINING_STATUS=${PIPESTATUS[0]}

if [ $TRAINING_STATUS -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Training failed at $(date)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: See ${LOG_FILE} for details."
  
  # Create error analysis
  ERROR_ANALYSIS="../logs/error_analysis_${TIMESTAMP}.txt"
  {
    echo "========== ERROR ANALYSIS =========="
    echo "Training failed at $(date)"
    echo "Configuration:"
    echo "  Config file: ${CONFIG_FILE}"
    echo "  Working directory: ${WORK_DIR}"
    echo "  GPU ID: ${GPU_ID}"
    echo ""
    echo "Last 20 lines of log:"
    tail -n 20 "${LOG_FILE}"
    echo ""
    echo "GPU status at failure:"
    nvidia-smi || echo "nvidia-smi not available"
  } > "${ERROR_ANALYSIS}"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Error analysis saved to: ${ERROR_ANALYSIS}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully at $(date)"
fi

exit $TRAINING_STATUS
