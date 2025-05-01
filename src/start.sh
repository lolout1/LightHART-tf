#!/bin/bash
# start.sh - Robust execution script for LightHART-TF training
# Optimized for transformer_optimized.py with raw accelerometer data

# Enable strict error handling
set -euo pipefail

# ===== CONFIGURATION =====
CONFIG_FILE="config/smartfallmm/optimized.yaml"
GPU_ID="0"
MAX_MEMORY="90"  # Maximum GPU memory percentage to use (0-100)
NUM_THREADS=8    # CPU threads for operations
USE_SMV=false    # Disable SMV calculation for raw accelerometer data

# ===== DIRECTORIES =====
# Create timestamp for this run
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
log_dir="logs"
work_dir="experiments/student_${timestamp}"
results_dir="${work_dir}/results"
viz_dir="${work_dir}/visualizations"
model_dir="${work_dir}/models"

# Create required directories
mkdir -p "${log_dir}"
mkdir -p "${work_dir}"
mkdir -p "${results_dir}"
mkdir -p "${viz_dir}"
mkdir -p "${model_dir}"

# ===== LOGGING =====
# Set up logfile
logfile="${log_dir}/training_${timestamp}.log"
error_log="${log_dir}/error_${timestamp}.log"

# Function to log messages to console and file
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "${message}" | tee -a "${logfile}"
}

# Function to log errors
log_error() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "${message}" | tee -a "${logfile}" "${error_log}"
}

# ===== ENVIRONMENT SETUP =====
# Setup Python environment variables
export PYTHONPATH="$(pwd):$(pwd)/.."
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered
export PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files

# Setup TensorFlow environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Avoid allocating all GPU memory at once
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"  # Set GPU device
export TF_DETERMINISTIC_OPS=1  # For reproducibility
export TF_MEMORY_USAGE="${MAX_MEMORY}"  # Control GPU memory usage
export OMP_NUM_THREADS="${NUM_THREADS}"  # OpenMP threads
export TF_NUM_INTRAOP_THREADS="${NUM_THREADS}"  # TF internal threads
export TF_NUM_INTEROP_THREADS="${NUM_THREADS}"  # TF thread pool

# ===== UTILITY FUNCTIONS =====
# Check for CUDA/GPU availability
check_gpu() {
    log "Checking GPU availability..."
    
    # Use nvidia-smi if available
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi | tee -a "${logfile}"
    fi
    
    # Check TensorFlow GPU access
    python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Found {len(gpus)} GPU(s):', [gpu.name for gpu in gpus])
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Memory growth enabled for {gpu.name}')
        except:
            print(f'Error configuring {gpu.name}')
else:
    print('No GPUs found, using CPU')
print(f'TensorFlow version: {tf.__version__}')
" | tee -a "${logfile}"
}

# Check required Python packages
check_dependencies() {
    log "Checking Python dependencies..."
    
    required_packages=("tensorflow" "numpy" "pandas" "matplotlib" "scikit-learn" "tqdm" "pyyaml" "scipy" "fastdtw")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        python -c "import $package" &> /dev/null || missing_packages+=("$package")
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_error "Missing Python packages: ${missing_packages[*]}"
        log "Installing missing packages..."
        pip install "${missing_packages[@]}" | tee -a "${logfile}"
    else
        log "All required Python packages are installed."
    fi
}

# Verify required files exist
check_required_files() {
    log "Verifying required files..."
    missing_files=0
    
    # Core Python files
    required_files=(
        "train.py"
        "trainer/base_trainer.py"
        "models/transformer_optimized.py"
        "utils/model_utils.py"
        "utils/metrics.py"
        "utils/tflite_converter.py"
        "feeder/make_dataset_tf.py"
        "${CONFIG_FILE}"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Missing required file: $file"
            missing_files=$((missing_files + 1))
        fi
    done
    
    if [ $missing_files -gt 0 ]; then
        log_error "$missing_files required files are missing. Aborting."
        exit 1
    else
        log "All required files verified."
    fi
}

# Backup code files
backup_code() {
    log "Backing up code files..."
    code_backup="${work_dir}/code_backup"
    mkdir -p "${code_backup}"
    
    cp train.py "${code_backup}/"
    cp -r trainer "${code_backup}/"
    cp -r utils "${code_backup}/"
    cp -r models "${code_backup}/"
    cp -r feeder "${code_backup}/"  
    cp "${CONFIG_FILE}" "${code_backup}/"
    cp "start.sh" "${code_backup}/"
    
    log "Code backup created at ${code_backup}"
}

# Update config file to disable SMV calculation
update_config() {
    if [ "$USE_SMV" = false ]; then
        log "Updating config to use raw accelerometer data (without SMV)..."
        
        # Create temp config
        temp_config="${work_dir}/optimized_temp.yaml"
        cp "${CONFIG_FILE}" "${temp_config}"
        
        # Add use_smv: false if not already present
        if ! grep -q "use_smv:" "${temp_config}"; then
            echo "use_smv: false" >> "${temp_config}"
            log "Added use_smv: false to config"
        else
            # Update existing use_smv value
            sed -i 's/use_smv: true/use_smv: false/g' "${temp_config}"
            log "Updated use_smv to false in config"
        fi
        
        # Save as the active config for this run
        active_config="${work_dir}/optimized_active.yaml"
        cp "${temp_config}" "${active_config}"
        CONFIG_FILE="${active_config}"
        log "Active config created at ${active_config}"
    fi
}

# Check model for compatibility with raw accelerometer data
check_model_compatibility() {
    log "Checking model compatibility for raw accelerometer data..."
    
    # Check if model is configured for raw accelerometer inputs
    python -c "
import sys
import yaml

try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    
    if 'model_args' in config and 'acc_coords' in config['model_args']:
        acc_coords = config['model_args']['acc_coords']
        if acc_coords == 3:
            print('Model configured for raw accelerometer data (3 channels)')
        else:
            print(f'Warning: Model configured for {acc_coords} channels, not raw accelerometer')
            sys.exit(1)
    else:
        print('Warning: Could not verify acc_coords in config file')
        sys.exit(1)
except Exception as e:
    print(f'Error checking model compatibility: {e}')
    sys.exit(1)
" || log_error "Model may not be compatible with raw accelerometer data"
}

# Handle errors
handle_error() {
    local error_code=$?
    local line_no=$1
    
    log_error "Error at line ${line_no}: Command exited with status ${error_code}"
    
    # Capture stack trace
    local stack_trace
    stack_trace=$(python -c "import traceback; traceback.print_stack()" 2>&1)
    log_error "Stack trace:\n${stack_trace}"
    
    # Check for common errors
    if grep -q "out of memory" "${logfile}"; then
        log_error "Detected GPU out of memory error. Try reducing batch size or model size."
    fi
    
    if grep -q "ImportError" "${logfile}"; then
        log_error "Detected missing module. Check Python environment and dependencies."
    fi
    
    if grep -q "shape mismatch" "${logfile}" || grep -q "incompatible dimensions" "${logfile}"; then
        log_error "Detected shape mismatch. Check data loading and model input compatibility."
        log_error "Ensure UTD_MM_TF is not adding SMV when use_smv=false in the config."
    fi
    
    log_error "See ${logfile} for details."
    log_error "Training failed at $(date)"
    
    # Create error analysis
    error_analysis_file="${log_dir}/error_analysis_${timestamp}.txt"
    {
        echo "Error Analysis for training run at ${timestamp}"
        echo "Exit code: ${error_code}"
        echo "--------------------------"
        echo "Common error analysis:"
        echo "1. GPU memory issues:"
        grep -i "out of memory" "${logfile}" || echo "None found"
        echo "2. Module import issues:"
        grep -i "no module named" "${logfile}" || echo "None found"
        grep -i "cannot import" "${logfile}" || echo "None found"
        echo "3. File not found issues:"
        grep -i "no such file or directory" "${logfile}" || echo "None found"
        echo "4. Tensor shape incompatibility:"
        grep -i "incompatible shapes" "${logfile}" || echo "None found"
        grep -i "dimension mismatch" "${logfile}" || echo "None found"
        echo "5. Data format issues:"
        grep -i "SMV" "${logfile}" | grep -i "error" || echo "None found"
        grep -i "accelerometer" "${logfile}" | grep -i "shape" || echo "None found"
        echo "6. Other TensorFlow errors:"
        grep -i "tensorflow" "${logfile}" | grep -i "error" || echo "None found"
    } > "${error_analysis_file}"
    
    log_error "Error analysis saved to: ${error_analysis_file}"
    
    exit "${error_code}"
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# ===== MAIN EXECUTION =====
# Initialize
log "Starting LightHART-TF training at $(date)"
log "Configuration:"
log "  Config file: ${CONFIG_FILE}"
log "  Working directory: ${work_dir}"
log "  Log file: ${logfile}"
log "  GPU ID: ${GPU_ID}"
log "  Using SMV: ${USE_SMV}"

# System checks
#check_dependencies
check_gpu
check_required_files
backup_code
update_config
check_model_compatibility

# Run training
log "Starting training process..."
python train.py \
  --config "${CONFIG_FILE}" \
  --work-dir "${work_dir}" \
  --model-saved-name "student_model" \
  --device "${GPU_ID}" 2>&1 | tee -a "${logfile}"

# Check training status
training_status=${PIPESTATUS[0]}

if [ $training_status -eq 0 ]; then
    log "Training completed successfully at $(date)"
    
    # Create symlink to latest experiment
    if [ -L "experiments/latest" ]; then
        rm "experiments/latest"
    fi
    ln -sf "student_${timestamp}" "experiments/latest"
    log "Created symlink: experiments/latest -> student_${timestamp}"
    
    # Export to TFLite
    log "Exporting best model to TFLite..."
    python -c "
import os
import tensorflow as tf
from utils.tflite_converter import convert_to_tflite

# Find best model
model_dir = '${model_dir}'
best_models = [f for f in os.listdir(model_dir) if f.endswith('.keras') or os.path.isdir(os.path.join(model_dir, f))]

if best_models:
    best_model_path = None
    for model_name in best_models:
        model_path = os.path.join(model_dir, model_name)
        if os.path.isdir(model_path):
            best_model_path = model_path
        elif model_name.endswith('.keras'):
            best_model_path = model_path
        if best_model_path:
            break
    
    if best_model_path:
        print(f'Found best model: {best_model_path}')
        
        # Load model
        model = tf.keras.models.load_model(best_model_path)
        
        # Export to TFLite
        tflite_path = os.path.join(model_dir, 'model.tflite')
        success = convert_to_tflite(
            model=model, 
            save_path=tflite_path,
            input_shape=(1, 64, 3),  # Raw accelerometer data
            quantize=False
        )
        print(f'TFLite export success: {success}')
    else:
        print('No suitable best model found for TFLite export')
else:
    print('No best model found for TFLite export')
" 2>&1 | tee -a "${logfile}"
    
    # Generate final report
    log "Generating final report..."
    python -c "
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_dir = '${results_dir}'
work_dir = '${work_dir}'
report_file = os.path.join(work_dir, 'final_report.html')

# Find all result files
result_files = [f for f in os.listdir(results_dir) if f.startswith('test_results_') and f.endswith('.json')]
results = []

for file in result_files:
    with open(os.path.join(results_dir, file), 'r') as f:
        try:
            result = json.load(f)
            results.append(result)
        except json.JSONDecodeError:
            print(f'Error parsing {file}')

if results:
    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(work_dir, 'results.csv'), index=False)
    
    # Calculate average results
    if 'subject' in df.columns:
        avg_row = {col: df[col].mean() for col in df.columns if col != 'subject'}
        avg_row['subject'] = 'Average'
        df_with_avg = pd.concat([df, pd.DataFrame([avg_row])])
    else:
        df_with_avg = df
    
    # Create report
    with open(report_file, 'w') as f:
        f.write('<html><head><title>LightHART-TF Training Report</title>')
        f.write('<style>body{font-family:Arial;max-width:900px;margin:auto;padding:20px}')
        f.write('table{border-collapse:collapse;width:100%;}')
        f.write('th,td{text-align:left;padding:8px;border:1px solid #ddd;}')
        f.write('tr:nth-child(even){background-color:#f2f2f2;}')
        f.write('th{background-color:#4CAF50;color:white;}')
        f.write('h1,h2{color:#4CAF50}</style></head><body>')
        f.write(f'<h1>LightHART-TF Training Report</h1>')
        f.write(f'<p>Training completed at {pd.Timestamp.now()}</p>')
        f.write(f'<h2>Configuration</h2>')
        f.write(f'<p>Raw accelerometer data (no SMV)</p>')
        f.write(f'<h2>Results Summary</h2>')
        f.write(df_with_avg.to_html(index=False))
        f.write('<h2>Visualizations</h2>')
        f.write('<p>See visualization directory for detailed plots</p>')
        f.write('</body></html>')
    
    print(f'Final report generated at {report_file}')
else:
    print('No results found for report generation')
" 2>&1 | tee -a "${logfile}"
    
    log "All processes completed successfully at $(date)"
else
    log_error "Training failed with exit code $training_status at $(date)"
fi

log "Done!"
exit $training_status
