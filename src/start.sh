#!/bin/bash
# start.sh - Robust execution script for LightHART-TF training
#
# This script handles the complete execution pipeline for training fall detection
# models with the LightHART-TF framework. It includes error handling, dependency
# checking, environment setup, and reporting.

# Enable strict error handling
set -euo pipefail

# ===== CONFIGURATION =====
# User configurable variables
CONFIG_FILE="config/smartfallmm/optimized.yaml"
GPU_ID="0"
MAX_MEMORY="90"  # Maximum GPU memory percentage to use (0-100)
NUM_THREADS=8    # CPU threads for operations

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

# Ensure module directories exist
mkdir -p utils/processor
mkdir -p trainer
mkdir -p models
mkdir -p config/smartfallmm
mkdir -p data/smartfallmm

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
        "trainer/training_loop.py"
        "trainer/evaluation.py"
        "utils/model_utils.py"
        "utils/metrics.py"
        "utils/visualization.py"
        "utils/dataset_tf.py"
        "utils/tflite_converter.py"
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
    cp "${CONFIG_FILE}" "${code_backup}/"
    cp "start.sh" "${code_backup}/"
    
    log "Code backup created at ${code_backup}"
}

# Create TFLite converter if missing
ensure_tflite_converter() {
    if [ ! -f "utils/tflite_converter.py" ]; then
        log "Creating TFLite converter..."
        cat > "utils/tflite_converter.py" << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFLite Converter for LightHART-TF

Handles converting TensorFlow models to TFLite format with proper preprocessing
for accelerometer-only input, supporting inference on mobile and edge devices.
"""
import os
import logging
import shutil
import traceback
from typing import Tuple, Optional, Union
import tensorflow as tf
import numpy as np

logger = logging.getLogger('lightheart-tf')

def convert_to_tflite(
    model: tf.keras.Model, 
    save_path: str, 
    input_shape: Tuple[int, int, int] = (1, 128, 3),
    quantize: bool = False,
    optimize: bool = True,
    include_metadata: bool = True
) -> bool:
    """Convert model to TFLite format with accelerometer-only input.
    
    This function creates a TFLite model that only takes accelerometer data as input,
    even if the original model was trained with both skeleton and accelerometer data.
    It adds signal magnitude vector (SMV) calculation as a preprocessing step.
    
    Args:
        model: TensorFlow model to convert
        save_path: Path to save the TFLite model
        input_shape: Input shape for accelerometer data (batch, frames, channels)
        quantize: Whether to apply quantization to reduce model size
        optimize: Whether to optimize model for inference
        include_metadata: Whether to include metadata in the model
        
    Returns:
        bool: Success status of the conversion
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Add .tflite extension if not present
        if not save_path.endswith('.tflite'):
            save_path += '.tflite'
        
        # Create specific preprocessing function for accelerometer data
        @tf.function(input_signature=[
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='accelerometer')
        ])
        def serving_function(accelerometer):
            # Calculate signal magnitude vector (SMV)
            mean = tf.reduce_mean(accelerometer, axis=1, keepdims=True)
            zero_mean = accelerometer - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            smv = tf.sqrt(sum_squared)
            
            # Concatenate SMV with original data
            acc_with_smv = tf.concat([smv, accelerometer], axis=-1)
            
            # Create a dictionary for the model if it expects one
            inputs = {'accelerometer': acc_with_smv}
            
            # Forward pass (handle both tuple and single output)
            outputs = model(inputs, training=False)
            
            # Return only logits if model returns (logits, features)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0]
            return outputs
        
        # Create temporary SavedModel with the concrete function
        saved_model_dir = os.path.join(os.path.dirname(save_path), "temp_saved_model")
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
            
        logger.info(f"Creating SavedModel with signature at {saved_model_dir}")
        
        # Save model with signature
        tf.saved_model.save(
            model, 
            saved_model_dir,
            signatures={
                'serving_default': serving_function
            }
        )
        
        # Create converter from saved model
        logger.info("Initializing TFLite converter")
        converter = tf.lite.TFLiteConverter.from_saved_model(
            saved_model_dir,
            signature_keys=['serving_default']
        )
        
        # Set optimization options
        if optimize:
            logger.info("Applying default optimizations")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying quantization")
            if optimize:
                # Already set optimizations above
                pass
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            # Set additional quantization options
            converter.target_spec.supported_types = [tf.float16]
            
            # Setup representative dataset for full integer quantization
            def representative_dataset():
                # Generate 100 random samples for calibration
                for _ in range(100):
                    sample = np.random.randn(*input_shape).astype(np.float32)
                    yield [sample]
            
            converter.representative_dataset = representative_dataset
        
        # Configure TFLite options
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Include TF ops for better compatibility
        ]
        
        # Enable custom ops if needed
        converter.allow_custom_ops = True
        
        # Set experimental flags
        converter.experimental_new_converter = True
        
        # Convert to TFLite format
        logger.info("Converting model to TFLite...")
        tflite_model = converter.convert()
        
        # Save the converted model
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to {save_path}")
        
        # Add metadata if requested
        if include_metadata:
            try:
                from tflite_support import metadata as tflite_metadata
                from tflite_support import metadata_schema_py_generated as schema_fb
                
                # Create metadata
                model_meta = schema_fb.ModelMetadataT()
                model_meta.name = "LightHART_Fall_Detection"
                model_meta.description = "Fall detection model for wearable devices"
                model_meta.version = "1.0"
                model_meta.author = "LightHART-TF"
                
                # Add input metadata
                input_meta = schema_fb.TensorMetadataT()
                input_meta.name = "accelerometer"
                input_meta.description = "Raw accelerometer data (x,y,z)"
                input_meta.content = schema_fb.ContentT()
                input_meta.content.contentProperties = schema_fb.FeaturePropertiesT()
                
                # Add output metadata
                output_meta = schema_fb.TensorMetadataT()
                output_meta.name = "fall_probability"
                output_meta.description = "Probability of fall (logits)"
                output_meta.content = schema_fb.ContentT()
                output_meta.content.contentProperties = schema_fb.FeaturePropertiesT()
                
                # Create metadata populator
                populator = tflite_metadata.MetadataPopulator.with_model_file(save_path)
                populator.load_metadata_buffer(tflite_metadata.Metadata.create_metadata_buffer(model_meta))
                populator.populate()
                
                logger.info("Added metadata to TFLite model")
            except ImportError:
                logger.warning("tflite-support not installed, skipping metadata")
            except Exception as e:
                logger.warning(f"Error adding metadata: {e}")
        
        # Test the model with sample input
        logger.info("Testing TFLite model inference...")
        interpreter = tf.lite.Interpreter(model_path=save_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Input details: {input_details}")
        logger.info(f"Output details: {output_details}")
        
        # Create sample input
        sample_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Test inference
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"TFLite test successful - output shape: {output.shape}")
        
        # Calculate model size
        model_size_bytes = os.path.getsize(save_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        logger.info(f"TFLite model size: {model_size_mb:.2f} MB")
        
        # Clean up temporary SavedModel
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
            
        return True
    
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        traceback.print_exc()
        
        # Try to clean up temporary files
        try:
            if 'saved_model_dir' in locals() and os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)
        except:
            pass
            
        return False

def load_tflite_model(model_path: str) -> Optional[tf.lite.Interpreter]:
    """Load a TFLite model for inference."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        logger.error(f"Error loading TFLite model: {e}")
        return None

def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    accelerometer_data: np.ndarray
) -> Union[np.ndarray, None]:
    """Run inference with a TFLite model."""
    try:
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ensure input has batch dimension
        if len(accelerometer_data.shape) == 2:
            # Add batch dimension
            accelerometer_data = np.expand_dims(accelerometer_data, axis=0)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], accelerometer_data.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return output
    except Exception as e:
        logger.error(f"Error during TFLite inference: {e}")
        return None
EOF
    fi
}

# Fix dataset loading for range error
ensure_dataset_fix() {
    dataset_file="utils/dataset_tf.py"
    if [ -f "$dataset_file" ]; then
        log "Checking for range error fix in dataset_tf.py..."
        if ! grep -q "if start_idx >= end_idx:" "$dataset_file"; then
            log "Adding range error fix to dataset_tf.py..."
            # Create backup
            cp "$dataset_file" "${dataset_file}.bak"
            
            # Insert fix using sed
            sed -i '/__getitem__/,/return/ s/batch_data\['\''accelerometer'\''\] = tf.gather(self.acc_data_with_smv, tf.range(start_idx, end_idx))/# Ensure valid range (fix for the tf.range error)\n        if start_idx >= end_idx:\n            start_idx = 0\n            end_idx = min(self.batch_size, self.num_samples)\n            \n        indices = tf.range(start_idx, end_idx)\n        \n        batch_data['\''accelerometer'\''] = tf.gather(self.acc_data_with_smv, indices)/' "$dataset_file"
            
            log "Range error fix applied to dataset_tf.py"
        else
            log "Range error fix already present in dataset_tf.py"
        fi
    else
        log_error "utils/dataset_tf.py not found. Cannot apply range error fix."
    fi
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
    
    if grep -q "Requires start <= limit when delta > 0" "${logfile}"; then
        log_error "Detected tf.range error in dataset. Applying fix..."
        ensure_dataset_fix
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
        echo "5. Range errors:"
        grep -i "Requires start <= limit when delta > 0" "${logfile}" || echo "None found"
        echo "6. Other TensorFlow errors:"
        grep -i "tensorflow" "${logfile}" | grep -i "error" || echo "None found"
        echo "7. Attribute errors:"
        grep -i "attributeerror" "${logfile}" || echo "None found"
        echo "8. Python exceptions:"
        grep -i "exception" "${logfile}" || echo "None found"
        grep -i "traceback" -A 10 "${logfile}" || echo "None found"
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

# System checks
check_dependencies
check_gpu
check_required_files
backup_code

# Ensure critical files are present
ensure_tflite_converter
ensure_dataset_fix

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
best_models = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

if best_models:
    best_model = os.path.join(model_dir, best_models[0])
    print(f'Found best model: {best_model}')
    
    # Load model
    model = tf.keras.models.load_model(best_model)
    
    # Export to TFLite
    tflite_path = os.path.join(model_dir, 'model.tflite')
    success = convert_to_tflite(model, tflite_path)
    print(f'TFLite export success: {success}')
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
        result = json.load(f)
        results.append(result)

if results:
    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(work_dir, 'results.csv'), index=False)
    
    # Calculate average results
    avg_row = {col: df[col].mean() for col in df.columns if col != 'subject'}
    avg_row['subject'] = 'Average'
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])])
    
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
