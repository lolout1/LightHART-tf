#!/bin/bash
# src/start.sh
set -e

# Create timestamp for this run
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
log_dir="logs"
mkdir -p $log_dir
mkdir -p experiments

# Setup environment
export PYTHONPATH="$(pwd):$(pwd)/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export CUDA_VISIBLE_DEVICES="0" # Use GPU 0

# Create a log file
logfile="${log_dir}/training_${timestamp}.log"
work_dir="experiments/student_${timestamp}"

echo "Starting training at $(date)" | tee -a $logfile
echo "Training logs will be saved to: $logfile" | tee -a $logfile
echo "Model will be saved to: $work_dir" | tee -a $logfile

# Check TensorFlow status
echo "Checking TensorFlow installation..." | tee -a $logfile
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}');" | tee -a $logfile

# Make sure required directories exist
mkdir -p utils
mkdir -p models
mkdir -p config/smartfallmm

# Run with error trapping
{
    python train.py \
      --config config/smartfallmm/optimized.yaml \
      --work-dir $work_dir \
      --model-saved-name student_model 2>&1 | tee -a $logfile
    
    training_status=${PIPESTATUS[0]}
    
    if [ $training_status -eq 0 ]; then
        echo "Training completed successfully at $(date)" | tee -a $logfile
        
        # Create symlink to latest experiment
        ln -sf "student_${timestamp}" "experiments/latest"
        echo "Created symlink: experiments/latest -> student_${timestamp}" | tee -a $logfile
    else
        echo "Training failed with exit code $training_status at $(date)" | tee -a $logfile
    fi
} || {
    error_code=$?
    echo "Script failed with error code $error_code" | tee -a $logfile
    exit $error_code
}

echo "Done!" | tee -a $logfile
