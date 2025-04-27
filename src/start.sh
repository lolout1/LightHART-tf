#!/bin/bash

# Set environment variables
export PYTHONPATH="$(pwd):$(pwd)/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optional: Set XLA flags for GPU computation
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib

# Run training script
python train.py \
  --config config/smartfallmm/student_improved.yaml \
  --work-dir ../experiments/student_$(date +%Y-%m-%d_%H-%M-%S) \
  --model-saved-name student_model \
  --device 0

# Make the script executable with: chmod +x start.sh
