#!/bin/bash
set -euo pipefail

MODEL_PATH="../experiments/student_2025-05-10_07-30-58_20250510_073100/models/student_model_32.weights.h5"
CONFIG_FILE="config/smartfallmm/student.yaml"
OUTPUT_DIR="../experiments/tflite_test_subjects"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "====== Testing Subjects 38, 46 ======"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "===================================="

mkdir -p "$OUTPUT_DIR"

# Test Keras model
echo -e "\n=== Testing Keras Model ==="
python3 test_subjects.py

# Convert to TFLite
echo -e "\n=== Converting to TFLite ==="
python3 -c "
import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from models.transformer_optimized import TransModel
from utils.tflite_converter import convert_to_tflite
import yaml

# Load config
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

# Create and load model
model = TransModel(**config['model_args'])
dummy_input = {'accelerometer': tf.zeros((1, 64, 3))}
_ = model(dummy_input, training=False)
model.load_weights('${MODEL_PATH}')

# Convert to TFLite
output_path = '${OUTPUT_DIR}/model_${TIMESTAMP}.tflite'
success = convert_to_tflite(
    model=model,
    save_path=output_path,
    input_shape=(1, 64, 3),
    quantize=False,
    optimize_for_mobile=True
)

if success:
    print(f'TFLite model saved to: {output_path}')
else:
    print('Conversion failed')
    sys.exit(1)
"

echo -e "\n✅ Testing completed!"
