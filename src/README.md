# LightHART-TF: TensorFlow Implementation of Human Activity Recognition and Fall Detection

This repository contains a TensorFlow implementation of the LightHART (Lightweight Human Activity Recognition Transformer) framework, originally implemented in PyTorch. LightHART-TF provides deep learning models for human activity recognition and fall detection using multimodal sensor data.

## Directory Structure

```
src/
├── base_trainer.py          # Main training framework
├── feeder/                  # Data loading modules
├── models/                  # Model implementations
├── trainer/                 # Training utilities
├── utils/                   # Utility functions
└── visualization/           # Visualization tools
```

## Core Files

### base_trainer.py

The backbone of the training pipeline, handling:
- Training loop implementation
- Cross-validation across subjects
- Model initialization and optimization
- Evaluation metrics calculation
- Visualization generation
- Result saving

**Usage:**
```bash
python src/train.py --config config/smartfallmm/student.yaml
```

## Directories

### feeder/

Contains data loading and preprocessing utilities for various datasets.

| File | Description |
|------|-------------|
| `Make_Dataset.py` | PyTorch Dataset implementations for different data sources |
| `make_dataset_tf.py` | TensorFlow equivalent data loaders compatible with tf.data API |

### models/

Contains neural network model implementations.

| File | Description |
|------|-------------|
| `__init__.py` | Module initialization |
| `modules.py` | Reusable model components (attention layers, MLP blocks) |
| `transformer.py` | Transformer model implementations |
| `transformer_student_tf.py` | Student model for knowledge distillation |
| `transformer_tf.py` | TensorFlow implementation of transformer models |
| `ConvTransformer.py` | Convolutional transformer models |
| `functional_transformer.py` | Functional API implementation of transformers |
| `InertialTransformer.py` | Specialized transformer for inertial sensor data |

### trainer/

Contains specialized training modules.

| File | Description |
|------|-------------|
| `__init__.py` | Module initialization |
| `base_trainer.py` | Base training class (moved to root in TF version) |
| `distiller.py` | Knowledge distillation training implementation |
| `train.py` | Training utilities for TensorFlow |

### utils/

Contains utility functions used throughout the codebase.

| File | Description |
|------|-------------|
| `__init__.py` | Module initialization |
| `callbacks.py` | Custom callbacks (early stopping, model checkpointing) |
| `callbacks_tf.py` | TensorFlow-specific callbacks |
| `dataset.py` | Dataset utilities for PyTorch |
| `dataset_tf.py` | Dataset utilities for TensorFlow |
| `loader.py` | Data loading functions |
| `loaders.py` | Additional data loading utilities for TensorFlow |
| `loss.py` | Custom loss functions (focal loss, distillation loss) |
| `loss_tf.py` | TensorFlow implementations of loss functions |
| `metrics.py` | Evaluation metrics calculations |
| `tf_utils.py` | TensorFlow-specific utilities |
| `processor/` | Data processing modules |
|   `├── base.py` | Base processing classes |
|   `└── base_tf.py` | TensorFlow implementations of processors |

### visualization/

Contains visualization tools and scripts.

| File | Description |
|------|-------------|
| `comparision.py` | Model performance comparison visualizations |
| `confusion_matrix.py` | Confusion matrix visualization |

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your experiment in YAML files in the `config/` directory

3. Train a model:
```bash
python src/train.py --config config/smartfallmm/student.yaml
```

4. Test a trained model:
```bash
python src/train.py --config config/smartfallmm/student.yaml --weights path/to/model.weights.h5 --phase test
```

## Configuration

Configuration is handled via YAML files with the following main sections:

- `model`: Model class to use
- `dataset`: Dataset to use
- `subjects`: List of subjects for cross-validation
- `model_args`: Model-specific arguments
- `dataset_args`: Dataset-specific arguments
- Optimization parameters (batch size, learning rate, etc.)

## Key Features

- **Cross-Subject Validation**: Leave-one-subject-out validation
- **TensorFlow Implementation**: Complete TF equivalent of PyTorch code
- **Multimodal Fusion**: Supports skeleton and inertial sensor data
- **Knowledge Distillation**: Supports training lightweight models
- **TFLite Export**: Models can be exported to TFLite format for deployment

## Utility Scripts

Several utility scripts are available:

- `train.sh`: Batch training of multiple models
- `exec.sh`: Execute distillation experiments
- `start.sh`: Quick-start script with configuration

## Example Results

Model performance varies by subject but typically achieves:
- Accuracy: 75-90%
- F1 Score: 75-95%
- AUC: 80-95%

For detailed results, see generated `scores.csv` and `scores.json` in the experiment output directory.

## Troubleshooting

- **GPU Memory Issues**: Reduce batch size or model size
- **Training Errors**: Check error logs in the experiment directory
- **Data Loading Errors**: Verify dataset paths and format
- **Model Saving Errors**: Ensure proper file extensions (.weights.h5)

---

For more details on using specific models or datasets, refer to the comments within the corresponding files.
