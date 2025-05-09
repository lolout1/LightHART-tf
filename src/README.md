# LightHART-TF: Deep Cross-Modal Knowledge Distillation for Fall Detection

## Overview

LightHART-TF is a TensorFlow implementation of the LightHART framework for human activity recognition and fall detection using wearable sensors and skeleton data. The framework employs a multi-modal architecture that allows training a resource-efficient model that can run on mobile/wearable devices while leveraging the high accuracy of more complex models during training.

The framework consists of three main components:

1. **Teacher Model**: A feature-rich multi-modal transformer trained on both skeleton and accelerometer data
2. **Student Model**: A lightweight transformer architecture optimized for running on wearable devices using only accelerometer data
3. **Knowledge Distillation**: A process that transfers knowledge from the teacher to the student model

## Data Pipeline

### Dataset Structure

The SmartFallMM dataset is structured as follows:

```
data/smartfallmm/
├── young/
│   ├── accelerometer/
│   │   ├── watch/
│   │   ├── phone/
│   │   └── meta_wrist/
│   ├── gyroscope/
│   │   ├── watch/
│   │   └── phone/
│   └── skeleton/
└── old/
    ├── accelerometer/
    │   ├── watch/
    │   └── phone/
    ├── gyroscope/
    └── skeleton/
```

Each modality contains different subjects performing various activities, including falls (action_id > 9) and activities of daily living (action_id ≤ 9).

### Data Loading and Preprocessing

#### 1. File Loading

- **CSV Loading**: Accelerometer, gyroscope, and skeleton data are loaded from CSV files
- **Filtering**: Data is filtered to remove NaN values and backfilled

#### 2. Modality Alignment

Accelerometer and skeleton data have different sampling rates and might not be perfectly synchronized. To align them:

1. **DTW Alignment**: Dynamic Time Warping is applied to align accelerometer and skeleton data
   - The left wrist joint (joint_id = 9) from skeleton data is used for alignment
   - DTW matches the Frobenius norm of accelerometer data with the Frobenius norm of the wrist joint trajectory
   - After alignment, data points are filtered to maintain consistent sequence lengths

#### 3. Windowing Strategies

Two windowing strategies are implemented:

- **Average Pooling**: Used to normalize sequence lengths by pooling and padding
- **Selective Sliding Window**: Activity recognition-focused approach that:
  - Detects peaks in acceleration magnitude
  - Creates windows around these peaks
  - Uses different height/distance parameters for fall (height=1.4, distance=50) vs. non-fall (height=1.2, distance=100)

#### 4. Signal Processing

- **Butterworth Filter**: Applied to reduce noise in accelerometer data (cutoff=7.5Hz, fs=25Hz)
- **Signal Magnitude Vector (SMV)**: Calculated to represent the overall intensity of motion
  ```
  SMV = sqrt(sum((acc - mean(acc))^2))
  ```

## Model Architectures

### Teacher Model (MM-Transformer)

The teacher model is a multi-modal transformer that processes both skeleton and accelerometer data:

- **Input**:
  - Skeleton data: [batch, frames, joints, coordinates]
  - Accelerometer data: [batch, frames, coordinates]

- **Architecture**:
  - **Spatial Encoder**: Processes skeleton data with 2D convolutions and extracts joint relationships
  - **Temporal Encoder**: Transformer blocks that capture temporal dependencies
  - **Joint Relationship Block**: Transformer that models relationships between joints
  - **Classification Head**: Outputs fall detection probability

- **Key Parameters**:
  - Embedding dimension: 16-32
  - Number of transformer layers: 2-4
  - Number of attention heads: 2-4

### Student Model (Transformer)

The student model is a lightweight transformer designed to run on wearable devices:

- **Input**:
  - Accelerometer data only: [batch, frames, coordinates]

- **Architecture**:
  - **Convolutional Projection**: Projects raw accelerometer data to embedding space
  - **Transformer Encoder**: Self-attention blocks that capture temporal patterns
  - **Global Pooling**: Averages features across time dimension
  - **Classification Head**: Outputs fall detection probability

- **Key Parameters**:
  - Embedding dimension: 32
  - Number of transformer layers: 2
  - Number of attention heads: 4

## Training Process

### Cross-Validation Setup

Leave-one-subject-out cross-validation is employed:

- **Training Subjects**: A subset of subjects excluding the test subject
- **Validation Subjects**: Fixed subjects [38, 46] for consistent validation
- **Test Subject**: One subject held out for testing

This ensures the model generalizes to new subjects and provides robust performance metrics.

### Teacher Model Training

1. **Data Preparation**:
   - Both accelerometer and skeleton data are loaded
   - DTW alignment is applied to synchronize modalities
   - Selective sliding window extracts relevant segments

2. **Training**:
   - Binary classification loss (BCEWithLogitsLoss) with positive class weighting
   - AdamW optimizer with weight decay for regularization
   - Early stopping based on validation loss

3. **Metrics**:
   - Accuracy, F1 score, precision, recall, and AUC-ROC

### Student Model Training

1. **Data Preparation**:
   - Only accelerometer data is used
   - Same preprocessing pipeline as teacher but without skeleton data
   - Optional SMV calculation to enhance features

2. **Training**:
   - Similar setup to teacher but with:
     - Reduced model complexity
     - Optimizations for mobile deployment

3. **Metrics**:
   - Same as teacher for direct comparison

### Knowledge Distillation

1. **Process**:
   - Train teacher model first
   - Initialize student model
   - Load teacher weights for specific test subject
   - Train student using combined loss function

2. **Distillation Loss**:
   ```
   Loss = α * Feature_Loss + (1-α) * Classification_Loss
   ```

   Where:
   - `α` controls the balance (default = 0.6)
   - `Feature_Loss` is the KL divergence between teacher and student features
   - `Classification_Loss` is the BCE loss on the labels

3. **Temperature Scaling**:
   - Features are scaled by temperature parameter (default = 4.5)
   - Higher temperature produces softer probability distributions

4. **Weighting Strategy**:
   - Teacher's correct predictions receive higher weight (1.0)
   - Incorrect predictions receive lower weight (0.5)
   - This ensures the student learns more from the teacher's correct decisions

## Results and Evaluation

### Metrics

Each model is evaluated on:

- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Visualization

- **Confusion Matrix**: Visual representation of prediction accuracy
- **Feature Distribution**: KDE plots comparing teacher and student feature spaces
- **Loss Curves**: Training and validation loss over epochs

### TFLite Export

Models are exported to TFLite format for deployment on mobile/wearable devices:

- **Quantization**: Optional int8 quantization for further model compression
- **Inference**: Optimized for accelerometer-only input
- **Size Reduction**: Final model size is ~100KB

## Usage

### Installation

```bash
# Clone repository
git clone https://github.com/username/LightHART-TF.git
cd LightHART-TF

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install ahrs  # For advanced sensor fusion
```

### Training Teacher Model

```bash
cd src
python train.py --config config/smartfallmm/teacher.yaml --phase train
```

### Training Student Model

```bash
cd src
python train.py --config config/smartfallmm/student.yaml --phase train
```

### Knowledge Distillation

```bash
cd src
python distiller.py --config config/smartfallmm/distill.yaml --phase distill --teacher-weight /path/to/teacher/model
```

### TFLite Export

```bash
cd src
python train.py --config config/smartfallmm/student.yaml --phase tflite --weights /path/to/student/model
```

## Implementation Details

### Key Components

1. **Dataset Module (`utils/dataset_tf.py`)**:
   - `SmartFallMM` class: Manages dataset loading and matching trials
   - `DatasetBuilder` class: Processes and prepares data for training
   - Alignment functions: DTW and filtering functions

2. **Data Loader (`feeder/make_dataset_tf.py`)**:
   - `UTD_MM_TF` class: TensorFlow data sequence implementation
   - SMV calculation and batch processing

3. **Model Implementations**:
   - `models/mm_transformer.py`: Teacher model implementation
   - `models/transformer_optimized.py`: Student model implementation

4. **Training Framework**:
   - `trainer/base_trainer.py`: Core training functionalities
   - `distiller.py`: Knowledge distillation framework
   - `train.py`: Main script for training and evaluation

### Code Organization

```
src/
├── config/
│   └── smartfallmm/
│       ├── teacher.yaml
│       ├── student.yaml
│       └── distill.yaml
├── models/
│   ├── mm_transformer.py
│   └── transformer_optimized.py
├── utils/
│   ├── dataset_tf.py
│   ├── loss_tf.py
│   └── tflite_converter.py
├── feeder/
│   └── make_dataset_tf.py
├── trainer/
│   └── base_trainer.py
├── train.py
└── distiller.py
```

## Conclusion

LightHART-TF demonstrates how knowledge distillation can be used to create lightweight models for wearable fall detection systems while maintaining high accuracy by leveraging more complex teacher models during training. The modular design allows for easy experimentation with different model architectures, distillation strategies, and deployment configurations.
