# Student model configuration - exactly matching PyTorch hyperparameters
model: src.models.transformer_tf.StudentTransformerTF
dataset: smartfallmm

# Subjects for fall detection
subjects: [32, 39, 30, 31, 33, 34, 35, 37, 43, 44, 45, 36, 29]

model_args:
  acc_frames: 128
  num_classes: 1
  num_heads: 2
  acc_coords: 4
  num_layers: 2
  embed_dim: 32
  dropout: 0.5  # Matching PyTorch exactly (0.5)

dataset_args: 
  mode: 'sliding_window'
  max_length: 128
  task: 'fd'
  modalities: ['accelerometer']
  age_group: ['young']
  sensors: ['watch']

# Training parameters - exact match to PyTorch
batch_size: 64  # Match PyTorch batch size
test_batch_size: 64
val_batch_size: 64
num_epoch: 80  # Match PyTorch epochs

# Optimization parameters - exact match to PyTorch
optimizer: adamw
base_lr: 1e-3  # Match PyTorch learning rate
weight_decay: 4e-4  # Match PyTorch weight decay

# Training phase
phase: 'train'
