#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import json
import traceback
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import shutil

logger = logging.getLogger('lightheart-tf')

class EarlyStopping:
    """Robust early stopping implementation with error handling"""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.wait = 0
        self.stopped_epoch = 0
        self.history = []
    
    def __call__(self, val_loss):
        """Check if training should early stop based on validation loss"""
        try:
            # Convert to float if tensor
            if isinstance(val_loss, tf.Tensor):
                val_loss = float(val_loss.numpy())
            else:
                val_loss = float(val_loss)
                
            # Record history
            self.history.append(val_loss)
            
            if self.best_loss is None:
                # First epoch
                self.best_loss = val_loss
                return False
                
            # Check if improved
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.wait = 0
                return False
            else:
                # No improvement
                self.counter += 1
                self.wait += 1
                
                # Check if patience exceeded
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.stopped_epoch = len(self.history)
                    return True
                
                return False
                
        except Exception as e:
            logging.error(f"Error in EarlyStopping: {e}")
            # Don't stop on errors
            return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.wait = 0
        self.best_loss = None
        self.early_stop = False
        self.history = []
        self.stopped_epoch = 0

class BaseTrainer:
    """Base trainer class for fall detection models with robust TFLite conversion support"""
    def __init__(self, arg):
        self.arg = arg
        
        # Initialize metrics tracking
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.test_accuracy = 0 
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0 
        self.test_auc = 0
        
        # Initialize subject splits
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        
        # Initialize model components
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = {}
        self.pos_weights = None
        
        # Setup early stopping
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        
        # Setup directories and load model
        self.setup_directories()
        self.model = self.load_model()
        
        # Log model information
        num_params = self.count_parameters(self.model)
        self.print_log(f"Model: {self.arg.model}")
        self.print_log(f"Parameters: {num_params:,}")
        self.print_log(f"Model size: {num_params * 4 / (1024**2):.2f} MB")
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'results'), exist_ok=True)
        
        self.model_path = os.path.join(
            self.arg.work_dir, 
            'models', 
            self.arg.model_saved_name
        )
    
    def import_class(self, import_str):
        """Dynamically import a class from a string path"""
        if import_str is None:
            raise ValueError("Import path cannot be None")
            
        mod_str, _sep, class_str = import_str.rpartition('.')
        
        for prefix in ['', 'src.']:
            try:
                import importlib
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
                
        raise ImportError(f"Cannot import {class_str} from {mod_str}")
    
    def print_log(self, message):
        """Print and log a message"""
        logger.info(message)
        
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(message, file=f)
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        total_params = 0
        for var in model.trainable_variables:
            total_params += tf.size(var).numpy()
        return total_params
    
    def load_model(self):
        """Load and initialize model"""
        try:
            if self.arg.phase == 'train':
                if self.arg.model is None:
                    raise ValueError("Model class path is required")
                    
                model_class = self.import_class(self.arg.model)
                model = model_class(**self.arg.model_args)
                self.print_log(f"Created model: {self.arg.model}")
                
                try:
                    # Build model with dummy accelerometer input
                    acc_frames = self.arg.model_args.get('acc_frames', 128)
                    acc_coords = 3  # Always use 3 for raw accelerometer data
                    
                    # Create dummy input
                    dummy_input = {
                        'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                    }
                    
                    # Run a forward pass to build the model
                    _ = model(dummy_input, training=False)
                    
                    self.print_log("Model built successfully")
                except Exception as e:
                    self.print_log(f"Warning: Could not pre-build model: {e}")
                    self.print_log(traceback.format_exc())
                
                return model
            else:
                # Handle loading for test/eval phase
                if hasattr(self.arg, 'weights') and self.arg.weights:
                    try:
                        # Try loading full model
                        model = tf.keras.models.load_model(self.arg.weights)
                        self.print_log(f"Loaded model from {self.arg.weights}")
                        return model
                    except Exception:
                        # Fall back to loading just weights
                        if self.arg.model is None:
                            raise ValueError("Model class path is required when weights cannot be directly loaded")
                            
                        model_class = self.import_class(self.arg.model)
                        model = model_class(**self.arg.model_args)
                        
                        # Build model with dummy input
                        acc_frames = self.arg.model_args.get('acc_frames', 128)
                        acc_coords = 3  # Always use 3 for raw accelerometer data
                        dummy_input = {'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)}
                        _ = model(dummy_input, training=False)
                        
                        # Load weights
                        model.load_weights(self.arg.weights)
                        self.print_log(f"Loaded weights from {self.arg.weights}")
                        return model
                else:
                    # Create new model when no weights specified
                    if self.arg.model is None:
                        raise ValueError("Model class path is required")
                        
                    model_class = self.import_class(self.arg.model)
                    model = model_class(**self.arg.model_args)
                    self.print_log(f"Created model: {self.arg.model}")
                    return model
        except Exception as e:
            self.print_log(f"Error loading model: {e}")
            self.print_log(traceback.format_exc())
            raise
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced datasets"""
        from collections import Counter
        
        counter = Counter(labels)
        
        if 1 in counter and 0 in counter:
            pos_weight = counter[0] / counter[1]
        else:
            pos_weight = 1.0
            
        self.print_log(f"Class balance - Negative: {counter.get(0, 0)}, Positive: {counter.get(1, 0)}")
        self.print_log(f"Positive class weight: {pos_weight:.4f}")
        
        return tf.constant(pos_weight, dtype=tf.float32)
    
    def load_optimizer(self):
        """Load optimizer based on configuration"""
        try:
            if not hasattr(self.arg, 'optimizer'):
                self.arg.optimizer = 'adam'
                
            if not hasattr(self.arg, 'base_lr'):
                self.arg.base_lr = 0.001
                
            if not hasattr(self.arg, 'weight_decay'):
                self.arg.weight_decay = 0.0004
            
            if self.arg.optimizer.lower() == "adam":
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.arg.base_lr
                )
            elif self.arg.optimizer.lower() == "adamw":
                self.optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay
                )
            elif self.arg.optimizer.lower() == "sgd":
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.arg.base_lr,
                    momentum=0.9
                )
            else:
                self.print_log(f"Unknown optimizer: {self.arg.optimizer}, using Adam")
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
                
            self.print_log(f"Optimizer: {self.optimizer.__class__.__name__}, LR={self.arg.base_lr}")
            return True
        except Exception as e:
            self.print_log(f"Error loading optimizer: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def load_loss(self):
        """Load loss function with robust shape handling"""
        try:
            if not hasattr(self, 'pos_weights') or self.pos_weights is None:
                self.pos_weights = tf.constant(1.0)
            
            def weighted_bce(y_true, y_pred):
                """Binary cross entropy with robust shape handling"""
                try:
                    # Cast to float32
                    y_true = tf.cast(y_true, tf.float32)
                    
                    # Ensure y_true has the same shape as y_pred
                    if len(y_pred.shape) > 1 and y_pred.shape[-1] == 1:
                        # If y_pred is [batch_size, 1], ensure y_true has same shape
                        if len(y_true.shape) == 1:
                            # Convert [batch_size] to [batch_size, 1]
                            y_true = tf.expand_dims(y_true, -1)
                    elif len(y_pred.shape) == 1 and len(y_true.shape) > 1:
                        # If predictions are [batch_size] but labels are [batch_size, 1]
                        y_pred = tf.expand_dims(y_pred, -1)
                    
                    # Final safety check - if shapes still don't match, reshape both
                    if y_true.shape != y_pred.shape:
                        batch_size = tf.shape(y_true)[0]
                        y_true = tf.reshape(y_true, [batch_size, 1])
                        y_pred = tf.reshape(y_pred, [batch_size, 1])
                    
                    # Compute BCE with correct shapes
                    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
                    weights = y_true * (self.pos_weights - 1.0) + 1.0
                    return tf.reduce_mean(weights * bce)
                    
                except Exception as e:
                    logging.error(f"Error in loss calculation: {e}")
                    logging.error(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                    
                    # Return a small constant loss to continue training
                    return tf.constant(0.1, dtype=tf.float32)
            
            self.criterion = weighted_bce
            self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")
            return True
        except Exception as e:
            self.print_log(f"Error loading loss function: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def load_data(self):
        """Load and prepare data for training/validation/testing"""
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
            
            # Get use_smv flag from config or default to False
            use_smv = getattr(self.arg, 'use_smv', False)
            self.print_log(f"Data loader configuration: use_smv={use_smv}")
            
            if self.arg.phase == 'train':
                builder = prepare_smartfallmm_tf(self.arg)
                
                if not self.train_subjects:
                    self.print_log("No training subjects specified")
                    return False
                
                self.print_log(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False)
                
                if any(len(x) == 0 for x in self.norm_train.values()):
                    self.print_log("Error: Training data is empty")
                    return False
                
                # Create data loader with proper error handling
                try:
                    # Try with use_smv parameter first
                    self.data_loader['train'] = Feeder(
                        dataset=self.norm_train,
                        batch_size=self.arg.batch_size,
                        use_smv=use_smv
                    )
                except TypeError as e:
                    if "unexpected keyword argument 'use_smv'" in str(e):
                        self.print_log("Data loader doesn't support use_smv parameter, using default")
                        self.data_loader['train'] = Feeder(
                            dataset=self.norm_train,
                            batch_size=self.arg.batch_size
                        )
                    else:
                        raise
                
                # Check sample batch
                try:
                    sample_batch = next(iter(self.data_loader['train']))
                    self.print_log(f"Train data sample shapes: {[(k, v.shape) for k, v in sample_batch[0].items()]}")
                    self.print_log(f"Train labels shape: {sample_batch[1].shape}")
                except Exception as e:
                    self.print_log(f"Warning: Could not check train data shapes: {e}")
                
                # Calculate class weights
                self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
                
                # Visualize class distribution
                try:
                    self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
                except Exception as e:
                    self.print_log(f"Error visualizing distribution: {e}")
                
                # Load validation data
                if self.val_subject:
                    self.print_log(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        self.print_log("Warning: Validation data is empty, using subset of training data")
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)
                        
                        self.norm_val = {
                            k: v[-val_size:].copy() for k, v in self.norm_train.items()
                        }
                        self.norm_train = {
                            k: v[:-val_size].copy() for k, v in self.norm_train.items()
                        }
                    
                    # Create validation data loader
                    try:
                        self.data_loader['val'] = Feeder(
                            dataset=self.norm_val,
                            batch_size=self.arg.val_batch_size,
                            use_smv=use_smv
                        )
                    except TypeError as e:
                        if "unexpected keyword argument 'use_smv'" in str(e):
                            self.data_loader['val'] = Feeder(
                                dataset=self.norm_val,
                                batch_size=self.arg.val_batch_size
                            )
                        else:
                            raise
                    
                    # Visualize validation distribution
                    try:
                        self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                    except Exception as e:
                        self.print_log(f"Error visualizing distribution: {e}")
                
                # Load test data
                if self.test_subject:
                    self.print_log(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        self.print_log("Warning: Test data is empty")
                        return False
                    
                    # Create test data loader
                    try:
                        self.data_loader['test'] = Feeder(
                            dataset=self.norm_test,
                            batch_size=self.arg.test_batch_size,
                            use_smv=use_smv
                        )
                    except TypeError as e:
                        if "unexpected keyword argument 'use_smv'" in str(e):
                            self.data_loader['test'] = Feeder(
                                dataset=self.norm_test,
                                batch_size=self.arg.test_batch_size
                            )
                        else:
                            raise
                    
                    # Visualize test distribution
                    subject_id = self.test_subject[0] if self.test_subject else 'unknown'
                    try:
                        self.distribution_viz(
                            self.norm_test['labels'],
                            self.arg.work_dir,
                            f'test_{subject_id}'
                        )
                    except Exception as e:
                        self.print_log(f"Error visualizing distribution: {e}")
                
                self.print_log("Data loading complete")
                return True
                
            elif self.arg.phase == 'test':
                # Load only test data for test phase
                if not self.test_subject:
                    self.print_log("No test subjects specified")
                    return False
                
                builder = prepare_smartfallmm_tf(self.arg)
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                
                if any(len(x) == 0 for x in self.norm_test.values()):
                    self.print_log("Error: Test data is empty")
                    return False
                
                # Create test data loader
                try:
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size,
                        use_smv=use_smv
                    )
                except TypeError as e:
                    if "unexpected keyword argument 'use_smv'" in str(e):
                        self.data_loader['test'] = Feeder(
                            dataset=self.norm_test,
                            batch_size=self.arg.test_batch_size
                        )
                    else:
                        raise
                
                self.print_log("Test data loading complete")
                return True
        
        except Exception as e:
            self.print_log(f"Error loading data: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def distribution_viz(self, labels, work_dir, mode):
        """Visualize class distribution"""
        try:
            values, count = np.unique(labels, return_counts=True)
            
            plt.figure(figsize=(8, 6))
            plt.bar(values, count, color=['blue', 'red'])
            plt.xlabel('Labels')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Label Distribution')
            plt.xticks(values)
            
            for i, v in enumerate(count):
                plt.text(values[i], v + 0.1, str(v), ha='center')
            
            viz_dir = os.path.join(work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'{mode}_distribution.png'))
            plt.close()
            
            dist_str = ", ".join([f"Label {int(v)}: {c}" for v, c in zip(values, count)])
            self.print_log(f"{mode} distribution: {dist_str}")
            
        except Exception as e:
            self.print_log(f"Error visualizing distribution: {e}")
            self.print_log(traceback.format_exc())
    
    def calculate_metrics(self, targets, predictions):
        """Calculate performance metrics"""
        try:
            # Convert to numpy arrays if needed
            if isinstance(targets, tf.Tensor):
                targets = targets.numpy()
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            
            # Flatten arrays
            targets = np.array(targets).flatten()
            predictions = np.array(predictions).flatten()
            
            # Calculate basic accuracy
            accuracy = accuracy_score(targets, predictions) * 100
            
            # Get unique values
            unique_targets = np.unique(targets)
            unique_preds = np.unique(predictions)
            
            # Handle edge cases with single class
            if len(unique_targets) <= 1 or len(unique_preds) <= 1:
                if len(unique_targets) == 1 and len(unique_preds) == 1 and unique_targets[0] == unique_preds[0]:
                    # All predictions match and are correct
                    if unique_targets[0] == 1:
                        precision = 100.0
                        recall = 100.0
                        f1 = 100.0
                    else:
                        precision = 0.0
                        recall = 0.0
                        f1 = 0.0
                    auc = 50.0
                else:
                    # Manual calculation from confusion matrix
                    tp = np.sum((predictions == 1) & (targets == 1))
                    fp = np.sum((predictions == 1) & (targets == 0))
                    fn = np.sum((predictions == 0) & (targets == 1))
                    
                    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    auc = 50.0
            else:
                # Normal case with both classes present
                precision = precision_score(targets, predictions, zero_division=0) * 100
                recall = recall_score(targets, predictions, zero_division=0) * 100
                f1 = f1_score(targets, predictions, zero_division=0) * 100
                try:
                    auc = roc_auc_score(targets, predictions) * 100
                except:
                    auc = 50.0
            
            return accuracy, f1, recall, precision, auc
        except Exception as e:
            self.print_log(f"Error calculating metrics: {e}")
            self.print_log(traceback.format_exc())
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    def train(self, epoch):
        """Training loop for a single epoch"""
        try:
            self.print_log(f"Starting epoch {epoch+1} training")
            start_time = time.time()
            
            # Get training data loader
            loader = self.data_loader['train']
            
            # Get number of batches from loader's __len__ method
            total_batches = len(loader)
            num_samples = total_batches * self.arg.batch_size
            
            self.print_log(f"Epoch {epoch+1}/{self.arg.num_epoch} - {total_batches} batches (~{num_samples} samples)")
            
            # Training metrics tracking
            train_loss = 0.0
            all_labels = []
            all_preds = []
            steps = 0
            
            # Add a timeout mechanism to prevent infinite loops
            max_batch_time = 300  # 5 minutes max per batch
            start_batch_time = time.time()
            
            # Loop through batches
            for batch_idx in range(total_batches):
                # Check for timeout
                batch_duration = time.time() - start_batch_time
                if batch_duration > max_batch_time:
                    self.print_log(f"WARNING: Batch {batch_idx} taking too long ({batch_duration:.1f}s), skipping")
                    start_batch_time = time.time()
                    continue
                
                # Log progress periodically
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Training epoch {epoch+1}: batch {batch_idx+1}/{total_batches}")
                
                try:
                    # Get batch data - use indexing to avoid iterator issues
                    inputs, targets, batch_indices = loader[batch_idx]
                    
                    # Debug first batch data
                    if batch_idx == 0:
                        self.print_log(f"First batch indices shape: {batch_indices.shape}")
                        for key, value in inputs.items():
                            self.print_log(f"First batch {key} shape: {value.shape}")
                        self.print_log(f"First batch labels shape: {targets.shape}")
                        self.print_log(f"First batch indices shape: {batch_indices.shape}")
                        for key, value in inputs.items():
                            self.print_log(f"Input '{key}' shape: {value.shape}")
                    
                    # Ensure targets are float32 for BCE loss
                    targets = tf.cast(targets, tf.float32)
                    
                    # Extract accelerometer data (primary modality)
                    if isinstance(inputs, dict) and 'accelerometer' in inputs:
                        acc_data = inputs['accelerometer']
                    else:
                        self.print_log(f"Error: No accelerometer data in batch {batch_idx}")
                        continue
                    
                    # Forward and backward pass with gradient tape
                    with tf.GradientTape() as tape:
                        # Forward pass with accelerometer data
                        outputs = self.model(inputs, training=True)
                        
                        # Handle different output formats (logits or logits+features)
                        if isinstance(outputs, tuple) and len(outputs) > 0:
                            logits = outputs[0]  # First element is logits
                        else:
                            logits = outputs  # Single output is logits
                        
                        # Ensure proper shapes for loss calculation
                        if len(logits.shape) > 1 and logits.shape[-1] > 0:
                            # If logits has a feature dimension, match target shape
                            if len(targets.shape) == 1:
                                targets = tf.reshape(targets, [-1, 1])
                            loss = self.criterion(targets, logits)
                        else:
                            # Reshape both for compatibility
                            batch_size = tf.shape(acc_data)[0]
                            targets_reshaped = tf.reshape(targets, [batch_size, 1])
                            logits_reshaped = tf.reshape(logits, [batch_size, 1])
                            loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    # Calculate and apply gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for grad in gradients:
                        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                            has_nan = True
                            break
                            
                    if has_nan:
                        self.print_log(f"WARNING: NaN gradients detected in batch {batch_idx}")
                        continue
                        
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Get predictions from logits
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        # Multi-class: use argmax
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        # Binary: threshold sigmoid output
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    # Update statistics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                    # Reset batch timer
                    start_batch_time = time.time()
                    
                except Exception as e:
                    self.print_log(f"Error in batch {batch_idx}: {e}")
                    self.print_log(traceback.format_exc())
                    start_batch_time = time.time()  # Reset timer and continue
                    continue
            
            # Calculate epoch metrics
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                # Save loss history
                self.train_loss_summary.append(float(train_loss))
                
                # Log training metrics
                epoch_time = time.time() - start_time
                self.print_log(
                    f"Epoch {epoch+1} results: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}% "
                    f"({epoch_time:.2f}s)"
                )
                
                # Run validation
                self.print_log(f"Running validation for epoch {epoch+1}")
                val_loss = self.eval(epoch, loader_name='val')
                
                # Save validation loss history
                self.val_loss_summary.append(float(val_loss))
                
                # Check early stopping
                if self.early_stop(val_loss):
                    self.print_log(f"Early stopping triggered at epoch {epoch+1}")
                    return True
                
                return False
            else:
                self.print_log(f"Warning: No steps completed in epoch {epoch+1}")
                return False
                
        except Exception as e:
            self.print_log(f"Critical error in epoch {epoch+1}: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def eval(self, epoch, loader_name='val', result_file=None):
        """Evaluation loop for validation or testing"""
        try:
            start_time = time.time()
            
            # Get the requested data loader
            loader = self.data_loader.get(loader_name)
            if loader is None:
                self.print_log(f"No data loader for {loader_name}")
                return float('inf')
            
            # Get number of batches
            total_batches = len(loader)
            num_samples = total_batches * (
                self.arg.val_batch_size if loader_name == 'val' else self.arg.test_batch_size
            )
            
            self.print_log(f"Evaluating {loader_name} (epoch {epoch+1}) - {total_batches} batches (~{num_samples} samples)")
            
            # Evaluation metrics tracking
            eval_loss = 0.0
            all_labels = []
            all_preds = []
            all_logits = []  # Keep logits for later comparison
            steps = 0
            
            # Add timeout protection
            max_batch_time = 300  # 5 minutes max per batch
            start_batch_time = time.time()
            
            # Loop through batches
            for batch_idx in range(total_batches):
                # Check for timeout
                batch_duration = time.time() - start_batch_time
                if batch_duration > max_batch_time:
                    self.print_log(f"WARNING: Evaluation batch {batch_idx} taking too long ({batch_duration:.1f}s), skipping")
                    start_batch_time = time.time()
                    continue
                
                # Log progress periodically
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Eval {loader_name} (epoch {epoch+1}): batch {batch_idx+1}/{total_batches}")
                
                try:
                    # Get batch data - use indexing to avoid iterator issues
                    inputs, targets, _ = loader[batch_idx]
                    
                    # Debug first batch data
                    if batch_idx == 0:
                        for key, value in inputs.items():
                            self.print_log(f"First batch {key} shape: {value.shape}")
                        self.print_log(f"First batch labels shape: {targets.shape}")
                    
                    # Ensure targets are float32 for BCE loss
                    targets = tf.cast(targets, tf.float32)
                    
                    # Forward pass with data
                    outputs = self.model(inputs, training=False)
                    
                    # Handle different output formats (logits or logits+features)
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]  # First element is logits
                    else:
                        logits = outputs  # Single output is logits
                    
                    # Store raw logits for later comparison
                    batch_logits = logits.numpy()
                    all_logits.append(batch_logits)
                    
                    # Ensure proper shapes for loss calculation
                    if len(logits.shape) > 1 and logits.shape[-1] > 0:
                        # If logits has a feature dimension, match target shape
                        if len(targets.shape) == 1:
                            targets = tf.reshape(targets, [-1, 1])
                        loss = self.criterion(targets, logits)
                    else:
                        # Reshape both for compatibility
                        batch_size = tf.shape(inputs['accelerometer'])[0]
                        targets_reshaped = tf.reshape(targets, [batch_size, 1])
                        logits_reshaped = tf.reshape(logits, [batch_size, 1])
                        loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    # Get predictions from logits
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        # Multi-class: use argmax
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        # Binary: threshold sigmoid output
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    # Update statistics
                    eval_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                    # Reset batch timer
                    start_batch_time = time.time()
                    
                except Exception as e:
                    self.print_log(f"Error in evaluation batch {batch_idx}: {e}")
                    self.print_log(traceback.format_exc())
                    start_batch_time = time.time()  # Reset timer and continue
                    continue
            
            # Calculate metrics
            if steps > 0:
                eval_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                # Log evaluation metrics
                self.print_log(
                    f"{loader_name.capitalize()}: "
                    f"Loss={eval_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}%"
                )
                
                # Log evaluation time
                epoch_time = time.time() - start_time
                self.print_log(f"{loader_name.capitalize()} time: {epoch_time:.2f}s")
                
                # Handle validation results
                if loader_name == 'val':
                    is_best = False
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        is_best = True
                        self.print_log(f"New best validation loss: {eval_loss:.4f}")
                    
                    if is_best:
                        self.save_model(epoch)
                
                # Handle test results
                elif loader_name.startswith('test'):
                    # Save metrics
                    self.test_accuracy = accuracy
                    self.test_f1 = f1
                    self.test_recall = recall
                    self.test_precision = precision
                    self.test_auc = auc_score
                    
                    # Create confusion matrix
                    subject_id = self.test_subject[0] if self.test_subject else None
                    if subject_id:
                        self.cm_viz(all_preds, all_labels, subject_id)
                    
                    # Save results to file
                    results = {
                        "subject": self.test_subject[0] if self.test_subject else "unknown",
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "auc": float(auc_score),
                        "loss": float(eval_loss),
                        "logits_sample": [float(x) for x in all_logits[0].flatten()[:5]] if len(all_logits) > 0 else []
                    }
                    
                    results_file = os.path.join(
                        self.arg.work_dir,
                        'results',
                        f'test_results_{self.test_subject[0] if self.test_subject else "unknown"}.json'
                    )
                    
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Save detailed predictions if requested
                    if result_file:
                        with open(result_file, 'w') as f:
                            for pred, true in zip(all_preds, all_labels):
                                f.write(f"{pred} ==> {true}\n")
                
                return eval_loss
            else:
                self.print_log(f"No evaluation steps for {loader_name}")
                return float('inf')
                
        except Exception as e:
            self.print_log(f"Error in evaluation: {e}")
            self.print_log(traceback.format_exc())
            return float('inf')
    
    def cm_viz(self, y_pred, y_true, subject_id=None):
        """Visualize confusion matrix"""
        try:
            if isinstance(y_pred, tf.Tensor):
                y_pred = y_pred.numpy()
            if isinstance(y_true, tf.Tensor):
                y_true = y_true.numpy()
            
            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix' + (f' (Subject {subject_id})' if subject_id else ''))
            plt.colorbar()
            
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            filename = f'confusion_matrix_{subject_id}' if subject_id else 'confusion_matrix'
            plt.savefig(os.path.join(viz_dir, f'{filename}.png'))
            plt.close()
            
        except Exception as e:
            self.print_log(f"Error visualizing confusion matrix: {e}")
            self.print_log(traceback.format_exc())
    
    def loss_viz(self, train_loss, val_loss, subject_id=None):
        """Visualize training and validation loss curves"""
        try:
            if not train_loss or not val_loss:
                self.print_log("No loss data to visualize")
                return
            
            epochs = range(1, len(train_loss) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
            
            title = 'Training vs Validation Loss'
            if subject_id:
                title += f' (Subject {subject_id})'
            plt.title(title)
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            filename = f'loss_curves_{subject_id}' if subject_id else 'loss_curves'
            plt.savefig(os.path.join(viz_dir, f'{filename}.png'))
            plt.close()
            
        except Exception as e:
            self.print_log(f"Error visualizing loss: {e}")
            self.print_log(traceback.format_exc())
    
    def save_model(self, epoch):
        """Save model weights and export to TFLite"""
        try:
            if self.test_subject:
                base_filename = f"{self.model_path}_{self.test_subject[0]}"
            else:
                base_filename = f"{self.model_path}_epoch{epoch}"
            
            # Save weights
            weights_path = f"{base_filename}.weights.h5"
            self.model.save_weights(weights_path)
            self.print_log(f"Saved model weights to {weights_path}")
            
            # Try to save full model
            try:
                model_path = f"{base_filename}.keras"
                self.model.save(model_path)
                self.print_log(f"Saved full model to {model_path}")
            except Exception as e:
                self.print_log(f"Warning: Could not save full model: {e}")
                self.print_log(traceback.format_exc())
            
            # Try to export TFLite model
            try:
                # Check if model has built-in export method
                if hasattr(self.model, 'export_to_tflite'):
                    self.print_log("Using model's built-in TFLite export method")
                    tflite_path = f"{base_filename}.tflite"
                    success = self.model.export_to_tflite(tflite_path)
                else:
                    # Use utility function
                    from utils.tflite_converter import convert_to_tflite
                    
                    acc_frames = self.arg.model_args.get('acc_frames', 128)
                    
                    tflite_path = f"{base_filename}.tflite"
                    success = convert_to_tflite(
                        model=self.model,
                        save_path=tflite_path,
                        input_shape=(1, acc_frames, 3),  # Always use 3 for raw accelerometer
                        quantize=getattr(self.arg, 'quantize', False)
                    )
                
                if success:
                    self.print_log(f"Exported TFLite model to {tflite_path}")
                else:
                    self.print_log("Warning: TFLite export failed")
            except Exception as e:
                self.print_log(f"Warning: TFLite export failed: {e}")
                self.print_log(traceback.format_exc())
            
            return True
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def compare_regular_and_tflite_inference(self, subject_id=None):
        """Compare regular and TFLite model inference on the same test set"""
        try:
            if not self.test_subject:
                self.print_log("No test subject specified for comparison")
                return False
                
            subject_id = self.test_subject[0] if self.test_subject else "unknown"
            self.print_log(f"Comparing regular model vs TFLite model for subject {subject_id}")
            
            # Check for regular model weights
            weights_path = f"{self.model_path}_{subject_id}.weights.h5"
            tflite_path = f"{self.model_path}_{subject_id}.tflite"
            
            if not os.path.exists(weights_path):
                self.print_log(f"Model weights not found at {weights_path}")
                return False
                
            if not os.path.exists(tflite_path):
                self.print_log(f"TFLite model not found at {tflite_path}")
                return False
                
            # Load best weights for regular model
            self.model.load_weights(weights_path)
            self.print_log(f"Loaded best weights from {weights_path}")
            
            # Get test data
            loader = self.data_loader.get('test')
            if loader is None:
                self.print_log("No test data loader available")
                return False
                
            # Run inference with regular model
            regular_preds = []
            regular_logits = []
            all_labels = []
            
            # Iterate through test data
            self.print_log("Running inference with regular model...")
            for inputs, targets, _ in loader:
                # Forward pass
                outputs = self.model(inputs, training=False)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                # Store logits
                regular_logits.extend(logits.numpy())
                
                # Get predictions
                if len(logits.shape) > 1 and logits.shape[-1] > 1:
                    predictions = tf.argmax(logits, axis=-1)
                else:
                    predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                # Store predictions and labels
                regular_preds.extend(predictions.numpy())
                all_labels.extend(targets.numpy())
            
            # Calculate metrics for regular model
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, regular_preds)
            
            self.print_log(
                f"Regular model metrics: "
                f"Acc={accuracy:.2f}%, "
                f"F1={f1:.2f}%, "
                f"Prec={precision:.2f}%, "
                f"Rec={recall:.2f}%, "
                f"AUC={auc_score:.2f}%"
            )
            
            # Sample of regular model logits
            if len(regular_logits) > 0:
                self.print_log(f"Regular model logits sample: {regular_logits[0].flatten()[:5]}")
            
            # Load and run TFLite model
            try:
                self.print_log("Running inference with TFLite model...")
                
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                self.print_log(f"TFLite input shape: {input_details[0]['shape']}")
                self.print_log(f"TFLite output shape: {output_details[0]['shape']}")
                
                # Run inference on test set
                tflite_preds = []
                tflite_logits = []
                
                # Iterate through test data again
                for inputs, _, _ in loader:
                    if 'accelerometer' in inputs:
                        # Get accelerometer data
                        acc_data = inputs['accelerometer']
                        
                        # Process one sample at a time for TFLite
                        for i in range(acc_data.shape[0]):
                            sample = acc_data[i:i+1].numpy()  # Add batch dimension
                            
                            # Set input tensor
                            interpreter.set_tensor(input_details[0]['index'], sample)
                            
                            # Run inference
                            interpreter.invoke()
                            
                            # Get output tensor
                            output = interpreter.get_tensor(output_details[0]['index'])
                            
                            # Store logits
                            tflite_logits.append(output)
                            
                            # Get prediction
                            pred = 1 if output.flatten()[0] > 0.5 else 0
                            tflite_preds.append(pred)
                
                # Calculate metrics for TFLite model
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, tflite_preds)
                
                self.print_log(
                    f"TFLite model metrics: "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}%"
                )
                
                # Sample of TFLite model logits
                if len(tflite_logits) > 0:
                    self.print_log(f"TFLite model logits sample: {tflite_logits[0].flatten()[:5]}")
                
                # Check for differences in predictions
                mismatches = sum(1 for a, b in zip(regular_preds, tflite_preds) if a != b)
                if mismatches > 0:
                    mismatch_percent = 100 * mismatches / len(regular_preds)
                    self.print_log(f"Found {mismatches} mismatches between regular and TFLite models ({mismatch_percent:.2f}%)")
                else:
                    self.print_log("No mismatches between regular and TFLite models!")
                    
                # Save comparison results
                comparison_file = os.path.join(
                    self.arg.work_dir,
                    'results',
                    f'tflite_comparison_{subject_id}.json'
                )
                
                comparison_results = {
                    "subject": subject_id,
                    "regular_model": {
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "auc": float(auc_score),
                        "logits_sample": [float(x) for x in regular_logits[0].flatten()[:5]] if len(regular_logits) > 0 else []
                    },
                    "tflite_model": {
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "auc": float(auc_score),
                        "logits_sample": [float(x) for x in tflite_logits[0].flatten()[:5]] if len(tflite_logits) > 0 else []
                    },
                    "mismatches": mismatches,
                    "mismatch_percent": float(mismatch_percent) if mismatches > 0 else 0.0
                }
                
                with open(comparison_file, 'w') as f:
                    json.dump(comparison_results, f, indent=2)
                
                return True
                
            except Exception as e:
                self.print_log(f"Error running TFLite inference: {e}")
                self.print_log(traceback.format_exc())
                return False
                
        except Exception as e:
            self.print_log(f"Error comparing models: {e}")
            self.print_log(traceback.format_exc())
            return False
    
    def add_avg_df(self, results):
        """Add average row to results dataframe"""
        if not results:
            return results
        
        avg_result = {'test_subject': 'Average'}
        
        for column in results[0].keys():
            if column != 'test_subject':
                values = [float(r[column]) for r in results]
                avg_result[column] = round(sum(values) / len(values), 2)
        
        results.append(avg_result)
        return results
    
    def evaluate_test_set(self, epoch=0):
        """Evaluate model on test set"""
        try:
            # Save model training state
            model_training = self.model.trainable
            self.model.trainable = False
            
            # Get test data loader
            loader = self.data_loader.get('test')
            if loader is None:
                self.print_log("No test data loader available")
                return None
            
            # Get batch count
            total_batches = len(loader)
            num_samples = total_batches * self.arg.test_batch_size
            
            subject_id = self.test_subject[0] if self.test_subject else "unknown"
            self.print_log(f"Testing subject {subject_id} - {total_batches} batches (~{num_samples} samples)")
            
            # Test metrics tracking
            test_loss = 0.0
            all_labels = []
            all_preds = []
            all_logits = []
            steps = 0
            
            # Add timeout protection
            max_batch_time = 300  # 5 minutes max per batch
            start_batch_time = time.time()
            
            # Test loop
            for batch_idx in range(total_batches):
                # Check for timeout
                batch_duration = time.time() - start_batch_time
                if batch_duration > max_batch_time:
                    self.print_log(f"WARNING: Test batch {batch_idx} taking too long ({batch_duration:.1f}s), skipping")
                    start_batch_time = time.time()
                    continue
                    
                # Log progress periodically
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Test batch {batch_idx+1}/{total_batches}")
                
                try:
                    # Get batch data
                    inputs, targets, _ = loader[batch_idx]
                    
                    # Ensure targets are float32
                    targets = tf.cast(targets, tf.float32)
                    
                    # Forward pass
                    outputs = self.model(inputs, training=False)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]  # First element is logits
                    else:
                        logits = outputs  # Single output is logits
                        
                    # Store logits for later analysis
                    all_logits.append(logits.numpy())
                    
                    # Ensure proper shapes for loss calculation
                    if len(logits.shape) > 1 and logits.shape[-1] > 0:
                        # If logits has a feature dimension, match target shape
                        if len(targets.shape) == 1:
                            targets = tf.reshape(targets, [-1, 1])
                        loss = self.criterion(targets, logits)
                    else:
                        # Reshape both for compatibility
                        batch_size = tf.shape(inputs['accelerometer'])[0]
                        targets_reshaped = tf.reshape(targets, [batch_size, 1])
                        logits_reshaped = tf.reshape(logits, [batch_size, 1])
                        loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    # Get predictions
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    # Update statistics
                    test_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                    # Reset batch timer
                    start_batch_time = time.time()
                    
                except Exception as e:
                    self.print_log(f"Error in test batch {batch_idx}: {e}")
                    self.print_log(traceback.format_exc())
                    start_batch_time = time.time()
                    continue
            
            # Calculate metrics
            if steps > 0:
                test_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                # Store metrics
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
                
                # Log test results
                self.print_log(
                    f"Test results for Subject {subject_id}: "
                    f"Loss={test_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}%"
                )
                
                # Create confusion matrix
                self.cm_viz(all_preds, all_labels, subject_id)
                
                # Sample of model logits
                if len(all_logits) > 0:
                    self.print_log(f"Model logits sample: {all_logits[0].flatten()[:5]}")
                
                # Save results
                results = {
                    "subject": subject_id,
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "auc": float(auc_score),
                    "loss": float(test_loss),
                    "logits_sample": [float(x) for x in all_logits[0].flatten()[:5]] if len(all_logits) > 0 else []
                }
                
                results_file = os.path.join(
                    self.arg.work_dir,
                    'results',
                    f'test_results_{subject_id}.json'
                )
                
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Compare with TFLite model
                self.compare_regular_and_tflite_inference(subject_id)
                
                # Restore model training state
                self.model.trainable = model_training
                
                return results
            else:
                self.print_log(f"No valid test steps completed for subject {subject_id}")
                
                # Restore model training state
                self.model.trainable = model_training
                
                return None
                
        except Exception as e:
            self.print_log(f"Error in test evaluation: {e}")
            self.print_log(traceback.format_exc())
            
            # Restore model training state
            if 'model_training' in locals():
                self.model.trainable = model_training
            
            return None
    
    def start(self):
        """Main entry point for training or evaluation"""
        # Set timeout for the entire process
        max_total_time = 24 * 3600  # 24 hours max for full training
        total_start_time = time.time()
        
        try:
            if self.arg.phase == 'train':
                self.print_log('Starting training with parameters:')
                for key, value in vars(self.arg).items():
                    self.print_log(f'  {key}: {value}')
                
                results = []
                
                # Define validation subjects
                val_subjects = [38, 46]
                
                # Process each test subject (leave-one-out cross-validation)
                for test_subject in self.arg.subjects:
                    if test_subject in val_subjects:
                        continue
                    
                    try:
                        # Check if we've exceeded max total time
                        if time.time() - total_start_time > max_total_time:
                            self.print_log("Maximum total training time exceeded, stopping")
                            break
                        
                        # Reset metrics for this subject
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                        self.best_loss = float('inf')
                        
                        # Set up subject partitioning for cross-validation
                        self.test_subject = [test_subject]
                        self.val_subject = val_subjects
                        self.train_subjects = [s for s in self.arg.subjects 
                                            if s != test_subject and s not in val_subjects]
                        
                        self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                        self.print_log(f"Train: {len(self.train_subjects)} subjects: {self.train_subjects}")
                        self.print_log(f"Val: {len(self.val_subject)} subjects: {self.val_subject}")
                        self.print_log(f"Test: Subject {test_subject}")
                        
                        # Create a fresh model for each fold
                        try:
                            self.model = self.load_model()
                        except Exception as model_error:
                            self.print_log(f"Failed to load model for subject {test_subject}: {model_error}")
                            self.print_log(traceback.format_exc())
                            continue
                        
                        # Load data for this fold
                        try:
                            if not self.load_data():
                                self.print_log(f"Skipping subject {test_subject} due to data loading issues")
                                continue
                        except Exception as data_error:
                            self.print_log(f"Failed to load data for subject {test_subject}: {data_error}")
                            self.print_log(traceback.format_exc())
                            continue
                        
                        # Initialize optimizer and loss
                        try:
                            if not self.load_optimizer() or not self.load_loss():
                                self.print_log(f"Skipping subject {test_subject} due to optimizer/loss issues")
                                continue
                        except Exception as opt_error:
                            self.print_log(f"Failed to initialize optimizer/loss: {opt_error}")
                            self.print_log(traceback.format_exc())
                            continue
                        
                        # Reset early stopping
                        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
                        
                        # Set up per-subject timeout
                        subject_start_time = time.time()
                        max_subject_time = 8 * 3600  # 8 hours max per subject
                        
                        # Training loop
                        early_stop = False
                        self.print_log(f"Starting training for subject {test_subject}")
                        
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            # Check subject timeout
                            if time.time() - subject_start_time > max_subject_time:
                                self.print_log(f"Maximum time per subject exceeded for subject {test_subject}")
                                break
                            
                            try:
                                # Train for one epoch
                                early_stop = self.train(epoch)
                                
                                # Check for early stopping
                                if early_stop:
                                    self.print_log(f"Early stopping at epoch {epoch+1}")
                                    break
                            except Exception as epoch_error:
                                self.print_log(f"Error in epoch {epoch+1}: {epoch_error}")
                                self.print_log(traceback.format_exc())
                                
                                # If first epoch fails, skip this subject
                                if epoch == 0:
                                    self.print_log(f"First epoch failed, skipping subject {test_subject}")
                                    break
                                continue  # Otherwise try next epoch
                        
                        # Load best weights for evaluation
                        best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                        if os.path.exists(best_weights):
                            try:
                                self.model.load_weights(best_weights)
                                self.print_log(f"Loaded best weights from {best_weights}")
                            except Exception as weight_error:
                                self.print_log(f"Error loading best weights: {weight_error}")
                                self.print_log(traceback.format_exc())
                        
                        # Final evaluation
                        self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                        result = self.evaluate_test_set()
                        
                        # Compare regular model with TFLite model
                        self.print_log(f"=== Comparing regular model and TFLite model for subject {test_subject} ===")
                        self.compare_regular_and_tflite_inference(test_subject)
                        
                        # Visualize loss curves
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                        
                        # Add results to summary
                        if result:
                            subject_result = {
                                'test_subject': str(test_subject),
                                'accuracy': round(self.test_accuracy, 2),
                                'f1_score': round(self.test_f1, 2),
                                'precision': round(self.test_precision, 2),
                                'recall': round(self.test_recall, 2),
                                'auc': round(self.test_auc, 2)
                            }
                            
                            results.append(subject_result)
                        
                        # Clear TensorFlow session to avoid memory issues
                        tf.keras.backend.clear_session()
                        
                    except Exception as subject_error:
                        self.print_log(f"Fatal error processing subject {test_subject}: {subject_error}")
                        self.print_log(traceback.format_exc())
                        tf.keras.backend.clear_session()
                        continue
                
                # Generate final report
                if results:
                    try:
                        # Add average results
                        results = self.add_avg_df(results)
                        
                        # Save results as CSV
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)
                        
                        # Save results as JSON
                        with open(os.path.join(self.arg.work_dir, 'scores.json'), 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        # Log final results summary
                        self.print_log("\n=== Final Results ===")
                        for result in results:
                            subject = result['test_subject']
                            accuracy = result['accuracy']
                            f1 = result['f1_score']
                            precision = result['precision']
                            recall = result['recall']
                            auc = result.get('auc', 'N/A')
                            
                            self.print_log(
                                f"Subject {subject}: "
                                f"Acc={accuracy:.2f}%, "
                                f"F1={f1:.2f}%, "
                                f"Prec={precision:.2f}%, "
                                f"Rec={recall:.2f}%, "
                                f"AUC={auc}"
                            )
                    except Exception as report_error:
                        self.print_log(f"Error generating final report: {report_error}")
                        self.print_log(traceback.format_exc())
                
                self.print_log("Training completed successfully")
                
            elif self.arg.phase == 'test':
                # Test phase - evaluate loaded model
                self.print_log('Testing with parameters:')
                for key, value in vars(self.arg).items():
                    self.print_log(f'  {key}: {value}')
                
                if hasattr(self.arg, 'weights') and self.arg.weights:
                    self.print_log(f"Using weights: {self.arg.weights}")
                
                self.test_subject = self.arg.subjects
                
                if not self.load_data():
                    self.print_log("Failed to load test data")
                    return
                
                self.load_loss()
                result = self.evaluate_test_set()
                
                if result:
                    self.print_log("Testing completed successfully")
                else:
                    self.print_log("Testing failed")
                
            elif self.arg.phase == 'tflite':
                # TFLite export phase
                self.print_log('Exporting TFLite model with parameters:')
                for key, value in vars(self.arg).items():
                    self.print_log(f'  {key}: {value}')
                
                if not hasattr(self.arg, 'weights') or not self.arg.weights:
                    self.print_log("Must specify weights for TFLite export")
                    return
                
                # Try model's built-in export method first
                if hasattr(self.model, 'export_to_tflite'):
                    self.print_log("Using model's built-in TFLite export method")
                    tflite_path = os.path.join(self.arg.work_dir, f"{self.arg.model_saved_name}.tflite")
                    success = self.model.export_to_tflite(tflite_path)
                else:
                    # Fall back to utility function
                    try:
                        from utils.tflite_converter import convert_to_tflite
                        
                        acc_frames = self.arg.model_args.get('acc_frames', 128)
                        
                        tflite_path = os.path.join(self.arg.work_dir, f"{self.arg.model_saved_name}.tflite")
                        
                        self.print_log(f"Exporting model with input shape: (1, {acc_frames}, 3)")
                        
                        success = convert_to_tflite(
                            model=self.model,
                            save_path=tflite_path,
                            input_shape=(1, acc_frames, 3),
                            quantize=getattr(self.arg, 'quantize', False)
                        )
                        
                    except Exception as e:
                        self.print_log(f"Error exporting TFLite model: {e}")
                        self.print_log(traceback.format_exc())
                        success = False
                
                if success:
                    self.print_log(f"TFLite model exported successfully to {tflite_path}")
                else:
                    self.print_log("TFLite export failed")
            
        except Exception as e:
            self.print_log(f"Fatal error in training process: {e}")
            self.print_log(traceback.format_exc())
