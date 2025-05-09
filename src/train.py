#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - Main training script for LightHART-TF
Fully compatible with PyTorch implementation with robust error handling
"""

import os
import logging
import argparse
import yaml
import json
import sys
import time
import traceback
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger('lightheart-tf')

class EarlyStopping:
    """Early stopping implementation to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = None
        self.history = []
    
    def __call__(self, val_loss, epoch=None):
        if isinstance(val_loss, tf.Tensor):
            val_loss = float(val_loss.numpy())
        else:
            val_loss = float(val_loss)
            
        self.history.append(val_loss)
        
        if self.best_loss is None:
            self.best_loss = val_loss
            if epoch is not None:
                self.best_epoch = epoch
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False
    
    def reset(self):
        """Reset early stopping state for new training run"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = None
        self.history = []

def import_class(import_str):
    """
    Import a class or module dynamically with robust fallbacks
    
    Args:
        import_str: Import path as string (e.g., 'models.transformer_optimized.TransModel')
        
    Returns:
        Imported class or raises ImportError
    """
    if import_str is None:
        raise ValueError("Import path cannot be None")
    
    logger.info(f"Attempting to import: {import_str}")
    
    # Handle relative vs absolute imports
    mod_str, _sep, class_str = import_str.rpartition('.')
    
    # Track import attempts for debugging
    attempted_paths = []
    
    # Try different import prefixes to handle various project structures
    for prefix in ['', 'src.', 'models.', 'utils.']:
        try:
            import_path = f"{prefix}{mod_str}"
            attempted_paths.append(import_path)
            
            import importlib
            logger.info(f"Trying to import from: {import_path}")
            module = importlib.import_module(import_path)
            
            if hasattr(module, class_str):
                logger.info(f"Successfully imported {class_str} from {import_path}")
                return getattr(module, class_str)
        except ImportError as e:
            logger.warning(f"Import failed from {prefix}{mod_str}: {e}")
            continue
    
    # Try filesystem-based import as last resort
    try:
        # Map import path to potential file locations
        possible_file_paths = []
        
        # Handle models
        if 'models.' in import_str:
            model_name = import_str.split('.')[-2]
            possible_file_paths.extend([
                os.path.join(os.getcwd(), 'models', f"{model_name}.py"),
                os.path.join(os.getcwd(), 'src', 'models', f"{model_name}.py")
            ])
        
        # Handle utils
        elif 'utils.' in import_str:
            util_name = import_str.split('.')[-2]
            possible_file_paths.extend([
                os.path.join(os.getcwd(), 'utils', f"{util_name}.py"),
                os.path.join(os.getcwd(), 'src', 'utils', f"{util_name}.py")
            ])
        
        # Try importing from files directly
        for file_path in possible_file_paths:
            if os.path.exists(file_path):
                logger.info(f"Found module file at: {file_path}")
                
                # Add parent directory to path
                dir_path = os.path.dirname(file_path)
                if dir_path not in sys.path:
                    sys.path.insert(0, dir_path)
                
                # Try direct file import
                try:
                    file_name = os.path.basename(file_path)[:-3]  # Remove .py
                    module = __import__(file_name)
                    if hasattr(module, class_str):
                        logger.info(f"Successfully imported {class_str} from file: {file_path}")
                        return getattr(module, class_str)
                except Exception as e:
                    logger.warning(f"Error importing from file {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error in file-based import attempt: {e}")
    
    # All import attempts failed
    error_msg = f"Failed to import {class_str}. Attempted: {attempted_paths}"
    logger.error(error_msg)
    raise ImportError(error_msg)

class Trainer:
    """Main training class for LightHART models"""
    def __init__(self, arg):
        """
        Initialize trainer with configuration
        
        Args:
            arg: Arguments from command line and config file
        """
        self.arg = arg
        self.setup_environment()
        self.setup_directories()
        self.setup_metrics()
        
        # Print training configuration
        logger.info(f"Training configuration: {self.arg}")
        
        # Initialize model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
            num_params = self.count_parameters(self.model)
            logger.info(f"Model: {arg.model}")
            logger.info(f"Parameters: {num_params:,}")
            logger.info(f"Model size: {num_params * 4 / (1024**2):.2f} MB")
        else:
            self.model = self.load_trained_model()
    
    def setup_environment(self):
        """Configure TensorFlow environment and GPU settings"""
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set visible devices
                if hasattr(self.arg, 'device'):
                    if isinstance(self.arg.device, list):
                        gpu_id = self.arg.device[0]
                    else:
                        gpu_id = self.arg.device
                    
                    # Configure CUDA visible devices
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    self.output_device = gpu_id
                    logger.info(f"Using GPU: {gpu_id}")
                else:
                    self.output_device = 0
                    logger.info("Using default GPU: 0")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPU found. Using CPU.")
            self.output_device = -1
        
        # Set random seeds for reproducibility
        if hasattr(self.arg, 'seed'):
            seed = self.arg.seed
            np.random.seed(seed)
            tf.random.set_seed(seed)
            logger.info(f"Random seed set to {seed}")
    
    def setup_directories(self):
        """Create working directories for outputs"""
        self.arg.work_dir = os.path.abspath(self.arg.work_dir)
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'results'), exist_ok=True)
        
        # Set model path with timestamp to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_path = os.path.join(
            self.arg.work_dir, 
            'models', 
            f"{self.arg.model_saved_name}_{timestamp}"
        )
        
        # Save config for reproducibility
        if hasattr(self.arg, 'config') and self.arg.config:
            config_name = os.path.basename(self.arg.config)
            shutil.copy(self.arg.config, os.path.join(self.arg.work_dir, config_name))
    
    def setup_metrics(self):
        """Initialize metrics tracking"""
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_recall = 0 
        self.test_precision = 0
        self.test_auc = None
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.early_stop = EarlyStopping(patience=15, min_delta=0.001)
        
        # For holding dataset
        self.data_loader = {}
        
        # For tensor transformations
        self.inertial_modality = None
        if hasattr(self.arg, 'dataset_args') and 'modalities' in self.arg.dataset_args:
            self.inertial_modality = [m for m in self.arg.dataset_args['modalities'] 
                                     if m != 'skeleton']
        
        # Configure multi-modality
        self.fuse = False
        if self.inertial_modality and len(self.inertial_modality) > 1:
            self.fuse = True
            logger.info(f"Multi-modal fusion enabled: {self.inertial_modality}")
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        try:
            total_params = 0
            for var in model.trainable_variables:
                total_params += tf.size(var).numpy()
            return total_params
        except Exception as e:
            logger.error(f"Error counting parameters: {e}")
            return 0
    
    def load_model(self, model_name, model_args):
        """
        Load and initialize model
        
        Args:
            model_name: Model class path (e.g., 'models.transformer_optimized.TransModel')
            model_args: Dictionary of model constructor arguments
            
        Returns:
            Initialized model
        """
        try:
            # Import model class
            logger.info(f"Importing model: {model_name}")
            Model = import_class(model_name)
            
            # Initialize model with args
            logger.info(f"Initializing model with args: {model_args}")
            model = Model(**model_args)
            
            # Initialize model by running a dummy forward pass
            try:
                acc_frames = model_args.get('acc_frames', 128)
                acc_coords = model_args.get('acc_coords', 3)
                dummy_acc = tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                
                # Check if model needs skeleton data
                if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                    num_joints = model_args.get('num_joints', 32)
                    dummy_skl = tf.zeros((2, acc_frames, num_joints, 3), dtype=tf.float32)
                    _ = model({'accelerometer': dummy_acc, 'skeleton': dummy_skl}, training=False)
                else:
                    _ = model({'accelerometer': dummy_acc}, training=False)
                logger.info("Model initialized with dummy input")
            except Exception as e:
                logger.warning(f"Error during model initialization with dummy data: {e}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_trained_model(self):
        """Load a trained model from weights file"""
        try:
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                raise ValueError("No weights specified for testing phase")
            
            weights_path = self.arg.weights
            logger.info(f"Loading model weights from: {weights_path}")
            
            # Try loading as full Keras model first
            try:
                model = tf.keras.models.load_model(weights_path)
                logger.info(f"Loaded complete model from {weights_path}")
                return model
            except Exception as e:
                logger.warning(f"Could not load as complete model: {e}")
            
            # Try loading as weights for model architecture
            if hasattr(self.arg, 'model') and hasattr(self.arg, 'model_args'):
                model = self.load_model(self.arg.model, self.arg.model_args)
                
                # Load weights
                try:
                    model.load_weights(weights_path)
                    logger.info(f"Loaded weights into model from {weights_path}")
                    return model
                except Exception as e:
                    logger.error(f"Error loading weights: {e}")
            
            raise ValueError(f"Failed to load model from {weights_path}")
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            raise
    
    def cal_weights(self):
        """Calculate class weights for imbalanced datasets"""
        try:
            if not hasattr(self, 'norm_train') or 'labels' not in self.norm_train:
                logger.warning("No training data for weight calculation")
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
                return
                
            # Count class instances
            from collections import Counter
            label_count = Counter(self.norm_train['labels'])
            
            # Calculate weight for positive class
            if 0 in label_count and 1 in label_count:
                weight_value = label_count[0] / label_count[1]
                logger.info(f"Class balance - Negative: {label_count[0]}, Positive: {label_count[1]}")
                logger.info(f"Positive class weight: {weight_value:.4f}")
            else:
                weight_value = 1.0
                logger.warning(f"Missing classes in labels. Using weight=1.0. Counts: {label_count}")
            
            self.pos_weights = tf.constant(weight_value, dtype=tf.float32)
        except Exception as e:
            logger.error(f"Error calculating class weights: {e}")
            self.pos_weights = tf.constant(1.0, dtype=tf.float32)
    
    def load_optimizer(self):
        """Configure optimizer based on arguments"""
        try:
            if not hasattr(self.arg, 'optimizer'):
                self.arg.optimizer = 'adam'
            if not hasattr(self.arg, 'base_lr'):
                self.arg.base_lr = 0.001
            if not hasattr(self.arg, 'weight_decay'):
                self.arg.weight_decay = 0.0004
            
            # Configure different optimizers
            if self.arg.optimizer.lower() == "adam":
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
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
                logger.warning(f"Unknown optimizer: {self.arg.optimizer}, using Adam")
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
            
            logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, LR: {self.arg.base_lr}")
            return True
        except Exception as e:
            logger.error(f"Error configuring optimizer: {e}")
            return False
    
    def load_loss(self):
        """Configure loss function"""
        try:
            # Set up pos_weights if not already done
            if not hasattr(self, 'pos_weights') or self.pos_weights is None:
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
            
            # Define weighted BCE loss
            def weighted_bce(y_true, y_pred):
                # Handle shape differences
                if len(tf.shape(y_pred)) > 1 and tf.shape(y_pred)[-1] == 1:
                    if len(tf.shape(y_true)) == 1:
                        y_true = tf.expand_dims(y_true, -1)
                elif len(tf.shape(y_pred)) == 1 and len(tf.shape(y_true)) > 1:
                    y_pred = tf.expand_dims(y_pred, -1)
                
                y_true = tf.cast(y_true, tf.float32)
                
                # Use sigmoid cross entropy with weights
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
                weights = y_true * (self.pos_weights - 1.0) + 1.0
                weighted_bce = weights * bce
                return tf.reduce_mean(weighted_bce)
            
            self.criterion = weighted_bce
            logger.info(f"Using weighted BCE loss with pos_weight={float(self.pos_weights):.4f}")
            return True
        except Exception as e:
            logger.error(f"Error configuring loss function: {e}")
            return False
    
    def load_data(self):
        """Load and prepare datasets"""
        try:
            # Import dataset utilities
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            
            # Import data feeder
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = import_class(feeder_class_path)
            
            # Extra params
            use_smv = getattr(self.arg, 'use_smv', False)
            
            if self.arg.phase == 'train':
                # Prepare dataset using SmartFallMM
                builder = prepare_smartfallmm_tf(self.arg)
                
                # Check training subjects
                if not self.train_subjects:
                    logger.error("No training subjects specified")
                    return False
                
                # Process training data
                logger.info(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, self.fuse)
                
                # Check for empty data
                if any(len(x) == 0 for x in self.norm_train.values()):
                    logger.error("Training data is empty")
                    return False
                
                # Create data loader
                try:
                    self.data_loader['train'] = Feeder(
                        dataset=self.norm_train,
                        batch_size=self.arg.batch_size,
                        use_smv=use_smv
                    )
                except TypeError as e:
                    # Handle case where feeder doesn't accept use_smv
                    if "unexpected keyword argument 'use_smv'" in str(e):
                        self.data_loader['train'] = Feeder(
                            dataset=self.norm_train,
                            batch_size=self.arg.batch_size
                        )
                    else:
                        raise
                
                # Calculate class weights
                self.cal_weights()
                
                # Load validation data
                if self.val_subject:
                    logger.info(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = split_by_subjects_tf(builder, self.val_subject, self.fuse)
                    
                    # Handle empty validation data
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        logger.warning("Validation data is empty, using subset of training data")
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)
                        
                        self.norm_val = {k: v[-val_size:].copy() for k, v in self.norm_train.items()}
                        self.norm_train = {k: v[:-val_size].copy() for k, v in self.norm_train.items()}
                    
                    # Create validation loader
                    try:
                        self.data_loader['val'] = Feeder(
                            dataset=self.norm_val,
                            batch_size=self.arg.val_batch_size,
                            use_smv=use_smv
                        )
                    except TypeError:
                        self.data_loader['val'] = Feeder(
                            dataset=self.norm_val,
                            batch_size=self.arg.val_batch_size
                        )
                
                # Load test data
                if self.test_subject:
                    logger.info(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.fuse)
                    
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        logger.error("Test data is empty")
                        return False
                    
                    # Create test loader
                    try:
                        self.data_loader['test'] = Feeder(
                            dataset=self.norm_test,
                            batch_size=self.arg.test_batch_size,
                            use_smv=use_smv
                        )
                    except TypeError:
                        self.data_loader['test'] = Feeder(
                            dataset=self.norm_test,
                            batch_size=self.arg.test_batch_size
                        )
                
                return True
            
            elif self.arg.phase == 'test':
                # For test phase, only load test data
                if not self.test_subject:
                    logger.error("No test subjects specified")
                    return False
                
                builder = prepare_smartfallmm_tf(self.arg)
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.fuse)
                
                if any(len(x) == 0 for x in self.norm_test.values()):
                    logger.error("Test data is empty")
                    return False
                
                # Create test loader
                try:
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size,
                        use_smv=use_smv
                    )
                except TypeError:
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size
                    )
                
                return True
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            traceback.print_exc()
            return False
    
    def print_log(self, message):
        """Print to console and log file"""
        print(message)
        
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            log_path = os.path.join(self.arg.work_dir, 'log.txt')
            with open(log_path, 'a') as f:
                print(message, file=f)
    
    def distribution_viz(self, labels, work_dir, mode):
        """Visualize data distribution by class"""
        try:
            # Create visualization directory
            viz_dir = os.path.join(work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Count class distribution
            values, counts = np.unique(labels, return_counts=True)
            
            # Create bar chart
            plt.figure(figsize=(8, 6))
            plt.bar(values, counts)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Class Distribution')
            plt.xticks(values)
            
            # Add count labels
            for i, v in enumerate(counts):
                plt.text(values[i], v + 5, str(v), ha='center')
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{mode}_distribution.png'))
            plt.close()
            
            # Log distribution
            dist_str = ", ".join([f"Class {int(v)}: {c}" for v, c in zip(values, counts)])
            logger.info(f"{mode} distribution: {dist_str}")
        except Exception as e:
            logger.error(f"Error visualizing distribution: {e}")
    
    def loss_viz(self, train_loss, val_loss, subject_id=None):
        """Visualize training and validation loss curves"""
        try:
            # Check for data
            if not train_loss or not val_loss:
                logger.warning("No loss data for visualization")
                return
            
            # Create epochs range
            epochs = range(1, len(train_loss) + 1)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
            
            # Add title and labels
            title = 'Training vs Validation Loss'
            if subject_id:
                title += f' (Subject {subject_id})'
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Save figure
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            filename = f'loss_curves_{subject_id}' if subject_id else 'loss_curves'
            plt.savefig(os.path.join(viz_dir, f'{filename}.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing loss: {e}")
    
    def cm_viz(self, y_pred, y_true, subject_id=None):
        """Visualize confusion matrix"""
        try:
            # Ensure numpy arrays
            if isinstance(y_pred, tf.Tensor):
                y_pred = y_pred.numpy()
            if isinstance(y_true, tf.Tensor):
                y_true = y_true.numpy()
            
            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()
            
            # Ensure same length
            if len(y_pred) != len(y_true):
                logger.warning(f"Length mismatch: y_pred={len(y_pred)}, y_true={len(y_true)}")
                min_len = min(len(y_pred), len(y_true))
                y_pred = y_pred[:min_len]
                y_true = y_true[:min_len]
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix' + (f' (Subject {subject_id})' if subject_id else ''))
            plt.colorbar()
            
            # Set labels
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Save figure
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            filename = f'confusion_matrix_{subject_id}' if subject_id else 'confusion_matrix'
            plt.savefig(os.path.join(viz_dir, f'{filename}.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing confusion matrix: {e}")
    
    def cal_prediction(self, logits):
        """Calculate binary predictions from logits"""
        if len(tf.shape(logits)) > 1 and tf.shape(logits)[-1] > 1:
            # Multi-class case
            return tf.argmax(logits, axis=-1)
        else:
            # Binary case
            return tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    
    def cal_metrics(self, targets, predictions, probabilities=None):
        """Calculate performance metrics"""
        try:
            # Convert to numpy arrays
            if isinstance(targets, tf.Tensor):
                targets = targets.numpy()
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            if probabilities is not None and isinstance(probabilities, tf.Tensor):
                probabilities = probabilities.numpy()
            
            targets = np.array(targets).flatten()
            predictions = np.array(predictions).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(targets, predictions) * 100
            
            # Handle potential division by zero
            try:
                precision = precision_score(targets, predictions, zero_division=0) * 100
            except:
                precision = 0
                
            try:
                recall = recall_score(targets, predictions, zero_division=0) * 100
            except:
                recall = 0
                
            try:
                f1 = f1_score(targets, predictions, zero_division=0) * 100
            except:
                f1 = 0
            
            # Calculate AUC if probabilities provided
            auc = None
            if probabilities is not None:
                probabilities = np.array(probabilities).flatten()
                try:
                    # AUC needs at least one instance of each class
                    unique_classes = np.unique(targets)
                    if len(unique_classes) > 1:
                        auc = roc_auc_score(targets, probabilities) * 100
                except Exception as e:
                    logger.warning(f"AUC calculation error: {e}")
            
            return accuracy, f1, recall, precision, auc
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0, None
    
    def save_model(self, epoch, subject_id=None):
        """Save model weights and full model"""
        try:
            if subject_id:
                base_filename = f"{self.model_path}_{subject_id}"
            else:
                base_filename = f"{self.model_path}_epoch{epoch}"
            
            # Save weights in multiple formats for compatibility
            weights_path = f"{base_filename}.weights.h5"
            self.model.save_weights(weights_path)
            logger.info(f"Saved model weights to {weights_path}")
            
            # Try saving full model (may fail with custom layers)
            try:
                model_path = f"{base_filename}.keras"
                self.model.save(model_path)
                logger.info(f"Saved full model to {model_path}")
            except Exception as e:
                logger.warning(f"Could not save full model: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_weights(self):
        """Load saved weights for the current test subject"""
        try:
            if not self.test_subject:
                logger.warning("No test subject specified for loading weights")
                return False
            
            subject_id = self.test_subject[0]
            weights_path = f"{self.model_path}_{subject_id}.weights.h5"
            
            if not os.path.exists(weights_path):
                logger.error(f"Weights file not found: {weights_path}")
                return False
            
            self.model.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return False
    
    def train(self, epoch):
        """Train for one epoch"""
        try:
            start_time = time.time()
            
            # Set model to training mode
            self.model.trainable = True
            
            # Get data loader
            loader = self.data_loader['train']
            total_batches = len(loader)
            
            # Initialize metrics
            train_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            steps = 0
            
            # Training loop
            for batch_idx in range(total_batches):
                try:
                    # Log progress periodically
                    if batch_idx % 10 == 0 or batch_idx + 1 == total_batches:
                        logger.info(f"Epoch {epoch+1}: batch {batch_idx+1}/{total_batches}")
                    
                    # Get batch data
                    inputs, targets, _ = loader[batch_idx]
                    targets = tf.cast(targets, tf.float32)
                    
                    # Forward and backward pass
                    with tf.GradientTape() as tape:
                        outputs = self.model(inputs, training=True)
                        
                        # Handle different output formats
                        if isinstance(outputs, tuple) and len(outputs) > 0:
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        # Calculate loss
                        if len(tf.shape(logits)) > 1 and tf.shape(logits)[-1] > 1:
                            # Multi-class case
                            if len(tf.shape(targets)) == 1:
                                targets = tf.reshape(targets, [-1, 1])
                            loss = self.criterion(targets, logits)
                        else:
                            # Binary case - ensure shapes match
                            batch_size = tf.shape(inputs['accelerometer'])[0]
                            targets_reshaped = tf.reshape(targets, [batch_size, 1])
                            logits_reshaped = tf.reshape(logits, [batch_size, 1])
                            loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    # Calculate gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for grad in gradients:
                        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                            has_nan = True
                            break
                    
                    if has_nan:
                        logger.warning(f"NaN gradients in batch {batch_idx}, skipping update")
                        continue
                    
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Calculate predictions
                    if len(tf.shape(logits)) > 1 and tf.shape(logits)[-1] > 1:
                        # Multi-class
                        predictions = tf.argmax(logits, axis=-1)
                        probabilities = tf.nn.softmax(logits)[:, 1]
                    else:
                        # Binary
                        logits_squeezed = tf.squeeze(logits)
                        probabilities = tf.sigmoid(logits_squeezed)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Accumulate batch metrics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            # Calculate epoch metrics
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.cal_metrics(
                    all_labels, all_preds, all_probs
                )
                
                # Store metrics
                self.train_loss_summary.append(float(train_loss))
                
                # Log results
                epoch_time = time.time() - start_time
                auc_str = f"{auc_score:.2f}%" if auc_score is not None else "N/A"
                
                logger.info(
                    f"Epoch {epoch+1} results: "
                    f"Loss={train_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_str} "
                    f"({epoch_time:.2f}s)"
                )
                
                # Run validation
                val_loss = self.eval(epoch, loader_name='val')
                self.val_loss_summary.append(float(val_loss))
                
                # Check for early stopping
                should_stop = self.early_stop(val_loss, epoch)
                
                return should_stop
            else:
                logger.warning(f"No valid steps completed in epoch {epoch+1}")
                return False
        
        except Exception as e:
            logger.error(f"Critical error in epoch {epoch+1}: {e}")
            traceback.print_exc()
            return False
    
    def eval(self, epoch, loader_name='val', result_file=None):
        """Evaluate model on validation or test data"""
        try:
            start_time = time.time()
            
            # Set model to evaluation mode
            self.model.trainable = False
            
            # Get data loader
            loader = self.data_loader.get(loader_name)
            if loader is None:
                logger.error(f"No data loader for {loader_name}")
                return float('inf')
            
            total_batches = len(loader)
            
            # Log start of evaluation
            logger.info(f"Evaluating {loader_name} data (epoch {epoch+1}): {total_batches} batches")
            
            # Initialize metrics
            eval_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            steps = 0
            
            # Evaluation loop
            for batch_idx in range(total_batches):
                try:
                    # Log progress
                    if batch_idx % 10 == 0 or batch_idx + 1 == total_batches:
                        logger.info(f"Eval batch {batch_idx+1}/{total_batches}")
                    
                    # Get batch data
                    inputs, targets, _ = loader[batch_idx]
                    targets = tf.cast(targets, tf.float32)
                    
                    # Forward pass
                    outputs = self.model(inputs, training=False)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Calculate loss
                    if len(tf.shape(logits)) > 1 and tf.shape(logits)[-1] > 1:
                        # Multi-class case
                        if len(tf.shape(targets)) == 1:
                            targets = tf.reshape(targets, [-1, 1])
                        loss = self.criterion(targets, logits)
                    else:
                        # Binary case
                        batch_size = tf.shape(inputs['accelerometer'])[0]
                        targets_reshaped = tf.reshape(targets, [batch_size, 1])
                        logits_reshaped = tf.reshape(logits, [batch_size, 1])
                        loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    # Calculate predictions
                    if len(tf.shape(logits)) > 1 and tf.shape(logits)[-1] > 1:
                        predictions = tf.argmax(logits, axis=-1)
                        probabilities = tf.nn.softmax(logits)[:, 1]
                    else:
                        logits_squeezed = tf.squeeze(logits)
                        probabilities = tf.sigmoid(logits_squeezed)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Accumulate batch metrics
                    eval_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch metrics
            if steps > 0:
                eval_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.cal_metrics(
                    all_labels, all_preds, all_probs
                )
                
                # Log results
                auc_str = f"{auc_score:.2f}%" if auc_score is not None else "N/A"
                logger.info(
                    f"{loader_name.capitalize()}: "
                    f"Loss={eval_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_str}"
                )
                
                # Save best model for validation
                if loader_name == 'val':
                    is_best = False
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        is_best = True
                        logger.info(f"New best validation loss: {eval_loss:.4f}")
                    
                    if is_best:
                        subject_id = self.test_subject[0] if self.test_subject else None
                        self.save_model(epoch, subject_id)
                
                # Store test metrics
                elif loader_name.startswith('test'):
                    self.test_accuracy = accuracy
                    self.test_f1 = f1
                    self.test_recall = recall
                    self.test_precision = precision
                    self.test_auc = auc_score
                    
                    # Visualize confusion matrix
                    subject_id = self.test_subject[0] if self.test_subject else None
                    if subject_id:
                        self.cm_viz(all_preds, all_labels, subject_id)
                    
                    # Save results to file
                    results = {
                        "subject": str(subject_id) if subject_id else "unknown",
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "auc": float(auc_score) if auc_score is not None else None,
                        "loss": float(eval_loss)
                    }
                    
                    results_file = os.path.join(
                        self.arg.work_dir,
                        'results',
                        f'test_results_{subject_id if subject_id else "unknown"}.json'
                    )
                    
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                
                return eval_loss
            else:
                logger.warning(f"No valid steps for {loader_name}")
                return float('inf')
        
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            traceback.print_exc()
            return float('inf')
    
    def add_avg_df(self, results):
        """Add average row to results dataframe"""
        if not results:
            return results
        
        # Metrics to average
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        metric_values = {metric: [] for metric in metrics}
        
        # Collect values
        for result in results:
            for metric in metrics:
                if metric in result and result[metric] is not None:
                    metric_values[metric].append(result[metric])
        
        # Calculate averages
        avg_result = {'test_subject': 'Average'}
        for metric in metrics:
            if metric_values[metric]:
                avg_result[metric] = round(sum(metric_values[metric]) / len(metric_values[metric]), 2)
            else:
                avg_result[metric] = None
        
        # Add to results
        results.append(avg_result)
        return results
    
    def start(self):
        """Main entry point for training or testing"""
        max_total_time = 24 * 3600  # 24 hours max runtime
        total_start_time = time.time()
        
        try:
            if self.arg.phase == 'train':
                logger.info('Starting training with parameters:')
                for key, value in vars(self.arg).items():
                    logger.info(f'  {key}: {value}')
                
                results = []
                
                # Determine test subjects
                test_subjects = self.arg.subjects
                
                # Define fixed validation subjects
                val_subjects = getattr(self.arg, 'val_subjects_fixed', [38, 46])
                
                # Fixed training subjects (no fall data)
                train_subjects_fixed = getattr(self.arg, 'train_subjects_fixed', [45, 36, 29])
                
                # Subjects eligible for testing
                test_eligible = getattr(self.arg, 'test_eligible_subjects', 
                                     [32, 39, 30, 31, 33, 34, 35, 37, 43, 44])
                
                # If no specific test subjects, use all eligible
                if not test_subjects:
                    test_subjects = test_eligible
                
                # Run cross-validation for each test subject
                for test_subject in test_subjects:
                    try:
                        # Check time limit
                        if time.time() - total_start_time > max_total_time:
                            logger.info("Maximum time limit exceeded, stopping")
                            break
                        
                        # Reset for new test subject
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                        self.best_loss = float('inf')
                        self.data_loader = {}
                        
                        # Set up subject splits
                        self.test_subject = [test_subject]
                        self.val_subject = val_subjects
                        
                        # Training subjects: all eligible except current test + fixed training subjects
                        self.train_subjects = [s for s in test_eligible if s != test_subject]
                        self.train_subjects.extend(train_subjects_fixed)
                        
                        logger.info(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                        logger.info(f"Train: {self.train_subjects}")
                        logger.info(f"Val: {self.val_subject}")
                        logger.info(f"Test: {self.test_subject}")
                        
                        # Reset TensorFlow state
                        tf.keras.backend.clear_session()
                        
                        # Create new model for this fold
                        self.model = self.load_model(self.arg.model, self.arg.model_args)
                        
                        # Load data for this split
                        if not self.load_data():
                            logger.error(f"Failed to load data for subject {test_subject}")
                            continue
                        
                        # Set up optimizer and loss
                        if not self.load_optimizer() or not self.load_loss():
                            logger.error(f"Failed to set up optimizer/loss for subject {test_subject}")
                            continue
                        
                        # Reset early stopping
                        self.early_stop.reset()
                        
                        # Training loop
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            try:
                                should_stop = self.train(epoch)
                                if should_stop:
                                    logger.info(f"Early stopping at epoch {epoch+1}")
                                    break
                            except Exception as e:
                                logger.error(f"Error in epoch {epoch+1}: {e}")
                                if epoch == 0:
                                    # If first epoch fails, skip this subject
                                    logger.error(f"First epoch failed, skipping subject {test_subject}")
                                    break
                                continue
                        
                        # Load best model for final evaluation
                        self.load_weights()
                        
                        # Final test evaluation
                        logger.info(f"=== Final evaluation for subject {test_subject} ===")
                        self.eval(epoch=0, loader_name='test')
                        
                        # Visualize training curves
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary, test_subject)
                        
                        # Store results
                        subject_result = {
                            'test_subject': str(test_subject),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2),
                            'auc': round(self.test_auc, 2) if self.test_auc is not None else None
                        }
                        results.append(subject_result)
                        
                        # Clean up for next fold
                        self.data_loader = {}
                        tf.keras.backend.clear_session()
                        
                    except Exception as e:
                        logger.error(f"Error processing subject {test_subject}: {e}")
                        traceback.print_exc()
                        continue
                
                # Generate final report
                if results:
                    # Add average row
                    results = self.add_avg_df(results)
                    
                    # Save as CSV
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)
                    
                    # Save as JSON
                    with open(os.path.join(self.arg.work_dir, 'scores.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Log final results
                    logger.info("\n=== Final Results ===")
                    for result in results:
                        subject = result['test_subject']
                        accuracy = result.get('accuracy', 'N/A')
                        f1 = result.get('f1_score', 'N/A')
                        precision = result.get('precision', 'N/A')
                        recall = result.get('recall', 'N/A')
                        auc = result.get('auc', 'N/A')
                        
                        logger.info(
                            f"Subject {subject}: "
                            f"Acc={accuracy}%, "
                            f"F1={f1}%, "
                            f"Prec={precision}%, "
                            f"Rec={recall}%, "
                            f"AUC={auc}%"
                        )
                
                logger.info("Training completed successfully")
                
            elif self.arg.phase == 'test':
                logger.info("Starting testing with parameters:")
                for key, value in vars(self.arg).items():
                    logger.info(f"  {key}: {value}")
                
                # Set test subject
                self.test_subject = self.arg.subjects
                
                # Load test data
                if not self.load_data():
                    logger.error("Failed to load test data")
                    return
                
                # Configure loss function
                self.load_loss()
                
                # Run evaluation
                logger.info(f"Running evaluation for subjects: {self.test_subject}")
                self.eval(epoch=0, loader_name='test')
                
                logger.info("Testing completed successfully")
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            traceback.print_exc()

def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Configure command line argument parser"""
    parser = argparse.ArgumentParser(description='LightHART-TF Fall Detection')
    
    # Base configuration
    parser.add_argument('--config', default='config/smartfallmm/student.yaml', help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='../experiments/default', help='Working directory')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='Training or testing phase')
    
    # Model parameters
    parser.add_argument('--model', type=str, default=None, help='Model class path')
    parser.add_argument('--model-args', type=str, default=None, help='Model arguments')
    parser.add_argument('--weights', type=str, help='Path to weights file for testing')
    parser.add_argument('--model-saved-name', type=str, default='model', help='Name for saving model')
    
    # Training parameters
    parser.add_argument('--num-epoch', type=int, default=80, help='Number of epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer: adam, adamw, sgd')
    parser.add_argument('--base-lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset name')
    parser.add_argument('--dataset-args', type=str, default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, help='Subject IDs to process')
    parser.add_argument('--feeder', type=str, help='Data feeder class path')
    parser.add_argument('--use-smv', type=str2bool, default=False, help='Use Signal Magnitude Vector')
    
    # Batch sizes
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, help='Testing batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, help='Validation batch size')
    
    # Other parameters
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--num-worker', type=int, default=4, help='Workers for data loading')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print logs to file')
    parser.add_argument('--result-file', type=str, help='File to save results')
    
    return parser

def parse_args():
    """Parse command line arguments and config file"""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config file
    if args.config is not None:
        try:
            with open(args.config, 'r') as f:
                config_args = yaml.safe_load(f)
            
            # Override args from config
            for k, v in config_args.items():
                if k not in ['device']:  # Don't override device
                    setattr(args, k, v)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Convert string args to proper types
    for arg_name in ['model_args', 'dataset_args']:
        if hasattr(args, arg_name) and isinstance(getattr(args, arg_name), str):
            try:
                setattr(args, arg_name, json.loads(getattr(args, arg_name)))
            except:
                pass
    
    return args

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Configure GPU
    if hasattr(args, 'device'):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    # Set random seeds
    if hasattr(args, 'seed'):
        seed = args.seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    # Create trainer and start
    try:
        logger.info("Initializing trainer...")
        trainer = Trainer(args)
        
        logger.info(f"Starting {args.phase} phase...")
        trainer.start()
        
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure proper imports
    import os
    import sys
    import shutil
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Create empty __init__.py files if they don't exist
    for dir_path in ['models', 'utils']:
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
    
    main()
