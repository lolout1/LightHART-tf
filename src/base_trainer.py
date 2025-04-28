#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Trainer for LightHART-TF

Implements core training, evaluation, and model management functionality.
"""
import os
import time
import logging
from datetime import datetime
import traceback
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger('lightheart-tf')

class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.wait = 0
    
    def __call__(self, val_loss):
        """Check if training should stop"""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.wait = 0
            return False
        
        self.counter += 1
        self.wait += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.wait = 0
        self.best_loss = None
        self.early_stop = False

class BaseTrainer:
    """Base trainer for fall detection models"""
    
    def __init__(self, arg):
        """Initialize trainer with arguments"""
        self.arg = arg
        
        # Initialize metrics and state
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.test_accuracy = 0 
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0 
        self.test_auc = 0
        
        # Initialize dataset splits
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        
        # Initialize data variables
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = {}
        self.pos_weights = None
        
        # Initialize early stopping
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        
        # Setup directories and model paths
        self.setup_directories()
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(self.arg.model, self.arg.model_args)
        else:
            if hasattr(self.arg, 'weights') and self.arg.weights:
                try:
                    self.model = tf.keras.models.load_model(self.arg.weights)
                    logger.info(f"Loaded model from {self.arg.weights}")
                except Exception:
                    # If loading full model fails, try loading model class and weights
                    self.model = self.load_model(self.arg.model, self.arg.model_args)
                    self.model.load_weights(self.arg.weights)
                    logger.info(f"Loaded weights from {self.arg.weights}")
            else:
                self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        # Report model information
        num_params = self.count_parameters(self.model)
        self.print_log(f"Model: {self.arg.model}")
        self.print_log(f"Parameters: {num_params:,}")
        self.print_log(f"Model size: {num_params * 4 / (1024**2):.2f} MB")
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        # Ensure work directory has timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if hasattr(self.arg, 'work_dir') and os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{timestamp}"
        
        # Create required directories
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'results'), exist_ok=True)
        
        # Set model path
        self.model_path = os.path.join(
            self.arg.work_dir, 
            'models', 
            self.arg.model_saved_name
        )
    
    def import_class(self, import_str):
        """Dynamically import a class"""
        mod_str, _sep, class_str = import_str.rpartition('.')
        
        # Try multiple import paths
        for prefix in ['', 'src.']:
            try:
                import importlib
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
                
        raise ImportError(f"Cannot import {class_str} from {mod_str}")
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        total_params = 0
        for var in model.trainable_variables:
            total_params += tf.size(var).numpy()
        return total_params
    
    def print_log(self, message):
        """Log message to console and file"""
        logger.info(message)
        
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(message, file=f)
    
    def load_model(self, model_name, model_args):
        """Load and initialize model"""
        try:
            ModelClass = self.import_class(model_name)
            model = ModelClass(**model_args)
            self.print_log(f"Created model: {model_name}")
            
            # Build model with dummy input to initialize weights
            try:
                if 'accelerometer' in self.arg.dataset_args['modalities']:
                    acc_frames = model_args.get('acc_frames', 128)
                    acc_coords = model_args.get('acc_coords', 3)
                    
                    # Create dummy input with batch size 2
                    dummy_input = {
                        'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                    }
                    
                    # Add skeleton if needed
                    if 'skeleton' in self.arg.dataset_args['modalities']:
                        dummy_input['skeleton'] = tf.zeros((2, acc_frames, 32, 3), dtype=tf.float32)
                    
                    # Forward pass to build model
                    _ = model(dummy_input, training=False)
                    
                    self.print_log("Model built successfully")
            except Exception as e:
                self.print_log(f"Warning: Could not pre-build model: {e}")
            
            return model
        except Exception as e:
            self.print_log(f"Error loading model {model_name}: {e}")
            traceback.print_exc()
            raise
    
    def cal_weights(self):
        """Calculate class weights for imbalanced training"""
        labels = self.norm_train.get('labels', [])
        if len(labels) == 0:
            self.pos_weights = tf.constant(1.0)
            self.print_log("No labels found, using default pos_weight=1.0")
            return
            
        label_count = Counter(labels)
        
        # Ensure there are positive examples
        if 1 in label_count and label_count[1] > 0:
            # Calculate ratio of negative to positive examples
            self.pos_weights = tf.constant(float(label_count[0]) / float(label_count[1]))
        else:
            self.pos_weights = tf.constant(1.0)
        
        # Log class distribution
        self.print_log(f"Class balance - Negative: {label_count.get(0, 0)}, Positive: {label_count.get(1, 0)}")
        self.print_log(f"Positive class weight: {self.pos_weights.numpy():.4f}")
    
    def load_optimizer(self):
        """Initialize optimizer based on configuration"""
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
    
    def load_loss(self):
        """Initialize loss function with class weights"""
        if not hasattr(self, 'pos_weights') or self.pos_weights is None:
            self.pos_weights = tf.constant(1.0)
        
        # Create weighted binary cross entropy loss
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * (self.pos_weights - 1.0) + 1.0
            return tf.reduce_mean(weights * bce)
        
        self.criterion = weighted_bce
        self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")
    
    def load_data(self):
        """Load and prepare datasets"""
        try:
            # Import data feeder
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
            
            # Import dataset preparation functions
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            
            if self.arg.phase == 'train':
                # Create dataset builder
                builder = prepare_smartfallmm_tf(self.arg)
                
                # Check if we have valid subject lists
                if not self.train_subjects:
                    self.print_log("No training subjects specified")
                    return False
                
                # Prepare training data
                self.print_log(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False)
                
                if any(len(x) == 0 for x in self.norm_train.values()):
                    self.print_log("Error: Training data is empty")
                    return False
                    
                # Create training data loader
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size
                )
                
                # Calculate class weights
                self.cal_weights()
                
                # Visualize training data distribution
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
                
                # Prepare validation data if available
                if self.val_subject:
                    self.print_log(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        self.print_log("Warning: Validation data is empty, using subset of training data")
                        # Use a subset of training data for validation
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)  # 20% or max 100 samples
                        
                        self.norm_val = {
                            k: v[-val_size:].copy() for k, v in self.norm_train.items()
                        }
                        self.norm_train = {
                            k: v[:-val_size].copy() for k, v in self.norm_train.items()
                        }
                    
                    # Create validation data loader
                    self.data_loader['val'] = Feeder(
                        dataset=self.norm_val,
                        batch_size=self.arg.val_batch_size
                    )
                    
                    # Visualize validation data distribution
                    self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                
                # Prepare test data if available
                if self.test_subject:
                    self.print_log(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        self.print_log("Warning: Test data is empty")
                        return False
                        
                    # Create test data loader
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size
                    )
                    
                    # Visualize test data distribution
                    subject_id = self.test_subject[0] if self.test_subject else 'unknown'
                    self.distribution_viz(
                        self.norm_test['labels'], 
                        self.arg.work_dir, 
                        f'test_{subject_id}'
                    )
                
                self.print_log("Data loading complete")
                return True
            elif self.arg.phase == 'test':
                # For test-only mode
                if not self.test_subject:
                    self.print_log("No test subjects specified")
                    return False
                
                builder = prepare_smartfallmm_tf(self.arg)
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                
                if any(len(x) == 0 for x in self.norm_test.values()):
                    self.print_log("Error: Test data is empty")
                    return False
                
                self.data_loader['test'] = Feeder(
                    dataset=self.norm_test,
                    batch_size=self.arg.test_batch_size
                )
                
                self.print_log("Test data loading complete")
                return True
                
        except Exception as e:
            self.print_log(f"Error loading data: {e}")
            traceback.print_exc()
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
            
            # Add count labels on bars
            for i, v in enumerate(count):
                plt.text(values[i], v + 0.1, str(v), ha='center')
            
            # Save visualization
            viz_dir = os.path.join(work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'{mode}_distribution.png'))
            plt.close()
            
            # Log distribution
            dist_str = ", ".join([f"Label {int(v)}: {c}" for v, c in zip(values, count)])
            self.print_log(f"{mode} distribution: {dist_str}")
            
        except Exception as e:
            self.print_log(f"Error visualizing distribution: {e}")
    
    def loss_viz(self, train_loss, val_loss):
        """Visualize training and validation loss curves"""
        try:
            if not train_loss or not val_loss:
                self.print_log("No loss data to visualize")
                return
                
            epochs = range(1, len(train_loss) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
            plt.title(f'Training vs Validation Loss (Subject {self.test_subject[0]})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Save visualization
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            plt.savefig(os.path.join(viz_dir, f'loss_curves_{self.test_subject[0]}.png'))
            plt.close()
            
        except Exception as e:
            self.print_log(f"Error visualizing loss: {e}")
    
    def cm_viz(self, y_pred, y_true):
        """Visualize confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix (Subject {self.test_subject[0]})')
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
            
            # Save visualization
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{self.test_subject[0]}.png'))
            plt.close()
            
        except Exception as e:
            self.print_log(f"Error visualizing confusion matrix: {e}")
    
    def cal_prediction(self, logits):
        """Calculate binary predictions from logits"""
        # Handle different output shapes
        if isinstance(logits, tuple) and len(logits) > 0:
            logits = logits[0]  # Sometimes model returns (logits, features)
            
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            # Multi-class case
            return tf.argmax(logits, axis=-1)
        else:
            # Binary case
            return tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    
    def cal_metrics(self, targets, predictions):
        """Calculate evaluation metrics"""
        # Convert to numpy arrays
        if isinstance(targets, tf.Tensor):
            targets = targets.numpy()
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()
            
        # Flatten arrays
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions) * 100
        
        # Handle edge cases for binary metrics
        unique_targets = np.unique(targets)
        unique_preds = np.unique(predictions)
        
        if len(unique_targets) <= 1 or len(unique_preds) <= 1:
            # Single class in targets or predictions
            if len(unique_targets) == 1 and len(unique_preds) == 1 and unique_targets[0] == unique_preds[0]:
                # Perfect prediction of a single class
                if unique_targets[0] == 1:  # All positive
                    precision = 100.0
                    recall = 100.0
                    f1 = 100.0
                else:  # All negative
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                auc = 50.0  # Undefined AUC for single class
            else:
                # Imperfect prediction with one class
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                auc = 50.0
        else:
            # Normal case with multiple classes
            precision = precision_score(targets, predictions, zero_division=0) * 100
            recall = recall_score(targets, predictions, zero_division=0) * 100
            f1 = f1_score(targets, predictions, zero_division=0) * 100
            
            try:
                auc = roc_auc_score(targets, predictions) * 100
            except:
                auc = 50.0  # Default AUC when calculation fails
        
        return accuracy, f1, recall, precision, auc
    
    @tf.function
    def train_step(self, inputs, targets):
        """Training step with gradient tape"""
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(inputs, training=True)
            
            # Extract logits (model may return (logits, features))
            if isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                logits = outputs
            
            # Reshape logits if needed
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                # Multi-class case
                pass
            else:
                # Binary case
                logits = tf.squeeze(logits)
            
            # Compute loss
            loss = self.criterion(targets, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate predictions
        predictions = self.cal_prediction(logits)
        
        return loss, predictions
    
    @tf.function
    def test_step(self, inputs, targets):
        """Evaluation step"""
        # Forward pass
        outputs = self.model(inputs, training=False)
        
        # Extract logits (model may return (logits, features))
        if isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs
        
        # Reshape logits if needed
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            # Multi-class case
            pass
        else:
            # Binary case
            logits = tf.squeeze(logits)
        
        # Compute loss
        loss = self.criterion(targets, logits)
        
        # Calculate predictions
        predictions = self.cal_prediction(logits)
        
        return loss, predictions
    
    def train(self, epoch):
        """Train model for one epoch"""
        self.model.trainable = True
        
        # Start timer
        start_time = time.time()
        
        # Get data loader
        loader = self.data_loader['train']
        
        # Initialize metrics
        train_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        # Create progress bar
        desc = f"Epoch {epoch+1}/{self.arg.num_epoch}"
        progress_bar = tqdm(loader, desc=desc)
        
        # Iterate through batches
        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            # Convert targets to float32
            targets = tf.cast(targets, tf.float32)
            
            # Train step
            loss, predictions = self.train_step(inputs, targets)
            
            # Update metrics
            loss_val = loss.numpy() if isinstance(loss, tf.Tensor) else float(loss)
            train_loss += loss_val
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{train_loss/steps:.4f}"
            })
        
        # Calculate average loss and metrics
        train_loss /= steps
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(all_labels, all_preds)
        
        # Save training loss
        self.train_loss_summary.append(train_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Log results
        self.print_log(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Acc={accuracy:.2f}%, "
            f"F1={f1:.2f}%, "
            f"Prec={precision:.2f}%, "
            f"Rec={recall:.2f}%, "
            f"AUC={auc_score:.2f}% "
            f"({epoch_time:.2f}s)"
        )
        
        # Validate model
        val_loss = self.eval(epoch, loader_name='val')
        
        # Save validation loss
        self.val_loss_summary.append(val_loss)
        
        # Check early stopping
        self.early_stop(val_loss)
    
    def eval(self, epoch, loader_name='val'):
        """Evaluate model on a dataset"""
        self.model.trainable = False
        
        # Start timer
        start_time = time.time()
        
        # Get data loader
        loader = self.data_loader.get(loader_name)
        if loader is None:
            self.print_log(f"No data loader for {loader_name}")
            return float('inf')
        
        # Initialize metrics
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_logits = []
        steps = 0
        
        # Create progress bar
        desc = f"Eval {loader_name} ({epoch+1})"
        progress_bar = tqdm(loader, desc=desc)
        
        # Iterate through batches
        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            # Convert targets to float32
            targets = tf.cast(targets, tf.float32)
            
            # Evaluation step
            loss, predictions = self.test_step(inputs, targets)
            
            # Extract logits for AUC calculation
            outputs = self.model(inputs, training=False)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                logits = outputs
            
            # Update metrics
            loss_val = loss.numpy() if isinstance(loss, tf.Tensor) else float(loss)
            eval_loss += loss_val
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            all_logits.extend(logits.numpy())
            steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{eval_loss/steps:.4f}"
            })
        
        # Calculate average loss and metrics
        if steps > 0:
            eval_loss /= steps
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(all_labels, all_preds)
            
            # Log results
            self.print_log(
                f"{loader_name.capitalize()}: "
                f"Loss={eval_loss:.4f}, "
                f"Acc={accuracy:.2f}%, "
                f"F1={f1:.2f}%, "
                f"Prec={precision:.2f}%, "
                f"Rec={recall:.2f}%, "
                f"AUC={auc_score:.2f}%"
            )
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            self.print_log(f"{loader_name.capitalize()} time: {epoch_time:.2f}s")
            
            # For validation, check if this is the best model
            if loader_name == 'val':
                is_best = False
                
                # Check if loss improved
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    is_best = True
                    self.print_log(f"New best validation loss: {eval_loss:.4f}")
                
                # Save model if best
                if is_best:
                    self.save_model(epoch)
            
            # For test, store metrics
            elif loader_name.startswith('test'):
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
                
                # Create confusion matrix visualization
                self.cm_viz(all_preds, all_labels)
                
                # Save test results
                results = {
                    "subject": self.test_subject[0] if self.test_subject else "unknown",
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "auc": float(auc_score),
                    "loss": float(eval_loss)
                }
                
                results_file = os.path.join(
                    self.arg.work_dir,
                    'results',
                    f'test_results_{self.test_subject[0] if self.test_subject else "unknown"}.json'
                )
                
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return eval_loss
        else:
            self.print_log(f"No evaluation steps for {loader_name}")
            return float('inf')
    
    def save_model(self, epoch):
        """Save model weights and full model"""
        try:
            # Generate model filename
            if self.test_subject:
                base_filename = f"{self.model_path}_{self.test_subject[0]}"
            else:
                base_filename = f"{self.model_path}_epoch{epoch}"
            
            # Save model weights
            weights_path = f"{base_filename}.weights.h5"
            self.model.save_weights(weights_path)
            self.print_log(f"Saved model weights to {weights_path}")
            
            # Save full model
            try:
                model_path = f"{base_filename}"
                self.model.save(model_path)
                self.print_log(f"Saved full model to {model_path}")
            except Exception as e:
                self.print_log(f"Warning: Could not save full model: {e}")
            
            # Try to export TFLite model
            try:
                from utils.tflite_converter import convert_to_tflite
                
                tflite_path = f"{base_filename}.tflite"
                success = convert_to_tflite(
                    model=self.model,
                    save_path=tflite_path,
                    input_shape=(1, 128, 3)  # Default shape for accelerometer data
                )
                
                if success:
                    self.print_log(f"Exported TFLite model to {tflite_path}")
                else:
                    self.print_log("Warning: TFLite export failed")
            except Exception as e:
                self.print_log(f"Warning: Could not export TFLite model: {e}")
            
            return True
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            traceback.print_exc()
            return False
    
    def create_results_df(self):
        """Create results dataframe"""
        return []
    
    def add_avg_df(self, results):
        """Add average row to results"""
        if not results:
            return results
            
        avg_result = {'test_subject': 'Average'}
        
        for column in results[0].keys():
            if column != 'test_subject':
                values = [float(r[column]) for r in results]
                avg_result[column] = round(sum(values) / len(values), 2)
        
        results.append(avg_result)
        return results
    
    def start(self):
        """Main execution method for training or testing"""
        if self.arg.phase == 'train':
            # Log training parameters
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            
            # Create results list
            results = self.create_results_df()
            
            # Define validation subjects
            val_subjects = [38, 46]  # Default validation subjects
            
            # Process each subject in leave-one-out cross-validation
            for i, test_subject in enumerate(self.arg.subjects):
                # Skip validation subjects
                if test_subject in val_subjects:
                    continue
                
                # Reset metrics for this fold
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                # Define train/val/test split
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects 
                                      if s != test_subject and s not in val_subjects]
                
                self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                self.print_log(f"Train: {len(self.train_subjects)} subjects")
                self.print_log(f"Val: {len(self.val_subject)} subjects")
                self.print_log(f"Test: Subject {test_subject}")
                
                # Create new model instance
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                
                # Load data
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                # Initialize optimizer and loss
                self.load_optimizer()
                self.load_loss()
                
                # Reset early stopping
                self.early_stop.reset()
                
                # Train for specified epochs
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    
                    # Check early stopping
                    if self.early_stop.early_stop:
                        self.print_log(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Load best model for testing
                best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                if os.path.exists(best_weights):
                    self.model.load_weights(best_weights)
                    self.print_log(f"Loaded best weights from {best_weights}")
                
                # Final evaluation on test set
                self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                self.eval(epoch=0, loader_name=f'test')
                
                # Visualize loss curves
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                
                # Store results
                subject_result = {
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                
                results.append(subject_result)
                
                # Clear memory
                tf.keras.backend.clear_session()
            
            # Calculate and save average results
            if results:
                # Add average row
                results = self.add_avg_df(results)
                
                # Save results as CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)
                
                # Save as JSON
                with open(os.path.join(self.arg.work_dir, 'scores.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Log final results
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
            
            self.print_log("Training completed successfully")
            
        elif self.arg.phase == 'test':
            # Testing only mode
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for testing")
                return
            
            # Set up subject for testing
            if not hasattr(self, 'test_subject') or not self.test_subject:
                if hasattr(self.arg, 'subjects') and self.arg.subjects:
                    self.test_subject = [self.arg.subjects[0]]
                else:
                    self.print_log("No test subject specified")
                    return
            
            # Load data
            if not self.load_data():
                self.print_log("Failed to load test data")
                return
            
            # Initialize loss function
            self.load_loss()
            
            # Evaluate on test data
            self.print_log(f"Testing on subject {self.test_subject[0]}")
            self.eval(epoch=0, loader_name='test')
            
            # Log results
            self.print_log(
                f"Test results: "
                f"Acc={self.test_accuracy:.2f}%, "
                f"F1={self.test_f1:.2f}%, "
                f"Prec={self.test_precision:.2f}%, "
                f"Rec={self.test_recall:.2f}%, "
                f"AUC={self.test_auc:.2f}%"
            )
            
        elif self.arg.phase == 'tflite':
            # TFLite export mode
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for TFLite export")
                return
            
            # Try to export TFLite model
            try:
                from utils.tflite_converter import convert_to_tflite
                
                # Generate output path
                if hasattr(self.arg, 'result_file') and self.arg.result_file:
                    tflite_path = self.arg.result_file
                else:
                    tflite_path = os.path.join(self.arg.work_dir, 'model.tflite')
                
                # Export model
                success = convert_to_tflite(
                    model=self.model,
                    save_path=tflite_path,
                    input_shape=(1, 128, 3),  # Default shape for accelerometer data
                    quantize=True
                )
                
                if success:
                    self.print_log(f"Successfully exported TFLite model to {tflite_path}")
                else:
                    self.print_log("Failed to export TFLite model")
                    
            except Exception as e:
                self.print_log(f"Error exporting TFLite model: {e}")
                traceback.print_exc()
        
        else:
            self.print_log(f"Unknown phase: {self.arg.phase}")
