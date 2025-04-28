# src/base_trainer.py
"""
Base Trainer for LightHART-TF
Handles training, evaluation, and TFLite export
"""
import os
import time
import datetime
import shutil
import argparse
import yaml
import json
import traceback
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

class EarlyStoppingTF:
    """Early stopping implementation for TensorFlow"""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

class BaseTrainer:
    """Base trainer class for LightHART-TF models"""
    
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.test_accuracy = 0 
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0 
        self.test_auc = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.early_stop = EarlyStoppingTF(patience=15, min_delta=.001)
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities'] 
                                  if modality != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1 
        
        # Setup working directory
        if os.path.exists(self.arg.work_dir):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.arg.work_dir = f"{self.arg.work_dir}_{timestamp}"
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        
        # Setup model path
        self.model_path = f'{self.arg.work_dir}/models/{self.arg.model_saved_name}'
        
        # Save configuration file
        self.save_config(arg.config, arg.work_dir)
        
        # Initialize model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            if hasattr(self.arg, 'weights') and self.arg.weights:
                try:
                    self.model = tf.keras.models.load_model(self.arg.weights)
                except:
                    self.model = self.load_model(arg.model, arg.model_args)
                    self.model.load_weights(self.arg.weights)
            else:
                self.model = self.load_model(arg.model, arg.model_args)
        
        # Count and log model parameters
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params/(1024**2):.2f} MB')

    def add_avg_df(self, results):
        """Add average row to results dataframe"""
        avg_result = {'test_subject': 'Average'}
        for column in results[0].keys():
            if column != 'test_subject':
                avg_result[column] = round(np.mean([float(r[column]) for r in results]), 2)
        results.append(avg_result)
        return results

    def save_config(self, src_path, desc_path):
        """Save configuration file to working directory"""
        try:
            config_filename = os.path.basename(src_path)
            shutil.copy(src_path, f'{desc_path}/{config_filename}')
            # Also save as JSON for easier reading
            with open(src_path, 'r') as f:
                config = yaml.safe_load(f)
            with open(f'{desc_path}/config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.print_log(f"Error saving config: {e}")

    def cal_weights(self):
        """Calculate class weights for imbalanced training"""
        label_count = Counter(self.norm_train['labels'])
        self.pos_weights = tf.constant(float(label_count[0]) / float(label_count[1]) 
                                      if 1 in label_count and label_count[1] > 0 else 1.0)
        self.print_log(f"Class balance - Negative: {label_count[0]}, Positive: {label_count[1]}")
        self.print_log(f"Positive class weight: {self.pos_weights.numpy():.4f}")

    def count_parameters(self, model):
        """Count trainable parameters in model"""
        return np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])

    def has_empty_value(self, *lists):
        """Check if any of the provided lists are empty"""
        return any(isinstance(lst, (list, np.ndarray)) and len(lst) == 0 for lst in lists)

    def import_class(self, import_str):
        """Dynamically import a class"""
        import importlib
        mod_str, _sep, class_str = import_str.rpartition('.')
        
        # Try to import with different prefixes
        for prefix in ['', 'src.']:
            try:
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
        
        self.print_log(f"Error importing {import_str}")
        raise ImportError(f"Cannot import {class_str} from {mod_str}")

    def load_model(self, model_name, model_args):
        """Load and initialize model"""
        try:
            ModelClass = self.import_class(model_name)
            model = ModelClass(**model_args)
            self.print_log(f"Created model: {model_name}")
            return model
        except Exception as e:
            self.print_log(f"Error loading model {model_name}: {e}")
            traceback.print_exc()
            raise

    def load_weights(self):
        """Load saved model weights"""
        try:
            weights_path = f'{self.model_path}_{self.test_subject[0]}.weights.h5'
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                self.print_log(f"Loaded weights: {weights_path}")
                return True
            else:
                self.print_log(f"Weights file not found: {weights_path}")
                return False
        except Exception as e:
            self.print_log(f"Error loading weights: {e}")
            return False

    def load_optimizer(self):
        """Initialize optimizer based on configuration"""
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
        if not hasattr(self, 'pos_weights'):
            self.pos_weights = tf.constant(1.0)
            
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * (self.pos_weights - 1.0) + 1.0
            return tf.reduce_mean(weights * bce)
            
        self.criterion = weighted_bce
        self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")

    def load_data(self):
        """Load and prepare training data"""
        try:
            # Import data feeder
            Feeder = self.import_class(self.arg.feeder)
            
            # Import dataset preparation functions
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            
            if self.arg.phase == 'train':
                # Create dataset builder
                builder = prepare_smartfallmm_tf(self.arg)
                
                # Prepare training data
                self.print_log(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, self.fuse)
                
                if self.has_empty_value(list(self.norm_train.values())):
                    self.print_log("Error: Training data is empty")
                    return False
                    
                # Create training data loader
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size
                )
                
                # Calculate class weights for imbalanced training
                self.cal_weights()
                
                # Visualize training data distribution
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
                
                # Prepare validation data
                self.print_log(f"Processing validation data for subjects: {self.val_subject}")
                self.norm_val = split_by_subjects_tf(builder, self.val_subject, self.fuse)
                
                if self.has_empty_value(list(self.norm_val.values())):
                    self.print_log("Error: Validation data is empty")
                    return False
                    
                # Create validation data loader
                self.data_loader['val'] = Feeder(
                    dataset=self.norm_val,
                    batch_size=self.arg.val_batch_size
                )
                
                # Visualize validation data distribution
                self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                
                # Prepare test data
                self.print_log(f"Processing test data for subjects: {self.test_subject}")
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.fuse)
                
                if self.has_empty_value(list(self.norm_test.values())):
                    self.print_log("Error: Test data is empty")
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
                
        except Exception as e:
            self.print_log(f"Error loading data: {e}")
            traceback.print_exc()
            return False

    def record_time(self):
        """Start timing measurement"""
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """Calculate elapsed time since last record_time call"""
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time=True):
        """Print and log message"""
        print(string)
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def distribution_viz(self, labels, work_dir, mode):
        """Visualize class distribution"""
        try:
            values, count = np.unique(labels, return_counts=True)
            
            plt.figure(figsize=(8, 6))
            plt.bar(values, count)
            plt.xlabel('Labels')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Label Distribution')
            
            # Add count labels on bars
            for i, v in enumerate(count):
                plt.text(values[i], v + 0.1, str(v), ha='center')
            
            # Save to visualizations directory
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
            epochs = range(1, len(train_loss) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
            plt.title(f'Training vs Validation Loss (Subject {self.test_subject[0]})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'trainvsval_{self.test_subject[0]}.png'))
            plt.close()
        except Exception as e:
            self.print_log(f"Error visualizing loss: {e}")

    def cm_viz(self, y_pred, y_true):
        """Visualize confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(f'Confusion Matrix (Subject {self.test_subject[0]})')
            
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{self.test_subject[0]}.png'))
            plt.close()
        except Exception as e:
            self.print_log(f"Error visualizing confusion matrix: {e}")

    def create_df(self):
        """Create empty dataframe for results"""
        return []

    def cal_prediction(self, logits):
        """Calculate binary predictions from logits"""
        return tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)

    def cal_metrics(self, targets, predictions):
        """Calculate evaluation metrics with robust handling for single-class data"""
        # Convert to numpy arrays
        if isinstance(targets, tf.Tensor):
            targets = targets.numpy()
        if isinstance(predictions, tf.Tensor):
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                # Handle multi-class predictions
                pred_probs = tf.nn.softmax(predictions, axis=-1).numpy()[:, 1]
                pred_classes = (pred_probs > 0.5).astype(np.int32)
            else:
                # Handle binary predictions
                pred_probs = tf.sigmoid(predictions).numpy().flatten()
                pred_classes = (pred_probs > 0.5).astype(np.int32)
        else:
            # Already processed predictions
            pred_classes = np.array(predictions).flatten()
            pred_probs = pred_classes.astype(np.float32)
        
        # Convert targets to numpy array
        targets = np.array(targets).flatten().astype(np.int32)
        
        # Calculate accuracy
        accuracy = accuracy_score(targets, pred_classes) * 100
        
        # Check for single-class data (subject 30 issue)
        unique_targets = np.unique(targets)
        
        if len(unique_targets) == 1:
            target_class = unique_targets[0]
            self.print_log(f"Single-class dataset detected (class {target_class})")
            
            if target_class == 1:  # All positive samples
                # Count true positives and false negatives
                tp = np.sum((pred_classes == 1) & (targets == 1))
                fn = np.sum((pred_classes == 0) & (targets == 1))
                total = tp + fn
                
                # Calculate precision
                if np.sum(pred_classes == 1) == 0:
                    precision = 100.0 if fn == 0 else 0.0
                else:
                    precision = 100.0 * tp / np.sum(pred_classes == 1)
                
                # Calculate recall
                recall = 0.0 if total == 0 else 100.0 * tp / total
                
                # Calculate F1
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                
                # Set default AUC
                auc = 50.0
            else:  # All negative samples
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                auc = 50.0
        else:
            # Standard metrics calculation
            try:
                precision = precision_score(targets, pred_classes, zero_division=0) * 100
                recall = recall_score(targets, pred_classes, zero_division=0) * 100
                f1 = f1_score(targets, pred_classes, zero_division=0) * 100
                
                if len(unique_targets) > 1:
                    auc = roc_auc_score(targets, pred_probs) * 100
                else:
                    auc = 50.0
            except Exception as e:
                self.print_log(f"Error calculating metrics: {e}")
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                auc = 50.0
        
        return accuracy, f1, recall, precision, auc

    @tf.function
    def train_step(self, inputs, targets):
        """Single training step with gradient updates"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits, _ = self.model(inputs, training=True)
            
            # Flatten logits if needed
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                logits = tf.squeeze(logits)
                
            # Compute loss
            loss = self.criterion(targets, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate predictions
        preds = self.cal_prediction(logits)
        
        return loss, preds

    @tf.function
    def test_step(self, inputs, targets):
        """Single evaluation step"""
        # Forward pass
        logits, _ = self.model(inputs, training=False)
        
        # Flatten logits if needed
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            logits = tf.squeeze(logits)
            
        # Compute loss
        loss = self.criterion(targets, logits)
        
        # Calculate predictions
        preds = self.cal_prediction(logits)
        
        return loss, preds

    def save_model(self, epoch, subject_id=None):
        """Save model with TFLite export
        
        Args:
            epoch: Current epoch number
            subject_id: Optional subject ID to include in filename
            
        Returns:
            tuple: (weights_path, tflite_path)
        """
        try:
            # Import TFLite converter
            from utils.tflite_converter import convert_to_tflite
            
            # Generate filenames
            if subject_id is not None:
                base_name = f"{self.model_path}_{subject_id}"
            else:
                base_name = f"{self.model_path}_{epoch}"
            
            # Create directory if needed
            os.makedirs(os.path.dirname(base_name), exist_ok=True)
            
            # Save model weights
            weights_path = f"{base_name}.weights.h5"
            self.model.save_weights(weights_path)
            self.print_log(f"Model weights saved to {weights_path}")
            
            # Save full model if possible
            try:
                model_path = base_name
                self.model.save(model_path)
                self.print_log(f"Full model saved to {model_path}")
            except Exception as e:
                self.print_log(f"Warning: Could not save full model: {e}")
            
            # Export TFLite model
            tflite_path = f"{base_name}.tflite"
            success = convert_to_tflite(
                model=self.model,
                save_path=tflite_path,
                input_shape=(1, 128, 3)  # Accelerometer-only shape
            )
            
            if success:
                self.print_log(f"TFLite model exported to {tflite_path}")
                self.print_log("New best model saved")
            else:
                self.print_log("Warning: TFLite conversion failed")
            
            return weights_path, tflite_path
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            traceback.print_exc()
            return None, None

    def train(self, epoch):
        """Train model for one epoch"""
        # Set model to training mode
        self.model.trainable = True
        
        # Start timing
        self.record_time()
        loader = self.data_loader['train']
        timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}
        
        # Initialize metrics
        train_loss = 0.0
        label_list = []
        pred_list = []
        cnt = 0
        
        # Create progress bar
        process = tqdm(loader, ncols=80, desc=f"Epoch {epoch+1}/{self.arg.num_epoch}")
        
        # Iterate through batches
        for batch_idx, (inputs, targets, _) in enumerate(process):
            # Prepare inputs and targets
            targets = tf.cast(targets, tf.float32)
            timer['dataloader'] += self.split_time()
            
            # Train step
            loss, preds = self.train_step(inputs, targets)
            timer['model'] += self.split_time()
            
            # Update metrics
            train_loss += loss.numpy() if isinstance(loss, tf.Tensor) else loss
            label_list.extend(targets.numpy())
            pred_list.extend(preds.numpy())
            cnt += 1
            
            # Update progress bar
            process.set_postfix({'loss': f"{train_loss/cnt:.4f}"})
            timer['stats'] += self.split_time()
        
        # Calculate final metrics
        if cnt > 0:
            train_loss /= cnt
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)
            
            # Save training loss
            self.train_loss_summary.append(train_loss)
            
            # Calculate time proportions
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
            
            # Log results
            self.print_log(
                f'Train Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, '
                f'F1={f1:.2f}%, Prec={precision:.2f}%, Rec={recall:.2f}%, AUC={auc_score:.2f}%'
            )
            self.print_log(f'Time: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')
            
            # Validate
            val_loss = self.eval(epoch, loader_name='val')
            self.val_loss_summary.append(val_loss)
            
            # Check early stopping
            self.early_stop(val_loss)
        else:
            self.print_log("No batches processed in epoch")

    def eval(self, epoch, loader_name='val', result_file=None):
        """Evaluate model on dataset"""
        # Set model to evaluation mode
        self.model.trainable = False
        
        # Open result file if specified
        result_file_handle = None
        if result_file is not None:
            result_file_handle = open(result_file, 'w', encoding='utf-8')
        
        self.print_log(f'Evaluating epoch {epoch+1} on {loader_name}')
        
        # Initialize metrics
        loss = 0.0
        cnt = 0
        all_preds = []
        all_labels = []
        wrong_idx = []
        
        # Create progress bar
        loader = self.data_loader[loader_name]
        process = tqdm(loader, ncols=80, desc=f"Eval {loader_name}")
        
        # Iterate through batches
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            try:
                # Prepare inputs and targets
                targets = tf.cast(targets, tf.float32)
                
                # Forward pass
                logits, _ = self.model(inputs, training=False)
                
                # Flatten logits if needed
                if len(logits.shape) > 1 and logits.shape[-1] > 1:
                    logits = tf.squeeze(logits)
                
                # Compute loss
                batch_loss = self.criterion(targets, logits)
                
                # Update metrics
                loss += batch_loss.numpy() if isinstance(batch_loss, tf.Tensor) else batch_loss
                all_preds.append(logits.numpy())
                all_labels.append(targets.numpy())
                cnt += 1
                
                # Track wrong predictions
                preds = self.cal_prediction(logits)
                for i in range(len(targets)):
                    if preds[i] != int(targets[i]):
                        wrong_idx.append(idx[i])
                
                # Update progress bar
                process.set_postfix({'loss': f"{loss/cnt:.4f}"})
                
            except Exception as e:
                self.print_log(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate final metrics
        if cnt > 0:
            loss /= cnt
            
            # Concatenate predictions and labels
            try:
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                accuracy, f1, recall, precision, auc_score = self.cal_metrics(all_labels, all_preds)
            except Exception as e:
                self.print_log(f"Error calculating metrics: {e}")
                accuracy, f1, recall, precision, auc_score = 0.0, 0.0, 0.0, 0.0, 50.0
            
            # Write results to file if provided
            if result_file_handle is not None:
                pred_classes = (tf.sigmoid(all_preds) > 0.5).numpy().astype(int)
                for i in range(len(pred_classes)):
                    result_file_handle.write(f"{pred_classes[i]} ==> {int(all_labels[i])}\n")
                result_file_handle.close()
            
            # Log results
            self.print_log(
                f'{loader_name.capitalize()}: Loss={loss:.4f}, '
                f'Acc={accuracy:.2f}%, F1={f1:.2f}%, '
                f'Prec={precision:.2f}%, Rec={recall:.2f}%, '
                f'AUC={auc_score:.2f}%'
            )
            
            # Save best model if validation
            if loader_name == 'val':
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model(epoch, self.test_subject[0] if self.test_subject else None)
            else:
                # Store test metrics
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
                
                # Visualize confusion matrix
                try:
                    self.cm_viz(
                        (tf.sigmoid(all_preds) > 0.5).numpy().astype(int),
                        all_labels.astype(int)
                    )
                except:
                    self.print_log("Error creating confusion matrix")
            
            return loss
        else:
            self.print_log(f"No batches processed in {loader_name}")
            return float('inf')

    def start(self):
        """Start training or testing process"""
        if self.arg.phase == 'train':
            # Log training parameters
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            
            # Initialize results dataframe
            results = self.create_df()
            
            # Define validation subjects
            val_subjects = [38, 46]
            
            # Run cross-validation for each test subject
            for test_subject in self.arg.subjects:
                # Skip validation subjects
                if test_subject in val_subjects:
                    continue
                
                # Reset metrics for this fold
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.best_f1 = 0.0
                
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
                self.print_log(f"Model Parameters: {self.count_parameters(self.model)}")
                
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
                    if self.early_stop.early_stop:
                        self.print_log(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Load best model for testing
                model_path = self.load_model(self.arg.model, self.arg.model_args)
                self.load_weights()
                
                # Final evaluation
                self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                self.eval(epoch=0, loader_name='test')
                
                # Visualize loss curves
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                
                # Save results
                subject_result = {
                    'test_subject': str(self.test_subject[0]),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                results.append(subject_result)
                
                self.print_log(f"Results for subject {test_subject}: "
                              f"Acc={subject_result['accuracy']}%, "
                              f"F1={subject_result['f1_score']}%")
            
            # Calculate and save average results
            if results:
                results = self.add_avg_df(results)
                
                # Save as CSV
                df = pd.DataFrame(results)
                df.to_csv(f'{self.arg.work_dir}/scores.csv', index=False)
                
                # Save as JSON
                with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Log final results
                self.print_log("\n=== Final Results ===")
                for r in results:
                    self.print_log(f"Subject {r['test_subject']}: "
                                  f"Acc={r['accuracy']}%, "
                                  f"F1={r['f1_score']}%")
            
            self.print_log("Training completed successfully")
            
        elif self.arg.phase == 'test':
            # Check if weights are specified
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for testing")
                return
            
            # Set up subject splits
            self.test_subject = [self.arg.subjects[0]]
            self.val_subject = [38, 46]
            self.train_subjects = [s for s in self.arg.subjects 
                                 if s != self.test_subject[0] and s not in self.val_subject]
            
            # Load data
            if not self.load_data():
                self.print_log("Failed to load test data")
                return
            
            # Initialize loss
            self.load_loss()
            
            # Run evaluation
            self.print_log(f"Testing on subject {self.test_subject[0]}")
            self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
            
            # Save results
            with open(f'{self.arg.work_dir}/test_results.json', 'w') as f:
                json.dump({
                    'test_subject': str(self.test_subject[0]),
                    'accuracy': float(self.test_accuracy),
                    'f1_score': float(self.test_f1),
                    'precision': float(self.test_precision),
                    'recall': float(self.test_recall),
                    'auc': float(self.test_auc)
                }, f, indent=2)
            
            self.print_log("Testing completed successfully")
