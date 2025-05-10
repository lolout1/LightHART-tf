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

logger = logging.getLogger('lightheart-tf')

class EarlyStopping:
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
        try:
            if isinstance(val_loss, tf.Tensor):
                val_loss = float(val_loss.numpy())
            else:
                val_loss = float(val_loss)
            self.history.append(val_loss)
            if self.best_loss is None:
                self.best_loss = val_loss
                return False
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.wait = 0
                return False
            else:
                self.counter += 1
                self.wait += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.stopped_epoch = len(self.history)
                    return True
                return False
        except Exception as e:
            logging.error(f"Error in EarlyStopping: {e}")
            return False
    
    def reset(self):
        self.counter = 0
        self.wait = 0
        self.best_loss = None
        self.early_stop = False
        self.history = []
        self.stopped_epoch = 0

class BaseTrainer:
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
        self.data_loader = {}
        self.pos_weights = None
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        self.setup_directories()
        self.print_log("Loading model...")
        self.model = self.load_model()
        num_params = self.count_parameters(self.model)
        self.print_log(f"Model: {self.arg.model}")
        self.print_log(f"Parameters: {num_params:,}")
        self.print_log(f"Model size: {num_params * 4 / (1024**2):.2f} MB")
    
    def setup_directories(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'results'), exist_ok=True)
        self.model_path = os.path.join(self.arg.work_dir, 'models', self.arg.model_saved_name)
    
    def import_class(self, import_str):
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
        logger.info(message)
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(message, file=f)
    
    def count_parameters(self, model):
        total_params = 0
        for var in model.trainable_variables:
            total_params += tf.size(var).numpy()
        return total_params
    
    def load_model(self):
        try:
            if self.arg.phase == 'train':
                if self.arg.model is None:
                    raise ValueError("Model class path is required")
                model_class = self.import_class(self.arg.model)
                model = model_class(**self.arg.model_args)
                self.print_log(f"Created model: {self.arg.model}")
                try:
                    acc_frames = self.arg.model_args.get('acc_frames', 128)
                    acc_coords = 3
                    dummy_input = {'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)}
                    _ = model(dummy_input, training=False)
                    self.print_log("Model built successfully")
                except Exception as e:
                    self.print_log(f"Warning: Could not pre-build model: {e}")
                return model
            else:
                if hasattr(self.arg, 'weights') and self.arg.weights:
                    try:
                        model = tf.keras.models.load_model(self.arg.weights)
                        self.print_log(f"Loaded model from {self.arg.weights}")
                        return model
                    except Exception:
                        if self.arg.model is None:
                            raise ValueError("Model class path is required when weights cannot be directly loaded")
                        model_class = self.import_class(self.arg.model)
                        model = model_class(**self.arg.model_args)
                        acc_frames = self.arg.model_args.get('acc_frames', 128)
                        acc_coords = 3
                        dummy_input = {'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)}
                        _ = model(dummy_input, training=False)
                        model.load_weights(self.arg.weights)
                        self.print_log(f"Loaded weights from {self.arg.weights}")
                        return model
                else:
                    if self.arg.model is None:
                        raise ValueError("Model class path is required")
                    model_class = self.import_class(self.arg.model)
                    model = model_class(**self.arg.model_args)
                    self.print_log(f"Created model: {self.arg.model}")
                    return model
        except Exception as e:
            self.print_log(f"Error loading model: {e}")
            raise
    
    def calculate_class_weights(self, labels):
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
        """Setup optimizer with learning rate scheduling"""
        try:
            if not hasattr(self.arg, 'optimizer'):
                self.arg.optimizer = 'adam'
            if not hasattr(self.arg, 'base_lr'):
                self.arg.base_lr = 0.001
            if not hasattr(self.arg, 'weight_decay'):
                self.arg.weight_decay = 0.0004
            
            # Create learning rate schedule - cosine decay with warmup
            total_steps = self.arg.num_epoch * len(self.data_loader['train'])
            warmup_steps = int(total_steps * 0.1)  # 10% warmup
            
            # Learning rate scheduler function for better convergence
            def lr_scheduler(step):
                if step < warmup_steps:
                    # Linear warmup
                    return self.arg.base_lr * (step / warmup_steps)
                else:
                    # Cosine decay
                    decay_steps = total_steps - warmup_steps
                    step_offset = step - warmup_steps
                    return self.arg.base_lr * 0.5 * (1 + tf.cos(
                        3.14159 * step_offset / decay_steps
                    ))
            
            # Create learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()
            lr_schedule.__call__ = lr_scheduler
            
            # Create optimizer based on configuration
            if self.arg.optimizer.lower() == "adam":
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    beta_1=0.9, 
                    beta_2=0.999
                )
            elif self.arg.optimizer.lower() == "adamw":
                self.optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=self.arg.weight_decay,
                    beta_1=0.9, 
                    beta_2=0.999
                )
            elif self.arg.optimizer.lower() == "sgd":
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=lr_schedule,
                    momentum=0.9
                )
            else:
                self.print_log(f"Unknown optimizer: {self.arg.optimizer}, using Adam")
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule
                )
            
            self.print_log(f"Optimizer: {self.optimizer.__class__.__name__}, base_lr={self.arg.base_lr}")
            return True
        except Exception as e:
            self.print_log(f"Error loading optimizer: {e}")
            return False

    def load_loss(self):
        """Load loss function with additional regularization to combat high validation loss"""
        try:
            if not hasattr(self, 'pos_weights') or self.pos_weights is None:
                self.pos_weights = tf.constant(1.0)
            
            # Create focal loss for better handling of imbalanced data
            def weighted_focal_bce(y_true, y_pred, gamma=2.0):
                y_true = tf.cast(y_true, tf.float32)
                
                # Handle dimension mismatches
                if len(y_pred.shape) > 1 and y_pred.shape[-1] == 1:
                    if len(y_true.shape) == 1:
                        y_true = tf.expand_dims(y_true, -1)
                elif len(y_pred.shape) == 1 and len(y_true.shape) > 1:
                    y_pred = tf.expand_dims(y_pred, -1)
                
                if y_true.shape != y_pred.shape:
                    batch_size = tf.shape(y_true)[0]
                    y_true = tf.reshape(y_true, [batch_size, 1])
                    y_pred = tf.reshape(y_pred, [batch_size, 1])
                
                # Calculate binary cross entropy
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
                
                # Apply focal weighting
                prob = tf.sigmoid(y_pred)
                p_t = tf.where(tf.equal(y_true, 1), prob, 1-prob)
                focal_weight = tf.pow(1-p_t, gamma)
                
                # Apply class weighting
                class_weight = y_true * (self.pos_weights - 1.0) + 1.0
                
                # Combine weights
                weights = focal_weight * class_weight
                
                return tf.reduce_mean(weights * bce)
            
            self.criterion = weighted_focal_bce
            self.print_log(f"Using focal BCE loss with pos_weight={self.pos_weights.numpy():.4f}, gamma=2.0")
            return True
        except Exception as e:
            self.print_log(f"Error loading loss function: {e}")
            # Fallback to standard BCE loss
            def weighted_bce(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
                weights = y_true * (self.pos_weights - 1.0) + 1.0
                return tf.reduce_mean(weights * bce)
            self.criterion = weighted_bce
            self.print_log(f"Using standard BCE loss with pos_weight={self.pos_weights.numpy():.4f}")
            return False
    
    def load_data(self):
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
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
                
                try:
                    self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, use_smv=use_smv)
                except TypeError as e:
                    if "unexpected keyword argument 'use_smv'" in str(e):
                        self.print_log("Data loader doesn't support use_smv parameter, using default")
                        self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size)
                    else:
                        raise
                
                try:
                    sample_batch = next(iter(self.data_loader['train']))
                    self.print_log(f"Train data sample shapes: {[(k, v.shape) for k, v in sample_batch[0].items()]}")
                    self.print_log(f"Train labels shape: {sample_batch[1].shape}")
                except Exception as e:
                    self.print_log(f"Warning: Could not check train data shapes: {e}")
                
                self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
                try:
                    self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
                except Exception as e:
                    self.print_log(f"Error visualizing distribution: {e}")
                
                if self.val_subject:
                    self.print_log(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        self.print_log("Warning: Validation data is empty, using subset of training data")
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)
                        self.norm_val = {k: v[-val_size:].copy() for k, v in self.norm_train.items()}
                        self.norm_train = {k: v[:-val_size].copy() for k, v in self.norm_train.items()}
                    
                    try:
                        self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size, use_smv=use_smv)
                    except TypeError as e:
                        if "unexpected keyword argument 'use_smv'" in str(e):
                            self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size)
                        else:
                            raise
                    
                    try:
                        self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                    except Exception as e:
                        self.print_log(f"Error visualizing distribution: {e}")
                
                if self.test_subject:
                    self.print_log(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        self.print_log("Warning: Test data is empty")
                        return False
                    
                    try:
                        self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv)
                    except TypeError as e:
                        if "unexpected keyword argument 'use_smv'" in str(e):
                            self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size)
                        else:
                            raise
                    
                    subject_id = self.test_subject[0] if self.test_subject else 'unknown'
                    try:
                        self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_{subject_id}')
                    except Exception as e:
                        self.print_log(f"Error visualizing distribution: {e}")
                
                self.print_log("Data loading complete")
                return True
                
            elif self.arg.phase == 'test':
                if not self.test_subject:
                    self.print_log("No test subjects specified")
                    return False
                
                builder = prepare_smartfallmm_tf(self.arg)
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                if any(len(x) == 0 for x in self.norm_test.values()):
                    self.print_log("Error: Test data is empty")
                    return False
                
                try:
                    self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv)
                except TypeError as e:
                    if "unexpected keyword argument 'use_smv'" in str(e):
                        self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size)
                    else:
                        raise
                
                self.print_log("Test data loading complete")
                return True
        
        except Exception as e:
            self.print_log(f"Error loading data: {e}")
            return False
    
    def distribution_viz(self, labels, work_dir, mode):
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
    
    def calculate_metrics(self, targets, predictions, probabilities=None):
        try:
            if isinstance(targets, tf.Tensor):
                targets = targets.numpy()
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            if probabilities is not None and isinstance(probabilities, tf.Tensor):
                probabilities = probabilities.numpy()
            
            targets = np.array(targets).flatten()
            predictions = np.array(predictions).flatten()
            
            # Calculate standard metrics with binary predictions
            accuracy = accuracy_score(targets, predictions) * 100
            precision = precision_score(targets, predictions, zero_division=0) * 100
            recall = recall_score(targets, predictions, zero_division=0) * 100
            f1 = f1_score(targets, predictions, zero_division=0) * 100
            
            # Calculate AUC with probability scores if available
            if probabilities is not None:
                probabilities = np.array(probabilities).flatten()
                try:
                    # Check if we have both classes present
                    if len(np.unique(targets)) > 1:
                        auc = roc_auc_score(targets, probabilities) * 100
                    else:
                        # Only one class present, AUC is undefined
                        auc = np.nan
                except Exception as e:
                    self.print_log(f"AUC calculation error: {e}")
                    auc = np.nan
            else:
                auc = np.nan
            
            return accuracy, f1, recall, precision, auc
        except Exception as e:
            self.print_log(f"Error calculating metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    def train(self, epoch):
        try:
            self.print_log(f"Starting epoch {epoch+1} training")
            start_time = time.time()
            
            loader = self.data_loader['train']
            total_batches = len(loader)
            num_samples = total_batches * self.arg.batch_size
            
            self.print_log(f"Epoch {epoch+1}/{self.arg.num_epoch} - {total_batches} batches (~{num_samples} samples)")
            
            train_loss = 0.0
            all_labels = []
            all_preds = []
            steps = 0
            
            for batch_idx in range(total_batches):
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Training epoch {epoch+1}: batch {batch_idx+1}/{total_batches}")
                
                try:
                    inputs, targets, batch_indices = loader[batch_idx]
                    
                    if batch_idx == 0:
                        for key, value in inputs.items():
                            self.print_log(f"First batch {key} shape: {value.shape}")
                        self.print_log(f"First batch labels shape: {targets.shape}")
                        self.print_log(f"First batch indices shape: {batch_indices.shape}")
                        for key, value in inputs.items():
                            self.print_log(f"Input '{key}' shape: {value.shape}")
                    
                    targets = tf.cast(targets, tf.float32)
                    
                    if isinstance(inputs, dict) and 'accelerometer' in inputs:
                        acc_data = inputs['accelerometer']
                    else:
                        self.print_log(f"Error: No accelerometer data in batch {batch_idx}")
                        continue
                    
                    with tf.GradientTape() as tape:
                        outputs = self.model(inputs, training=True)
                        
                        if isinstance(outputs, tuple) and len(outputs) > 0:
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        if len(logits.shape) > 1 and logits.shape[-1] > 0:
                            if len(targets.shape) == 1:
                                targets = tf.reshape(targets, [-1, 1])
                            loss = self.criterion(targets, logits)
                        else:
                            batch_size = tf.shape(acc_data)[0]
                            targets_reshaped = tf.reshape(targets, [batch_size, 1])
                            logits_reshaped = tf.reshape(logits, [batch_size, 1])
                            loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    
                    has_nan = False
                    for grad in gradients:
                        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                            has_nan = True
                            break
                    
                    if has_nan:
                        self.print_log(f"WARNING: NaN gradients detected in batch {batch_idx}")
                        continue
                    
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                except Exception as e:
                    self.print_log(f"Error in batch {batch_idx}: {e}")
                    continue
            
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                self.train_loss_summary.append(float(train_loss))
                
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
                
                self.print_log(f"Running validation for epoch {epoch+1}")
                val_loss = self.eval(epoch, loader_name='val')
                
                self.val_loss_summary.append(float(val_loss))
                
                if self.early_stop(val_loss):
                    self.print_log(f"Early stopping triggered at epoch {epoch+1}")
                    return True
                
                return False
            else:
                self.print_log(f"Warning: No steps completed in epoch {epoch+1}")
                return False
                
        except Exception as e:
            self.print_log(f"Critical error in epoch {epoch+1}: {e}")
            return False
    
    def eval(self, epoch, loader_name='val', result_file=None):
        try:
            start_time = time.time()
            
            loader = self.data_loader.get(loader_name)
            if loader is None:
                self.print_log(f"No data loader for {loader_name}")
                return float('inf')
            
            total_batches = len(loader)
            num_samples = total_batches * (
                self.arg.val_batch_size if loader_name == 'val' else self.arg.test_batch_size
            )
            
            self.print_log(f"Evaluating {loader_name} (epoch {epoch+1}) - {total_batches} batches (~{num_samples} samples)")
            
            eval_loss = 0.0
            all_labels = []
            all_preds = []
            all_logits = []
            steps = 0
            
            for batch_idx in range(total_batches):
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Eval {loader_name} (epoch {epoch+1}): batch {batch_idx+1}/{total_batches}")
                
                try:
                    inputs, targets, _ = loader[batch_idx]
                    
                    if batch_idx == 0:
                        for key, value in inputs.items():
                            self.print_log(f"First batch {key} shape: {value.shape}")
                        self.print_log(f"First batch labels shape: {targets.shape}")
                    
                    targets = tf.cast(targets, tf.float32)
                    
                    outputs = self.model(inputs, training=False)
                    
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    batch_logits = logits.numpy()
                    all_logits.append(batch_logits)
                    
                    if len(logits.shape) > 1 and logits.shape[-1] > 0:
                        if len(targets.shape) == 1:
                            targets = tf.reshape(targets, [-1, 1])
                        loss = self.criterion(targets, logits)
                    else:
                        batch_size = tf.shape(inputs['accelerometer'])[0]
                        targets_reshaped = tf.reshape(targets, [batch_size, 1])
                        logits_reshaped = tf.reshape(logits, [batch_size, 1])
                        loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    eval_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                except Exception as e:
                    self.print_log(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
            
            if steps > 0:
                eval_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                self.print_log(
                    f"{loader_name.capitalize()}: "
                    f"Loss={eval_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}%"
                )
                
                epoch_time = time.time() - start_time
                self.print_log(f"{loader_name.capitalize()} time: {epoch_time:.2f}s")
                
                if loader_name == 'val':
                    is_best = False
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        is_best = True
                        self.print_log(f"New best validation loss: {eval_loss:.4f}")
                    
                    if is_best:
                        self.save_model(epoch)
                
                elif loader_name.startswith('test'):
                    self.test_accuracy = accuracy
                    self.test_f1 = f1
                    self.test_recall = recall
                    self.test_precision = precision
                    self.test_auc = auc_score
                    
                    subject_id = self.test_subject[0] if self.test_subject else None
                    if subject_id:
                        self.cm_viz(all_preds, all_labels, subject_id)
                    
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
            return float('inf')
    
    def cm_viz(self, y_pred, y_true, subject_id=None):
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
    
    def loss_viz(self, train_loss, val_loss, subject_id=None):
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
    
    def save_model(self, epoch):
        try:
            if self.test_subject:
                base_filename = f"{self.model_path}_{self.test_subject[0]}"
            else:
                base_filename = f"{self.model_path}_epoch{epoch}"
            
            # Save weights
            weights_path = f"{base_filename}.weights.h5"
            self.model.save_weights(weights_path)
            self.print_log(f"Saved model weights to {weights_path}")
            
            # Save full model (if possible)
            try:
                model_path = f"{base_filename}.keras"
                self.model.save(model_path)
                self.print_log(f"Saved full model to {model_path}")
            except Exception as e:
                self.print_log(f"Warning: Could not save full model: {e}")
            
            return True
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            return False
    
    def evaluate_test_set(self, epoch=0):
        try:
            model_training = self.model.trainable
            self.model.trainable = False
            
            loader = self.data_loader.get('test')
            if loader is None:
                self.print_log("No test data loader available")
                return None
            
            total_batches = len(loader)
            num_samples = total_batches * self.arg.test_batch_size
            
            subject_id = self.test_subject[0] if self.test_subject else "unknown"
            self.print_log(f"Testing subject {subject_id} - {total_batches} batches (~{num_samples} samples)")
            
            test_loss = 0.0
            all_labels = []
            all_preds = []
            all_logits = []
            steps = 0
            
            for batch_idx in range(total_batches):
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Test batch {batch_idx+1}/{total_batches}")
                
                try:
                    inputs, targets, _ = loader[batch_idx]
                    
                    targets = tf.cast(targets, tf.float32)
                    
                    outputs = self.model(inputs, training=False)
                    
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    all_logits.append(logits.numpy())
                    
                    if len(logits.shape) > 1 and logits.shape[-1] > 0:
                        if len(targets.shape) == 1:
                            targets = tf.reshape(targets, [-1, 1])
                        loss = self.criterion(targets, logits)
                    else:
                        batch_size = tf.shape(inputs['accelerometer'])[0]
                        targets_reshaped = tf.reshape(targets, [batch_size, 1])
                        logits_reshaped = tf.reshape(logits, [batch_size, 1])
                        loss = self.criterion(targets_reshaped, logits_reshaped)
                    
                    if len(logits.shape) > 1 and logits.shape[-1] > 1:
                        predictions = tf.argmax(logits, axis=-1)
                    else:
                        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                    
                    test_loss += loss.numpy()
                    all_labels.extend(targets.numpy())
                    all_preds.extend(predictions.numpy())
                    steps += 1
                    
                except Exception as e:
                    self.print_log(f"Error in test batch {batch_idx}: {e}")
                    continue
            
            if steps > 0:
                test_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
                
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
                
                self.print_log(
                    f"Test results for Subject {subject_id}: "
                    f"Loss={test_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_score:.2f}%"
                )
                
                self.cm_viz(all_preds, all_labels, subject_id)
                
                if len(all_logits) > 0:
                    self.print_log(f"Model logits sample: {all_logits[0].flatten()[:5]}")
                
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
                
                self.model.trainable = model_training
                
                return results
            else:
                self.print_log(f"No valid test steps completed for subject {subject_id}")
                self.model.trainable = model_training
                return None
                
        except Exception as e:
            self.print_log(f"Error in test evaluation: {e}")
            if 'model_training' in locals():
                self.model.trainable = model_training
            return None
    
    def add_avg_df(self, results):
        if not results:
            return results
        
        avg_results = {key: 0 for key in results[0].keys() if key != 'test_subject'}
        for result in results:
            for key, value in result.items():
                if key != 'test_subject':
                    avg_results[key] += value
        
        avg_count = len(results)
        if avg_count > 0:
            for key in avg_results:
                avg_results[key] = round(avg_results[key] / avg_count, 2)
            
            avg_results['test_subject'] = 'Average'
            results.append(avg_results)
        
        return results
    
    def start(self):
        max_total_time = 24 * 3600
        total_start_time = time.time()
        
        try:
            if self.arg.phase == 'train':
                self.print_log('Starting training with parameters:')
                for key, value in vars(self.arg).items():
                    self.print_log(f'  {key}: {value}')
                
                results = []
                val_subjects = [38, 46]
                
                for test_subject in self.arg.subjects:
                    if test_subject in val_subjects:
                        continue
                    
                    try:
                        if time.time() - total_start_time > max_total_time:
                            self.print_log("Maximum total training time exceeded, stopping")
                            break
                        
                        # Reset state for each fold
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                        self.best_loss = float('inf')
                        self.data_loader = {}
                        
                        # Set up subjects for this fold
                        self.test_subject = [test_subject]
                        self.val_subject = val_subjects
                        self.train_subjects = [s for s in self.arg.subjects 
                                            if s != test_subject and s not in val_subjects]
                        
                        self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                        self.print_log(f"Train: {len(self.train_subjects)} subjects: {self.train_subjects}")
                        self.print_log(f"Val: {len(self.val_subject)} subjects: {self.val_subject}")
                        self.print_log(f"Test: Subject {test_subject}")
                        
                        # Create fresh model for each fold
                        try:
                            # Clear old model resources
                            tf.keras.backend.clear_session()
                            # Recreate model
                            self.model = self.load_model()
                            self.print_log(f"Model loaded successfully for subject {test_subject}")
                        except Exception as model_error:
                            self.print_log(f"Failed to load model for subject {test_subject}: {model_error}")
                            continue
                        
                        # Load data for this fold
                        try:
                            if not self.load_data():
                                self.print_log(f"Skipping subject {test_subject} due to data loading issues")
                                continue
                            self.print_log(f"Data loaded successfully for subject {test_subject}")
                        except Exception as data_error:
                            self.print_log(f"Failed to load data for subject {test_subject}: {data_error}")
                            continue
                        
                        # Set up optimizer and loss
                        try:
                            if not self.load_optimizer() or not self.load_loss():
                                self.print_log(f"Skipping subject {test_subject} due to optimizer/loss issues")
                                continue
                            self.print_log(f"Optimizer and loss initialized for subject {test_subject}")
                        except Exception as opt_error:
                            self.print_log(f"Failed to initialize optimizer/loss: {opt_error}")
                            continue
                        
                        # Reset early stopping
                        self.early_stop.reset()
                        
                        # Train for this fold
                        subject_start_time = time.time()
                        max_subject_time = 8 * 3600
                        self.print_log(f"Starting training for subject {test_subject}")
                        
                        # Training loop
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            if time.time() - subject_start_time > max_subject_time:
                                self.print_log(f"Maximum time per subject exceeded for subject {test_subject}")
                                break
                            
                            try:
                                early_stop = self.train(epoch)
                                if early_stop:
                                    self.print_log(f"Early stopping at epoch {epoch+1}")
                                    break
                            except Exception as epoch_error:
                                self.print_log(f"Error in epoch {epoch+1}: {epoch_error}")
                                if epoch == 0:
                                    self.print_log(f"First epoch failed, skipping subject {test_subject}")
                                    break
                                continue
                        
                        # Load best weights for evaluation
                        best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                        if os.path.exists(best_weights):
                            try:
                                self.model.load_weights(best_weights)
                                self.print_log(f"Loaded best weights from {best_weights}")
                            except Exception as weight_error:
                                self.print_log(f"Error loading best weights: {weight_error}")
                        
                        # Evaluate on test set
                        self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                        result = self.evaluate_test_set()
                        
                        # Visualize results
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                        
                        # Save results
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
                            self.print_log(f"Completed fold for subject {test_subject}")
                        
                        # Clean up resources
                        self.data_loader = {}
                        tf.keras.backend.clear_session()
                        
                    except Exception as subject_error:
                        self.print_log(f"Fatal error processing subject {test_subject}: {subject_error}")
                        tf.keras.backend.clear_session()
                        continue
                
                # Generate final summary report
                if results:
                    try:
                        results = self.add_avg_df(results)
                        
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)
                        
                        with open(os.path.join(self.arg.work_dir, 'scores.json'), 'w') as f:
                            json.dump(results, f, indent=2)
                        
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
                
                self.print_log("Training completed successfully")
                
            elif self.arg.phase == 'test':
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
            
        except Exception as e:
            self.print_log(f"Fatal error in training process: {e}")
