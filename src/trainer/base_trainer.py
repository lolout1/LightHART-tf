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
    
    def __call__(self, val_loss):
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
        self.counter = 0
        self.wait = 0
        self.best_loss = None
        self.early_stop = False

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
        
        self.model_path = os.path.join(
            self.arg.work_dir, 
            'models', 
            self.arg.model_saved_name
        )
    
    def import_class(self, import_str):
        if import_str is None:
            raise ValueError("Model path cannot be None")
            
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
        if self.arg.phase == 'train':
            if self.arg.model is None:
                raise ValueError("Model class path is required")
                
            model_class = self.import_class(self.arg.model)
            model = model_class(**self.arg.model_args)
            self.print_log(f"Created model: {self.arg.model}")
            
            try:
                if 'accelerometer' in self.arg.dataset_args['modalities']:
                    acc_frames = self.arg.model_args.get('acc_frames', 128)
                    acc_coords = self.arg.model_args.get('acc_coords', 3)
                    
                    dummy_input = {
                        'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                    }
                    
                    if 'skeleton' in self.arg.dataset_args['modalities']:
                        dummy_input['skeleton'] = tf.zeros((2, acc_frames, 32, 3), dtype=tf.float32)
                    
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
                    acc_coords = self.arg.model_args.get('acc_coords', 3)
                    dummy_input = {'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)}
                    if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                        dummy_input['skeleton'] = tf.zeros((2, acc_frames, 32, 3), dtype=tf.float32)
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
        if not hasattr(self, 'pos_weights') or self.pos_weights is None:
            self.pos_weights = tf.constant(1.0)
        
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * (self.pos_weights - 1.0) + 1.0
            return tf.reduce_mean(weights * bce)
        
        self.criterion = weighted_bce
        self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")
    
    def load_data(self):
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
            
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
                    
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size
                )
                
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
                        
                        self.norm_val = {
                            k: v[-val_size:].copy() for k, v in self.norm_train.items()
                        }
                        self.norm_train = {
                            k: v[:-val_size].copy() for k, v in self.norm_train.items()
                        }
                    
                    self.data_loader['val'] = Feeder(
                        dataset=self.norm_val,
                        batch_size=self.arg.val_batch_size
                    )
                    
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
                        
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size
                    )
                    
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
    
    def calculate_metrics(self, targets, predictions):
        if isinstance(targets, tf.Tensor):
            targets = targets.numpy()
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()
            
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        
        accuracy = accuracy_score(targets, predictions) * 100
        
        unique_targets = np.unique(targets)
        unique_preds = np.unique(predictions)
        
        if len(unique_targets) <= 1 or len(unique_preds) <= 1:
            if len(unique_targets) == 1 and len(unique_preds) == 1 and unique_targets[0] == unique_preds[0]:
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
                tp = np.sum((predictions == 1) & (targets == 1))
                fp = np.sum((predictions == 1) & (targets == 0))
                fn = np.sum((predictions == 0) & (targets == 1))
                
                precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                auc = 50.0
        else:
            precision = precision_score(targets, predictions, zero_division=0) * 100
            recall = recall_score(targets, predictions, zero_division=0) * 100
            f1 = f1_score(targets, predictions, zero_division=0) * 100
            try:
                auc = roc_auc_score(targets, predictions) * 100
            except:
                auc = 50.0
        
        return accuracy, f1, recall, precision, auc
    
    def train(self, epoch):
        start_time = time.time()
        
        # Get actual number of batches by iterating through data loader once
        loader = self.data_loader['train']
        num_samples = 0
        for _, _, indices in loader:
            num_samples += len(indices)
        
        # Calculate true number of batches
        batch_size = self.arg.batch_size
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        # Reset data loader
        loader = self.data_loader['train']
        
        train_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        self.print_log(f"Starting epoch {epoch+1}/{self.arg.num_epoch} - {total_batches} batches ({num_samples} samples)")
        
        # Training loop
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 20 == 0 or batch_idx + 1 == total_batches:
                self.print_log(f"Epoch {epoch+1}: batch {batch_idx+1}/{total_batches}")
                
            targets = tf.cast(targets, tf.float32)
            
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    logits = outputs
                
                if len(logits.shape) > 1 and logits.shape[-1] > 1:
                    loss = self.criterion(targets, logits)
                else:
                    loss = self.criterion(targets, tf.squeeze(logits))
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                predictions = tf.argmax(logits, axis=-1)
            else:
                predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
            
            train_loss += loss.numpy()
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
            
            # Don't continue after we've seen all samples
            if batch_idx + 1 >= total_batches:
                break
        
        # Calculate average loss and metrics
        train_loss /= steps
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
        
        self.train_loss_summary.append(float(train_loss))
        
        epoch_time = time.time() - start_time
        
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
        
        val_loss = self.eval(epoch, loader_name='val')
        
        self.val_loss_summary.append(float(val_loss))
        
        self.early_stop(val_loss)
    
    def eval(self, epoch, loader_name='val', result_file=None):
        start_time = time.time()
        
        loader = self.data_loader.get(loader_name)
        if loader is None:
            self.print_log(f"No data loader for {loader_name}")
            return float('inf')
        
        # Get actual number of batches by iterating through data loader once
        num_samples = 0
        for _, _, indices in loader:
            num_samples += len(indices)
        
        # Calculate true number of batches
        batch_size = self.arg.val_batch_size if loader_name == 'val' else self.arg.test_batch_size
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        # Reset data loader
        loader = self.data_loader[loader_name]
        
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        self.print_log(f"Evaluating {loader_name} (epoch {epoch+1}) - {total_batches} batches ({num_samples} samples)")
        
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                self.print_log(f"Eval {loader_name} (epoch {epoch+1}): batch {batch_idx+1}/{total_batches}")
                
            targets = tf.cast(targets, tf.float32)
            
            outputs = self.model(inputs, training=False)
            
            if isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                logits = outputs
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                loss = self.criterion(targets, logits)
            else:
                loss = self.criterion(targets, tf.squeeze(logits))
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                predictions = tf.argmax(logits, axis=-1)
            else:
                predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
            
            eval_loss += loss.numpy()
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
            
            # Don't continue after we've seen all samples
            if batch_idx + 1 >= total_batches:
                break
        
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
                    "loss": float(eval_loss)
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
            
            weights_path = f"{base_filename}.weights.h5"
            self.model.save_weights(weights_path)
            self.print_log(f"Saved model weights to {weights_path}")
            
            model_path = f"{base_filename}.keras"
            try:
                self.model.save(model_path)
                self.print_log(f"Saved full model to {model_path}")
            except Exception as e:
                self.print_log(f"Warning: Could not save full model: {e}")
            
            try:
                # Create input
                acc_frames = self.arg.model_args.get('acc_frames', 128)
                acc_coords = self.arg.model_args.get('acc_coords', 3)
                
                # Create standalone model for TFLite
                inputs = tf.keras.Input(shape=(acc_frames, acc_coords), name='input')
                
                # Calculate SMV
                mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
                zero_mean = inputs - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                smv = tf.sqrt(sum_squared)
                processed = tf.concat([smv, inputs], axis=-1)
                
                # Create model inputs dict
                model_inputs = {'accelerometer': processed}
                
                # Forward pass
                outputs = self.model(model_inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Create new model
                tflite_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # Save TFLite model
                tflite_path = f"{base_filename}.tflite"
                converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
                tflite_buffer = converter.convert()
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_buffer)
                
                self.print_log(f"Exported TFLite model to {tflite_path}")
            except Exception as e:
                self.print_log(f"Warning: TFLite export failed: {e}")
            
            return True
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            traceback.print_exc()
            return False
    
    def add_avg_df(self, results):
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
        model_training = self.model.trainable
        self.model.trainable = False
        
        # Get test data loader
        loader = self.data_loader.get('test')
        if loader is None:
            self.print_log("No test data loader available")
            return
        
        # Get actual number of batches by iterating through data loader once
        num_samples = 0
        for _, _, indices in loader:
            num_samples += len(indices)
        
        # Calculate true number of batches
        batch_size = self.arg.test_batch_size
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        # Reset data loader
        loader = self.data_loader['test']
        
        subject_id = self.test_subject[0] if self.test_subject else "unknown"
        self.print_log(f"Testing subject {subject_id} - {total_batches} batches ({num_samples} samples)")
        
        test_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                self.print_log(f"Test batch {batch_idx+1}/{total_batches}")
                
            targets = tf.cast(targets, tf.float32)
            
            outputs = self.model(inputs, training=False)
            
            if isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                logits = outputs
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                loss = self.criterion(targets, logits)
            else:
                loss = self.criterion(targets, tf.squeeze(logits))
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                predictions = tf.argmax(logits, axis=-1)
            else:
                predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
            
            test_loss += loss.numpy()
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
            
            # Don't continue after we've seen all samples
            if batch_idx + 1 >= total_batches:
                break
        
        # Calculate metrics
        test_loss /= steps
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
        
        # Store metrics
        self.test_accuracy = accuracy
        self.test_f1 = f1
        self.test_recall = recall
        self.test_precision = precision
        self.test_auc = auc_score
        
        # Log results
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
        
        # Save results
        results = {
            "subject": subject_id,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc_score),
            "loss": float(test_loss)
        }
        
        results_file = os.path.join(
            self.arg.work_dir,
            'results',
            f'test_results_{subject_id}.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Restore model training state
        self.model.trainable = model_training
        
        return results
    
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            
            results = []
            
            val_subjects = [38, 46]
            
            for test_subject in self.arg.subjects:
                if test_subject in val_subjects:
                    continue
                
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects 
                                      if s != test_subject and s not in val_subjects]
                
                self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                self.print_log(f"Train: {len(self.train_subjects)} subjects")
                self.print_log(f"Val: {len(self.val_subject)} subjects")
                self.print_log(f"Test: Subject {test_subject}")
                
                self.model = self.load_model()
                
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                self.load_optimizer()
                self.load_loss()
                
                self.early_stop = EarlyStopping(patience=15, min_delta=.001)
                
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    
                    if self.early_stop.early_stop:
                        self.print_log(f"Early stopping at epoch {epoch+1}")
                        break
                
                best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                if os.path.exists(best_weights):
                    self.model.load_weights(best_weights)
                    self.print_log(f"Loaded best weights from {best_weights}")
                
                self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                self.evaluate_test_set()
                
                self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                
                subject_result = {
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                
                results.append(subject_result)
                
                tf.keras.backend.clear_session()
            
            if results:
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
            
            self.print_log("Training completed successfully")
