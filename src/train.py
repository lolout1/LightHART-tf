import os
import sys
import logging
import argparse
import yaml
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from collections import Counter
import shutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('training_debug.log')])
logger = logging.getLogger('lightheart-tf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EarlyStopping:
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
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_auc = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.data_loader = {}
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.arg.work_dir:
            self.arg.work_dir = f"{self.arg.work_dir}_{timestamp}"
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        self.model_path = os.path.join(self.arg.work_dir, 'models', self.arg.model_saved_name)
        self.setup_logging()
        self.save_config(arg.config, arg.work_dir)
        self.setup_cross_validation()
        self.model = None
        self.print_log("Trainer initialized successfully")
        self.print_log(f"Working directory: {self.arg.work_dir}")
    def setup_logging(self):
        log_file = os.path.join(self.arg.work_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    def setup_cross_validation(self):
        if hasattr(self.arg, 'subjects') and self.arg.subjects is not None:
            self.fixed_val_subjects = getattr(self.arg, 'val_subjects_fixed', None) or [38, 46]
            self.fixed_train_subjects = getattr(self.arg, 'train_subjects_fixed', None) or [45, 36, 29]
            if self.fixed_val_subjects is None:
                self.fixed_val_subjects = [38, 46]
            if self.fixed_train_subjects is None:
                self.fixed_train_subjects = [45, 36, 29]
            self.test_eligible_subjects = [s for s in self.arg.subjects if s not in self.fixed_val_subjects and s not in self.fixed_train_subjects]
            self.total_folds = len(self.test_eligible_subjects)
            self.print_log("=== Cross-Validation Setup ===")
            self.print_log(f"Total subjects: {self.arg.subjects}")
            self.print_log(f"Fixed validation subjects: {self.fixed_val_subjects}")
            self.print_log(f"Fixed training subjects: {self.fixed_train_subjects}")
            self.print_log(f"Test eligible subjects: {self.test_eligible_subjects}")
            self.print_log(f"Total folds: {self.total_folds}")
            self.print_log("============================")
        else:
            self.total_folds = 1
            self.fixed_val_subjects = []
            self.fixed_train_subjects = []
            self.test_eligible_subjects = []
            self.print_log("No cross-validation setup (subjects not specified)")
    def save_config(self, src_path, dest_path):
        config_dest = os.path.join(dest_path, 'config')
        os.makedirs(config_dest, exist_ok=True)
        if os.path.exists(src_path):
            dest_file = os.path.join(config_dest, os.path.basename(src_path))
            if os.path.abspath(src_path) != os.path.abspath(dest_file):
                try:
                    shutil.copy(src_path, dest_file)
                    self.print_log(f"Config saved to: {dest_file}")
                except Exception as e:
                    self.print_log(f"Error copying config: {e}")
            else:
                self.print_log(f"Config already exists at destination: {dest_file}")
        else:
            self.print_log(f"Warning: Config file not found: {src_path}")
    def count_parameters(self):
        if self.model is None:
            return 0
        return sum(np.prod(v.shape.as_list()) for v in self.model.trainable_variables)
    def import_class(self, import_str):
        mod_str, _, class_str = import_str.rpartition('.')
        import importlib
        try:
            module = importlib.import_module(mod_str)
            return getattr(module, class_str)
        except Exception as e:
            self.print_log(f"Error importing class {import_str}: {e}")
            raise
    def load_model(self):
        self.print_log(f"Loading model: {self.arg.model}")
        try:
            model_class = self.import_class(self.arg.model)
            model = model_class(**self.arg.model_args)
            acc_frames = self.arg.model_args.get('acc_frames', 64)
            acc_coords = self.arg.model_args.get('acc_coords', 3)
            if hasattr(self.arg, 'use_smv') and self.arg.use_smv:
                acc_coords += 1
            dummy_input = {'accelerometer': tf.zeros((1, acc_frames, acc_coords))}
            if hasattr(self.arg, 'dataset_args') and 'skeleton' in self.arg.dataset_args.get('modalities', []):
                mocap_frames = self.arg.model_args.get('mocap_frames', 64)
                num_joints = self.arg.model_args.get('num_joints', 32)
                dummy_input['skeleton'] = tf.zeros((1, mocap_frames, num_joints, 3))
            _ = model(dummy_input, training=False)
            self.print_log(f"Model loaded successfully with {self.count_parameters()} parameters")
            return model
        except Exception as e:
            self.print_log(f"Error loading model: {e}")
            raise
    def calculate_class_weights(self, labels):
        from collections import Counter
        counter = Counter(labels)
        self.print_log(f"Label distribution: {dict(counter)}")
        if 0 not in counter or 1 not in counter:
            self.print_log("Warning: Not all classes present in training data!")
            return tf.constant(1.0, dtype=tf.float32)
        pos_weight = counter[0] / counter[1]
        self.print_log(f'Class weights - neg: {counter[0]}, pos: {counter[1]}, pos_weight: {pos_weight:.4f}')
        return tf.constant(pos_weight, dtype=tf.float32)
    def load_optimizer(self):
        base_lr = self.arg.base_lr
        self.print_log(f"Loading optimizer: {self.arg.optimizer} with lr={base_lr}")
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")
    def load_loss(self):
        from utils.loss import BinaryFocalLoss
        self.pos_weights = getattr(self, 'pos_weights', tf.constant(1.0))
        self.print_log(f"Loading loss function with pos_weight: {self.pos_weights.numpy()}")
        self.criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    def load_data(self):
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
        self.print_log("=== Loading Data ===")
        self.print_log(f"Train subjects: {self.train_subjects}")
        self.print_log(f"Val subjects: {self.val_subject}")
        self.print_log(f"Test subjects: {self.test_subject}")
        try:
            Feeder = self.import_class(self.arg.feeder)
            if self.arg.dataset == 'smartfallmm':
                builder = prepare_smartfallmm_tf(self.arg)
            else:
                raise ValueError(f"Unsupported dataset: {self.arg.dataset}")
            if self.arg.phase == 'train':
                all_subjects = self.train_subjects + self.val_subject + self.test_subject
                self.print_log(f"Computing global statistics from {len(all_subjects)} subjects")
                all_data = split_by_subjects_tf(builder, all_subjects, False, compute_stats_only=True)
                self.acc_mean = all_data.get('acc_mean')
                self.acc_std = all_data.get('acc_std')
                self.skl_mean = all_data.get('skl_mean')
                self.skl_std = all_data.get('skl_std')
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False, acc_mean=self.acc_mean, acc_std=self.acc_std, skl_mean=self.skl_mean, skl_std=self.skl_std)
                if not self.norm_train or 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0:
                    self.print_log(f'ERROR: No training data for subjects {self.train_subjects}')
                    return False
                self.print_log(f"Training data loaded: {len(self.norm_train['labels'])} samples")
                self.norm_val = split_by_subjects_tf(builder, self.val_subject, False, acc_mean=self.acc_mean, acc_std=self.acc_std, skl_mean=self.skl_mean, skl_std=self.skl_std)
                if not self.norm_val or 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                    self.print_log(f'ERROR: No validation data for subjects {self.val_subject}')
                    return False
                self.print_log(f"Validation data loaded: {len(self.norm_val['labels'])} samples")
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False, acc_mean=self.acc_mean, acc_std=self.acc_std, skl_mean=self.skl_mean, skl_std=self.skl_std)
                if not self.norm_test or 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                    self.print_log(f'ERROR: No test data for subject {self.test_subject}')
                    return False
                self.print_log(f"Test data loaded: {len(self.norm_test['labels'])} samples")
                self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
                use_smv = getattr(self.arg, 'use_smv', False)
                window_size = self.arg.dataset_args.get('max_length', 64)
                self.print_log(f"Creating data loaders with batch_size={self.arg.batch_size}, use_smv={use_smv}, window_size={window_size}")
                self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, use_smv=use_smv, window_size=window_size)
                self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size, use_smv=use_smv, window_size=window_size)
                self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv, window_size=window_size)
                self.print_log(f"Train batches: {len(self.data_loader['train'])}")
                self.print_log(f"Val batches: {len(self.data_loader['val'])}")
                self.print_log(f"Test batches: {len(self.data_loader['test'])}")
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, f'train_s{self.test_subject[0]}')
                self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, f'val_s{self.test_subject[0]}')
                self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_s{self.test_subject[0]}')
                self.print_log("=== Data Loading Complete ===")
                return True
        except Exception as e:
            self.print_log(f"ERROR in load_data: {e}")
            import traceback
            self.print_log(traceback.format_exc())
            return False
    def distribution_viz(self, labels, work_dir, mode):
        vis_dir = os.path.join(work_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        unique, counts = np.unique(labels, return_counts=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        bars = ax1.bar(unique, counts, color=['blue', 'red'])
        ax1.set_xlabel('Labels')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{mode} Label Distribution')
        ax1.set_xticks(unique)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{count} ({count/sum(counts)*100:.1f}%)', ha='center', va='bottom')
        ax2.pie(counts, labels=unique, autopct='%1.1f%%', colors=['blue', 'red'])
        ax2.set_title(f'{mode} Label Proportion')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{mode}_label_distribution.png'), dpi=300)
        plt.close()
        stats_path = os.path.join(vis_dir, f'{mode}_distribution_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Distribution statistics for {mode}:\n")
            f.write(f"Total samples: {sum(counts)}\n")
            for label, count in zip(unique, counts):
                f.write(f"Label {label}: {count} ({count/sum(counts)*100:.2f}%)\n")
    def calculate_metrics(self, targets, predictions, probabilities=None):
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        self.print_log(f"Calculating metrics for {len(targets)} samples")
        self.print_log(f"Predictions - 0: {np.sum(predictions == 0)}, 1: {np.sum(predictions == 1)}")
        self.print_log(f"Targets - 0: {np.sum(targets == 0)}, 1: {np.sum(targets == 1)}")
        accuracy = accuracy_score(targets, predictions) * 100
        f1 = f1_score(targets, predictions, zero_division=0) * 100
        precision = precision_score(targets, predictions, zero_division=0) * 100
        recall = recall_score(targets, predictions, zero_division=0) * 100
        try:
            if probabilities is not None and len(np.unique(targets)) > 1:
                auc = roc_auc_score(targets, probabilities) * 100
            else:
                if len(np.unique(targets)) == 1:
                    self.print_log("Warning: Only one class in targets, AUC cannot be calculated")
                auc = 0.0
        except Exception as e:
            self.print_log(f"Warning: AUC calculation failed: {e}")
            auc = 0.0
        cm = confusion_matrix(targets, predictions)
        self.print_log(f"Confusion Matrix:\n{cm}")
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            self.print_log(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        return accuracy, f1, recall, precision, auc
    def train_step(self, inputs, targets):
        try:
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = self.criterion(targets, logits)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, logits
        except Exception as e:
            self.print_log(f"Error in train_step: {e}")
            raise
    def train(self, epoch):
        self.print_log(f'Starting Epoch: {epoch+1}/{self.arg.num_epoch}')
        loader = self.data_loader['train']
        train_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        start_time = time.time()
        from tqdm import tqdm
        progress = tqdm(range(len(loader)), ncols=80, desc=f'Train epoch {epoch+1}')
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                loss, logits = self.train_step(inputs, targets)
                probabilities = tf.sigmoid(logits)
                predictions = tf.cast(probabilities > 0.5, tf.int32)
                train_loss += loss.numpy()
                all_labels.extend(targets.numpy().flatten())
                all_preds.extend(predictions.numpy().flatten())
                all_probs.extend(probabilities.numpy().flatten())
                steps += 1
                progress.set_postfix({'loss': f'{loss.numpy():.4f}'})
            except Exception as e:
                self.print_log(f"Error in training batch {batch_idx}: {e}")
                continue
        if steps > 0:
            train_loss /= steps
            train_time = time.time() - start_time
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
            self.train_loss_summary.append(float(train_loss))
            self.print_log(f'Epoch {epoch+1} Training Results:')
            self.print_log(f'  Loss: {train_loss:.4f}')
            self.print_log(f'  Accuracy: {accuracy:.2f}%')
            self.print_log(f'  F1 Score: {f1:.2f}%')
            self.print_log(f'  Precision: {precision:.2f}%')
            self.print_log(f'  Recall: {recall:.2f}%')
            self.print_log(f'  AUC: {auc_score:.2f}%')
            self.print_log(f'  Time: {train_time:.2f}s')
            self.print_log(f'  Batches: {steps}/{len(loader)}')
        else:
            self.print_log("Warning: No valid training batches!")
            return True
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))
        self.early_stop(val_loss)
        if self.early_stop.early_stop:
            self.print_log(f"Early stopping triggered at epoch {epoch+1}")
            return True
        return False
    def eval(self, epoch, loader_name='val', result_file=None):
        self.print_log(f'Evaluating {loader_name} at epoch {epoch+1}')
        loader = self.data_loader[loader_name]
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        from tqdm import tqdm
        progress = tqdm(range(len(loader)), ncols=80, desc=f'{loader_name.capitalize()}')
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                outputs = self.model(inputs, training=False)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = self.criterion(targets, logits)
                probabilities = tf.sigmoid(logits)
                predictions = tf.cast(probabilities > 0.5, tf.int32)
                eval_loss += loss.numpy()
                all_labels.extend(targets.numpy().flatten())
                all_preds.extend(predictions.numpy().flatten())
                all_probs.extend(probabilities.numpy().flatten())
                steps += 1
                progress.set_postfix({'loss': f'{loss.numpy():.4f}'})
            except Exception as e:
                self.print_log(f"Error in evaluation batch {batch_idx}: {e}")
                continue
        if steps > 0:
            eval_loss /= steps
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
            self.print_log(f'{loader_name.capitalize()} Results:')
            self.print_log(f'  Loss: {eval_loss:.4f}')
            self.print_log(f'  Accuracy: {accuracy:.2f}%')
            self.print_log(f'  F1 Score: {f1:.2f}%')
            self.print_log(f'  Precision: {precision:.2f}%')
            self.print_log(f'  Recall: {recall:.2f}%')
            self.print_log(f'  AUC: {auc_score:.2f}%')
            self.print_log(f'  Batches: {steps}/{len(loader)}')
        else:
            self.print_log(f"Warning: No valid {loader_name} batches!")
            return float('inf')
        if result_file is not None:
            with open(result_file, 'w') as f:
                f.write(f"Predictions for {loader_name} epoch {epoch+1}\n")
                f.write("true,predicted,probability\n")
                for true, pred, prob in zip(all_labels, all_preds, all_probs):
                    f.write(f'{true},{pred},{prob:.4f}\n')
        if loader_name == 'val' and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.save_model()
            self.print_log(f'New best model saved with loss: {eval_loss:.4f}')
        if loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
            self.cm_viz(all_preds, all_labels)
        return eval_loss
    def save_model(self):
        try:
            weight_path = f'{self.model_path}_{self.test_subject[0]}.weights.h5'
            self.model.save_weights(weight_path)
            self.print_log(f"Model saved to: {weight_path}")
            file_size = os.path.getsize(weight_path)
            self.print_log(f"Saved model size: {file_size/1024/1024:.2f} MB")
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
    def load_weights(self):
        weight_path = f'{self.model_path}_{self.test_subject[0]}.weights.h5'
        if os.path.exists(weight_path):
            try:
                self.model.load_weights(weight_path)
                self.print_log(f"Weights loaded from: {weight_path}")
            except Exception as e:
                self.print_log(f"Error loading weights: {e}")
                raise
        else:
            self.print_log(f"Warning: Weight file not found: {weight_path}")
    def loss_viz(self, train_loss, val_loss):
        if not train_loss or not val_loss:
            self.print_log("Warning: No loss data to visualize")
            return
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'Training vs Validation Loss - Subject {self.test_subject[0]}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        loss_diff = np.array(val_loss) - np.array(train_loss)
        plt.plot(epochs, loss_diff, 'g-', label='Val-Train Difference', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Validation-Training Loss Difference')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        vis_path = os.path.join(self.arg.work_dir, 'visualizations', f'loss_curves_s{self.test_subject[0]}.png')
        plt.savefig(vis_path, dpi=300)
        plt.close()
        loss_data = pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss})
        loss_data.to_csv(os.path.join(self.arg.work_dir, 'visualizations', f'loss_data_s{self.test_subject[0]}.csv'), index=False)
    def cm_viz(self, y_pred, y_true):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Subject {self.test_subject[0]}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j+0.5, i+0.7, f'{cm_norm[i,j]:.1%}', ha='center', va='center', color='red', fontsize=10)
        vis_path = os.path.join(self.arg.work_dir, 'visualizations', f'confusion_matrix_s{self.test_subject[0]}.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
    def print_log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        log_file = os.path.join(self.arg.work_dir, 'training.log')
        with open(log_file, 'a') as f:
            f.write(full_message + '\n')
        logger.info(message)
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('=== Starting Training ===')
            self.print_log('Configuration:')
            self.print_log(yaml.dump(vars(self.arg), default_flow_style=False))
            results = []
            for fold_idx, test_subject in enumerate(self.test_eligible_subjects):
                fold_start_time = time.time()
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = self.fixed_val_subjects
                remaining_eligible = [s for s in self.test_eligible_subjects if s != test_subject]
                self.train_subjects = self.fixed_train_subjects + remaining_eligible
                self.print_log(f'\n{"="*60}')
                self.print_log(f'FOLD {fold_idx+1}/{self.total_folds}: Test Subject {test_subject}')
                self.print_log(f'Train: {self.train_subjects}')
                self.print_log(f'Val: {self.val_subject}')
                self.print_log(f'Test: {self.test_subject}')
                self.print_log(f'{"="*60}')
                try:
                    tf.keras.backend.clear_session()
                    self.model = self.load_model()
                    self.print_log(f'Model Parameters: {self.count_parameters()}')
                    if not self.load_data():
                        self.print_log(f"Failed to load data for subject {test_subject}")
                        continue
                    self.load_optimizer()
                    self.load_loss()
                    self.early_stop.reset()
                    for epoch in range(self.arg.num_epoch):
                        if self.train(epoch):
                            break
                    self.model = self.load_model()
                    self.load_weights()
                    self.print_log(f'\n=== Testing Subject {self.test_subject[0]} ===')
                    self.eval(epoch=0, loader_name='test')
                    self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    subject_result = {'test_subject': str(self.test_subject[0]), 'accuracy': round(self.test_accuracy, 2), 'f1_score': round(self.test_f1, 2), 'precision': round(self.test_precision, 2), 'recall': round(self.test_recall, 2), 'auc': round(self.test_auc, 2), 'fold_time': round(time.time() - fold_start_time, 2)}
                    results.append(subject_result)
                    pd.DataFrame(results).to_csv(os.path.join(self.arg.work_dir, 'interim_results.csv'), index=False)
                    self.print_log(f'\nFold {fold_idx+1} completed in {subject_result["fold_time"]:.2f}s')
                    self.print_log(f'Results: Acc={self.test_accuracy:.2f}%, F1={self.test_f1:.2f}%')
                except Exception as e:
                    self.print_log(f"Error in fold {fold_idx+1}: {e}")
                    import traceback
                    self.print_log(traceback.format_exc())
                    continue
            if results:
                df_results = pd.DataFrame(results)
                stats = df_results.describe().round(2)
                avg_row = df_results.mean(numeric_only=True).round(2)
                avg_row['test_subject'] = 'Average'
                df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
                df_results.to_csv(os.path.join(self.arg.work_dir, 'final_results.csv'), index=False)
                stats.to_csv(os.path.join(self.arg.work_dir, 'results_statistics.csv'))
                self.print_log("\n" + "="*60)
                self.print_log("FINAL RESULTS SUMMARY")
                self.print_log("="*60)
                self.print_log(df_results.to_string(index=False))
                self.print_log("\nStatistics:")
                self.print_log(stats.to_string())
                self.print_log("="*60)
                self.create_overall_visualization(df_results)
                self.create_summary_report(df_results, stats)
            else:
                self.print_log("Warning: No results collected!")
        else:
            self.print_log(f"Phase {self.arg.phase} not implemented")
    def create_overall_visualization(self, results_df):
        plot_df = results_df[results_df['test_subject'] != 'Average'].copy()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Validation Results Overview', fontsize=16)
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(plot_df['test_subject'], plot_df[metric])
            ax.axhline(y=plot_df[metric].mean(), color='r', linestyle='--', label='Mean')
            ax.set_xlabel('Test Subject')
            ax.set_ylabel(f'{metric} (%)')
            ax.set_title(f'{metric.replace("_", " ").title()} by Subject')
            ax.legend()
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, plot_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.1f}%', ha='center', va='bottom')
        ax = axes[1, 2]
        ax.bar(plot_df['test_subject'], plot_df['fold_time'])
        ax.set_xlabel('Test Subject')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Fold Processing Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'visualizations', 'overall_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    def create_summary_report(self, results_df, stats):
        report_path = os.path.join(self.arg.work_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write("LIGHTHEART-TF TRAINING SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            f.write("CONFIGURATION:\n")
            f.write("-"*30 + "\n")
            config_items = {'Model': self.arg.model, 'Dataset': self.arg.dataset, 'Optimizer': self.arg.optimizer, 'Learning Rate': self.arg.base_lr, 'Batch Size': self.arg.batch_size, 'Epochs': self.arg.num_epoch, 'Window Size': self.arg.dataset_args.get('max_length', 'N/A'), 'Task': self.arg.dataset_args.get('task', 'N/A')}
            for key, value in config_items.items():
                f.write(f"{key}: {value}\n")
            f.write("\nCROSS-VALIDATION SETUP:\n")
            f.write("-"*30 + "\n")
            f.write(f"Total Subjects: {len(self.arg.subjects)}\n")
            f.write(f"Fixed Train: {self.fixed_train_subjects}\n")
            f.write(f"Fixed Val: {self.fixed_val_subjects}\n")
            f.write(f"Test Eligible: {self.test_eligible_subjects}\n")
            f.write(f"Total Folds: {self.total_folds}\n")
            f.write("\nRESULTS:\n")
            f.write("-"*30 + "\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\nSTATISTICS:\n")
            f.write("-"*30 + "\n")
            f.write(stats.to_string())
            f.write("\n\nPERFORMANCE ANALYSIS:\n")
            f.write("-"*30 + "\n")
            analysis_df = results_df[results_df['test_subject'] != 'Average'].copy()
            for metric in ['accuracy', 'f1_score', 'auc']:
                best_idx = analysis_df[metric].idxmax()
                worst_idx = analysis_df[metric].idxmin()
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Best: Subject {analysis_df.loc[best_idx, 'test_subject']} ({analysis_df.loc[best_idx, metric]:.2f}%)\n")
                f.write(f"  Worst: Subject {analysis_df.loc[worst_idx, 'test_subject']} ({analysis_df.loc[worst_idx, metric]:.2f}%)\n")
                f.write(f"  Range: {analysis_df[metric].max() - analysis_df[metric].min():.2f}%\n")
                f.write(f"  Std Dev: {analysis_df[metric].std():.2f}%\n")
            f.write("\nTIMING:\n")
            f.write("-"*30 + "\n")
            total_time = analysis_df['fold_time'].sum()
            avg_time = analysis_df['fold_time'].mean()
            f.write(f"Total Training Time: {total_time:.2f}s ({total_time/3600:.2f} hours)\n")
            f.write(f"Average Fold Time: {avg_time:.2f}s\n")
            f.write("\nGENERATED FILES:\n")
            f.write("-"*30 + "\n")
            f.write(f"Working Directory: {self.arg.work_dir}\n")
            f.write("- final_results.csv\n")
            f.write("- results_statistics.csv\n")
            f.write("- training.log\n")
            f.write("- visualizations/\n")
            f.write("- models/\n")
            f.write("\n" + "="*50 + "\n")
            f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='LightHART-TF Training')
    parser.add_argument('--config', default='./config/smartfallmm/student.yaml')
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=80)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--model-args', default=None, type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--model-saved-name', type=str, default='student_model')
    parser.add_argument('--loss', default='loss.BCE')
    parser.add_argument('--loss-args', default="{}", type=str)
    parser.add_argument('--dataset-args', default=None, type=str)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--work-dir', type=str, default='simple')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--use-smv', type=str2bool, default=False)
    parser.add_argument('--train-subjects-fixed', nargs='+', type=int)
    parser.add_argument('--val-subjects-fixed', nargs='+', type=int)
    parser.add_argument('--test-eligible-subjects', nargs='+', type=int)
    return parser

def main():
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.device[0])
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    np.random.seed(arg.seed)
    tf.random.set_seed(arg.seed)
    try:
        trainer = Trainer(arg)
        trainer.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
