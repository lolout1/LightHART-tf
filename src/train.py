# src/train.py
import os
import sys
import logging
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lightheart-tf')

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
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.model_path = os.path.join(self.arg.work_dir, 'models', self.arg.model_saved_name)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_config(arg.config, arg.work_dir)
        self.model = self.load_model()
        self.print_log(f'# Parameters: {self.count_parameters()}')
    
    def save_config(self, src_path, dest_path):
        import shutil
        config_dest = os.path.join(dest_path, 'config')
        os.makedirs(config_dest, exist_ok=True)
        dest_file = os.path.join(config_dest, os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(dest_file):
            shutil.copy(src_path, dest_file)
    
    def count_parameters(self):
        return sum(np.prod(v.shape.as_list()) for v in self.model.trainable_variables)
    
    def import_class(self, import_str):
        mod_str, _, class_str = import_str.rpartition('.')
        import importlib
        module = importlib.import_module(mod_str)
        return getattr(module, class_str)
    
    def load_model(self):
        model_class = self.import_class(self.arg.model)
        model = model_class(**self.arg.model_args)
        acc_frames = self.arg.model_args.get('acc_frames', 64)
        acc_coords = self.arg.model_args.get('acc_coords', 3)
        if getattr(self.arg, 'use_smv', False):
            acc_coords += 1
        dummy_input = {'accelerometer': tf.zeros((1, acc_frames, acc_coords)), 'skeleton': tf.zeros((1, self.arg.model_args.get('mocap_frames', 64), self.arg.model_args.get('num_joints', 32), 3))}
        _ = model(dummy_input, training=False)
        return model
    
    def calculate_class_weights(self, labels):
        from collections import Counter
        counter = Counter(labels)
        if 0 not in counter:
            counter[0] = 1
        if 1 not in counter:
            counter[1] = 1
        pos_weight = counter[0] / counter[1]
        self.print_log(f'Class distribution - 0: {counter[0]}, 1: {counter[1]}, pos_weight: {pos_weight:.4f}')
        return tf.constant(pos_weight, dtype=tf.float32)
    
    def load_optimizer(self):
        base_lr = self.arg.base_lr
        # Create new optimizer for each fold to avoid TF Variable issues
        tf.keras.backend.clear_session()
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)
    
    def load_loss(self):
        self.pos_weights = getattr(self, 'pos_weights', tf.constant(1.0))
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.squeeze(y_pred)
            y_true = tf.squeeze(y_true)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=self.pos_weights)
            return tf.reduce_mean(loss)
        self.criterion = weighted_bce
    
    def load_data(self):
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
        Feeder = self.import_class(self.arg.feeder)
        builder = prepare_smartfallmm_tf(self.arg)
        if self.arg.phase == 'train':
            self.print_log(f'Loading data for train: {self.train_subjects}, val: {self.val_subject}, test: {self.test_subject}')

            # Important: Get all data first to compute global statistics for normalization
            self.print_log('Computing global normalization statistics for consistent preprocessing...')
            all_data = split_by_subjects_tf(builder, self.train_subjects + self.val_subject + self.test_subject, True, compute_stats_only=True)

            # Store global stats to ensure consistent normalization across splits
            self.acc_mean = all_data.get('acc_mean', None)
            self.acc_std = all_data.get('acc_std', None)
            self.skl_mean = all_data.get('skl_mean', None)
            self.skl_std = all_data.get('skl_std', None)

            # Now split with consistent normalization
            self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False,
                                                 acc_mean=self.acc_mean, acc_std=self.acc_std,
                                                 skl_mean=self.skl_mean, skl_std=self.skl_std)
            self.norm_val = split_by_subjects_tf(builder, self.val_subject, False,
                                               acc_mean=self.acc_mean, acc_std=self.acc_std,
                                               skl_mean=self.skl_mean, skl_std=self.skl_std)
            self.norm_test = split_by_subjects_tf(builder, self.test_subject, False,
                                               acc_mean=self.acc_mean, acc_std=self.acc_std,
                                               skl_mean=self.skl_mean, skl_std=self.skl_std)

            # Data validation checks
            if not self.norm_train or 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0:
                self.print_log(f'No training data for subjects {self.train_subjects}')
                return False
            if not self.norm_val or 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                self.print_log(f'No validation data for subjects {self.val_subject}')
                return False
            if not self.norm_test or 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                self.print_log(f'No test data for subject {self.test_subject}')
                return False

            # Log data distribution
            train_size = len(self.norm_train.get('labels', []))
            val_size = len(self.norm_val.get('labels', []))
            test_size = len(self.norm_test.get('labels', []))
            self.print_log(f'Data sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}')

            # Log class distribution
            train_pos = sum(1 for l in self.norm_train['labels'] if l == 1)
            train_neg = train_size - train_pos
            val_pos = sum(1 for l in self.norm_val['labels'] if l == 1)
            val_neg = val_size - val_pos
            test_pos = sum(1 for l in self.norm_test['labels'] if l == 1)
            test_neg = test_size - test_pos

            self.print_log(f'Class distribution:')
            self.print_log(f'Train - Positive: {train_pos} ({train_pos/train_size:.2%}), Negative: {train_neg} ({train_neg/train_size:.2%})')
            self.print_log(f'Val - Positive: {val_pos} ({val_pos/val_size:.2%}), Negative: {val_neg} ({val_neg/val_size:.2%})')
            self.print_log(f'Test - Positive: {test_pos} ({test_pos/test_size:.2%}), Negative: {test_neg} ({test_neg/test_size:.2%})')

            # Calculate class weights
            self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])

            # Create data loaders
            use_smv = getattr(self.arg, 'use_smv', False)
            window_size = self.arg.dataset_args.get('max_length', 64)
            self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, use_smv=use_smv, window_size=window_size)
            self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size, use_smv=use_smv, window_size=window_size)
            self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv, window_size=window_size)
            self.print_log(f'Batches - Train: {len(self.data_loader["train"])}, Val: {len(self.data_loader["val"])}, Test: {len(self.data_loader["test"])}')
            return True
    
    def calculate_metrics(self, targets, predictions, probabilities=None):
        """Calculate performance metrics with improved handling of edge cases.

        Args:
            targets: Ground truth labels (0/1)
            predictions: Binary predictions after thresholding
            probabilities: Probability scores before thresholding (for AUC)

        Returns:
            tuple: (accuracy, f1, recall, precision, auc)
        """
        # Convert to numpy arrays and flatten
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()

        # Ensure binary targets/predictions (avoid double thresholding)
        # Only threshold if values aren't already binary (0/1)
        if not np.all(np.isin(targets, [0, 1])):
            targets = (targets > 0.5).astype(int)
        if not np.all(np.isin(predictions, [0, 1])):
            predictions = (predictions > 0.5).astype(int)

        # Handle edge cases: if all predictions are the same class
        if len(np.unique(predictions)) == 1:
            self.print_log(f"WARNING: All predictions are the same class ({predictions[0]}). This may indicate a problem.")

        # Calculate metrics with proper error handling
        try:
            accuracy = accuracy_score(targets, predictions) * 100
        except Exception as e:
            self.print_log(f"Error calculating accuracy: {e}")
            accuracy = 0.0

        try:
            f1 = f1_score(targets, predictions, average='binary', zero_division=0) * 100
        except Exception as e:
            self.print_log(f"Error calculating F1: {e}")
            f1 = 0.0

        try:
            precision = precision_score(targets, predictions, average='binary', zero_division=0) * 100
        except Exception as e:
            self.print_log(f"Error calculating precision: {e}")
            precision = 0.0

        try:
            recall = recall_score(targets, predictions, average='binary', zero_division=0) * 100
        except Exception as e:
            self.print_log(f"Error calculating recall: {e}")
            recall = 0.0

        # Calculate AUC with proper error handling
        auc = 0.0
        try:
            if probabilities is not None:
                probabilities = np.array(probabilities).flatten()
                # Check if we have multiple classes in targets (AUC requires both classes)
                if len(np.unique(targets)) > 1:
                    auc = roc_auc_score(targets, probabilities) * 100
                else:
                    self.print_log("WARNING: Only one class in targets, AUC can't be calculated")
            else:
                if len(np.unique(targets)) > 1:
                    auc = roc_auc_score(targets, predictions) * 100
                else:
                    self.print_log("WARNING: Only one class in targets, AUC can't be calculated")
        except Exception as e:
            self.print_log(f"Error calculating AUC: {e}")

        # Generate and log confusion matrix for detailed analysis
        try:
            cm = confusion_matrix(targets, predictions)
            tn, fp, fn, tp = cm.ravel()
            self.print_log(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        except Exception as e:
            self.print_log(f"Error calculating confusion matrix: {e}")

        return accuracy, f1, recall, precision, auc
    
    # Remove tf.function decorator to fix optimizer variable creation error
    def train_step(self, inputs, targets):
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
    
    def train(self, epoch):
        # Print start of epoch message with more visibility
        total_batches = len(self.data_loader['train'])
        epoch_header = f'Epoch: {epoch+1} - Starting training ({total_batches} batches)'
        self.print_log(epoch_header)

        # Make training start more visible in console
        print("\n" + "="*len(epoch_header))
        print(epoch_header)
        print("="*len(epoch_header))

        # Set up print interval for progress updates
        print_interval = max(1, min(10, total_batches // 10))  # Dynamic interval based on batch count

        loader = self.data_loader['train']
        train_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        start_time = time.time()

        # Training loop
        for batch_idx in range(len(loader)):
            inputs, targets, _ = loader[batch_idx]
            targets = tf.cast(targets, tf.float32)

            # Forward and backward pass
            loss, logits = self.train_step(inputs, targets)
            probabilities = tf.sigmoid(logits)
            predictions = tf.cast(probabilities > 0.5, tf.int32)

            # Update metrics
            train_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            all_probs.extend(probabilities.numpy().flatten())
            steps += 1

            # Log progress more frequently with percentage complete
            if batch_idx % print_interval == 0 or batch_idx == len(loader) - 1:
                percent_complete = (batch_idx + 1) / len(loader) * 100
                progress_msg = f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)} ({percent_complete:.1f}%), Loss: {loss:.4f}'
                print(progress_msg)
                self.print_log(progress_msg)

        # Calculate average loss and metrics
        train_loss /= steps
        train_time = time.time() - start_time

        # Calculate and log all metrics
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
        self.train_loss_summary.append(float(train_loss))

        # Print detailed training results
        results_msg = f'\tTraining Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%'
        time_msg = f'\tTime: {train_time:.2f}s'

        print(results_msg)
        print(time_msg)
        self.print_log(results_msg)
        self.print_log(time_msg)

        # Evaluate on validation set
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))

        # Check early stopping
        if self.early_stop(val_loss):
            self.print_log("Early stopping triggered")
            print("Early stopping triggered")
            return True

        return False
    
    # Remove tf.function decorator to fix consistency with train_step
    def eval_step(self, inputs, targets):
        outputs = self.model(inputs, training=False)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        loss = self.criterion(targets, logits)
        return loss, logits
    
    def eval(self, epoch, loader_name='val'):
        """Evaluate the model on validation or test data with detailed logging.

        Args:
            epoch: Current training epoch
            loader_name: Dataset to evaluate on ('val' or 'test')

        Returns:
            float: Average evaluation loss
        """
        # Print start of evaluation with more visibility
        total_batches = len(self.data_loader[loader_name])
        eval_header = f"\n{'='*20} {loader_name.upper()} EVALUATION {'='*20}"
        self.print_log(eval_header)
        print(eval_header)

        # Make start of evaluation more prominent in console
        eval_start_banner = f"STARTING {loader_name.upper()} EVALUATION - EPOCH {epoch+1}"
        print("\n" + "="*len(eval_start_banner))
        print(eval_start_banner)
        print("="*len(eval_start_banner))

        start_msg = f"Evaluating on {total_batches} batches..."
        self.print_log(start_msg)
        print(start_msg)

        # Load data and initialize metrics
        loader = self.data_loader[loader_name]
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        all_ids = []  # Track sample IDs for per-sample analysis
        steps = 0

        # Print progress indicator every few batches
        print_interval = max(1, min(5, total_batches // 10))  # Dynamic interval based on batch count

        # Start timing
        start_time = time.time()

        # Gather predictions
        for batch_idx in range(len(loader)):
            inputs, targets, sample_ids = loader[batch_idx]
            targets = tf.cast(targets, tf.float32)

            # Get predictions and loss
            loss, logits = self.eval_step(inputs, targets)
            probabilities = tf.sigmoid(logits)
            predictions = tf.cast(probabilities > 0.5, tf.int32)

            # Accumulate results
            eval_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            all_probs.extend(probabilities.numpy().flatten())
            if sample_ids is not None:
                # Check if sample_ids is already a numpy array or a tensor
                if isinstance(sample_ids, np.ndarray):
                    all_ids.extend(sample_ids.flatten())
                else:
                    all_ids.extend(sample_ids.numpy().flatten())
            steps += 1

            # Print progress more frequently with percentage complete
            if batch_idx % print_interval == 0 or batch_idx == total_batches - 1:
                percent_complete = (batch_idx + 1) / total_batches * 100
                progress_msg = f"Batch {batch_idx+1}/{total_batches} ({percent_complete:.1f}%), Loss: {loss.numpy():.4f}"
                print(progress_msg)
                self.print_log(progress_msg)

        # Return early if no data
        if steps == 0:
            warning_msg = f"WARNING: No batches to evaluate in {loader_name} loader!"
            print(warning_msg)
            self.print_log(warning_msg)
            return float('inf')

        # Calculate average loss and evaluation time
        eval_loss /= steps
        eval_time = time.time() - start_time

        # Calculate and log all metrics
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)

        # Format a detailed report
        report_header = f"EPOCH {epoch+1} - {loader_name.upper()} METRICS:"
        eval_metrics = [
            f"Loss: {eval_loss:.4f}",
            f"Accuracy: {accuracy:.2f}%",
            f"F1 Score: {f1:.2f}%",
            f"Precision: {precision:.2f}%",
            f"Recall: {recall:.2f}%",
            f"AUC: {auc_score:.2f}%",
            f"Time: {eval_time:.2f}s"
        ]

        # Print detailed evaluation report - more visibly
        separator = "="*70
        results_banner = f"RESULTS SUMMARY - {loader_name.upper()} - EPOCH {epoch+1}"

        # Make results highly visible in console output
        print("\n" + separator)
        print(results_banner)
        print(separator)
        for metric in eval_metrics:
            print(f"  {metric}")
        print(separator)

        # Also log to file
        self.print_log("\n" + separator)
        self.print_log(results_banner)
        self.print_log(separator)
        for metric in eval_metrics:
            self.print_log(f"  {metric}")
        self.print_log(separator)

        # Save per-sample predictions to CSV for error analysis
        if hasattr(self.arg, 'save_predictions') and self.arg.save_predictions:
            try:
                import pandas as pd
                preds_df = pd.DataFrame({
                    'sample_id': all_ids if all_ids else range(len(all_labels)),
                    'true_label': all_labels,
                    'prediction': all_preds,
                    'probability': all_probs
                })
                pred_path = os.path.join(self.arg.work_dir, f'{loader_name}_predictions_epoch{epoch+1}_subj{self.test_subject[0]}.csv')
                preds_df.to_csv(pred_path, index=False)
                saved_msg = f"Saved detailed predictions to {pred_path}"
                print(saved_msg)
                self.print_log(saved_msg)
            except Exception as e:
                error_msg = f"Failed to save predictions to CSV: {e}"
                print(error_msg)
                self.print_log(error_msg)

        # Handle model saving
        if loader_name == 'val':
            # Save model if validation improves
            if eval_loss < self.best_loss:
                prev_best = self.best_loss
                self.best_loss = eval_loss
                self.save_model()
                improved_msg = f'Model saved! Validation loss improved: {prev_best:.4f} → {eval_loss:.4f}'
                print(improved_msg)
                self.print_log(improved_msg)

                # Save validation metrics for best model
                self.best_val_metrics = {
                    'epoch': epoch + 1,
                    'loss': eval_loss,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc_score
                }
            else:
                no_improve_msg = f'Validation loss did not improve. Best: {self.best_loss:.4f}, Current: {eval_loss:.4f}'
                print(no_improve_msg)
                self.print_log(no_improve_msg)

        # Store test metrics for final reporting
        elif loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
            self.test_loss = eval_loss

            # Save test metrics for current epoch
            test_metrics_path = os.path.join(self.arg.work_dir, f'test_metrics_subject{self.test_subject[0]}.csv')
            try:
                import pandas as pd
                # Check if file exists
                if os.path.exists(test_metrics_path):
                    metrics_df = pd.read_csv(test_metrics_path)
                else:
                    metrics_df = pd.DataFrame()

                # Append new metrics
                new_metrics = pd.DataFrame({
                    'epoch': [epoch + 1],
                    'subject': [self.test_subject[0]],
                    'loss': [eval_loss],
                    'accuracy': [accuracy],
                    'f1': [f1],
                    'precision': [precision],
                    'recall': [recall],
                    'auc': [auc_score]
                })
                metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
                metrics_df.to_csv(test_metrics_path, index=False)
                saved_metrics_msg = f"Saved test metrics to {test_metrics_path}"
                print(saved_metrics_msg)
                self.print_log(saved_metrics_msg)
            except Exception as e:
                metrics_error_msg = f"Failed to save test metrics to CSV: {e}"
                print(metrics_error_msg)
                self.print_log(metrics_error_msg)

        return eval_loss
    
    def save_model(self):
        self.model.save_weights(f'{self.model_path}_{self.test_subject[0]}.weights.h5')
    
    def loss_viz(self, train_loss, val_loss):
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title(f'Training vs Validation Loss - Subject {self.test_subject[0]}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.arg.work_dir, f'loss_curves_subject_{self.test_subject[0]}.png'))
        plt.close()
    
    def print_log(self, message):
        print(message)
        with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
            print(message, file=f)
    
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            results = []
            fixed_train_subjects = [45, 36, 29]
            fixed_val_subjects = [38, 46]
            test_eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
            total_subjects = len(test_eligible_subjects)

            # Print overall training plan upfront
            overall_plan = f'''
======================================================
TRAINING PLAN: LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
======================================================
Total test subjects: {total_subjects}
Fixed validation subjects: {fixed_val_subjects}
Fixed train subjects: {fixed_train_subjects}
Test eligible subjects: {test_eligible_subjects}
======================================================
'''
            print(overall_plan)
            self.print_log(overall_plan)

            for i, test_subject in enumerate(test_eligible_subjects):
                self.print_log(f'\n----------- Test Subject {test_subject} ({i+1}/{total_subjects}) -----------')
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = fixed_val_subjects
                remaining_test_subjects = [s for s in test_eligible_subjects if s != test_subject]
                self.train_subjects = fixed_train_subjects + remaining_test_subjects
                tf.keras.backend.clear_session()
                self.model = self.load_model()
                if not self.load_data():
                    self.print_log(f"Failed to load data for subject {test_subject}")
                    continue
                self.load_optimizer()
                self.load_loss()
                self.early_stop.reset()
                best_epoch = 0
                for epoch in range(self.arg.num_epoch):
                    # Print progress update before training each epoch
                    epoch_progress = f"Subject {test_subject} - Training epoch {epoch+1}/{self.arg.num_epoch}"
                    print("="*len(epoch_progress))
                    print(epoch_progress)
                    print("="*len(epoch_progress))

                    if self.train(epoch):
                        self.print_log(f'Early stopping triggered at epoch {epoch+1}')
                        best_epoch = epoch
                        break
                    best_epoch = epoch
                self.model.load_weights(f'{self.model_path}_{test_subject}.weights.h5')
                self.print_log(f'\n---------- Test Results for Subject {test_subject} ----------')
                test_loss = self.eval(epoch=best_epoch, loader_name='test')
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)

                # Get final train metrics for this fold
                train_loss = self.train_loss_summary[-1] if self.train_loss_summary else float('nan')
                val_loss = self.val_loss_summary[-1] if self.val_loss_summary else float('nan')

                # Create a comprehensive result dictionary with all metrics
                result = {
                    'test_subject': str(test_subject),
                    # Train metrics
                    'train_loss': round(train_loss, 4),
                    # Validation metrics
                    'val_loss': round(val_loss, 4),
                    # Test metrics
                    'test_loss': round(test_loss, 4),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2) if self.test_auc is not None else None
                }
                results.append(result)

                # Print current results table after each fold completes
                current_df = pd.DataFrame(results)
                if len(current_df) > 1:  # Only calculate averages if we have more than one fold
                    current_avg = current_df.mean(numeric_only=True).round(2)
                    current_avg['test_subject'] = 'Running Avg'
                    current_df = pd.concat([current_df, pd.DataFrame([current_avg])], ignore_index=True)

                self.print_log("\n========== Current Results Summary ==========")
                self.print_log(current_df.to_string(index=False))
                self.print_log(f"\nCompleted {i+1}/{total_subjects} test subjects\n")

                # Save comprehensive metrics for this fold to a dedicated file
                fold_metrics_path = os.path.join(self.arg.work_dir, f'metrics_subject{test_subject}.csv')
                fold_metrics = {
                    'epoch': list(range(1, len(self.train_loss_summary) + 1)),
                    'train_loss': self.train_loss_summary,
                    'val_loss': self.val_loss_summary
                }
                # Add final test metrics to the last row
                fold_metrics_df = pd.DataFrame(fold_metrics)
                last_row = len(fold_metrics_df) - 1
                if last_row >= 0:
                    fold_metrics_df.loc[last_row, 'test_loss'] = test_loss
                    fold_metrics_df.loc[last_row, 'test_accuracy'] = self.test_accuracy
                    fold_metrics_df.loc[last_row, 'test_f1'] = self.test_f1
                    fold_metrics_df.loc[last_row, 'test_precision'] = self.test_precision
                    fold_metrics_df.loc[last_row, 'test_recall'] = self.test_recall
                    fold_metrics_df.loc[last_row, 'test_auc'] = self.test_auc
                fold_metrics_df.to_csv(fold_metrics_path, index=False)
                self.print_log(f"Saved detailed fold metrics to {fold_metrics_path}")

                # Also save intermediate results
                current_df.to_csv(os.path.join(self.arg.work_dir, 'current_scores.csv'), index=False)

                # Save all metrics across all folds in a single file
                all_metrics_path = os.path.join(self.arg.work_dir, 'all_fold_metrics.csv')
                if os.path.exists(all_metrics_path):
                    all_metrics_df = pd.read_csv(all_metrics_path)
                else:
                    all_metrics_df = pd.DataFrame()

                # Add this fold's metrics
                new_row = pd.DataFrame([{
                    'fold': i+1,
                    'test_subject': test_subject,
                    'epochs': len(self.train_loss_summary),
                    'final_train_loss': train_loss,
                    'final_val_loss': val_loss,
                    'final_test_loss': test_loss,
                    'test_accuracy': self.test_accuracy,
                    'test_f1': self.test_f1,
                    'test_precision': self.test_precision,
                    'test_recall': self.test_recall,
                    'test_auc': self.test_auc
                }])
                all_metrics_df = pd.concat([all_metrics_df, new_row], ignore_index=True)
                all_metrics_df.to_csv(all_metrics_path, index=False)
                self.print_log(f"Updated all fold metrics in {all_metrics_path}")

                # Print fold summary to be extra visible in terminal output
                fold_summary = f'''
==========================================
COMPLETED FOLD {i+1}/{total_subjects}: Subject {test_subject}
Accuracy: {self.test_accuracy:.2f}%
F1 Score: {self.test_f1:.2f}%
Precision: {self.test_precision:.2f}%
Recall: {self.test_recall:.2f}%
AUC: {self.test_auc:.2f}%
==========================================
'''
                print(fold_summary)
                self.print_log(fold_summary)

            # Final results processing with enhanced reporting
            if results:  # Check if we have any results before creating DataFrame
                df_results = pd.DataFrame(results)

                # Calculate average row with more descriptive statistics
                avg_row = df_results.mean(numeric_only=True).round(2)
                avg_row['test_subject'] = 'Average'

                # Calculate standard deviation for uncertainty estimation
                std_row = df_results.std(numeric_only=True).round(2)
                std_row['test_subject'] = 'Std Dev'

                # Calculate min/max for range reporting
                min_row = df_results.min(numeric_only=True).round(2)
                min_row['test_subject'] = 'Min'

                max_row = df_results.max(numeric_only=True).round(2)
                max_row['test_subject'] = 'Max'
            else:
                self.print_log("Warning: No results collected for final processing")
                return

            # Add all summary statistics to the results
            summary_rows = pd.DataFrame([avg_row, std_row, min_row, max_row])
            df_results = pd.concat([df_results, summary_rows], ignore_index=True)

            # Save complete results
            df_results.to_csv(os.path.join(self.arg.work_dir, 'final_scores.csv'), index=False)

            # Create a more detailed final report with separate tables
            final_report_path = os.path.join(self.arg.work_dir, 'final_report.txt')
            with open(final_report_path, 'w') as f:
                # Header
                header = "="*80 + "\n" + "FINAL CROSS-VALIDATION RESULTS REPORT\n" + "="*80 + "\n"
                f.write(header)

                # Summary statistics table
                f.write("\nSUMMARY STATISTICS:\n")
                f.write("-"*80 + "\n")
                summary_df = pd.DataFrame([avg_row, std_row, min_row, max_row])
                f.write(summary_df.to_string(index=False))

                # Per-subject results
                f.write("\n\nPER-SUBJECT RESULTS:\n")
                f.write("-"*80 + "\n")
                per_subject_df = df_results[df_results['test_subject'].apply(lambda x: x not in ['Average', 'Std Dev', 'Min', 'Max'])]
                f.write(per_subject_df.to_string(index=False))

                # Training/Validation vs Test Loss Comparison
                f.write("\n\nTRAINING/VALIDATION/TEST LOSS COMPARISON:\n")
                f.write("-"*80 + "\n")
                loss_df = per_subject_df[['test_subject', 'train_loss', 'val_loss', 'test_loss']]
                f.write(loss_df.to_string(index=False))

                # Accuracy Metrics Comparison
                f.write("\n\nACCURACY METRICS COMPARISON:\n")
                f.write("-"*80 + "\n")
                metrics_df = per_subject_df[['test_subject', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']]
                f.write(metrics_df.to_string(index=False))

                # Footer with timestamp
                f.write(f"\n\nReport generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Print final results summary to console and log
            self.print_log("\n========== Final Results Summary ==========")
            self.print_log(df_results.to_string(index=False))
            self.print_log(f"\nDetailed final report saved to: {final_report_path}")

            # Also generate a plots directory with visualizations
            plots_dir = os.path.join(self.arg.work_dir, 'summary_plots')
            os.makedirs(plots_dir, exist_ok=True)

            try:
                # Load the all_fold_metrics.csv which has data for all folds
                all_metrics_path = os.path.join(self.arg.work_dir, 'all_fold_metrics.csv')
                if os.path.exists(all_metrics_path):
                    all_metrics = pd.read_csv(all_metrics_path)

                    # Generate bar chart of test accuracy across subjects
                    plt.figure(figsize=(12, 6))
                    plt.bar(all_metrics['test_subject'].astype(str), all_metrics['test_accuracy'])
                    plt.axhline(y=all_metrics['test_accuracy'].mean(), color='r', linestyle='--', label=f'Mean: {all_metrics["test_accuracy"].mean():.2f}%')
                    plt.title('Test Accuracy Across Subjects')
                    plt.xlabel('Subject ID')
                    plt.ylabel('Accuracy (%)')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, 'test_accuracy_by_subject.png'))
                    plt.close()

                    # Generate another chart for F1 scores
                    plt.figure(figsize=(12, 6))
                    plt.bar(all_metrics['test_subject'].astype(str), all_metrics['test_f1'])
                    plt.axhline(y=all_metrics['test_f1'].mean(), color='r', linestyle='--', label=f'Mean: {all_metrics["test_f1"].mean():.2f}%')
                    plt.title('F1 Score Across Subjects')
                    plt.xlabel('Subject ID')
                    plt.ylabel('F1 Score (%)')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, 'f1_score_by_subject.png'))
                    plt.close()

                    # Generate loss comparison plot
                    plt.figure(figsize=(12, 6))
                    x = all_metrics['test_subject'].astype(str)
                    width = 0.25
                    plt.bar(x, all_metrics['final_train_loss'], width, label='Train Loss')
                    plt.bar([p + width for p in range(len(x))], all_metrics['final_val_loss'], width, label='Validation Loss')
                    plt.bar([p + width*2 for p in range(len(x))], all_metrics['final_test_loss'], width, label='Test Loss')
                    plt.xticks([p + width for p in range(len(x))], x)
                    plt.title('Loss Comparison Across Subjects')
                    plt.xlabel('Subject ID')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(plots_dir, 'loss_comparison_by_subject.png'))
                    plt.close()

                    self.print_log(f"Generated summary plots in {plots_dir}")
            except Exception as e:
                self.print_log(f"Error generating summary plots: {e}")

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
    parser.add_argument('--train-feeder-args', default=None, type=str)
    parser.add_argument('--val-feeder-args', default=None, type=str)
    parser.add_argument('--test-feeder-args', default=None, type=str)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--work-dir', type=str, default='simple')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--use-smv', type=str2bool, default=False)
    return parser

def main():
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.device[0])
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    np.random.seed(arg.seed)
    tf.random.set_seed(arg.seed)
    trainer = Trainer(arg)
    trainer.start()

if __name__ == "__main__":
    main()
