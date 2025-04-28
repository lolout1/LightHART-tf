import os
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
import traceback
import importlib
import sys
import csv
import shutil
from sklearn.metrics import confusion_matrix
import seaborn as sns

class EarlyStoppingTF:
    def __init__(self, patience=15, min_delta=0.00001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.wait = 0

    def __call__(self, current_value):
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
            self.wait = 0
            return False
        else:
            self.counter += 1
            self.wait += 1
            if self.verbose:
                logging.info(f"Early stopping: val_loss={current_value:.6f} (no improvement, counter: {self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"Early stopping triggered after {self.wait} epochs without improvement")
                return True
            return False

    def reset(self):
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.wait = 0

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
        self.data_loader = dict()
        self.early_stop = EarlyStoppingTF(patience=15, min_delta=0.001)
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities']
                                 if modality != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{timestamp}"
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(f"{self.arg.work_dir}/models", exist_ok=True)
        os.makedirs(f"{self.arg.work_dir}/visualizations", exist_ok=True)

        self.model_path = f'{self.arg.work_dir}/models/{self.arg.model_saved_name}'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.arg.work_dir}/training.log'),
                logging.StreamHandler()
            ]
        )

        self.print_log(f"LightHART-TF starting: {datetime.datetime.now()}")
        self.print_log(f"TensorFlow version: {tf.__version__}")

        self._setup_gpu()
        self.save_config(arg.config, arg.work_dir)
        self._load_model()

        num_params = self.count_parameters()
        self.print_log(f"Model parameters: {num_params:,} ({num_params/(1024**2):.2f} unieke MB)")

    def _setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.print_log(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                self.print_log(f"Error configuring GPU: {e}")
        else:
            self.print_log("No GPU found, using CPU")

    def _load_model(self):
        if self.arg.phase == 'train':
            self.model = self.load_model(self.arg.model, self.arg.model_args)
            input_shapes = {
                'accelerometer': (self.arg.model_args.get('acc_frames', 64),
                                 self.arg.model_args.get('acc_coords', 3))
            }
            if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                input_shapes['skeleton'] = (self.arg.model_args.get('acc_frames', 64), 32, 3)
            self._build_model(input_shapes)
        else:
            if hasattr(self.arg, 'weights') and self.arg.weights:
                self.model = self.load_weights(self.arg.weights)
            else:
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                input_shapes = {
                    'accelerometer': (self.arg.model_args.get('acc_frames', 64),
                                     self.arg.model_args.get('acc_coords', 3))
                }
                if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                    input_shapes['skeleton'] = (self.arg.model_args.get('acc_frames', 64), 32, 3)
                self._build_model(input_shapes)

    def _build_model(self, input_shapes):
        try:
            inputs = {}
            for name, shape in input_shapes.items():
                inputs[name] = tf.zeros((2,) + shape, dtype=tf.float32)
            _ = self.model(inputs, training=False)
            self.print_log("Model successfully built")
        except Exception as e:
            self.print_log(f"Error building model: {e}")
            traceback.print_exc()

    def import_class(self, import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        for prefix in ['', 'src.']:
            try:
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
        self.print_log(f"Error importing {import_str}")
        raise ImportError(f"Cannot import {class_str} from {mod_str}")

    def load_model(self, model_name, model_args):
        try:
            ModelClass = self.import_class(model_name)
            model = ModelClass(**model_args)
            self.print_log(f"Created model: {model_name}")
            return model
        except Exception as e:
            self.print_log(f"Error loading model {model_name}: {e}")
            traceback.print_exc()
            raise

    def load_weights(self, weights_path):
        try:
            if not os.path.exists(weights_path):
                self.print_log(f"Weights file not found: {weights_path}")
                return self.load_model(self.arg.model, self.arg.model_args)

            if weights_path.endswith('.weights.h5') or weights_path.endswith('.h5'):
                model = self.load_model(self.arg.model, self.arg.model_args)
                input_shapes = {
                    'accelerometer': (self.arg.model_args.get('acc_frames', 64),
                                    self.arg.model_args.get('acc_coords', 3))
                }
                if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                    input_shapes['skeleton'] = (self.arg.model_args.get('acc_frames', 64), 32, 3)
                self._build_model(input_shapes)
                model.load_weights(weights_path)
                self.print_log(f"Loaded weights from {weights_path}")
                return model
            else:
                model = tf.keras.models.load_model(weights_path)
                self.print_log(f"Loaded model from {weights_path}")
                return model
        except Exception as e:
            self.print_log(f"Error loading weights: {e}")
            return self.load_model(self.arg.model, self.arg.model_args)

    def save_config(self, src_path, dest_path):
        try:
            config_file = os.path.basename(src_path)
            dest_file = os.path.join(dest_path, config_file)
            with open(src_path, 'r') as src_f:
                with open(dest_file, 'w') as dst_f:
                    dst_f.write(src_f.read())
            with open(os.path.join(dest_path, 'args.json'), 'w') as f:
                json.dump(vars(self.arg), f, indent=2, default=str)
            self.print_log(f"Configuration saved to {dest_file}")
        except Exception as e:
            self.print_log(f"Error saving configuration: {e}")

    def count_parameters(self):
        try:
            return int(np.sum([tf.size(var).numpy() for var in self.model.trainable_variables]))
        except Exception as e:
            self.print_log(f"Error counting parameters: {e}")
            return 0

    def cal_weights(self):
        try:
            if not hasattr(self, 'norm_train') or 'labels' not in self.norm_train:
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
                return

            labels = self.norm_train['labels']
            neg_count = np.sum(labels == 0)
            pos_count = np.sum(labels == 1)

            if pos_count == 0:
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
                self.print_log(f"Warning: No positive samples found, using weight=1.0")
                return

            self.pos_weights = tf.constant(float(neg_count) / float(pos_count), dtype=tf.float32)
            self.print_log(f"Class balance - Negative: {neg_count}, Positive: {pos_count}")
            self.print_log(f"Positive class weight: {self.pos_weights.numpy():.4f}")
        except Exception as e:
            self.print_log(f"Error calculating weights: {e}")
            self.pos_weights = tf.constant(1.0, dtype=tf.float32)

    def has_empty_value(self, *lists):
        return any(isinstance(lst, (list, np.ndarray)) and len(lst) == 0 for lst in lists)

    def load_data(self):
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects, UTD_MM_TF

            self.print_log(f"Creating dataset: {self.arg.dataset}")
            builder = prepare_smartfallmm_tf(self.arg)

            self.print_log(f"Processing training data for subjects: {self.train_subjects}")
            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse)

            if self.has_empty_value(list(self.norm_train.values())):
                self.print_log("Error: Empty training data")
                return False

            self.data_loader['train'] = UTD_MM_TF(
                dataset=self.norm_train,
                batch_size=self.arg.batch_size,
                modalities=self.arg.dataset_args.get('modalities', ['accelerometer'])
            )

            self.cal_weights()
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')

            self.print_log(f"Processing validation data for subjects: {self.val_subject}")
            self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)

            if self.has_empty_value(list(self.norm_val.values())):
                self.print_log("Error: Empty validation data")
                return False

            self.data_loader['val'] = UTD_MM_TF(
                dataset=self.norm_val,
                batch_size=self.arg.val_batch_size,
                modalities=self.arg.dataset_args.get('modalities', ['accelerometer'])
            )

            self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')

            self.print_log(f"Processing test data for subjects: {self.test_subject}")
            self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)

            if self.has_empty_value(list(self.norm_test.values())):
                self.print_log("Error: Empty test data")
                return False

            self.data_loader['test'] = UTD_MM_TF(
                dataset=self.norm_test,
                batch_size=self.arg.test_batch_size,
                modalities=self.arg.dataset_args.get('modalities', ['accelerometer'])
                )

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
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time=True):
        logging.info(string)
        if self.arg.print_log:
            log_file = f'{self.arg.work_dir}/log.txt'
            with open(log_file, 'a') as f:
                print(string, file=f)

    def distribution_viz(self, labels, work_dir, mode):
        try:
            values, count = np.unique(labels, return_counts=True)

            plt.figure(figsize=(8, 6))
            plt.bar(values, count)
            plt.xlabel('Labels')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Label Distribution')

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

    def loss_viz(self, train_loss, val_loss):
        try:
            if len(train_loss) == 0 or len(val_loss) == 0:
                self.print_log("Not enough data for loss visualization")
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

            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'loss_curves_{self.test_subject[0]}.png'))
            plt.close()
        except Exception as e:
            self.print_log(f"Error visualizing loss: {e}")

    def cm_viz(self, y_pred, y_true):
        try:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Subject {self.test_subject[0]})')

            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{self.test_subject[0]}.png'))
            plt.close()
        except Exception as e:
            self.print_log(f"Error visualizing confusion matrix: {e}")

    def create_df(self):
        return []

    def cal_prediction(self, logits):
        return tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)

    def cal_metrics(self, targets, predictions):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        try:
            targets = np.array(targets, dtype=np.int32).flatten()
            predictions = np.array(predictions, dtype=np.int32).flatten()

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
                    tn, fp, fn, tp = 0, 0, 0, 0
                    for t, p in zip(targets, predictions):
                        if t == 1 and p == 1:
                            tp += 1
                        elif t == 1 and p == 0:
                            fn += 1
                        elif t == 0 and p == 1:
                            fp += 1
                        else:
                            tn += 1

                    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    auc = 50.0
            else:
                precision = precision_score(targets, predictions) * 100
                recall = recall_score(targets, predictions) * 100
                f1 = f1_score(targets, predictions) * 100

                try:
                    auc = roc_auc_score(targets, predictions) * 100
                except:
                    auc = 50.0

            return accuracy, f1, recall, precision, auc
        except Exception as e:
            self.print_log(f"Error calculating metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def load_optimizer(self):
        optimizer_name = self.arg.optimizer.lower()
        lr = self.arg.base_lr

        if optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.arg.weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            self.print_log(f"Unknown optimizer: {optimizer_name}, using Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.print_log(f"Optimizer: {self.optimizer.__class__.__name__}, LR={lr}")

    def load_loss(self):
        loss_name = getattr(self.arg, 'loss', 'bce').lower()

        if not hasattr(self, 'pos_weights'):
            self.pos_weights = tf.constant(1.0, dtype=tf.float32)

        if loss_name == "bce":
            def weighted_bce(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
                weights = y_true * (self.pos_weights - 1.0) + 1.0
                return tf.reduce_mean(weights * bce)

            self.criterion = weighted_bce
            self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")

        elif loss_name == "binary_focal":
            try:
                from utils.loss import BinaryFocalLossTF
                self.criterion = BinaryFocalLossTF(alpha=0.75)
                self.print_log("Using Binary Focal Loss")
            except ImportError:
                def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
                    y_true = tf.cast(y_true, tf.float32)
                    prob = tf.sigmoid(y_pred)
                    pt = tf.where(tf.equal(y_true, 1.0), prob, 1.0 - prob)
                    alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
                    focal_loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
                    return tf.reduce_mean(focal_loss)

                self.criterion = binary_focal_loss
                self.print_log("Using built-in Binary Focal Loss")
        else:
            def bce_loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

            self.criterion = bce_loss
            self.print_log("Using default BCE loss")

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            logits, features = self.model(inputs, training=True)
            logits = tf.reshape(logits, [-1])
            loss = self.criterion(targets, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        preds = self.cal_prediction(logits)

        return loss, preds, features

    def test_step(self, inputs, targets):
        logits, features = self.model(inputs, training=False)
        logits = tf.reshape(logits, [-1])

        loss = self.criterion(targets, logits)
        preds = self.cal_prediction(logits)

        return loss, preds, features

    def train(self, epoch):
        self.model.trainable = True
        self.record_time()

        loader = self.data_loader['train']
        timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}

        train_loss = 0.0
        label_list = []
        pred_list = []
        batch_count = 0

        progress = tqdm(range(len(loader)), desc=f"Epoch {epoch+1}/{self.arg.num_epoch}")

        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)

                timer['dataloader'] += self.split_time()

                loss, preds, _ = self.train_step(inputs, targets)

                timer['model'] += self.split_time()

                train_loss += loss.numpy()
                label_list.extend(targets.numpy())
                pred_list.extend(preds.numpy())
                batch_count += 1

                progress.set_postfix({'loss': f"{train_loss/batch_count:.4f}"})

                timer['stats'] += self.split_time()

            except Exception as e:
                self.print_log(f"Error in batch {batch_idx}: {e}")
                continue

        if batch_count > 0:
            train_loss /= batch_count
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

            self.train_loss_summary.append(train_loss)

            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }

            self.print_log(
                f'Train Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, '
                f'F1={f1:.2f}%, Prec={precision:.2f}%, Rec={recall:.2f}%, AUC={auc_score:.2f}%'
            )
            self.print_log(f'Time: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

            val_loss = self.eval(epoch, loader_name='val')
            self.val_loss_summary.append(val_loss)

            self.early_stop(val_loss)

            return self.early_stop.early_stop
        else:
            self.print_log("No valid batches processed in epoch")
            return False

    def eval(self, epoch, loader_name='val', result_file=None):
        self.model.trainable = False

        result_file_handle = None
        if result_file is not None:
            result_file_handle = open(result_file, 'w', encoding='utf-8')

        self.print_log(f'Evaluating epoch {epoch+1} on {loader_name}')

        loss = 0.0
        batch_count = 0
        label_list = []
        pred_list = []
        wrong_idx = []

        loader = self.data_loader[loader_name]
        progress = tqdm(range(len(loader)), desc=f"Eval {loader_name}")

        for batch_idx in progress:
            try:
                inputs, targets, idx = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)

                batch_loss, preds, _ = self.test_step(inputs, targets)

                loss += batch_loss.numpy()
                label_list.extend(targets.numpy())
                pred_list.extend(preds.numpy())
                batch_count += 1

                for i in range(len(targets)):
                    if preds[i] != int(targets[i]):
                        wrong_idx.append(idx[i])

                progress.set_postfix({'loss': f"{loss/batch_count:.4f}"})

            except Exception as e:
                self.print_log(f"Error in batch {batch_idx}: {e}")
                continue

        if batch_count > 0:
            loss /= batch_count
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

            if result_file_handle is not None:
                for i in range(len(pred_list)):
                    result_file_handle.write(f"{pred_list[i]} ==> {label_list[i]}\n")
                result_file_handle.close()

            self.print_log(
                f'{loader_name.capitalize()}: Loss={loss:.4f}, Acc={accuracy:.2f}%, '
                f'F1={f1:.2f}%, Prec={precision:.2f}%, Rec={recall:.2f}%, AUC={auc_score:.2f}%'
            )

            if loader_name == 'val':
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model()
                    self.print_log('New best model saved')

                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.print_log(f'New best F1 score: {f1:.2f}%')
            else:
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score

                self.cm_viz(pred_list, label_list)

            return loss
        else:
            self.print_log(f"No valid batches processed in {loader_name}")
            return float('inf')

    def save_model(self):
        try:
            weights_path = f"{self.model_path}_{self.test_subject[0]}.weights.h5"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)

            self.model.save_weights(weights_path)
            self.print_log(f"Model weights saved to {weights_path}")

            # Export to TFLite using concrete function
            tflite_path = f"{self.model_path}_{self.test_subject[0]}.tflite"
            acc_frames = self.arg.model_args.get('acc_frames', 64)
            acc_coords = self.arg.model_args.get('acc_coords', 3)

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, acc_frames, acc_coords], dtype=tf.float32, name='accelerometer')
            ])
            def serving_fn(accelerometer):
                mean = tf.reduce_mean(accelerometer, axis=1, keepdims=True)
                zero_mean = accelerometer - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                smv = tf.sqrt(sum_squared)
                acc_with_smv = tf.concat([smv, accelerometer], axis=-1)
                inputs = {'accelerometer': acc_with_smv}
                outputs = self.model(inputs, training=False)
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    return outputs[0]
                return outputs

            concrete_func = serving_fn.get_concrete_function()

            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.experimental_enable_resource_variables = True

            self.print_log("Converting model to TFLite...")
            tflite_model = converter.convert()

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            self.print_log(f"TFLite model exported to {tflite_path}")

            return True
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            traceback.print_exc()
            return False

    def add_avg_df(self, results):
        avg_result = {'test_subject': 'Average'}
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            values = [float(r[metric]) for r in results]
            avg_result[metric] = round(sum(values) / len(values), 2)
        results.append(avg_result)
        return results

    def eval_tflite(self, tflite_path, loader_name='test'):
        self.print_log(f'Evaluating TFLite model on {loader_name}')

        try:
            try:
                from ai_edge_litert import Interpreter as LiteRTInterpreter
                interpreter = LiteRTInterpreter(model_path=tflite_path)
                use_litert = True
                self.print_log("Using LiteRT interpreter")
            except ImportError:
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                use_litert = False
                self.print_log("Using tf.lite.Interpreter")

            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_index = input_details[0]['index']
            output_index = output_details[0]['index']

            loader = self.data_loader[loader_name]
            label_list = []
            pred_list = []

            for batch_idx in tqdm(range(len(loader)), desc=f"TFLite Eval {loader_name}"):
                inputs, targets, _ = loader[batch_idx]
                acc_data = inputs['accelerometer'].numpy()
                if acc_data.shape[2] == 4:
                    acc_data = acc_data[:, :, :3]
                elif acc_data.shape[2] != 3:
                    raise ValueError(f"Unexpected number of features: {acc_data.shape[2]}, expected 3 or 4")

                for i in range(acc_data.shape[0]):
                    input_data = acc_data[i:i+1]
                    interpreter.set_tensor(input_index, input_data.astype(np.float32))
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_index)
                    logits = output_data[0]
                    pred = int(logits > 0.5)
                    label_list.append(int(targets[i].numpy()))
                    pred_list.append(pred)

            accuracy, f1, recall, precision, auc = self.cal_metrics(label_list, pred_list)
            self.print_log(
                f'TFLite {loader_name.capitalize()}: Acc={accuracy:.2f}%, F1={f1:.2f}%, '
                f'Prec={precision:.2f}%, Rec={recall:.2f}%, AUC={auc:.2f}%'
            )

            if loader_name == 'test':
                self.cm_viz(pred_list, label_list)

        except Exception as e:
            self.print_log(f"Error evaluating TFLite model: {e}")
            traceback.print_exc()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Training parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')

            results = self.create_df()
            val_subjects = [38, 46]

            for test_subject in self.arg.subjects:
                if test_subject in val_subjects:
                    continue

                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.best_f1 = 0.0

                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects
                                     if s != test_subject and s not in val_subjects]

                self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                self.print_log(f"Train: {len(self.train_subjects)} subjects")
                self.print_log(f"Val: {len(self.val_subject)} subjects")
                self.print_log(f"Test: Subject {test_subject}")

                self.model = self.load_model(self.arg.model, self.arg.model_args)
                input_shapes = {
                    'accelerometer': (self.arg.model_args.get('acc_frames', 64),
                                    self.arg.model_args.get('acc_coords', 3)),
                }
                if 'skeleton' in self.arg.dataset_args.get('modalities', []):
                    input_shapes['skeleton'] = (self.arg.model_args.get('acc_frames', 64), 32, 3)
                self._build_model(input_shapes)

                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data issues")
                    continue

                self.load_optimizer()
                self.load_loss()
                self.early_stop.reset()

                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    early_stop = self.train(epoch)
                    if early_stop:
                        self.print_log(f"Early stopping at epoch {epoch+1}")
                        break

                best_weights = f"{self.model_path}_{self.test_subject[0]}.weights.h5"
                if os.path.exists(best_weights):
                    try:
                        self.model.load_weights(best_weights)
                        self.print_log(f"Loaded best weights from {best_weights}")
                    except Exception as e:
                        self.print_log(f"Error loading best weights: {e}")

                self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                self.eval(epoch=0, loader_name='test')

                self.loss_viz(self.train_loss_summary, self.val_loss_summary)

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

                tflite_path = f"{self.model_path}_{self.test_subject[0]}.tflite"
                if os.path.exists(tflite_path):
                    self.eval_tflite(tflite_path, loader_name='test')
                else:
                    self.print_log(f"TFLite model not found for subject {test_subject}")

            if results:
                results = self.add_avg_df(results)

                with open(f'{self.arg.work_dir}/scores.csv', 'w', newline='') as csvfile:
                    fieldnames = list(results[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)

                with open(f'{self.arg.work_dir}/scores.json', 'w') as jsonfile:
                    json.dump(results, jsonfile, indent=2)

                self.print_log("\n=== Final Results ===")
                for r in results:
                    self.print_log(f"Subject {r['test_subject']}: "
                                  f"Acc={r['accuracy']}%, "
                                  f"F1={r['f1_score']}%")

            self.print_log("Training completed successfully")

        elif self.arg.phase == 'test':
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for testing")
                return

            self.test_subject = [self.arg.subjects[0]]
            self.val_subject = [38, 46]
            self.train_subjects = [s for s in self.arg.subjects
                                 if s != self.test_subject[0] and s not in self.val_subject]

            if not self.load_data():
                self.print_log("Failed to load test data")
                return

            self.load_loss()

            self.print_log(f"Testing on subject {self.test_subject[0]}")
            self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)

            with open(f'{self.arg.work_dir}/test_results.json', 'w') as f:
                json.dump({
                    'test_subject': str(self.test_subject[0]),
                    'accuracy': self.test_accuracy,
                    'f1_score': self.test_f1,
                    'precision': self.test_precision,
                    'recall': self.test_recall,
                    'auc': self.test_auc
                }, f, indent=2)

            self.print_log("Testing completed successfully")

        elif self.arg.phase == 'tflite':
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for TFLite export")
                return

            self.test_subject = [self.arg.subjects[0]]

            self.print_log(f"Exporting TFLite model for subject {self.test_subject[0]}")

            acc_frames = self.arg.model_args.get('acc_frames', 64)
            acc_coords = self.arg.model_args.get('acc_coords', 3)

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, acc_frames, acc_coords], dtype=tf.float32, name='accelerometer')
            ])
            def export_fn(accelerometer):
                mean = tf.reduce_mean(accelerometer, axis=1, keepdims=True)
                zero_mean = accelerometer - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                smv = tf.sqrt(sum_squared)
                acc_with_smv = tf.concat([smv, accelerometer], axis=-1)
                logits, _ = self.model({'accelerometer': acc_with_smv}, training=False)
                return logits

            concrete_func = export_fn.get_concrete_function()

            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.experimental_enable_resource_variables = True

            self.print_log("Converting model to TFLite...")
            tflite_model = converter.convert()

            tflite_path = os.path.join(self.arg.work_dir, f'{self.arg.model_saved_name}_{self.test_subject[0]}.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            self.print_log(f"TFLite model exported to {tflite_path}")

            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            self.print_log(f"Input details: {input_details}")
            self.print_log(f"Output details: {output_details}")

            sample_input = np.random.rand(1, acc_frames, acc_coords).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            self.print_log(f"TFLite test successful: Output shape {output.shape}")

        self.print_log("TFLite export completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
