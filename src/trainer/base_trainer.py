import os
import time
import datetime
import yaml
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import traceback

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.test_metrics = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{self.timestamp}"
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                           handlers=[logging.FileHandler(f"{self.arg.work_dir}/train.log"), logging.StreamHandler()])
        self.model_path = f'{self.arg.work_dir}/models/{self.arg.model_saved_name}'
        self.save_config(arg.config, arg.work_dir)
        self._setup_tensorflow()
        self._import_modules()
        if self.arg.phase == 'train':
            self.model = self._load_model(arg.model, arg.model_args)
        else:
            self.model = self._load_weights(arg.weights)
        num_params = self._count_parameters()
        logging.info(f'Model parameters: {num_params:,} ({num_params/(1024**2):.2f} MB)')
    
    def _setup_tensorflow(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Found GPU: {gpu.name}")
                except RuntimeError as e:
                    logging.warning(f"Error configuring GPU: {e}")
        if getattr(self.arg, 'mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info(f"Mixed precision enabled: {policy}")
    
    def _import_modules(self):
        import importlib
        def import_class(name):
            module_name, class_name = name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        self.import_class = import_class
        from utils.dataset_tf import prepare_dataset, process_dataset, DatasetTF
        from utils.metrics import calculate_metrics
        self.prepare_dataset = prepare_dataset
        self.process_dataset = process_dataset
        self.DatasetTF = DatasetTF
        self.calculate_metrics = calculate_metrics
    
    def _load_model(self, model_name, model_args):
        try:
            ModelClass = self.import_class(model_name)
            model = ModelClass(**model_args)
            sample_shape = (1, model_args.get('acc_frames', 64), model_args.get('acc_coords', 4))
            sample_input = tf.ones(sample_shape, dtype=tf.float32)
            _ = model(sample_input)
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def _load_weights(self, weights_path):
        try:
            if weights_path.endswith('.h5') or weights_path.endswith('.weights.h5'):
                model = self._load_model(self.arg.model, self.arg.model_args)
                model.load_weights(weights_path)
                return model
            else:
                return tf.keras.models.load_model(weights_path)
        except Exception as e:
            logging.error(f"Error loading weights: {e}")
            raise
    
    def _count_parameters(self):
        return np.sum([np.prod(v.shape) for v in self.model.trainable_variables])
    
    def save_config(self, src_path, dest_dir):
        config_filename = os.path.basename(src_path)
        with open(src_path, 'r') as f_src:
            config = f_src.read()
            with open(f'{dest_dir}/{config_filename}', 'w') as f_dst:
                f_dst.write(config)
    
    def save_model(self, epoch, val_loss, metrics=None):
        try:
            save_dir = os.path.dirname(self.model_path)
            os.makedirs(save_dir, exist_ok=True)
            weights_path = f"{self.model_path}_{self.test_subject[0]}.weights.h5"
            self.model.save_weights(weights_path)
            logging.info(f"Model weights saved to {weights_path}")
            self._export_tflite()
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def _export_tflite(self):
        try:
            tflite_path = f"{self.model_path}_{self.test_subject[0]}.tflite"
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 64, 4], dtype=tf.float32)])
            def serve_function(x):
                return {'output': tf.sigmoid(self.model(x, training=False)[0])}
            concrete_func = serve_function.get_concrete_function()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logging.info(f"Successfully exported model to TFLite: {tflite_path}")
            return True
        except Exception as e:
            logging.error(f"Error exporting to TFLite: {e}")
            return False
    
    def load_data(self):
        try:
            root_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
            age_groups = self.arg.dataset_args.get('age_group', ['young'])
            modalities = self.arg.dataset_args.get('modalities', ['accelerometer'])
            sensors = self.arg.dataset_args.get('sensors', ['watch'])
            window_size = self.arg.dataset_args.get('max_length', 64)
            dataset = self.prepare_dataset(root_dir, age_groups, modalities, sensors)
            self.norm_train = self.process_dataset(dataset, self.train_subjects, window_size)
            if not self.norm_train:
                logging.error("Failed to process training data")
                return False
            self.data_loader = {}
            self.data_loader['train'] = self.DatasetTF(self.norm_train, self.arg.batch_size)
            self.norm_val = self.process_dataset(dataset, self.val_subject, window_size)
            if not self.norm_val:
                logging.error("Failed to process validation data")
                return False
            self.data_loader['val'] = self.DatasetTF(self.norm_val, self.arg.val_batch_size)
            self._calculate_class_weights()
            self.norm_test = self.process_dataset(dataset, self.test_subject, window_size)
            if not self.norm_test:
                logging.error("Failed to process test data")
                return False
            self.data_loader['test'] = self.DatasetTF(self.norm_test, self.arg.test_batch_size)
            self._plot_distributions()
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def _calculate_class_weights(self):
        labels = self.norm_train['labels']
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        if num_pos > 0 and num_neg > 0:
            self.pos_weight = float(num_neg) / num_pos
        else:
            self.pos_weight = 1.0
        logging.info(f"Class weights: positive={self.pos_weight:.2f}")
    
    def _plot_distributions(self):
        plt.figure(figsize=(10, 6))
        for i, split in enumerate(['train', 'val', 'test']):
            data = getattr(self, f'norm_{split}')
            if 'labels' in data:
                labels = data['labels']
                values, counts = np.unique(labels, return_counts=True)
                plt.subplot(1, 3, i+1)
                plt.bar(values, counts)
                plt.title(f'{split.capitalize()} Distribution')
                plt.xlabel('Labels')
                plt.ylabel('Count')
                for j, count in enumerate(counts):
                    plt.text(values[j], count + 5, str(count), ha='center')
        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/label_distributions.png')
        plt.close()
    
    def load_optimizer(self):
        name = self.arg.optimizer.lower()
        lr = self.arg.base_lr
        if name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == 'adamw':
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.arg.weight_decay)
        elif name == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            logging.warning(f"Unknown optimizer {name}, using Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        logging.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
    
    def load_loss(self):
        name = getattr(self.arg, 'loss', 'bce').lower()
        if name == 'bce':
            def weighted_bce(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
                weights = y_true * (self.pos_weight - 1.0) + 1.0
                return tf.reduce_mean(weights * bce)
            self.criterion = weighted_bce
        elif name == 'binary_focal':
            def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
                y_true = tf.cast(y_true, tf.float32)
                prob = tf.sigmoid(y_pred)
                pt = tf.where(tf.equal(y_true, 1.0), prob, 1.0 - prob)
                alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
                focal_loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
                return tf.reduce_mean(focal_loss)
            self.criterion = binary_focal_loss
        else:
            logging.warning(f"Unknown loss {name}, using BCE")
            self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        logging.info(f"Using loss: {name}")
    
    def train(self, epoch):
        self.model.trainable = True
        total_loss = 0.0
        all_labels = []
        all_preds = []
        batch_count = 0
        loader = self.data_loader['train']
        progress = tqdm(range(len(loader)), desc=f"Epoch {epoch+1}/{self.arg.num_epoch}")
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                with tf.GradientTape() as tape:
                    logits, _ = self.model(inputs, training=True)
                    logits = tf.reshape(logits, [-1])
                    loss = self.criterion(targets, logits)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss.numpy()
                batch_count += 1
                preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32).numpy()
                all_labels.extend(targets.numpy())
                all_preds.extend(preds)
                progress.set_postfix({'loss': f"{total_loss/batch_count:.4f}"})
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            # Important fix: properly handle the return value from calculate_metrics
            metrics_dict, metrics_values = self.calculate_metrics(all_labels, all_preds)
            accuracy, f1, recall, precision, auc = metrics_values  # Properly unpack the tuple
            
            logging.info(f"Train Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%, AUC={auc:.2f}%")
            self.train_loss_summary.append(avg_loss)
            val_loss, val_metrics = self.evaluate(epoch, 'val')
            self.val_loss_summary.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                logging.info(f"New best validation loss: {val_loss:.4f}")
                self.save_model(epoch, val_loss, val_metrics)
            # Get f1 from the metrics tuple
            if val_metrics[1] > self.best_f1:  # f1 is at index 1
                self.best_f1 = val_metrics[1]
                logging.info(f"New best validation F1: {val_metrics[1]:.2f}%")
            if len(self.val_loss_summary) >= 15:
                min_idx = np.argmin(self.val_loss_summary[-15:])
                if min_idx == 0:
                    logging.info("Early stopping: no improvement in last 15 epochs")
                    return True
            return False
        else:
            logging.error("No valid batches processed in epoch")
            return False
    
    def evaluate(self, epoch, split='val'):
        self.model.trainable = False
        total_loss = 0.0
        all_labels = []
        all_preds = []
        batch_count = 0
        loader = self.data_loader[split]
        progress = tqdm(range(len(loader)), desc=f"Eval {split}")
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                logits, _ = self.model(inputs, training=False)
                logits = tf.reshape(logits, [-1])
                loss = self.criterion(targets, logits)
                total_loss += loss.numpy()
                batch_count += 1
                preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32).numpy()
                all_labels.extend(targets.numpy())
                all_preds.extend(preds)
                progress.set_postfix({'loss': f"{total_loss/batch_count:.4f}"})
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            # Important fix: properly handle the return value from calculate_metrics
            metrics_dict, metrics_values = self.calculate_metrics(all_labels, all_preds)
            accuracy, f1, recall, precision, auc = metrics_values  # Properly unpack the tuple
            
            logging.info(f"{split.capitalize()} Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%, AUC={auc:.2f}%")
            if split == 'test':
                self.test_metrics = metrics_dict
            return avg_loss, metrics_values  # Return the tuple directly for easier access
        else:
            logging.error(f"No valid batches processed in {split} evaluation")
            return float('inf'), (0, 0, 0, 0, 0)  # Return a tuple of zeros for metrics_values
    
    def start(self):
        if self.arg.phase == 'train':
            logging.info("Training parameters:")
            for key, value in vars(self.arg).items():
                logging.info(f"  {key}: {value}")
            results = []
            val_subjects = [38, 46]
            for test_subject in self.arg.subjects:
                if test_subject in val_subjects:
                    continue
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.best_f1 = 0.0
                self.train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in val_subjects]
                self.val_subject = val_subjects
                self.test_subject = [test_subject]
                logging.info(f"=== Testing on subject {test_subject} ===")
                logging.info(f"Train subjects: {self.train_subjects}")
                logging.info(f"Val subjects: {self.val_subject}")
                self.model = self._load_model(self.arg.model, self.arg.model_args)
                if not self.load_data():
                    logging.error(f"Skipping subject {test_subject} due to data loading failure")
                    continue
                try:
                    self.load_optimizer()
                    self.load_loss()
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        early_stop = self.train(epoch)
                        if early_stop:
                            logging.info(f"Early stopping at epoch {epoch+1}")
                            break
                    best_model_path = f"{self.model_path}_{self.test_subject[0]}.weights.h5"
                    if os.path.exists(best_model_path):
                        try:
                            self.model.load_weights(best_model_path)
                            logging.info(f"Loaded best model weights from {best_model_path}")
                        except Exception as e:
                            logging.error(f"Error loading best model: {e}")
                    test_loss, test_metrics = self.evaluate(0, 'test')
                    self._plot_loss_curves()
                    # Access metrics from the tuple
                    accuracy, f1, recall, precision, auc = test_metrics
                    results.append({
                        'test_subject': str(self.test_subject[0]),
                        'accuracy': round(accuracy, 2),
                        'f1_score': round(f1, 2), 
                        'precision': round(precision, 2),
                        'recall': round(recall, 2),
                        'auc': round(auc, 2)
                    })
                except Exception as e:
                    logging.error(f"Error during training for subject {test_subject}: {e}")
                    logging.error(traceback.format_exc())
            if results:
                avg_result = {'test_subject': 'Average'}
                for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
                    avg_result[metric] = round(sum(float(r[metric]) for r in results) / len(results), 2)
                results.append(avg_result)
                with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                    json.dump(results, f, indent=4)
                logging.info(f"Results saved to {self.arg.work_dir}/scores.json")
                logging.info("Final results:")
                for r in results:
                    logging.info(f"Subject {r['test_subject']}: Acc={r['accuracy']}%, F1={r['f1_score']}%, Prec={r['precision']}%, Rec={r['recall']}%, AUC={r['auc']}%")
        else:
            if not self.arg.weights:
                logging.error("No weights provided for test mode")
                return
            self.test_subject = [self.arg.subjects[0]]
            if not self.load_data():
                logging.error("Failed to load test data")
                return
            test_loss, test_metrics = self.evaluate(0, 'test')
            # Access metrics from the tuple
            accuracy, f1, recall, precision, auc = test_metrics
            with open(f'{self.arg.work_dir}/test_results.json', 'w') as f:
                json.dump({
                    'test_subject': str(self.test_subject[0]),
                    'loss': float(test_loss),
                    'metrics': {
                        'accuracy': float(accuracy),
                        'f1': float(f1),
                        'precision': float(precision),
                        'recall': float(recall),
                        'auc': float(auc)
                    }
                }, f, indent=4)
    
    def _plot_loss_curves(self):
        if not self.train_loss_summary or not self.val_loss_summary:
            return
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_loss_summary) + 1)
        plt.plot(epochs, self.train_loss_summary, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_loss_summary, 'r-', label='Validation Loss')
        plt.title(f'Training and Validation Loss (Subject {self.test_subject[0]})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.arg.work_dir}/loss_curves_{self.test_subject[0]}.png')
        plt.close()
