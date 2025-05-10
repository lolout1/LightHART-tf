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
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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

class MetricLogger:
    def __init__(self):
        self.metrics = {'train': [], 'val': [], 'test': []}
    def log(self, phase, epoch, loss, accuracy, f1, precision, recall, auc):
        metrics = {'epoch': epoch, 'loss': loss, 'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}
        self.metrics[phase].append(metrics)
    def save(self, work_dir, subject):
        for phase in ['train', 'val', 'test']:
            if self.metrics[phase]:
                df = pd.DataFrame(self.metrics[phase])
                df.to_csv(os.path.join(work_dir, f'{phase}_metrics_subject_{subject}.csv'), index=False)

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
        self.metric_logger = MetricLogger()
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.model_path = os.path.join(self.arg.work_dir, 'models', self.arg.model_saved_name)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_config(arg.config, arg.work_dir)
        self.model = self.load_model()
        self.print_log(f'Model Parameters: {self.count_parameters()}')
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
        dummy_input = {'accelerometer': tf.zeros((1, self.arg.model_args.get('acc_frames', 64), self.arg.model_args.get('acc_coords', 3))), 'skeleton': tf.zeros((1, self.arg.model_args.get('mocap_frames', 64), self.arg.model_args.get('num_joints', 32), 3))}
        _ = model(dummy_input, training=False)
        return model
    def calculate_class_weights(self, labels):
        from collections import Counter
        counter = Counter(labels)
        total = sum(counter.values())
        weight_for_0 = total / (2.0 * counter[0]) if counter[0] > 0 else 1.0
        weight_for_1 = total / (2.0 * counter[1]) if counter[1] > 0 else 1.0
        class_weight = weight_for_1 / weight_for_0 if weight_for_0 > 0 else 1.0
        self.print_log(f'Class distribution - 0: {counter[0]}, 1: {counter[1]}, pos_weight: {class_weight:.4f}')
        return tf.constant(class_weight, dtype=tf.float32)
    def load_optimizer(self):
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=self.arg.base_lr, decay_steps=self.arg.num_epoch * len(self.data_loader['train']), alpha=0.1)
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    def load_loss(self):
        self.pos_weights = getattr(self, 'pos_weights', tf.constant(1.0))
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            if len(y_true.shape) == 1:
                y_true = tf.expand_dims(y_true, -1)
            if len(y_pred.shape) == 1:
                y_pred = tf.expand_dims(y_pred, -1)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * self.pos_weights + (1 - y_true)
            return tf.reduce_mean(weights * bce)
        self.criterion = weighted_bce
    def load_data(self):
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
        Feeder = self.import_class(self.arg.feeder)
        builder = prepare_smartfallmm_tf(self.arg)
        if self.arg.phase == 'train':
            self.print_log(f'Loading data for train: {self.train_subjects}, val: {self.val_subject}, test: {self.test_subject}')
            self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False)
            self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
            self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
            train_size = len(self.norm_train.get('labels', []))
            val_size = len(self.norm_val.get('labels', []))
            test_size = len(self.norm_test.get('labels', []))
            self.print_log(f'Data sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}')
            if not all([train_size > 0, val_size > 0, test_size > 0]):
                self.print_log(f'Insufficient data for subject {self.test_subject[0]}')
                return False
            self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
            use_smv = getattr(self.arg, 'use_smv', False)
            self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, use_smv=use_smv)
            self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size, use_smv=use_smv)
            self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv)
            self.print_log(f'Train batches: {len(self.data_loader["train"])}, Val batches: {len(self.data_loader["val"])}, Test batches: {len(self.data_loader["test"])}')
            return True
    def calculate_metrics(self, targets, predictions, probabilities=None):
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        accuracy = accuracy_score(targets, predictions) * 100
        f1 = f1_score(targets, predictions, zero_division=0) * 100
        precision = precision_score(targets, predictions, zero_division=0) * 100
        recall = recall_score(targets, predictions, zero_division=0) * 100
        try:
            if probabilities is not None:
                auc = roc_auc_score(targets, probabilities) * 100
            else:
                auc = roc_auc_score(targets, predictions) * 100
        except:
            auc = 0.0
        return accuracy, f1, recall, precision, auc
    @tf.function
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
        loader = self.data_loader['train']
        train_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        start_time = time.time()
        for batch_idx in range(len(loader)):
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
            if batch_idx % 10 == 0:
                self.print_log(f'Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, Loss: {loss:.4f}')
        train_loss /= steps
        train_time = time.time() - start_time
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
        self.train_loss_summary.append(float(train_loss))
        self.metric_logger.log('train', epoch, train_loss, accuracy, f1, precision, recall, auc_score)
        self.print_log(f'Epoch {epoch+1} Train - Time: {train_time:.2f}s, Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%')
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))
        if self.early_stop(val_loss):
            return True
        return False
    @tf.function
    def eval_step(self, inputs, targets):
        outputs = self.model(inputs, training=False)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        loss = self.criterion(targets, logits)
        return loss, logits
    def eval(self, epoch, loader_name='val'):
        loader = self.data_loader[loader_name]
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        start_time = time.time()
        for batch_idx in range(len(loader)):
            inputs, targets, _ = loader[batch_idx]
            targets = tf.cast(targets, tf.float32)
            loss, logits = self.eval_step(inputs, targets)
            probabilities = tf.sigmoid(logits)
            predictions = tf.cast(probabilities > 0.5, tf.int32)
            eval_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            all_probs.extend(probabilities.numpy().flatten())
            steps += 1
        if steps == 0:
            return float('inf')
        eval_loss /= steps
        eval_time = time.time() - start_time
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
        phase = loader_name
        self.metric_logger.log(phase, epoch, eval_loss, accuracy, f1, precision, recall, auc_score)
        self.print_log(f'Epoch {epoch+1} {phase.capitalize()} - Time: {eval_time:.2f}s, Loss: {eval_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%')
        if loader_name == 'val' and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.save_model()
        elif loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
            self.save_test_predictions(all_labels, all_preds, all_probs)
            self.plot_confusion_matrix(all_labels, all_preds)
        return eval_loss
    def save_model(self):
        self.model.save_weights(f'{self.model_path}_{self.test_subject[0]}.weights.h5')
        self.print_log(f'Model weights saved for subject {self.test_subject[0]}')
    def save_test_predictions(self, labels, predictions, probabilities):
        df = pd.DataFrame({'true': labels, 'pred': predictions, 'prob': probabilities})
        df.to_csv(os.path.join(self.arg.work_dir, f'test_predictions_subject_{self.test_subject[0]}.csv'), index=False)
    def plot_confusion_matrix(self, labels, predictions):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        classes = ['Non-Fall', 'Fall'] if self.arg.dataset_args.get('task') == 'fd' else [str(i) for i in range(self.arg.model_args.get('num_classes', 2))]
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Subject {self.test_subject[0]}')
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.savefig(os.path.join(self.arg.work_dir, f'confusion_matrix_subject_{self.test_subject[0]}.png'))
        plt.close()
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
            results = []
            train_subjects_fixed = self.arg.dataset_args.get('train_subjects_fixed', [45, 36, 29])
            val_subjects_fixed = self.arg.dataset_args.get('val_subjects_fixed', [38, 46])
            test_eligible_subjects = self.arg.dataset_args.get('test_eligible_subjects', [32, 39, 30, 31, 33, 34, 35, 37, 43, 44])
            self.print_log(f'Starting cross-validation with fixed train: {train_subjects_fixed}, fixed val: {val_subjects_fixed}, test candidates: {test_eligible_subjects}')
            for test_subject in test_eligible_subjects:
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = val_subjects_fixed
                self.train_subjects = train_subjects_fixed
                self.metric_logger = MetricLogger()
                self.print_log(f'\n=== Testing on subject {test_subject} ===')
                self.print_log(f'Train subjects: {self.train_subjects}')
                self.print_log(f'Val subjects: {self.val_subject}')
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
                    if self.train(epoch):
                        self.print_log(f'Early stopping triggered at epoch {epoch+1}')
                        best_epoch = epoch
                        break
                    best_epoch = epoch
                self.model.load_weights(f'{self.model_path}_{test_subject}.weights.h5')
                self.eval(epoch=best_epoch, loader_name='test')
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                self.metric_logger.save(self.arg.work_dir, test_subject)
                result = {'test_subject': str(test_subject), 'best_epoch': best_epoch + 1, 'accuracy': round(self.test_accuracy, 2), 'f1_score': round(self.test_f1, 2), 'precision': round(self.test_precision, 2), 'recall': round(self.test_recall, 2), 'auc': round(self.test_auc, 2)}
                results.append(result)
                self.print_log(f'Subject {test_subject} Results: Acc={self.test_accuracy:.2f}%, F1={self.test_f1:.2f}%, Precision={self.test_precision:.2f}%, Recall={self.test_recall:.2f}%, AUC={self.test_auc:.2f}%')
            df_results = pd.DataFrame(results)
            avg_row = df_results.mean(numeric_only=True).round(2)
            avg_row['test_subject'] = 'Average'
            df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
            df_results.to_csv(os.path.join(self.arg.work_dir, 'final_scores.csv'), index=False)
            self.print_log("\nFinal Results Summary:")
            self.print_log(df_results.to_string(index=False))
            self.print_log(f'\nAverage Metrics:')
            self.print_log(f'Accuracy: {avg_row["accuracy"]:.2f}%')
            self.print_log(f'F1-Score: {avg_row["f1_score"]:.2f}%')
            self.print_log(f'Precision: {avg_row["precision"]:.2f}%')
            self.print_log(f'Recall: {avg_row["recall"]:.2f}%')
            self.print_log(f'AUC: {avg_row["auc"]:.2f}%')

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
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if not hasattr(args, k) or getattr(args, k) is None:
                    setattr(args, k, v)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[0])
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    trainer = Trainer(args)
    trainer.start()

if __name__ == "__main__":
    main()
