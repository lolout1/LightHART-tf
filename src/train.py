#!/usr/bin/env python
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

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.model_path = os.path.join(self.arg.work_dir, 'models', self.arg.model_saved_name)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model = self.load_model()
    def import_class(self, import_str):
        mod_str, _, class_str = import_str.rpartition('.')
        if not mod_str:
            raise ImportError(f"Invalid import string: {import_str}")
        try:
            import importlib
            module = importlib.import_module(mod_str)
            return getattr(module, class_str)
        except Exception as e:
            logger.error(f"Failed to import {class_str} from {mod_str}: {e}")
            raise ImportError(f"Cannot import {class_str} from {mod_str}")
    def load_model(self):
        model_class = self.import_class(self.arg.model)
        return model_class(**self.arg.model_args)
    def calculate_class_weights(self, labels):
        from collections import Counter
        counter = Counter(labels)
        pos_weight = counter[0] / counter[1] if 1 in counter and 0 in counter else 1.0
        logger.info(f"Class distribution - 0: {counter[0]}, 1: {counter[1]}, pos_weight: {pos_weight:.4f}")
        return tf.constant(pos_weight, dtype=tf.float32)
    def load_optimizer(self):
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.arg.base_lr)
    def load_loss(self):
        self.pos_weights = getattr(self, 'pos_weights', tf.constant(1.0))
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            if len(y_true.shape) == 1 and len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
                y_true = tf.expand_dims(y_true, -1)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * (self.pos_weights - 1.0) + 1.0
            return tf.reduce_mean(weights * bce)
        self.criterion = weighted_bce
    def load_data(self):
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
        Feeder = self.import_class(self.arg.feeder)
        builder = prepare_smartfallmm_tf(self.arg)
        if self.arg.phase == 'train':
            self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False)
            self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
            self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
            if not all([self.norm_train.get('labels', []).size > 0, self.norm_val.get('labels', []).size > 0, self.norm_test.get('labels', []).size > 0]):
                logger.warning(f"Data loading failed - Train: {self.norm_train.get('labels', []).size}, Val: {self.norm_val.get('labels', []).size}, Test: {self.norm_test.get('labels', []).size}")
                return False
            self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
            use_smv = getattr(self.arg, 'use_smv', False)
            self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, use_smv=use_smv)
            self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size, use_smv=use_smv)
            self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size, use_smv=use_smv)
            return True
    def calculate_metrics(self, targets, predictions):
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        accuracy = accuracy_score(targets, predictions) * 100
        f1 = f1_score(targets, predictions, zero_division=0) * 100
        precision = precision_score(targets, predictions, zero_division=0) * 100
        recall = recall_score(targets, predictions, zero_division=0) * 100
        try:
            auc = roc_auc_score(targets, predictions) * 100
        except:
            auc = 0.0
        return accuracy, f1, recall, precision, auc
    def train(self, epoch):
        loader = self.data_loader['train']
        train_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        for batch_idx in range(len(loader)):
            inputs, targets, _ = loader[batch_idx]
            targets = tf.cast(targets, tf.float32)
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                if len(targets.shape) == 1:
                    targets = tf.expand_dims(targets, -1)
                if len(logits.shape) == 1:
                    logits = tf.expand_dims(logits, -1)
                loss = self.criterion(targets, logits)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
            train_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            steps += 1
        train_loss /= steps
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
        self.train_loss_summary.append(float(train_loss))
        logger.info(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}%')
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))
        if self.early_stop(val_loss):
            return True
        return False
    def eval(self, epoch, loader_name='val'):
        loader = self.data_loader[loader_name]
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        for batch_idx in range(len(loader)):
            inputs, targets, _ = loader[batch_idx]
            targets = tf.cast(targets, tf.float32)
            outputs = self.model(inputs, training=False)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            if len(targets.shape) == 1:
                targets = tf.expand_dims(targets, -1)
            if len(logits.shape) == 1:
                logits = tf.expand_dims(logits, -1)
            loss = self.criterion(targets, logits)
            predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
            eval_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            steps += 1
        if steps == 0:
            logger.warning(f"No data in {loader_name} loader")
            return float('inf')
        eval_loss /= steps
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds)
        logger.info(f'{loader_name.capitalize()}: Loss={eval_loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}%')
        if loader_name == 'val' and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.save_model()
        elif loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
        return eval_loss
    def save_model(self):
        self.model.save_weights(f'{self.model_path}_{self.test_subject[0]}.weights.h5')
        logger.info('Model weights saved')
    def start(self):
        if self.arg.phase == 'train':
            results = []
            for test_subject in self.arg.subjects:
                if test_subject in [38, 46]: continue
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = [38, 46]
                self.train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in [38, 46]]
                logger.info(f'\n=== Testing on subject {test_subject} ===')
                self.model = self.load_model()
                if not self.load_data(): 
                    logger.warning(f"Failed to load data for subject {test_subject}")
                    continue
                self.load_optimizer()
                self.load_loss()
                self.early_stop.reset()
                for epoch in range(self.arg.num_epoch):
                    if self.train(epoch): break
                self.model.load_weights(f'{self.model_path}_{test_subject}.weights.h5')
                self.eval(epoch=0, loader_name='test')
                result = {
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                results.append(result)
            pd.DataFrame(results).to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)

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
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    trainer = Trainer(args)
    trainer.start()

if __name__ == "__main__":
    main()
