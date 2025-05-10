#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import argparse
import yaml
from train import Trainer, str2bool
from utils.loss import DistillationLoss

logger = logging.getLogger('distiller')

class Distiller(Trainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = None
        self.distillation_loss = None
    def load_teacher_model(self):
        teacher_class = self.import_class(self.arg.teacher_model)
        teacher_model = teacher_class(**self.arg.teacher_args)
        subject_id = self.test_subject[0] if hasattr(self, 'test_subject') and self.test_subject else None
        if subject_id:
            weight_path = f"{self.arg.teacher_weight}_{subject_id}.weights.h5"
            if os.path.exists(weight_path):
                teacher_model.load_weights(weight_path)
            else:
                logger.warning(f"Teacher weights not found: {weight_path}")
        teacher_model.trainable = False
        return teacher_model
    def load_distillation_loss(self):
        temperature = getattr(self.arg, 'temperature', 4.5)
        alpha = getattr(self.arg, 'alpha', 0.6)
        pos_weight = self.pos_weights if hasattr(self, 'pos_weights') else None
        self.distillation_loss = DistillationLoss(temperature=temperature, alpha=alpha, pos_weight=pos_weight)
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
                teacher_outputs = self.teacher_model(inputs, training=False)
                student_outputs = self.model(inputs, training=True)
                if isinstance(teacher_outputs, tuple) and len(teacher_outputs) > 1:
                    teacher_logits, teacher_features = teacher_outputs
                else:
                    teacher_logits = teacher_outputs
                    teacher_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                if isinstance(student_outputs, tuple) and len(student_outputs) > 1:
                    student_logits, student_features = student_outputs
                else:
                    student_logits = student_outputs
                    student_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                loss = self.distillation_loss(
                    student_logits, teacher_logits, targets,
                    teacher_features, student_features
                )
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            predictions = tf.cast(tf.sigmoid(student_logits) > 0.5, tf.int32)
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
    def start(self):
        if self.arg.phase == 'distill':
            results = []
            for test_subject in [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]:
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = [38, 46]
                self.train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in [38, 46]]
                logger.info(f'\n=== Distilling for subject {test_subject} ===')
                tf.keras.backend.clear_session()
                self.teacher_model = self.load_teacher_model()
                self.model = self.load_model()
                if not self.load_data(): continue
                self.load_optimizer()
                self.load_distillation_loss()
                self.early_stop.reset()
                for epoch in range(self.arg.num_epoch):
                    if self.train(epoch): break
                self.model.save_weights(f'{self.model_path}_{test_subject}.weights.h5')
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
            pd.DataFrame(results).to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)

def get_distill_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--work-dir', type=str, default='distilled')
    parser.add_argument('--model-saved-name', type=str, default='distilled_model')
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--phase', type=str, default='distill')
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--teacher-args', type=str, default=None)
    parser.add_argument('--teacher-weight', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--model-args', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=4.5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=80)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    parser.add_argument('--dataset-args', type=str, default=None)
    parser.add_argument('--subjects', nargs='+', type=int, default=None)
    parser.add_argument('--feeder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--use-smv', type=str2bool, default=False)
    return parser

def main():
    parser = get_distill_args()
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
    distiller = Distiller(args)
    distiller.start()

if __name__ == "__main__":
    main()
