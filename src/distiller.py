import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import argparse
import yaml
import time
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
        dummy_input = {'accelerometer': tf.zeros((1, self.arg.teacher_args['acc_frames'], self.arg.teacher_args['acc_coords'])), 'skeleton': tf.zeros((1, self.arg.teacher_args['mocap_frames'], self.arg.teacher_args['num_joints'], 3))}
        _ = teacher_model(dummy_input, training=False)
        subject_id = self.test_subject[0] if hasattr(self, 'test_subject') and self.test_subject else None
        if subject_id:
            weight_path = f"{self.arg.teacher_weight}_{subject_id}.weights.h5"
            if os.path.exists(weight_path):
                teacher_model.load_weights(weight_path)
                logger.info(f"Loaded teacher weights from {weight_path}")
            else:
                model_path = f"{self.arg.teacher_weight}_{subject_id}.keras"
                if os.path.exists(model_path):
                    teacher_model = tf.keras.models.load_model(model_path)
                    logger.info(f"Loaded teacher model from {model_path}")
                else:
                    logger.warning(f"Teacher weights not found: {weight_path}")
        teacher_model.trainable = False
        return teacher_model
    def load_distillation_loss(self):
        temperature = getattr(self.arg, 'temperature', 4.5)
        alpha = getattr(self.arg, 'alpha', 0.6)
        pos_weight = self.pos_weights if hasattr(self, 'pos_weights') else None
        self.distillation_loss = DistillationLoss(temperature=temperature, alpha=alpha, pos_weight=pos_weight)
    @tf.function
    def distill_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            teacher_outputs = self.teacher_model(inputs, training=False)
            student_outputs = self.model(inputs, training=True)
            if isinstance(teacher_outputs, tuple) and len(teacher_outputs) > 1:
                teacher_logits, teacher_features = teacher_outputs
            else:
                teacher_logits = teacher_outputs
                teacher_features = None
            if isinstance(student_outputs, tuple) and len(student_outputs) > 1:
                student_logits, student_features = student_outputs
            else:
                student_logits = student_outputs
                student_features = None
            loss = self.distillation_loss(student_logits, teacher_logits, targets, teacher_features, student_features)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, student_logits
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
            loss, student_logits = self.distill_step(inputs, targets)
            probabilities = tf.sigmoid(student_logits)
            predictions = tf.cast(probabilities > 0.5, tf.int32)
            train_loss += loss.numpy()
            all_labels.extend(targets.numpy().flatten())
            all_preds.extend(predictions.numpy().flatten())
            all_probs.extend(probabilities.numpy().flatten())
            steps += 1
        train_loss /= steps
        train_time = time.time() - start_time
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
        self.train_loss_summary.append(float(train_loss))
        self.print_log(f'Epoch {epoch+1} Train - Time: {train_time:.2f}s, Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%')
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))
        if self.early_stop(val_loss):
            return True
        return False
    def start(self):
        if self.arg.phase in ['distill', 'train']:
            results = []
            train_subjects_fixed = self.arg.dataset_args.get('train_subjects_fixed', [45, 36, 29])
            val_subjects_fixed = self.arg.dataset_args.get('val_subjects_fixed', [38, 46])
            test_eligible_subjects = self.arg.dataset_args.get('test_eligible_subjects', [32, 39, 30, 31, 33, 34, 35, 37, 43, 44])
            for test_subject in test_eligible_subjects:
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.test_subject = [test_subject]
                self.val_subject = val_subjects_fixed
                self.train_subjects = train_subjects_fixed
                self.print_log(f'\n=== Distilling for subject {test_subject} ===')
                tf.keras.backend.clear_session()
                self.teacher_model = self.load_teacher_model()
                self.model = self.load_model()
                if not self.load_data():
                    continue
                self.load_optimizer()
                self.load_distillation_loss()
                self.early_stop.reset()
                best_epoch = 0
                for epoch in range(self.arg.num_epoch):
                    if self.train(epoch):
                        self.print_log(f'Early stopping triggered at epoch {epoch+1}')
                        best_epoch = epoch
                        break
                    best_epoch = epoch
                self.model.save_weights(f'{self.model_path}_{test_subject}.weights.h5')
                self.model.load_weights(f'{self.model_path}_{test_subject}.weights.h5')
                self.eval(epoch=best_epoch, loader_name='test')
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                result = {'test_subject': str(test_subject), 'accuracy': round(self.test_accuracy, 2), 'f1_score': round(self.test_f1, 2), 'precision': round(self.test_precision, 2), 'recall': round(self.test_recall, 2), 'auc': round(self.test_auc, 2)}
                results.append(result)
                self.print_log(f'Subject {test_subject} Results: Acc={self.test_accuracy:.2f}%, F1={self.test_f1:.2f}%, Precision={self.test_precision:.2f}%, Recall={self.test_recall:.2f}%, AUC={self.test_auc:.2f}%')
            df_results = pd.DataFrame(results)
            avg_row = df_results.mean(numeric_only=True).round(2)
            avg_row['test_subject'] = 'Average'
            df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
            df_results.to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)
            self.print_log("\nDistillation Results Summary:")
            self.print_log(df_results.to_string(index=False))

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
    parser.add_argument('--print-log', type=str2bool, default=True)
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
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    distiller = Distiller(args)
    distiller.start()

if __name__ == "__main__":
    main()
