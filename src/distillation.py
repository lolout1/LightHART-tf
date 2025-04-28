import argparse
import yaml
import os
import logging
import tensorflow as tf
import numpy as np
import signal
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from base_trainer import BaseTrainer
from utils.dataset_tf import UTD_MM_TF
import pandas as pd
from datetime import datetime

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"experiments/distill_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/metrics_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Training step timed out")

signal.signal(signal.SIGALRM, timeout_handler)

class Distillation(BaseTrainer):
    def __init__(self, arg):
        """Initialize with arguments."""
        super().__init__(arg)
        self.teacher_model = None
        self.student_model = self.model
        self.best_loss = float('inf')
        self.timeout_seconds = 300  # 5-minute timeout per step
        self.metrics_log = []  # Store per-epoch metrics
        self.validate_config()

    def validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['phase', 'model', 'teacher_model', 'dataset', 'subjects', 'model_args', 'teacher_args', 'dataset_args']
        for key in required_keys:
            if not hasattr(self.arg, key):
                raise ValueError(f"Configuration missing required key: {key}")
        if self.arg.phase != 'distill':
            raise ValueError("Configuration 'phase' must be 'distill'")
        if not self.arg.subjects:
            raise ValueError("Configuration 'subjects' must be non-empty")
        if 'modalities' not in self.arg.dataset_args:
            raise ValueError("Configuration 'dataset_args.modalities' must be specified")
        logger.info("Configuration validated successfully")

    def compute_metrics(self, y_true, y_pred):
        """Compute classification metrics."""
        y_true = np.array(y_true).flatten()
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int).flatten()
        y_pred_proba = np.array(y_pred).flatten()
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        return metrics

    def evaluate_model(self, model, loader, phase='val', modalities=['skeleton']):
        """Evaluate model on given loader."""
        y_true, y_pred = [], []
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            inputs, targets, _ = batch
            try:
                input_dict = {modality: inputs[modality] for modality in modalities}
                outputs = model(input_dict, training=False)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                if isinstance(logits, tf.Tensor):
                    logits = logits.numpy()
                loss = tf.keras.losses.binary_crossentropy(targets, logits).numpy()
                total_loss += loss
                batch_count += 1
                y_true.extend(targets.numpy() if isinstance(targets, tf.Tensor) else targets)
                y_pred.extend(logits.flatten())
            except Exception as e:
                logger.error(f"Error in evaluation for {phase}: {e}")
                continue
        if batch_count == 0:
            logger.warning(f"No batches processed for {phase}")
            return float('inf'), {}
        avg_loss = total_loss / batch_count
        metrics = self.compute_metrics(y_true, y_pred)
        return avg_loss, metrics

    def load_teacher_weights(self):
        """Load or train teacher weights."""
        weights_path = f"{self.arg.teacher_weight}_{self.test_subject[0]}.weights.h5"
        try:
            if os.path.exists(weights_path):
                self.teacher_model = self.load_model(self.arg.teacher_model, self.arg.teacher_args)
                self.teacher_model.load_weights(weights_path)
                logger.info(f"Loaded teacher weights from {weights_path}")
            else:
                logger.info(f"Teacher weights not found at {weights_path}, training teacher...")
                self.train_teacher()
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                self.teacher_model.save_weights(weights_path)
                logger.info(f"Saved teacher weights to {weights_path}")
        except Exception as e:
            logger.error(f"Error loading/training teacher: {str(e)}")
            raise

    def train_teacher(self):
        """Train teacher model with skeleton data."""
        try:
            train_loader = UTD_MM_TF(
                dataset=self.norm_train,
                batch_size=self.arg.batch_size,
                modalities=['skeleton']
            )
            val_loader = UTD_MM_TF(
                dataset=self.norm_val,
                batch_size=self.arg.val_batch_size,
                modalities=['skeleton']
            )

            self.teacher_model = self.load_model(self.arg.teacher_model, self.arg.teacher_args)
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
            self.teacher_model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            for epoch in range(self.arg.num_epoch):
                logger.info(f"Starting teacher training epoch {epoch+1}/{self.arg.num_epoch}")
                train_loss = 0.0
                batch_count = 0
                y_true, y_pred = [], []
                for batch in train_loader:
                    inputs, targets, _ = batch
                    logger.debug(f"Batch: inputs={inputs['skeleton'].shape}, targets={targets.shape}")

                    if np.any(np.isnan(inputs['skeleton'])) or np.any(np.isnan(targets)):
                        raise ValueError("NaN detected in inputs or targets")

                    signal.alarm(self.timeout_seconds)
                    try:
                        loss_values = self.teacher_model.train_on_batch(inputs, targets)
                        outputs = self.teacher_model.predict_on_batch(inputs)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        if isinstance(logits, tf.Tensor):
                            logits = logits.numpy()
                    except TimeoutException:
                        logger.error(f"Training step timed out after {self.timeout_seconds} seconds")
                        raise
                    finally:
                        signal.alarm(0)

                    loss = float(loss_values[0])
                    if np.isnan(loss):
                        raise ValueError("NaN loss detected")
                    train_loss += loss
                    batch_count += 1
                    y_true.extend(targets.numpy() if isinstance(targets, tf.Tensor) else targets)
                    y_pred.extend(logits.flatten())
                if batch_count == 0:
                    logger.warning("No batches processed")
                    continue
                train_loss /= batch_count
                train_metrics = self.compute_metrics(y_true, y_pred)

                # Evaluate on validation set
                val_loss, val_metrics = self.evaluate_model(self.teacher_model, val_loader, phase='val', modalities=['skeleton'])

                # Log metrics
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                           f"Train Acc={train_metrics['accuracy']:.4f}, "
                           f"Train F1={train_metrics['f1_score']:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                           f"Val F1={val_metrics['f1_score']:.4f}")
                self.metrics_log.append({
                    'fold': self.test_subject[0],
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    logger.info("Teacher improved")
                if self.early_stop(val_loss):
                    logger.info("Early stopping teacher")
                    break
        except Exception as e:
            logger.error(f"Error training teacher: {str(e)}")
            raise

    def distillation_loss(self, teacher_features, student_features, targets, student_logits):
        """Compute distillation loss."""
        alpha = self.arg.alpha
        kd_loss = tf.reduce_mean(tf.square(teacher_features - student_features))
        ce_loss = tf.keras.losses.binary_crossentropy(targets, student_logits)
        return (1 - alpha) * ce_loss + alpha * kd_loss

    def start(self):
        """Run distillation with cross-validation."""
        if self.arg.phase != 'distill':
            logger.info("Phase must be 'distill'")
            return

        results = []
        val_subjects = [38, 46]
        try:
            for fold, test_subject in enumerate(self.arg.subjects, 1):
                if test_subject in val_subjects:
                    continue
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in val_subjects]
                logger.info(f"Starting fold {fold}: Test subject {test_subject}, Val subjects {val_subjects}, "
                           f"Train subjects {self.train_subjects}")

                if not self.load_data():
                    logger.info(f"Skipping subject {test_subject} due to data loading failure")
                    continue

                self.load_teacher_weights()

                train_loader = UTD_MM_TF(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size,
                    modalities=['accelerometer', 'skeleton']
                )
                val_loader = UTD_MM_TF(
                    dataset=self.norm_val,
                    batch_size=self.arg.val_batch_size,
                    modalities=['accelerometer', 'skeleton']
                )

                self.student_model = self.load_model(self.arg.model, self.arg.model_args)
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay
                )
                self.student_model.compile(optimizer=optimizer)

                for epoch in range(self.arg.num_epoch):
                    logger.info(f"Fold {fold}, Epoch {epoch+1}/{self.arg.num_epoch}")
                    train_loss = 0.0
                    batch_count = 0
                    y_true, y_pred = [], []
                    for batch in train_loader:
                        inputs, targets, _ = batch
                        logger.debug(f"Batch: skeleton={inputs['skeleton'].shape}, accel={inputs['accelerometer'].shape}")

                        if np.any(np.isnan(inputs['skeleton'])) or np.any(np.isnan(inputs['accelerometer'])) or np.any(np.isnan(targets)):
                            raise ValueError("NaN detected in inputs or targets")

                        with tf.GradientTape() as tape:
                            teacher_outputs = self.teacher_model({'skeleton': inputs['skeleton']}, training=False)
                            student_outputs = self.student_model({'accelerometer': inputs['accelerometer']}, training=True)
                            teacher_logits, teacher_features = teacher_outputs if isinstance(teacher_outputs, tuple) else (teacher_outputs, None)
                            student_logits, student_features = student_outputs if isinstance(student_outputs, tuple) else (student_outputs, None)
                            if teacher_features is None or student_features is None:
                                raise ValueError("Model must return (logits, features)")
                            loss = self.distillation_loss(teacher_features, student_features, targets, student_logits)

                        if np.isnan(loss.numpy()):
                            raise ValueError("NaN loss detected")
                        gradients = tape.gradient(loss, self.student_model.trainable_variables)
                        if any(g is None or np.any(np.isnan(g.numpy())) for g in gradients):
                            raise ValueError("Invalid gradients")
                        optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
                        train_loss += loss.numpy()
                        batch_count += 1
                        y_true.extend(targets.numpy() if isinstance(targets, tf.Tensor) else targets)
                        y_pred.extend(student_logits.numpy().flatten() if isinstance(student_logits, tf.Tensor) else student_logits.flatten())
                    if batch_count == 0:
                        logger.warning("No batches processed")
                        continue
                    train_loss /= batch_count
                    train_metrics = self.compute_metrics(y_true, y_pred)

                    # Evaluate on validation set
                    val_loss, val_metrics = self.evaluate_model(self.student_model, val_loader, phase='val', modalities=['accelerometer'])

                    # Log metrics
                    logger.info(f"Fold {fold}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                               f"Train Acc={train_metrics['accuracy']:.4f}, "
                               f"Train F1={train_metrics['f1_score']:.4f}, "
                               f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                               f"Val F1={val_metrics['f1_score']:.4f}")
                    self.metrics_log.append({
                        'fold': test_subject,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        'val_loss': val_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    })

                    if self.early_stop(val_loss):
                        logger.info(f"Early stopping distillation for fold {fold}")
                        break

                # Evaluate on test set
                test_loader = UTD_MM_TF(
                    dataset=self.norm_test,
                    batch_size=self.arg.test_batch_size,
                    modalities=['accelerometer', 'skeleton']
                )
                test_loss, test_metrics = self.evaluate_model(self.student_model, test_loader, phase='test', modalities=['accelerometer'])
                logger.info(f"Fold {fold} Test Metrics: Loss={test_loss:.4f}, "
                           f"Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_score']:.4f}, "
                           f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, "
                           f"AUC={test_metrics['auc']:.4f}")
                results.append({
                    'fold': fold,
                    'test_subject': str(test_subject),
                    'test_loss': test_loss,
                    **{f'test_{k}': v for k, v in test_metrics.items()}
                })

            if results:
                self.save_results(results)
                self.save_metrics_log()
                logger.info(f"Distillation completed, results saved to {self.arg.work_dir}/scores.csv")
            else:
                logger.warning("No results generated")
        except Exception as e:
            logger.error(f"Distillation failed: {str(e)}")
            raise

    def save_results(self, results):
        """Save fold results to CSV."""
        os.makedirs(self.arg.work_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(f'{self.arg.work_dir}/scores.csv', index=False)
        # Compute and save average metrics
        avg_metrics = df[[col for col in df.columns if col.startswith('test_')]].mean().to_dict()
        avg_df = pd.DataFrame([{
            'fold': 'average',
            'test_subject': 'all',
            **avg_metrics
        }])
        avg_df.to_csv(f'{self.arg.work_dir}/scores.csv', mode='a', index=False)

    def save_metrics_log(self):
        """Save per-epoch metrics to CSV."""
        df = pd.DataFrame(self.metrics_log)
        df.to_csv(f'{self.arg.work_dir}/epoch_metrics.csv', index=False)

def get_args():
    """Parse arguments and load config."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--config', type=str, default='./config/smartfallmm/distill.yaml', help='Config file')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    return args

if __name__ == "__main__":
    try:
        args = get_args()
        trainer = Distillation(args)
        trainer.start()
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise
