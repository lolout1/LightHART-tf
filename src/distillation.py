import argparse
import yaml
import os
import logging
import tensorflow as tf
from base_trainer import BaseTrainer
from utils.dataset_tf import UTD_MM_TF

class Distillation(BaseTrainer):
    def __init__(self, arg):
        """Initialize the Distillation class with arguments."""
        super().__init__(arg)
        self.teacher_model = None
        self.student_model = self.model  # Student model from BaseTrainer
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')

    def load_teacher_weights(self):
        """Load teacher weights or train if not available."""
        weights_path = f"{self.arg.teacher_weight}_{self.test_subject[0]}.weights.h5"
        try:
            if os.path.exists(weights_path):
                self.teacher_model = self.load_model(self.arg.teacher_model, self.arg.teacher_args)
                self.teacher_model.load_weights(weights_path)
                self.print_log(f"Loaded teacher weights from {weights_path}")
            else:
                self.print_log(f"Teacher weights not found at {weights_path}, training teacher...")
                self.train_teacher()
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                self.teacher_model.save_weights(weights_path)
                self.print_log(f"Saved teacher weights to {weights_path}")
        except Exception as e:
            self.print_log(f"Error loading/training teacher: {str(e)}")
            raise

    def train_teacher(self):
        """Train the teacher model using skeleton data."""
        try:
            train_loader = UTD_MM_TF(
                dataset=self.norm_train,
                batch_size=self.arg.batch_size,
                modalities=['skeleton','accelerometer']
            )
            val_loader = UTD_MM_TF(
                dataset=self.norm_val,
                batch_size=self.arg.val_batch_size,
                modalities=['skeleton','accelerometer']
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
                train_loss = 0.0
                batch_count = 0
                for batch in train_loader:
                    inputs, targets, _ = batch
                    loss = self.teacher_model.train_on_batch(inputs, targets)
                    if tf.math.is_nan(loss):
                        raise ValueError("NaN loss detected during teacher training")
                    train_loss += loss
                    batch_count += 1
                train_loss /= batch_count
                self.print_log(f"Teacher Epoch {epoch+1}: Loss={train_loss:.4f}")
                
                val_loss = self.evaluate_model(self.teacher_model, val_loader)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.print_log("Teacher improved, updating best loss")
                if self.early_stop(val_loss):
                    self.print_log("Early stopping teacher training")
                    break
        except Exception as e:
            self.print_log(f"Error training teacher: {str(e)}")
            raise

    def distillation_loss(self, teacher_features, student_features, targets, student_logits):
        """Compute combined loss: feature-based KD (MSE) and cross-entropy."""
        alpha = self.arg.alpha
        kd_loss = tf.reduce_mean(tf.square(teacher_features - student_features))
        ce_loss = tf.keras.losses.binary_crossentropy(targets, student_logits)
        return (1 - alpha) * ce_loss + alpha * kd_loss

    def start(self):
        """Execute distillation with cross-validation."""
        if self.arg.phase != 'distill':
            self.print_log("Phase must be 'distill' for distillation")
            return
        
        results = []
        val_subjects = [38, 46]  # Fixed validation subjects
        try:
            for test_subject in self.arg.subjects:
                if test_subject in val_subjects:
                    continue
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in val_subjects]
                
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                self.load_teacher_weights()
                
                # Data loaders for distillation
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
                
                # Initialize student model
                self.student_model = self.load_model(self.arg.model, self.arg.model_args)
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay
                )
                self.student_model.compile(optimizer=optimizer)
                
                # Training loop
                for epoch in range(self.arg.num_epoch):
                    train_loss = 0.0
                    batch_count = 0
                    for batch in train_loader:
                        inputs, targets, _ = batch
                        with tf.GradientTape() as tape:
                            teacher_logits, teacher_features = self.teacher_model(
                                {'skeleton': inputs['skeleton']}, training=False
                            )
                            student_logits, student_features = self.student_model(
                                {'accelerometer': inputs['accelerometer']}, training=True
                            )
                            loss = self.distillation_loss(
                                teacher_features, student_features, targets, student_logits
                            )
                        if tf.math.is_nan(loss):
                            raise ValueError("NaN loss detected during distillation")
                        gradients = tape.gradient(loss, self.student_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
                        train_loss += loss.numpy()
                        batch_count += 1
                    train_loss /= batch_count
                    self.print_log(f"Distill Epoch {epoch+1}: Loss={train_loss:.4f}")
                    
                    val_loss = self.eval(epoch, loader_name='val')
                    if self.early_stop(val_loss):
                        self.print_log("Early stopping distillation")
                        break
                
                # Evaluate and store results
                self.eval(epoch=0, loader_name='test')
                results.append({
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                })
            
            if results:
                self.add_avg_df(results)
                os.makedirs(self.arg.work_dir, exist_ok=True)
                with open(f'{self.arg.work_dir}/scores.csv', 'w') as f:
                    import csv
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                self.print_log("Distillation completed successfully")
        except Exception as e:
            self.print_log(f"Distillation process failed: {str(e)}")
            raise

def get_args():
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation in TensorFlow')
    parser.add_argument('--config', type=str, default='./config/smartfallmm/distill.yaml', help='Path to config file')
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
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        args = get_args()
        trainer = Distillation(args)
        trainer.start()
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise
