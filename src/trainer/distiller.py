# trainer/distiller.py
import os
import logging
import time
import json
import numpy as np
import tensorflow as tf
from utils.callbacks import EarlyStopping
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_loss_curves

logger = logging.getLogger('lightheart-tf.distiller')

class DistillationTrainer:
    def __init__(self, args):
        self.args = args
        
        # Initialize metrics and data holders before model loading
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_auc = 0
        
        # Initialize data
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.data_loader = {}
        self.pos_weights = None
        
        # Setup directories
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'results'), exist_ok=True)
        
        self.model_path = os.path.join(self.work_dir, 'models', args.model_saved_name)
        
        # Initialize early stopping
        self.early_stop = EarlyStopping(patience=15, min_delta=0.001)
        
        # Initialize models
        self.teacher_model = self.load_teacher_model()
        self.student_model = self.load_student_model()
        
        logger.info(f"Initialized distillation trainer - Teacher: {self.args.teacher_model}, Student: {self.args.model}")
    
    def import_class(self, import_str):
        """Dynamically import a class"""
        mod_str, _sep, class_str = import_str.rpartition('.')
        
        # Try multiple import paths
        for prefix in ['', 'src.']:
            try:
                import importlib
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
                
        raise ImportError(f"Cannot import {class_str} from {mod_str}")
    
    def load_teacher_model(self):
        """Load teacher model"""
        # Import MMTransformer directly from models
        from models.mm_transformer import MMTransformer
        
        # Create model instance
        model = MMTransformer(**self.args.teacher_args)
        
        # Build model with dummy input
        try:
            acc_frames = self.args.teacher_args.get('acc_frames', 128)
            acc_coords = self.args.teacher_args.get('acc_coords', 3)
            num_joints = self.args.teacher_args.get('num_joints', 32)
            
            # Create dummy inputs
            acc_data = tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
            skl_data = tf.zeros((2, acc_frames, num_joints, 3), dtype=tf.float32)
            
            # Forward pass to build model
            _ = model((acc_data, skl_data), training=False)
            
            logger.info("Teacher model built successfully")
        except Exception as e:
            logger.warning(f"Warning: Could not pre-build teacher model: {e}")
        
        # Load weights if provided and test_subject is set
        if hasattr(self.args, 'teacher_weight') and self.args.teacher_weight:
            # Don't append subject ID yet, do that in start() method
            teacher_weight = self.args.teacher_weight
            if os.path.exists(teacher_weight):
                try:
                    model.load_weights(teacher_weight)
                    logger.info(f"Loaded teacher weights from {teacher_weight}")
                except Exception as e:
                    logger.warning(f"Could not load teacher weights: {e}")
            else:
                logger.warning(f"Teacher weights not found at {teacher_weight}")
        
        # Set teacher model to non-trainable
        model.trainable = False
        
        return model
    
    def load_student_model(self):
        """Load student model"""
        ModelClass = self.import_class(self.args.model)
        model = ModelClass(**self.args.model_args)
        
        # Build model with dummy input
        try:
            acc_frames = self.args.model_args.get('acc_frames', 128)
            acc_coords = self.args.model_args.get('acc_coords', 3)
            
            # Create dummy inputs
            acc_data = tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
            
            # Forward pass to build model
            _ = model({'accelerometer': acc_data}, training=False)
            
            logger.info("Student model built successfully")
        except Exception as e:
            logger.warning(f"Warning: Could not pre-build student model: {e}")
        
        return model
    
    def distillation_loss(self, y_true, student_logits, teacher_logits, student_features, teacher_features):
        """Custom distillation loss combining CE and KL divergence"""
        temperature = self.args.distill_args.get('T', 2.0)
        alpha = self.args.distill_args.get('alpha', 0.7)
        
        # Hard target loss (binary cross entropy with logits)
        hard_loss = tf.keras.losses.binary_crossentropy(
            y_true, 
            student_logits, 
            from_logits=True
        )
        
        # Feature-based distillation
        # Apply softmax with temperature
        teacher_feat_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
        student_feat_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
        
        # Normalize features for better comparison
        teacher_norm = tf.nn.l2_normalize(teacher_feat_flat, axis=1)
        student_norm = tf.nn.l2_normalize(student_feat_flat, axis=1)
        
        # Cosine similarity loss
        cosine_dist = 1.0 - tf.reduce_sum(teacher_norm * student_norm, axis=1)
        feature_loss = tf.reduce_mean(cosine_dist)
        
        # Combine losses with weighting factor alpha
        total_loss = (alpha * hard_loss) + ((1 - alpha) * feature_loss)
        
        return total_loss
    
    def load_optimizer(self):
        """Initialize optimizer for student model"""
        if not hasattr(self.args, 'optimizer'):
            self.args.optimizer = 'adam'
            
        if not hasattr(self.args, 'base_lr'):
            self.args.base_lr = 0.001
            
        if not hasattr(self.args, 'weight_decay'):
            self.args.weight_decay = 0.0004
        
        if self.args.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.args.base_lr
            )
        elif self.args.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.args.base_lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.args.base_lr,
                momentum=0.9
            )
        else:
            logger.warning(f"Unknown optimizer: {self.args.optimizer}, using Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.base_lr)
            
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, LR={self.args.base_lr}")
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced data"""
        from collections import Counter
        counter = Counter(labels)
        
        if 1 in counter and 0 in counter:
            pos_weight = counter[0] / counter[1]
        else:
            pos_weight = 1.0
            
        logger.info(f"Class balance - Negative: {counter.get(0, 0)}, Positive: {counter.get(1, 0)}")
        logger.info(f"Positive class weight: {pos_weight:.4f}")
        
        return tf.constant(pos_weight, dtype=tf.float32)
    
    def load_data(self):
        """Load and prepare datasets"""
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects
            
            feeder_class_path = getattr(self.args, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
            
            if self.args.phase == 'train':
                builder = prepare_smartfallmm_tf(self.args)
                
                if not self.train_subjects:
                    logger.error("No training subjects specified")
                    return False
                
                logger.info(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects(builder, self.train_subjects, self.args.distill_args.get('fuse', False))
                
                if any(len(x) == 0 for x in self.norm_train.values()):
                    logger.error("Training data is empty")
                    return False
                    
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.args.batch_size
                )
                
                self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
                
                if self.val_subject:
                    logger.info(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = split_by_subjects(builder, self.val_subject, self.args.distill_args.get('fuse', False))
                    
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        logger.warning("Validation data is empty, using subset of training data")
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)
                        
                        self.norm_val = {
                            k: v[-val_size:].copy() for k, v in self.norm_train.items()
                        }
                        self.norm_train = {
                            k: v[:-val_size].copy() for k, v in self.norm_train.items()
                        }
                    
                    self.data_loader['val'] = Feeder(
                        dataset=self.norm_val,
                        batch_size=self.args.val_batch_size
                    )
                
                if self.test_subject:
                    logger.info(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = split_by_subjects(builder, self.test_subject, self.args.distill_args.get('fuse', False))
                    
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        logger.warning("Test data is empty")
                        return False
                        
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.args.test_batch_size
                    )
                
                logger.info("Data loading complete")
                return True
            
            elif self.args.phase == 'test':
                if not self.test_subject:
                    logger.error("No test subjects specified")
                    return False
                    
                builder = prepare_smartfallmm_tf(self.args)
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.args.distill_args.get('fuse', False))
                
                if any(len(x) == 0 for x in self.norm_test.values()):
                    logger.error("Test data is empty")
                    return False
                
                self.data_loader['test'] = Feeder(
                    dataset=self.norm_test,
                    batch_size=self.args.test_batch_size
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @tf.function
    def train_step(self, inputs, targets):
        """Execute a single training step with distillation"""
        with tf.GradientTape() as tape:
            # Forward pass through teacher model (no gradients)
            teacher_logits, teacher_features = self.teacher_model(inputs, training=False)
            
            # Forward pass through student model (with gradients)
            student_logits, student_features = self.student_model(inputs, training=True)
            
            # Compute distillation loss
            loss = self.distillation_loss(
                targets, 
                student_logits, 
                teacher_logits, 
                student_features, 
                teacher_features
            )
        
        # Compute gradients and update student model only
        gradients = tape.gradient(loss, self.student_model.trainable_variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        # Calculate predictions from student model
        predictions = tf.cast(tf.sigmoid(student_logits) > 0.5, tf.int32)
        
        return loss, predictions
    
    def train_epoch(self, epoch):
        """Train for one epoch with distillation"""
        start_time = time.time()
        
        # Set models to appropriate modes
        self.teacher_model.trainable = False
        self.student_model.trainable = True
        
        # Get data loader
        loader = self.data_loader['train']
        
        # Initialize metrics
        train_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        # Training loop
        batch_count = len(loader)
        logger.info(f"Training epoch {epoch+1}/{self.args.num_epoch} - {batch_count} batches")
        
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}: batch {batch_idx+1}/{batch_count}")
                
            targets = tf.cast(targets, tf.float32)
            
            # Train step with distillation
            loss, predictions = self.train_step(inputs, targets)
            
            # Update metrics
            train_loss += loss.numpy()
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
        
        # Calculate average loss and metrics
        train_loss /= steps
        accuracy, f1, recall, precision, auc = self.calculate_metrics(all_labels, all_preds)
        
        self.train_loss_summary.append(float(train_loss))
        
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Acc={accuracy:.2f}%, "
            f"F1={f1:.2f}%, "
            f"Prec={precision:.2f}%, "
            f"Rec={recall:.2f}%, "
            f"AUC={auc:.2f}% "
            f"({epoch_time:.2f}s)"
        )
        
        # Evaluate on validation set
        val_loss = self.eval(epoch, 'val')
        self.val_loss_summary.append(float(val_loss))
        
        # Check early stopping
        return self.early_stop(val_loss)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        # Handle edge cases
        unique_labels = np.unique(y_true)
        if len(unique_labels) <= 1:
            # Single class in dataset
            if unique_labels[0] == 1:
                precision = 100.0 if np.all(y_pred == 1) else 0.0
                recall = 100.0 if np.all(y_pred == 1) else 0.0
                f1 = 100.0 if np.all(y_pred == 1) else 0.0
            else:
                precision = 100.0 if np.all(y_pred == 0) else 0.0
                recall = 100.0 if np.all(y_pred == 0) else 0.0
                f1 = 100.0 if np.all(y_pred == 0) else 0.0
            auc = 50.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0) * 100
            recall = recall_score(y_true, y_pred, zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, zero_division=0) * 100
            try:
                auc = roc_auc_score(y_true, y_pred) * 100
            except Exception:
                auc = 50.0
                
        return accuracy, f1, recall, precision, auc
    
    def eval(self, epoch, loader_name='val', result_file=None):
        """Evaluate model"""
        start_time = time.time()
        
        # Set models to evaluation mode
        self.teacher_model.trainable = False
        self.student_model.trainable = False
        
        # Get data loader
        loader = self.data_loader.get(loader_name)
        if loader is None:
            logger.error(f"No data loader for {loader_name}")
            return float('inf')
        
        # Initialize metrics
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        # Evaluation loop
        batch_count = len(loader)
        logger.info(f"Evaluating {loader_name} (epoch {epoch+1}) - {batch_count} batches")
        
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 5 == 0:
                logger.info(f"Eval {loader_name} (epoch {epoch+1}): batch {batch_idx+1}/{batch_count}")
                
            targets = tf.cast(targets, tf.float32)
            
            # Get teacher predictions
            teacher_logits, teacher_features = self.teacher_model(inputs, training=False)
            
            # Get student predictions
            student_logits, student_features = self.student_model(inputs, training=False)
            
            # Calculate distillation loss
            loss = self.distillation_loss(
                targets,
                student_logits,
                teacher_logits,
                student_features,
                teacher_features
            )
            
            # Calculate predictions
            predictions = tf.cast(tf.sigmoid(student_logits) > 0.5, tf.int32)
            
            # Update metrics
            eval_loss += loss.numpy()
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1
        
        # Calculate average loss and metrics
        eval_loss /= steps
        accuracy, f1, recall, precision, auc = self.calculate_metrics(all_labels, all_preds)
        
        logger.info(
            f"{loader_name.capitalize()}: "
            f"Loss={eval_loss:.4f}, "
            f"Acc={accuracy:.2f}%, "
            f"F1={f1:.2f}%, "
            f"Prec={precision:.2f}%, "
            f"Rec={recall:.2f}%, "
            f"AUC={auc:.2f}%"
        )
        
        # Save best model for validation
        if loader_name == 'val':
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.save_model()
                logger.info(f"New best validation loss: {eval_loss:.4f}")
        
        # Save test metrics
        elif loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_precision = precision
            self.test_recall = recall
            self.test_auc = auc
            
            # Create confusion matrix
            subject_id = self.test_subject[0] if self.test_subject else None
            if subject_id:
                plot_confusion_matrix(all_preds, all_labels, self.work_dir, subject_id, logger.info)
            
            # Save results to file
            results = {
                "subject": self.test_subject[0] if self.test_subject else "unknown",
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "loss": float(eval_loss)
            }
            
            results_file = os.path.join(
                self.work_dir,
                'results',
                f'test_results_{self.test_subject[0] if self.test_subject else "unknown"}.json'
            )
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            if result_file:
                with open(result_file, 'w') as f:
                    for pred, true in zip(all_preds, all_labels):
                        f.write(f"{pred} ==> {true}\n")
        
        return eval_loss
    
    def save_model(self):
        """Save student model weights and TFLite model"""
        # Save weights
        if self.test_subject:
            weights_path = f"{self.model_path}_{self.test_subject[0]}"
        else:
            weights_path = f"{self.model_path}"
            
        # Save weights
        self.student_model.save_weights(f"{weights_path}.weights.h5")
        logger.info(f"Saved model weights to {weights_path}.weights.h5")
        
        # Try to save full model
        try:
            self.student_model.save(weights_path)
            logger.info(f"Saved full model to {weights_path}")
        except Exception as e:
            logger.warning(f"Could not save full model: {e}")
        
        # Try to export TFLite model
        try:
            from utils.tflite_converter import convert_to_tflite_model
            
            # Export TFLite model
            tflite_path = f"{weights_path}.tflite"
            
            # Get input shape
            acc_frames = self.args.model_args.get('acc_frames', 128)
            acc_coords = self.args.model_args.get('acc_coords', 3)
            
            success = convert_to_tflite_model(
                self.student_model,
                tflite_path,
                input_shape=(1, acc_frames, acc_coords)
            )
            
            if success:
                logger.info(f"Exported TFLite model to {tflite_path}")
            else:
                logger.warning(f"Failed to export TFLite model")
        except Exception as e:
            logger.warning(f"Error exporting TFLite model: {e}")
    
    def add_avg_df(self, results):
        """Add average row to results dataframe"""
        if not results:
            return results
            
        avg_result = {'test_subject': 'Average'}
        
        for column in results[0].keys():
            if column != 'test_subject':
                values = [float(r[column]) for r in results]
                avg_result[column] = round(sum(values) / len(values), 2)
        
        results.append(avg_result)
        return results
    
    def start(self):
        """Main training loop with cross-validation"""
        if self.args.phase == 'train':
            logger.info('Parameters:')
            for key, value in vars(self.args).items():
                logger.info(f'  {key}: {value}')
            
            results = []
            
            # Use same validation subjects as in base_trainer
            val_subjects = [38, 46]
            
            for test_subject in self.args.subjects:
                if test_subject in val_subjects:
                    continue
                
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.args.subjects 
                                      if s != test_subject and s not in val_subjects]
                
                logger.info(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                logger.info(f"Train: {len(self.train_subjects)} subjects")
                logger.info(f"Val: {len(self.val_subject)} subjects")
                logger.info(f"Test: Subject {test_subject}")
                
                # Load models for this fold
                self.teacher_model = self.load_teacher_model()
                self.student_model = self.load_student_model()
                
                # Look for subject-specific teacher weights
                if hasattr(self.args, 'teacher_weight') and self.args.teacher_weight:
                    teacher_weight = f"{self.args.teacher_weight}_{test_subject}.pth"
                    if os.path.exists(teacher_weight):
                        try:
                            self.teacher_model.load_weights(teacher_weight)
                            logger.info(f"Loaded subject-specific teacher weights from {teacher_weight}")
                        except Exception as e:
                            logger.warning(f"Could not load subject-specific teacher weights: {e}")
                
                # Load data for this fold
                if not self.load_data():
                    logger.warning(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                # Initialize optimizer
                self.load_optimizer()
                
                # Reset early stopping
                self.early_stop = EarlyStopping(patience=15, min_delta=.001)
                
                # Train for specified number of epochs
                for epoch in range(self.args.start_epoch, self.args.num_epoch):
                    should_stop = self.train_epoch(epoch)
                    
                    if should_stop:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Load best model for testing
                best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                if os.path.exists(best_weights):
                    self.student_model.load_weights(best_weights)
                    logger.info(f"Loaded best weights from {best_weights}")
                
                # Evaluate on test set
                logger.info(f"=== Final evaluation on subject {test_subject} ===")
                self.eval(0, 'test')
                
                # Visualize loss curves
                plot_loss_curves(
                    self.train_loss_summary,
                    self.val_loss_summary,
                    self.work_dir,
                    test_subject
                )
                
                # Save results for this fold
                subject_result = {
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                
                results.append(subject_result)
                
                # Clear session to free memory
                tf.keras.backend.clear_session()
            
            # Save overall results
            if results:
                results = self.add_avg_df(results)
                
                import pandas as pd
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(self.work_dir, 'scores.csv'), index=False)
                
                with open(os.path.join(self.work_dir, 'scores.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info("\n=== Final Results ===")
                for result in results:
                    subject = result['test_subject']
                    accuracy = result['accuracy']
                    f1 = result['f1_score']
                    precision = result['precision']
                    recall = result['recall']
                    auc = result.get('auc', 'N/A')
                    
                    logger.info(
                        f"Subject {subject}: "
                        f"Acc={accuracy:.2f}%, "
                        f"F1={f1:.2f}%, "
                        f"Prec={precision:.2f}%, "
                        f"Rec={recall:.2f}%, "
                        f"AUC={auc}"
                    )
            
            logger.info("Training completed successfully")
