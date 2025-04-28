import os
import logging
import time
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger('distiller')

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.00001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.verbose = verbose
    
    def __call__(self, current_value):
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def reset(self):
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False

class DistillationTrainer:
    def __init__(self, args):
        self.args = args
        
        # Setup output directories
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'results'), exist_ok=True)
        
        # Initialize metrics
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_auc = 0
        
        # Initialize dataset variables
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.data_loader = {}
        self.pos_weights = None
        
        # Model saving path
        self.model_path = os.path.join(self.work_dir, 'models', args.model_saved_name)
        
        # Early stopping
        self.early_stop = EarlyStopping(patience=15, min_delta=0.001)
        
        # Setup modalities for teacher and student
        self._setup_modalities()
        
        # Load models
        logger.info("Loading teacher model...")
        self.teacher_model = self.load_teacher_model()
        
        logger.info("Loading student model...")
        self.student_model = self.load_student_model()
    
    def _setup_modalities(self):
        # Initialize distill_args if not present
        if not hasattr(self.args, 'distill_args') or self.args.distill_args is None:
            self.args.distill_args = {}
        
        # Set default modalities
        self.teacher_modalities = self.args.distill_args.get('teacher_modalities', ['accelerometer', 'skeleton'])
        self.student_modalities = self.args.distill_args.get('student_modalities', ['accelerometer'])
        
        # Ensure dataset args includes all needed modalities
        if not hasattr(self.args, 'dataset_args') or self.args.dataset_args is None:
            self.args.dataset_args = {}
        
        # Update dataset modalities
        all_modalities = list(set(self.teacher_modalities + self.student_modalities))
        self.args.dataset_args['modalities'] = all_modalities
        
        logger.info(f"Teacher modalities: {self.teacher_modalities}")
        logger.info(f"Student modalities: {self.student_modalities}")
    
    def import_class(self, import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        
        # Try multiple import paths
        for prefix in ['', 'models.', 'src.models.']:
            try:
                import importlib
                module = importlib.import_module(f"{prefix}{mod_str}")
                return getattr(module, class_str)
            except (ImportError, AttributeError):
                continue
        
        logger.error(f"Failed to import {class_str} from {mod_str}")
        raise ImportError(f"Cannot import {class_str} from {mod_str}")
    
    def load_teacher_model(self):
        try:
            TeacherModelClass = self.import_class(self.args.teacher_model)
            teacher_model = TeacherModelClass(**self.args.teacher_args)
            
            # Set to non-trainable
            teacher_model.trainable = False
            
            return teacher_model
        except Exception as e:
            logger.error(f"Error loading teacher model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_student_model(self):
        try:
            StudentModelClass = self.import_class(self.args.model)
            student_model = StudentModelClass(**self.args.model_args)
            return student_model
        except Exception as e:
            logger.error(f"Error loading student model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_teacher_weights(self):
        if not hasattr(self.args, 'teacher_weight') or not self.args.teacher_weight:
            logger.warning("No teacher weights specified")
            return False
        
        test_subject = self.test_subject[0] if self.test_subject else None
        
        # Try subject-specific weights first
        if test_subject is not None:
            for ext in ['.weights.h5', '.h5', '.pth', '']:
                weights_path = f"{self.args.teacher_weight}_{test_subject}{ext}"
                if os.path.exists(weights_path):
                    try:
                        self.teacher_model.load_weights(weights_path)
                        logger.info(f"Loaded subject-specific teacher weights from {weights_path}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load weights from {weights_path}: {e}")
        
        # Try generic weights
        for ext in ['.weights.h5', '.h5', '.pth', '']:
            weights_path = f"{self.args.teacher_weight}{ext}"
            if os.path.exists(weights_path):
                try:
                    self.teacher_model.load_weights(weights_path)
                    logger.info(f"Loaded teacher weights from {weights_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load weights from {weights_path}: {e}")
        
        logger.warning(f"No valid teacher weights found at {self.args.teacher_weight}")
        return False
    
    def load_optimizer(self):
        optimizer_name = getattr(self.args, 'optimizer', 'adam').lower()
        lr = self.args.base_lr
        
        if optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=self.args.weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.9
            )
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, LR={lr}")
    
    def create_distillation_loss(self):
        temperature = float(self.args.distill_args.get('T', 2.0))
        alpha = float(self.args.distill_args.get('alpha', 0.7))
        
        def distillation_loss(y_true, student_logits, teacher_logits, student_features, teacher_features):
            # Hard target loss
            y_true = tf.cast(y_true, tf.float32)
            student_logits = tf.reshape(student_logits, [-1])
            teacher_logits = tf.reshape(teacher_logits, [-1])
            
            hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true,
                logits=student_logits
            )
            hard_loss = tf.reduce_mean(hard_loss)
            
            # Soft target loss (knowledge distillation)
            teacher_probs = tf.nn.sigmoid(teacher_logits / temperature)
            soft_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=teacher_probs,
                logits=student_logits / temperature
            )
            soft_loss = tf.reduce_mean(soft_loss) * (temperature ** 2)
            
            # Feature distillation
            if student_features is not None and teacher_features is not None:
                # Normalize features
                student_feat_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
                teacher_feat_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
                
                student_norm = tf.nn.l2_normalize(student_feat_flat, axis=1)
                teacher_norm = tf.nn.l2_normalize(teacher_feat_flat, axis=1)
                
                # Cosine similarity loss
                cos_sim = tf.reduce_sum(student_norm * teacher_norm, axis=1)
                feature_loss = tf.reduce_mean(1.0 - cos_sim)
                
                # Weight by teacher correctness
                teacher_correct = tf.cast(tf.equal(
                    tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32),
                    y_true
                ), tf.float32)
                
                # Higher weight to correct teacher predictions
                teacher_weights = 0.8 * teacher_correct + 0.2 * (1.0 - teacher_correct)
                feature_loss = feature_loss * tf.reduce_mean(teacher_weights)
                
                # Combine losses
                total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss + 0.3 * feature_loss
            else:
                # Without feature distillation
                total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
            
            return total_loss
        
        return distillation_loss
    
    def load_data(self):
        try:
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects, UTD_MM_TF
            
            logger.info(f"Creating dataset: {self.args.dataset}")
            builder = prepare_smartfallmm_tf(self.args)
            
            logger.info(f"Processing training data for subjects: {self.train_subjects}")
            self.norm_train = split_by_subjects(
                builder, 
                self.train_subjects, 
                self.args.distill_args.get('fuse', False)
            )
            
            if not self.norm_train or all(len(v) == 0 for k, v in self.norm_train.items() if k != 'labels'):
                logger.error("Empty training data")
                return False
            
            self.data_loader['train'] = UTD_MM_TF(
                dataset=self.norm_train,
                batch_size=self.args.batch_size,
                modalities=self.args.dataset_args.get('modalities', ['accelerometer'])
            )
            
            # Calculate class weights
            labels = self.norm_train['labels']
            neg_count = np.sum(labels == 0)
            pos_count = np.sum(labels == 1)
            
            if pos_count > 0:
                self.pos_weights = tf.constant(float(neg_count) / float(pos_count), dtype=tf.float32)
            else:
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
            
            logger.info(f"Class balance - Negative: {neg_count}, Positive: {pos_count}")
            logger.info(f"Positive class weight: {self.pos_weights.numpy():.4f}")
            
            logger.info(f"Processing validation data for subjects: {self.val_subject}")
            self.norm_val = split_by_subjects(
                builder, 
                self.val_subject, 
                self.args.distill_args.get('fuse', False)
            )
            
            if not self.norm_val or all(len(v) == 0 for k, v in self.norm_val.items() if k != 'labels'):
                logger.warning("Empty validation data, using subset of training data")
                train_size = len(self.norm_train['labels'])
                val_size = min(train_size // 5, 100)
                
                self.norm_val = {}
                for k, v in self.norm_train.items():
                    if len(v) > 0:
                        self.norm_val[k] = v[-val_size:].copy()
                        self.norm_train[k] = v[:-val_size].copy()
            
            self.data_loader['val'] = UTD_MM_TF(
                dataset=self.norm_val,
                batch_size=self.args.val_batch_size,
                modalities=self.args.dataset_args.get('modalities', ['accelerometer'])
            )
            
            logger.info(f"Processing test data for subjects: {self.test_subject}")
            self.norm_test = split_by_subjects(
                builder, 
                self.test_subject, 
                self.args.distill_args.get('fuse', False)
            )
            
            if not self.norm_test or all(len(v) == 0 for k, v in self.norm_test.items() if k != 'labels'):
                logger.error("Empty test data")
                return False
            
            self.data_loader['test'] = UTD_MM_TF(
                dataset=self.norm_test,
                batch_size=self.args.test_batch_size,
                modalities=self.args.dataset_args.get('modalities', ['accelerometer'])
            )
            
            logger.info("Data loading complete")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_teacher_inputs(self, inputs):
        """Prepare inputs for teacher model (MMTransformer)"""
        # For teacher model, prepare either tuple or dict
        if self.teacher_modalities == ['accelerometer', 'skeleton'] and 'accelerometer' in inputs and 'skeleton' in inputs:
            # Return as tuple (acc_data, skl_data) for MMTransformer
            return (inputs['accelerometer'], inputs['skeleton'])
        else:
            # Return as dict with only needed modalities
            return {k: v for k, v in inputs.items() if k in self.teacher_modalities}
    
    def prepare_student_inputs(self, inputs):
        """Prepare inputs for student model"""
        # Return as dict with only needed modalities
        return {k: v for k, v in inputs.items() if k in self.student_modalities}
    
    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            # Prepare inputs for both models
            teacher_inputs = self.prepare_teacher_inputs(inputs)
            student_inputs = self.prepare_student_inputs(inputs)
            
            # Forward pass through teacher model (no gradients)
            teacher_logits, teacher_features = self.teacher_model(teacher_inputs, training=False)
            
            # Forward pass through student model (with gradients)
            student_logits, student_features = self.student_model(student_inputs, training=True)
            
            # Calculate distillation loss
            loss = self.distillation_loss(
                targets, 
                student_logits, 
                teacher_logits, 
                student_features, 
                teacher_features
            )
        
        # Compute gradients and update student model
        gradients = tape.gradient(loss, self.student_model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        # Calculate predictions
        predictions = tf.cast(tf.sigmoid(student_logits) > 0.5, tf.int32)
        
        return loss, predictions
    
    def eval_step(self, inputs, targets):
        # Prepare inputs for both models
        teacher_inputs = self.prepare_teacher_inputs(inputs)
        student_inputs = self.prepare_student_inputs(inputs)
        
        # Forward pass through teacher model
        teacher_logits, teacher_features = self.teacher_model(teacher_inputs, training=False)
        
        # Forward pass through student model
        student_logits, student_features = self.student_model(student_inputs, training=False)
        
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
        
        return loss, predictions
    
    def train_epoch(self, epoch):
        # Set models to appropriate modes
        self.teacher_model.trainable = False
        self.student_model.trainable = True
        
        # Get data loader
        loader = self.data_loader['train']
        
        # Initialize metrics
        epoch_loss = 0.0
        all_targets = []
        all_preds = []
        batch_count = 0
        
        # Progress tracking
        progress = tqdm(range(len(loader)), desc=f"Epoch {epoch+1}/{self.args.num_epoch}")
        
        # Training loop
        for batch_idx in progress:
            try:
                # Get batch data
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                
                # Execute training step
                loss, preds = self.train_step(inputs, targets)
                
                # Update metrics
                epoch_loss += loss.numpy()
                all_targets.extend(targets.numpy())
                all_preds.extend(preds.numpy())
                batch_count += 1
                
                # Update progress bar
                progress.set_postfix({'loss': f"{epoch_loss/batch_count:.4f}"})
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        if batch_count > 0:
            epoch_loss /= batch_count
            
            # Calculate metrics
            accuracy, f1, recall, precision, auc = self.calculate_metrics(all_targets, all_preds)
            
            # Store loss for plotting
            self.train_loss_summary.append(float(epoch_loss))
            
            # Log results
            logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss={epoch_loss:.4f}, "
                f"Acc={accuracy:.2f}%, "
                f"F1={f1:.2f}%, "
                f"Prec={precision:.2f}%, "
                f"Rec={recall:.2f}%, "
                f"AUC={auc:.2f}%"
            )
            
            # Evaluate on validation set
            val_loss = self.evaluate(epoch, loader_name='val')
            
            # Store validation loss for plotting
            self.val_loss_summary.append(float(val_loss))
            
            # Check early stopping
            return self.early_stop(val_loss)
        else:
            logger.error("No valid batches processed in epoch")
            return True
    
    def evaluate(self, epoch, loader_name='val'):
        # Set models to evaluation mode
        self.teacher_model.trainable = False
        self.student_model.trainable = False
        
        # Get data loader
        loader = self.data_loader[loader_name]
        
        # Initialize metrics
        eval_loss = 0.0
        all_targets = []
        all_preds = []
        batch_count = 0
        
        # Progress tracking
        progress = tqdm(range(len(loader)), desc=f"Eval {loader_name}")
        
        # Evaluation loop
        for batch_idx in progress:
            try:
                # Get batch data
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                
                # Execute evaluation step
                loss, preds = self.eval_step(inputs, targets)
                
                # Update metrics
                eval_loss += loss.numpy()
                all_targets.extend(targets.numpy())
                all_preds.extend(preds.numpy())
                batch_count += 1
                
                # Update progress bar
                progress.set_postfix({'loss': f"{eval_loss/batch_count:.4f}"})
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate metrics
        if batch_count > 0:
            eval_loss /= batch_count
            
            # Calculate metrics
            accuracy, f1, recall, precision, auc = self.calculate_metrics(all_targets, all_preds)
            
            # Log results
            logger.info(
                f"{loader_name.capitalize()}: "
                f"Loss={eval_loss:.4f}, "
                f"Acc={accuracy:.2f}%, "
                f"F1={f1:.2f}%, "
                f"Prec={precision:.2f}%, "
                f"Rec={recall:.2f}%, "
                f"AUC={auc:.2f}%"
            )
            
            # Save best model if validation
            if loader_name == 'val':
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_model()
                    logger.info(f'New best model saved with validation loss: {eval_loss:.4f}')
            
            # Save test metrics
            if loader_name == 'test':
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_precision = precision
                self.test_recall = recall
                self.test_auc = auc
                
                # Plot confusion matrix
                self.plot_confusion_matrix(all_preds, all_targets)
            
            return eval_loss
        else:
            logger.error(f"No valid batches processed in {loader_name}")
            return float('inf')
    
    def calculate_metrics(self, y_true, y_pred):
        try:
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred) * 100
            
            # Check for edge cases
            unique_labels = np.unique(y_true)
            unique_preds = np.unique(y_pred)
            
            if len(unique_labels) <= 1 or len(unique_preds) <= 1:
                # Single class case
                if len(unique_labels) == 1 and len(unique_preds) == 1 and unique_labels[0] == unique_preds[0]:
                    # Perfect prediction of single class
                    if unique_labels[0] == 1:
                        precision = 100.0
                        recall = 100.0
                        f1 = 100.0
                    else:
                        precision = 0.0
                        recall = 0.0
                        f1 = 0.0
                    auc = 50.0
                else:
                    # Calculate metrics manually
                    tn, fp, fn, tp = 0, 0, 0, 0
                    for t, p in zip(y_true, y_pred):
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
                    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    auc = 50.0
            else:
                # Normal case
                precision = precision_score(y_true, y_pred) * 100
                recall = recall_score(y_true, y_pred) * 100
                f1 = f1_score(y_true, y_pred) * 100
                
                try:
                    auc = roc_auc_score(y_true, y_pred) * 100
                except:
                    auc = 50.0
            
            return accuracy, f1, recall, precision, auc
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    def plot_confusion_matrix(self, y_pred, y_true):
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), 
                            ha="center", va="center", 
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(f'Confusion Matrix (Subject {self.test_subject[0]})')
            
            # Save figure
            plt.savefig(os.path.join(self.work_dir, 'visualizations', 
                                    f'confusion_matrix_{self.test_subject[0]}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def plot_loss_curves(self):
        try:
            if len(self.train_loss_summary) < 2:
                logger.warning("Not enough data for loss visualization")
                return
                
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.train_loss_summary) + 1)
            plt.plot(epochs, self.train_loss_summary, 'b-', label='Training Loss')
            
            if len(self.val_loss_summary) > 0:
                plt.plot(epochs[:len(self.val_loss_summary)], self.val_loss_summary, 'r-', label='Validation Loss')
            
            plt.title(f'Training vs Validation Loss (Subject {self.test_subject[0]})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            plt.savefig(os.path.join(self.work_dir, 'visualizations', 
                                   f'loss_curves_{self.test_subject[0]}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting loss curves: {e}")
    
    def save_model(self):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save weights
            weights_path = f"{self.model_path}_{self.test_subject[0]}.weights.h5"
            self.student_model.save_weights(weights_path)
            logger.info(f"Model weights saved to {weights_path}")
            
            # Export TFLite model
            try:
                # Define preprocessing function for accelerometer data
                acc_frames = self.args.model_args.get('acc_frames', 64)
                acc_coords = self.args.model_args.get('acc_coords', 3)
                
                @tf.function(input_signature=[
                    tf.TensorSpec(shape=[1, acc_frames, acc_coords], dtype=tf.float32, name='accelerometer')
                ])
                def process_input(accelerometer):
                    # Calculate signal magnitude vector
                    mean = tf.reduce_mean(accelerometer, axis=1, keepdims=True)
                    zero_mean = accelerometer - mean
                    sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                    smv = tf.sqrt(sum_squared)
                    acc_with_smv = tf.concat([smv, accelerometer], axis=-1)
                    return {'accelerometer': acc_with_smv}
                
                @tf.function(input_signature=[
                    tf.TensorSpec(shape=[1, acc_frames, acc_coords], dtype=tf.float32, name='accelerometer')
                ])
                def serving_fn(accelerometer):
                    processed = process_input(accelerometer)
                    outputs = self.student_model(processed, training=False)
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        return outputs[0]
                    return outputs
                
                # Save to TFLite
                tflite_path = f"{self.model_path}_{self.test_subject[0]}.tflite"
                
                converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                
                tflite_model = converter.convert()
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger.info(f"TFLite model exported to {tflite_path}")
                
            except Exception as e:
                logger.warning(f"Error exporting TFLite model: {e}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def add_avg_df(self, results):
        if not results:
            return results
            
        avg_result = {'test_subject': 'Average'}
        
        for column in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            values = [float(r[column]) for r in results]
            avg_result[column] = round(sum(values) / len(values), 2)
        
        results.append(avg_result)
        return results
    
    def start(self):
        if self.args.phase == 'train':
            logger.info("Starting distillation training")
            
            # Initialize distillation loss
            self.distillation_loss = self.create_distillation_loss()
            
            # Results storage
            results = []
            
            # Default validation subjects
            val_subjects = [38, 46]
            
            # Loop through subjects for cross-validation
            for test_subject in self.args.subjects:
                # Skip validation subjects
                if test_subject in val_subjects:
                    continue
                
                # Reset metrics for this fold
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.early_stop.reset()
                
                # Set subjects for this fold
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.args.subjects 
                                      if s != test_subject and s not in val_subjects]
                
                logger.info(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                logger.info(f"Train: {len(self.train_subjects)} subjects")
                logger.info(f"Val: {len(self.val_subject)} subjects")
                logger.info(f"Test: Subject {test_subject}")
                
                # Load teacher weights for this subject
                self.load_teacher_weights()
                
                # Load data for this fold
                if not self.load_data():
                    logger.warning(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                # Initialize optimizer
                self.load_optimizer()
                
                # Train for specified number of epochs
                for epoch in range(self.args.start_epoch, self.args.num_epoch):
                    should_stop = self.train_epoch(epoch)
                    
                    if should_stop:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Plot loss curves
                self.plot_loss_curves()
                
                # Load best model for testing
                best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                if os.path.exists(best_weights):
                    try:
                        self.student_model.load_weights(best_weights)
                        logger.info(f"Loaded best weights from {best_weights}")
                    except Exception as e:
                        logger.error(f"Error loading best weights: {e}")
                
                # Evaluate on test set
                logger.info(f"=== Final evaluation on subject {test_subject} ===")
                self.evaluate(0, 'test')
                
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
                
                # Save to CSV
                import pandas as pd
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(self.work_dir, 'scores.csv'), index=False)
                
                # Save to JSON
                with open(os.path.join(self.work_dir, 'scores.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Log final results
                logger.info("\n=== Final Results ===")
                for r in results:
                    if r['test_subject'] != 'Average':
                        logger.info(f"Subject {r['test_subject']}: "
                                  f"Acc={r['accuracy']}%, "
                                  f"F1={r['f1_score']}%")
                
                avg = next((r for r in results if r['test_subject'] == 'Average'), None)
                if avg:
                    logger.info(f"Average performance: "
                              f"Acc={avg['accuracy']}%, "
                              f"F1={avg['f1_score']}%")
            
            logger.info("Training completed successfully")
            
        elif self.args.phase == 'test':
            logger.info("Starting testing mode")
            
            # Initialize distillation loss
            self.distillation_loss = self.create_distillation_loss()
            
            # Set test subject
            self.test_subject = [self.args.subjects[0]]
            
            # Load teacher weights
            self.load_teacher_weights()
            
            # Load student weights
            if hasattr(self.args, 'weights') and self.args.weights:
                try:
                    self.student_model.load_weights(self.args.weights)
                    logger.info(f"Loaded student weights from {self.args.weights}")
                except Exception as e:
                    logger.error(f"Error loading student weights: {e}")
                    return
            else:
                logger.error("No student weights specified for testing")
                return
            
            # Load test data
            self.val_subject = [38, 46]  # needed for data loading
            self.train_subjects = [s for s in self.args.subjects 
                                 if s != self.test_subject[0] and s not in self.val_subject]
            
            if not self.load_data():
                logger.error("Failed to load test data")
                return
            
            # Evaluate on test set
            logger.info(f"Testing on subject {self.test_subject[0]}")
            self.evaluate(0, 'test')
            
            # Save results
            results_file = os.path.join(self.work_dir, 'results', 
                                      f'test_results_{self.test_subject[0]}.json')
            
            with open(results_file, 'w') as f:
                json.dump({
                    'subject': str(self.test_subject[0]),
                    'accuracy': float(self.test_accuracy),
                    'f1_score': float(self.test_f1),
                    'precision': float(self.test_precision),
                    'recall': float(self.test_recall),
                    'auc': float(self.test_auc)
                }, f, indent=2)
            
            logger.info("Testing completed successfully")
