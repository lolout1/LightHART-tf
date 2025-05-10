#!/usr/bin/env python
import os, logging, json, traceback, time, argparse, yaml
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trainer.base_trainer import BaseTrainer, EarlyStopping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Distiller(BaseTrainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = None
        self.distillation_loss = None
        logger.info("Distiller base initialized successfully")
    
    def load_teacher_model(self):
        try:
            if not hasattr(self.arg, 'teacher_model') or not hasattr(self.arg, 'teacher_args'):
                raise ValueError("Teacher model and args must be specified")
            
            teacher_class = self.import_class(self.arg.teacher_model)
            teacher_model = teacher_class(**self.arg.teacher_args)
            
            if not hasattr(self.arg, 'teacher_weight'):
                raise ValueError("Teacher weights path must be specified")
            
            teacher_weight_path = self.arg.teacher_weight
            
            # Build dummy inputs to initialize teacher model
            try:
                dummy_acc = tf.zeros((2, self.arg.teacher_args.get('acc_frames', 128), 
                                    self.arg.teacher_args.get('acc_coords', 3)), dtype=tf.float32)
                dummy_skl = tf.zeros((2, self.arg.teacher_args.get('acc_frames', 128), 
                                     self.arg.teacher_args.get('num_joints', 32), 3), dtype=tf.float32)
                
                _ = teacher_model({'accelerometer': dummy_acc, 'skeleton': dummy_skl}, training=False)
                logger.info("Teacher model initialized with dummy inputs")
            except Exception as e:
                logger.warning(f"Error initializing teacher model with dummy inputs: {e}")
            
            # Find the correct teacher weights file with subject ID
            if hasattr(self, 'test_subject') and self.test_subject:
                subject_id = self.test_subject[0] if isinstance(self.test_subject, list) else self.test_subject
                
                # Try different weight file patterns
                weight_formats = [
                    f"{teacher_weight_path}_{subject_id}.weights.h5",
                    f"{teacher_weight_path}_{subject_id}.h5",
                    f"{teacher_weight_path}_{subject_id}.keras"
                ]
                
                found_weight_path = None
                for weight_path in weight_formats:
                    if os.path.exists(weight_path):
                        found_weight_path = weight_path
                        logger.info(f"Found subject-specific weights: {found_weight_path}")
                        break
                
                if found_weight_path:
                    teacher_weight_path = found_weight_path
                else:
                    logger.error(f"No teacher weights found for subject {subject_id}")
                    raise FileNotFoundError(f"No teacher weights found for subject {subject_id}")
            else:
                # Find any available weights for initialization
                base_path = teacher_weight_path
                found_any = False
                
                for subject in [32, 39, 30, 31, 33, 34, 35, 37, 43, 44, 29, 36, 45]:
                    test_paths = [
                        f"{base_path}_{subject}.weights.h5",
                        f"{base_path}_{subject}.h5",
                        f"{base_path}_{subject}.keras"
                    ]
                    
                    for test_path in test_paths:
                        if os.path.exists(test_path):
                            teacher_weight_path = test_path
                            found_any = True
                            logger.info(f"Using weights from subject {subject} for model initialization")
                            break
                    
                    if found_any:
                        break
                
                if not found_any:
                    logger.error("No teacher weights found for any subject")
                    raise FileNotFoundError("No teacher weights found for any subject")
            
            # Load weights
            logger.info(f"Loading teacher weights from {teacher_weight_path}")
            teacher_model.load_weights(teacher_weight_path)
            teacher_model.trainable = False
            
            return teacher_model
            
        except Exception as e:
            logger.error(f"Error loading teacher model: {e}")
            traceback.print_exc()
            raise
    
    def load_distillation_loss(self):
        try:
            from utils.loss import DistillationLoss
            temperature = getattr(self.arg, 'temperature', 4.5)
            alpha = getattr(self.arg, 'alpha', 0.6)
            
            self.distillation_loss = DistillationLoss(
                temperature=temperature,
                alpha=alpha,
                pos_weight=self.pos_weights if hasattr(self, 'pos_weights') else None
            )
            
            logger.info(f"Distillation loss initialized with temperature={temperature}, alpha={alpha}")
            
        except ImportError as e:
            logger.warning(f"Failed to import DistillationLoss, creating custom implementation: {e}")
            
            temperature = getattr(self.arg, 'temperature', 4.5)
            alpha = getattr(self.arg, 'alpha', 0.6)
            
            def distillation_loss(student_logits, teacher_logits, labels, teacher_features, student_features):
                # Hard loss (student vs true labels)
                student_logits = tf.squeeze(student_logits)
                teacher_logits = tf.squeeze(teacher_logits)
                labels = tf.cast(tf.squeeze(labels), tf.float32)
                
                hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student_logits)
                
                if hasattr(self, 'pos_weights') and self.pos_weights is not None:
                    hard_loss = hard_loss * (self.pos_weights * labels + (1 - labels))
                
                # Teacher confidence weighting
                teacher_probs = tf.sigmoid(teacher_logits)
                teacher_pred = tf.cast(teacher_probs > 0.5, tf.float32)
                correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
                
                weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
                
                # Feature distillation
                teacher_features_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
                student_features_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
                
                teacher_probs_feat = tf.nn.softmax(teacher_features_flat / temperature, axis=-1)
                student_log_probs_feat = tf.nn.log_softmax(student_features_flat / temperature, axis=-1)
                
                feature_loss = tf.reduce_sum(
                    teacher_probs_feat * (tf.math.log(teacher_probs_feat + 1e-10) - student_log_probs_feat), 
                    axis=-1
                )
                feature_loss = feature_loss * (temperature ** 2)
                
                total_loss = alpha * tf.reduce_mean(weights * feature_loss) + (1.0 - alpha) * tf.reduce_mean(weights * hard_loss)
                return total_loss
            
            self.distillation_loss = distillation_loss
            logger.info(f"Custom distillation loss created with temperature={temperature}, alpha={alpha}")
    
    def load_data(self):
        """Override parent load_data to add better debugging"""
        try:
            logger.info(f"Loading data for subjects - Train: {self.train_subjects}, Val: {self.val_subject}, Test: {self.test_subject}")
            
            # Verify dataset configuration
            if not hasattr(self.arg, 'dataset_args'):
                logger.error("No dataset_args found in configuration")
                return False
            
            logger.info(f"Dataset args: {self.arg.dataset_args}")
            
            # Import data preparation functions
            try:
                from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
                logger.info("Successfully imported dataset_tf functions")
            except Exception as e:
                logger.error(f"Failed to import dataset_tf: {e}")
                return False
            
            # Import feeder class
            try:
                Feeder = self.import_class(self.arg.feeder)
                logger.info(f"Successfully imported feeder: {self.arg.feeder}")
            except Exception as e:
                logger.error(f"Failed to import feeder {self.arg.feeder}: {e}")
                return False
            
            # Prepare dataset
            try:
                builder = prepare_smartfallmm_tf(self.arg)
                logger.info("Successfully prepared SmartFallMM dataset")
            except Exception as e:
                logger.error(f"Failed to prepare SmartFallMM dataset: {e}")
                traceback.print_exc()
                return False
            
            # Process training data
            try:
                logger.info(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False)
                logger.info(f"Training data shapes: {[(k, v.shape) for k, v in self.norm_train.items() if isinstance(v, np.ndarray)]}")
                
                if not self.norm_train or all(len(v) == 0 for v in self.norm_train.values() if isinstance(v, np.ndarray)):
                    logger.error("Training data is empty")
                    return False
            except Exception as e:
                logger.error(f"Failed to process training data: {e}")
                traceback.print_exc()
                return False
            
            # Create training data loader
            try:
                use_smv = getattr(self.arg, 'use_smv', False)
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size,
                    use_smv=use_smv
                )
                logger.info(f"Training data loader created with {len(self.data_loader['train'])} batches")
            except Exception as e:
                logger.error(f"Failed to create training data loader: {e}")
                traceback.print_exc()
                return False
            
            # Calculate class weights
            try:
                if 'labels' in self.norm_train:
                    from collections import Counter
                    label_counts = Counter(self.norm_train['labels'])
                    logger.info(f"Training label distribution: {label_counts}")
                    
                    if 1 in label_counts and 0 in label_counts:
                        self.pos_weights = tf.constant(label_counts[0] / label_counts[1], dtype=tf.float32)
                    else:
                        self.pos_weights = tf.constant(1.0, dtype=tf.float32)
                    logger.info(f"Positive class weight: {self.pos_weights}")
            except Exception as e:
                logger.warning(f"Failed to calculate class weights: {e}")
                self.pos_weights = tf.constant(1.0, dtype=tf.float32)
            
            # Process validation data
            try:
                logger.info(f"Processing validation data for subjects: {self.val_subject}")
                self.norm_val = split_by_subjects_tf(builder, self.val_subject, False)
                logger.info(f"Validation data shapes: {[(k, v.shape) for k, v in self.norm_val.items() if isinstance(v, np.ndarray)]}")
                
                if not self.norm_val or all(len(v) == 0 for v in self.norm_val.values() if isinstance(v, np.ndarray)):
                    logger.warning("Validation data is empty, using subset of training data")
                    # Use a small subset of training data for validation
                    train_size = len(self.norm_train['labels'])
                    val_size = min(train_size // 5, 100)
                    self.norm_val = {k: v[-val_size:].copy() for k, v in self.norm_train.items()}
                
                self.data_loader['val'] = Feeder(
                    dataset=self.norm_val,
                    batch_size=self.arg.val_batch_size,
                    use_smv=use_smv
                )
                logger.info(f"Validation data loader created with {len(self.data_loader['val'])} batches")
            except Exception as e:
                logger.error(f"Failed to process validation data: {e}")
                traceback.print_exc()
                return False
            
            # Process test data
            try:
                logger.info(f"Processing test data for subjects: {self.test_subject}")
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False)
                logger.info(f"Test data shapes: {[(k, v.shape) for k, v in self.norm_test.items() if isinstance(v, np.ndarray)]}")
                
                if not self.norm_test or all(len(v) == 0 for v in self.norm_test.values() if isinstance(v, np.ndarray)):
                    logger.error("Test data is empty")
                    return False
                
                self.data_loader['test'] = Feeder(
                    dataset=self.norm_test,
                    batch_size=self.arg.test_batch_size,
                    use_smv=use_smv
                )
                logger.info(f"Test data loader created with {len(self.data_loader['test'])} batches")
            except Exception as e:
                logger.error(f"Failed to process test data: {e}")
                traceback.print_exc()
                return False
            
            logger.info("Data loading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Critical error in data loading: {e}")
            traceback.print_exc()
            return False
    
    def train(self, epoch):
        try:
            logger.info(f"Starting distillation epoch {epoch+1}/{self.arg.num_epoch}")
            start_time = time.time()
            
            self.model.trainable = True
            self.teacher_model.trainable = False
            
            loader = self.data_loader['train']
            total_batches = len(loader)
            
            train_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            steps = 0
            
            for batch_idx in range(total_batches):
                if batch_idx % 10 == 0 or batch_idx + 1 == total_batches:
                    logger.info(f"Batch {batch_idx+1}/{total_batches}")
                
                try:
                    inputs, targets, _ = loader[batch_idx]
                    targets = tf.cast(targets, tf.float32)
                    
                    with tf.GradientTape() as tape:
                        # Get teacher predictions
                        teacher_outputs = self.teacher_model(inputs, training=False)
                        if isinstance(teacher_outputs, tuple):
                            teacher_logits, teacher_features = teacher_outputs
                        else:
                            teacher_logits = teacher_outputs
                            teacher_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                        
                        # Get student predictions
                        student_outputs = self.model(inputs, training=True)
                        if isinstance(student_outputs, tuple):
                            student_logits, student_features = student_outputs
                        else:
                            student_logits = student_outputs
                            student_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                        
                        # Calculate distillation loss
                        loss = self.distillation_loss(
                            student_logits, teacher_logits, targets, 
                            teacher_features, student_features
                        )
                    
                    # Compute gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for grad in gradients:
                        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                            has_nan = True
                            break
                    
                    if has_nan:
                        logger.warning(f"NaN gradients detected in batch {batch_idx}, skipping update")
                        continue
                    
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Calculate predictions
                    if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] > 1:
                        probabilities = tf.nn.softmax(student_logits)[:, 1]
                        predictions = tf.argmax(student_logits, axis=1)
                    else:
                        student_logits = tf.squeeze(student_logits)
                        probabilities = tf.sigmoid(student_logits)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Accumulate metrics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(
                    all_labels, all_preds, all_probs
                )
                
                self.train_loss_summary.append(float(train_loss))
                
                epoch_time = time.time() - start_time
                auc_str = f"{auc_score:.2f}%" if auc_score is not None else "N/A"
                
                logger.info(
                    f"Epoch {epoch+1} results: "
                    f"Loss={train_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_str} "
                    f"({epoch_time:.2f}s)"
                )
                
                # Validation
                val_loss = self.eval(epoch, loader_name='val')
                self.val_loss_summary.append(float(val_loss))
                
                # Check early stopping
                if self.early_stop(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    return True
                
                return False
            else:
                logger.warning(f"No steps completed in epoch {epoch+1}")
                return False
                
        except Exception as e:
            logger.error(f"Critical error in epoch {epoch+1}: {e}")
            traceback.print_exc()
            return False
    
    def start(self):
        try:
            if self.arg.phase == 'distill':
                logger.info("Starting knowledge distillation")
                
                results = []
                
                # Define subject splits
                val_subjects = [38, 46]
                train_subjects_fixed = [45, 36, 29]
                test_eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
                
                # Cross-validation across test subjects
                for i, test_subject in enumerate(test_eligible_subjects):
                    try:
                        # Reset for new fold
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                        self.best_loss = float('inf')
                        self.data_loader = {}
                        
                        # Set up subjects for this fold
                        self.test_subject = [test_subject]
                        self.val_subject = val_subjects
                        self.train_subjects = [s for s in test_eligible_subjects if s != test_subject]
                        self.train_subjects.extend(train_subjects_fixed)
                        
                        logger.info(f"\n=== Cross-validation fold {i+1}: Testing on subject {test_subject} ===")
                        logger.info(f"Train subjects: {self.train_subjects}")
                        logger.info(f"Val subjects: {self.val_subject}")
                        logger.info(f"Test subject: {test_subject}")
                        
                        # Clear session
                        tf.keras.backend.clear_session()
                        
                        # Load models
                        self.teacher_model = self.load_teacher_model()
                        self.model = self.load_model()
                        
                        # Load data
                        if not self.load_data():
                            logger.warning(f"Skipping subject {test_subject} due to data loading issues")
                            continue
                        
                        # Initialize training
                        self.load_optimizer()
                        self.load_distillation_loss()  # Make sure distillation loss is loaded
                        self.early_stop.reset()
                        
                        # Training loop
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            try:
                                early_stop = self.train(epoch)
                                if early_stop:
                                    logger.info(f"Early stopping at epoch {epoch+1}")
                                    break
                            except Exception as epoch_error:
                                logger.error(f"Error in epoch {epoch+1}: {epoch_error}")
                                if epoch == 0:
                                    logger.error(f"First epoch failed, skipping subject {test_subject}")
                                    break
                                continue
                        
                        # Save model
                        model_file = f"{self.model_path}_{test_subject}.weights.h5"
                        self.model.save_weights(model_file)
                        logger.info(f"Model weights saved to {model_file}")
                        
                        # Final evaluation
                        logger.info(f"=== Final evaluation on subject {test_subject} ===")
                        
                        # Load best weights
                        if os.path.exists(model_file):
                            self.model.load_weights(model_file)
                            logger.info(f"Loaded best weights from {model_file}")
                        
                        # Evaluate on test set
                        test_loss = self.eval(epoch=0, loader_name='test')
                        
                        # Save loss curves
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                        
                        # Record results
                        subject_result = {
                            'test_subject': str(test_subject),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2),
                            'auc': round(self.test_auc, 2) if self.test_auc is not None else None
                        }
                        results.append(subject_result)
                        
                    except Exception as e:
                        logger.error(f"Error processing subject {test_subject}: {e}")
                        traceback.print_exc()
                        continue
                    finally:
                        # Clean up
                        self.data_loader = {}
                        tf.keras.backend.clear_session()
                
                # Save results
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)
                    
                    with open(os.path.join(self.arg.work_dir, 'distillation_scores.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Print summary
                    logger.info("\n=== Final Distillation Results ===")
                    for result in results:
                        subject = result['test_subject']
                        acc = result.get('accuracy', 'N/A')
                        f1 = result.get('f1_score', 'N/A')
                        precision = result.get('precision', 'N/A')
                        recall = result.get('recall', 'N/A')
                        auc = result.get('auc', 'N/A')
                        
                        logger.info(
                            f"Subject {subject}: "
                            f"Acc={acc}%, "
                            f"F1={f1}%, "
                            f"Prec={precision}%, "
                            f"Rec={recall}%, "
                            f"AUC={auc}%"
                        )
                    
                    # Calculate averages
                    avg_acc = sum(r.get('accuracy', 0) for r in results) / len(results)
                    avg_f1 = sum(r.get('f1_score', 0) for r in results) / len(results)
                    avg_prec = sum(r.get('precision', 0) for r in results) / len(results)
                    avg_rec = sum(r.get('recall', 0) for r in results) / len(results)
                    avg_auc = sum(r.get('auc', 0) for r in results if r.get('auc') is not None) / len(results)
                    
                    logger.info(
                        f"Average: "
                        f"Acc={avg_acc:.2f}%, "
                        f"F1={avg_f1:.2f}%, "
                        f"Prec={avg_prec:.2f}%, "
                        f"Rec={avg_rec:.2f}%, "
                        f"AUC={avg_auc:.2f}%"
                    )
                else:
                    logger.warning("No results were generated")
                
                logger.info("Distillation completed successfully")
            else:
                logger.warning(f"Phase '{self.arg.phase}' not supported by distiller, use 'distill'")
        
        except Exception as e:
            logger.error(f"Fatal error in distillation process: {e}")
            traceback.print_exc()

def get_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation for Fall Detection')
    
    # Configuration
    parser.add_argument('--config', default='config/smartfallmm/distill.yaml', help='Config file path')
    parser.add_argument('--work-dir', type=str, default='../experiments/distill', help='Working directory')
    parser.add_argument('--model-saved-name', type=str, default='student_model', help='Model save name')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--phase', type=str, default='distill', choices=['distill'], help='Phase')
    
    # Model configuration
    parser.add_argument('--teacher-model', type=str, default=None, help='Teacher model class path')
    parser.add_argument('--teacher-args', type=str, default=None, help='Teacher model arguments')
    parser.add_argument('--teacher-weight', type=str, default=None, help='Teacher weights path')
    
    parser.add_argument('--model', type=str, default=None, help='Student model class path')
    parser.add_argument('--model-args', type=str, default=None, help='Student model arguments')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=4.5, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.6, help='Distillation alpha weight')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, help='Test batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--num-epoch', type=int, default=80, help='Number of epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--num-worker', type=int, default=40, help='Number of worker processes')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer type')
    parser.add_argument('--base-lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset')
    parser.add_argument('--dataset-args', type=str, default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=None, help='Subject IDs')
    parser.add_argument('--feeder', type=str, default=None, help='Data feeder class')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--result-file', type=str, default=None, help='Results output file')
    parser.add_argument('--print-log', type=bool, default=True, help='Print logs')
    parser.add_argument('--use-smv', type=bool, default=False, help='Use Signal Magnitude Vector')
    
    return parser

def main():
    parser = get_args()
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                for k, v in config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
            except yaml.YAMLError as e:
                logger.error(f"Error loading config file: {e}")
                return
    
    # Set up environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Using GPU: {args.device}")
    except Exception as e:
        logger.warning(f"GPU configuration error: {e}")
    
    # Set random seeds
    try:
        import random
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    except Exception as e:
        logger.warning(f"Error setting random seeds: {e}")
    
    # Run distillation
    try:
        distiller = Distiller(args)
        distiller.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
