#!/usr/bin/env python
import os, logging, json, traceback, time, argparse, yaml
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import custom modules
from trainer.base_trainer import BaseTrainer, EarlyStopping

class Distiller(BaseTrainer):
    """
    Knowledge distillation implementation for LightHART-TF
    Transfers knowledge from a teacher model (skeleton+accelerometer) to a student model (accelerometer only)
    """
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = None
        self.early_stop = EarlyStopping(patience=15, min_delta=0.001)
        
        # Load teacher model first
        self.load_teacher()
        
        # Then load student model
        self.model = self.load_model(arg.model, arg.model_args)
        num_params = self.count_parameters(self.model)
        self.print_log(f"Student model parameters: {num_params:,}")
        
        # Load distillation loss
        self.load_distillation_loss()
        
        self.print_log("Distiller initialized successfully")
    
    def load_teacher(self):
        """Load the teacher model from config or weights"""
        try:
            if not hasattr(self.arg, 'teacher_model') or not hasattr(self.arg, 'teacher_args'):
                raise ValueError("Teacher model and args must be specified")
            
            # Import the teacher model class
            model_class = self.import_class(self.arg.teacher_model)
            self.teacher_model = model_class(**self.arg.teacher_args)
            
            # Check for teacher weights
            if not hasattr(self.arg, 'teacher_weight'):
                raise ValueError("Teacher weights path must be specified")
            
            try:
                # Try loading as full model first
                self.teacher_model = tf.keras.models.load_model(self.arg.teacher_weight)
                self.print_log(f"Loaded teacher model from {self.arg.teacher_weight}")
            except:
                try:
                    # If full model loading fails, try loading weights
                    if hasattr(self, 'test_subject') and self.test_subject:
                        subject_id = self.test_subject[0] if isinstance(self.test_subject, list) else self.test_subject
                        weight_path = f"{self.arg.teacher_weight}_{subject_id}.weights.h5"
                        
                        if os.path.exists(weight_path):
                            # Build the model with dummy inputs first
                            acc_frames = self.arg.teacher_args.get('acc_frames', 128)
                            acc_coords = self.arg.teacher_args.get('acc_coords', 3)
                            num_joints = self.arg.teacher_args.get('num_joints', 32)
                            in_chans = self.arg.teacher_args.get('in_chans', 3)
                            
                            dummy_input = {
                                'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32),
                                'skeleton': tf.zeros((2, acc_frames, num_joints, in_chans), dtype=tf.float32)
                            }
                            
                            _ = self.teacher_model(dummy_input, training=False)
                            
                            # Load weights
                            self.teacher_model.load_weights(weight_path)
                            self.print_log(f"Loaded teacher weights from {weight_path}")
                        else:
                            # Fallback to general weights file
                            self.teacher_model.load_weights(f"{self.arg.teacher_weight}")
                            self.print_log(f"Loaded teacher weights from {self.arg.teacher_weight}")
                    else:
                        # No test subject specified, load general weights
                        self.teacher_model.load_weights(f"{self.arg.teacher_weight}")
                        self.print_log(f"Loaded teacher weights from {self.arg.teacher_weight}")
                except Exception as e:
                    raise ValueError(f"Failed to load teacher weights: {e}")
            
            # Mark teacher as non-trainable
            self.teacher_model.trainable = False
            num_params = self.count_parameters(self.teacher_model)
            self.print_log(f"Teacher model parameters: {num_params:,}")
            
        except Exception as e:
            self.print_log(f"Error loading teacher model: {e}")
            traceback.print_exc()
            raise
    
    def load_distillation_loss(self):
        """Load distillation loss function with temperature and alpha parameters"""
        try:
            # Default parameters
            temperature = getattr(self.arg, 'temperature', 4.5)
            alpha = getattr(self.arg, 'alpha', 0.6)
            
            # Override from distiller_args if available
            if hasattr(self.arg, 'distiller_args'):
                temperature = self.arg.distiller_args.get('temperature', temperature)
                alpha = self.arg.distiller_args.get('alpha', alpha)
            
            # Define distillation loss function
            def distillation_loss(student_logits, teacher_logits, labels, teacher_features, student_features):
                # Hard target loss (BCE)
                # Ensure proper shapes for binary classification
                if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] == 1:
                    student_logits = tf.squeeze(student_logits, axis=1)
                if len(tf.shape(teacher_logits)) > 1 and tf.shape(teacher_logits)[1] == 1:
                    teacher_logits = tf.squeeze(teacher_logits, axis=1)
                if len(tf.shape(labels)) > 1 and tf.shape(labels)[1] == 1:
                    labels = tf.squeeze(labels, axis=1)
                
                # Binary cross-entropy for hard labels
                hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student_logits)
                
                # Weight based on teacher's correct predictions
                teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
                correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
                
                # Higher weight (1.0) for correct predictions, lower (0.5) for incorrect
                weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
                weighted_hard_loss = weights * hard_loss
                
                # Feature distillation using KL divergence
                # Apply softmax with temperature
                teacher_features_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
                student_features_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
                
                teacher_probs = tf.nn.softmax(teacher_features_flat / temperature, axis=-1)
                student_log_probs = tf.nn.log_softmax(student_features_flat / temperature, axis=-1)
                
                # Calculate KL divergence
                feature_loss = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs), axis=-1)
                feature_loss = feature_loss * (temperature ** 2)
                
                # Final loss combining hard and soft components
                total_loss = alpha * tf.reduce_mean(feature_loss) + (1.0 - alpha) * tf.reduce_mean(weighted_hard_loss)
                
                return total_loss
            
            self.distillation_loss = distillation_loss
            self.print_log(f"Distillation loss initialized with temperature={temperature}, alpha={alpha}")
            return True
        except Exception as e:
            self.print_log(f"Error loading distillation loss: {e}")
            return False
    
    def viz_feature(self, teacher_features, student_features, epoch, max_samples=8):
        """Visualize feature representations from teacher and student models"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            if isinstance(teacher_features, tf.Tensor):
                teacher_features = teacher_features.numpy()
            if isinstance(student_features, tf.Tensor):
                student_features = student_features.numpy()
            
            # Flatten features if needed
            if len(teacher_features.shape) > 2:
                teacher_features = teacher_features.reshape(teacher_features.shape[0], -1)
            if len(student_features.shape) > 2:
                student_features = student_features.reshape(student_features.shape[0], -1)
            
            # Limit number of samples for visualization
            num_samples = min(max_samples, teacher_features.shape[0])
            teacher_features = teacher_features[:num_samples]
            student_features = student_features[:num_samples]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            for i in range(num_samples):
                plt.subplot(2, 4, i+1)
                sns.kdeplot(teacher_features[i, :], bw_adjust=0.5, color='blue', label='Teacher')
                sns.kdeplot(student_features[i, :], bw_adjust=0.5, color='red', label='Student')
                if i == 0:
                    plt.legend()
            
            plt.savefig(os.path.join(viz_dir, f'feature_distribution_epoch_{epoch}.png'))
            plt.close()
            
            # Feature similarity metrics (cosine similarity)
            teacher_norm = tf.norm(teacher_features, axis=1, keepdims=True)
            student_norm = tf.norm(student_features, axis=1, keepdims=True)
            teacher_normalized = teacher_features / (teacher_norm + 1e-10)
            student_normalized = student_features / (student_norm + 1e-10)
            cosine_sim = tf.reduce_mean(tf.reduce_sum(teacher_normalized * student_normalized, axis=1))
            
            self.print_log(f"Epoch {epoch+1}: Feature cosine similarity: {cosine_sim:.4f}")
            
        except Exception as e:
            self.print_log(f"Error visualizing features: {e}")
    
    def train(self, epoch):
        """Run training with knowledge distillation for one epoch"""
        try:
            self.print_log(f"Starting distillation epoch {epoch+1}")
            start_time = time.time()
            
            # Set models to appropriate modes
            self.model.trainable = True
            self.teacher_model.trainable = False
            
            loader = self.data_loader['train']
            total_batches = len(loader)
            
            self.print_log(f"Epoch {epoch+1}/{self.arg.num_epoch} - {total_batches} batches")
            
            train_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            steps = 0
            
            for batch_idx in range(total_batches):
                if batch_idx % 5 == 0 or batch_idx + 1 == total_batches:
                    self.print_log(f"Distillation epoch {epoch+1}: batch {batch_idx+1}/{total_batches}")
                
                try:
                    # Get batch data
                    inputs, targets, _ = loader[batch_idx]
                    targets = tf.cast(targets, tf.float32)
                    
                    # Run teacher and student models
                    with tf.GradientTape() as tape:
                        # Get teacher predictions (no gradients)
                        teacher_outputs = self.teacher_model(inputs, training=False)
                        if isinstance(teacher_outputs, tuple) and len(teacher_outputs) > 0:
                            teacher_logits, teacher_features = teacher_outputs
                        else:
                            # Fallback if teacher doesn't return features
                            teacher_logits = teacher_outputs
                            teacher_features = tf.zeros((tf.shape(targets)[0], 32), dtype=tf.float32)
                        
                        # Get student predictions (with gradients)
                        student_outputs = self.model(inputs, training=True)
                        if isinstance(student_outputs, tuple) and len(student_outputs) > 0:
                            student_logits, student_features = student_outputs
                        else:
                            # Fallback if student doesn't return features
                            student_logits = student_outputs
                            student_features = tf.zeros((tf.shape(targets)[0], 32), dtype=tf.float32)
                        
                        # Calculate distillation loss
                        loss = self.distillation_loss(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits, 
                            labels=targets,
                            teacher_features=teacher_features, 
                            student_features=student_features
                        )
                    
                    # Visualize features periodically
                    if epoch % 10 == 0 and batch_idx == 0:
                        self.viz_feature(
                            teacher_features=teacher_features,
                            student_features=student_features,
                            epoch=epoch
                        )
                    
                    # Compute and apply gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for grad in gradients:
                        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                            has_nan = True
                            break
                    
                    if has_nan:
                        self.print_log(f"WARNING: NaN gradients detected in batch {batch_idx}")
                        continue
                    
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Calculate predictions for metrics
                    if len(student_logits.shape) > 1 and student_logits.shape[-1] > 1:
                        # Multi-class
                        probabilities = tf.nn.softmax(student_logits)[:, 1]
                        predictions = tf.argmax(student_logits, axis=-1)
                    else:
                        # Binary
                        logits_squeezed = tf.squeeze(student_logits)
                        probabilities = tf.sigmoid(logits_squeezed)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Collect statistics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    self.print_log(f"Error in distillation batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            # Calculate and report metrics
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
                
                self.train_loss_summary.append(float(train_loss))
                
                epoch_time = time.time() - start_time
                auc_str = f"{auc_score:.2f}%" if auc_score is not None else "N/A"
                self.print_log(
                    f"Distillation Epoch {epoch+1} results: "
                    f"Loss={train_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_str} "
                    f"({epoch_time:.2f}s)"
                )
                
                # Run validation
                self.print_log(f"Running validation for epoch {epoch+1}")
                val_loss = self.eval(epoch, loader_name='val')
                
                self.val_loss_summary.append(float(val_loss))
                
                # Check for early stopping
                if self.early_stop(val_loss):
                    self.print_log(f"Early stopping triggered at epoch {epoch+1}")
                    return True
                
                return False
            else:
                self.print_log(f"Warning: No steps completed in epoch {epoch+1}")
                return False
                
        except Exception as e:
            self.print_log(f"Critical error in epoch {epoch+1}: {e}")
            traceback.print_exc()
            return False
    
    def start(self):
        """Run full distillation process with cross-validation"""
        try:
            if self.arg.phase == 'distill':
                self.print_log("Starting knowledge distillation with parameters:")
                for key, value in vars(self.arg).items():
                    if key not in ['teacher_args', 'student_args', 'model_args', 'dataset_args']:
                        self.print_log(f"  {key}: {value}")
                
                results = []
                
                # Define cross-validation splits
                val_subjects = getattr(self.arg, 'val_subjects_fixed', [38, 46])
                train_subjects_fixed = getattr(self.arg, 'train_subjects_fixed', [45, 36, 29])
                test_eligible_subjects = getattr(self.arg, 'test_eligible_subjects', 
                                               [32, 39, 30, 31, 33, 34, 35, 37, 43, 44])
                
                if not hasattr(self.arg, 'subjects') or not self.arg.subjects:
                    self.arg.subjects = test_eligible_subjects + train_subjects_fixed + val_subjects
                
                # Run cross-validation
                for i, test_subject in enumerate(test_eligible_subjects):
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    self.data_loader = {}
                    
                    # Set up this fold
                    self.test_subject = [test_subject]
                    self.val_subject = val_subjects
                    self.train_subjects = [s for s in test_eligible_subjects if s != test_subject]
                    self.train_subjects.extend(train_subjects_fixed)
                    
                    self.print_log(f"\n=== Cross-validation fold {i+1}: Testing on subject {test_subject} ===")
                    self.print_log(f"Train: {len(self.train_subjects)} subjects: {self.train_subjects}")
                    self.print_log(f"Val: {len(self.val_subject)} subjects: {self.val_subject}")
                    self.print_log(f"Test: Subject {test_subject}")
                    
                    # Reset models for each fold
                    tf.keras.backend.clear_session()
                    self.load_teacher()
                    self.model = self.load_model(self.arg.model, self.arg.model_args)
                    
                    if not self.load_data():
                        self.print_log(f"Skipping subject {test_subject} due to data loading issues")
                        continue
                    
                    # Initialize optimizer and loss
                    self.load_optimizer()
                    self.load_distillation_loss()
                    self.early_stop.reset()
                    
                    # Training loop
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        try:
                            early_stop = self.train(epoch)
                            if early_stop:
                                self.print_log(f"Early stopping at epoch {epoch+1}")
                                break
                        except Exception as epoch_error:
                            self.print_log(f"Error in epoch {epoch+1}: {epoch_error}")
                            if epoch == 0:
                                self.print_log(f"First epoch failed, skipping subject {test_subject}")
                                break
                            continue
                    
                    # Load best weights for evaluation
                    best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                    if os.path.exists(best_weights):
                        try:
                            self.model.load_weights(best_weights)
                            self.print_log(f"Loaded best weights from {best_weights}")
                        except Exception as weight_error:
                            self.print_log(f"Error loading best weights: {weight_error}")
                    
                    # Final evaluation
                    self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                    result = self.evaluate_test_set()
                    
                    # Visualize training progress
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                    
                    # Save results
                    if result:
                        subject_result = {
                            'test_subject': str(test_subject),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2),
                            'auc': round(self.test_auc, 2) if self.test_auc is not None else None
                        }
                        results.append(subject_result)
                        self.print_log(f"Completed fold for subject {test_subject}")
                    
                    # Clean up for next fold
                    self.data_loader = {}
                    tf.keras.backend.clear_session()
                
                # Report final results
                if results:
                    # Add average row
                    results = self.add_avg_df(results)
                    
                    # Save as CSV and JSON
                    import pandas as pd
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)
                    
                    with open(os.path.join(self.arg.work_dir, 'distillation_scores.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Print final results
                    self.print_log("\n=== Final Distillation Results ===")
                    for result in results:
                        subject = result['test_subject']
                        accuracy = result.get('accuracy', 'N/A')
                        f1 = result.get('f1_score', 'N/A')
                        precision = result.get('precision', 'N/A')
                        recall = result.get('recall', 'N/A')
                        auc = result.get('auc', 'N/A')
                        
                        self.print_log(
                            f"Subject {subject}: "
                            f"Acc={accuracy}%, "
                            f"F1={f1}%, "
                            f"Prec={precision}%, "
                            f"Rec={recall}%, "
                            f"AUC={auc}%"
                        )
                
                self.print_log("Distillation completed successfully")
            else:
                self.print_log(f"Phase '{self.arg.phase}' not supported by distiller, use 'distill'")
        
        except Exception as e:
            self.print_log(f"Fatal error in distillation process: {e}")
            traceback.print_exc()

def get_args():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Knowledge Distillation for Fall Detection')
    
    # Basic configuration
    parser.add_argument('--config', default='config/smartfallmm/distill.yaml', help='Config file path')
    parser.add_argument('--work-dir', type=str, default='../experiments/distill', help='Working directory')
    parser.add_argument('--model-saved-name', type=str, default='student_model', help='Model save name')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--phase', type=str, default='distill', choices=['distill'], help='Phase')
    
    # Teacher model configuration
    parser.add_argument('--teacher-model', type=str, default=None, help='Teacher model class path')
    parser.add_argument('--teacher-args', type=str, default=None, help='Teacher model arguments')
    parser.add_argument('--teacher-weight', type=str, default=None, help='Teacher weights path')
    
    # Student model configuration
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
    
    # Dataset configuration
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
    """Main entry point for the distillation script"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                for k, v in config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
            except yaml.YAMLError as e:
                logging.error(f"Error loading config file: {e}")
                exit(1)
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure GPU growth
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logging.info(f"Using GPU(s): {[d.name for d in physical_devices]}")
        else:
            logging.info("No GPU found, using CPU")
    except Exception as e:
        logging.warning(f"GPU configuration error: {e}")
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create and run distiller
    distiller = Distiller(args)
    distiller.start()

if __name__ == "__main__":
    main()
