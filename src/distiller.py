#!/usr/bin/env python
import os, logging, json, traceback, time, argparse, yaml
import numpy as np
import tensorflow as tf
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("distiller.log")
    ]
)

# Import custom modules
from trainer.base_trainer import BaseTrainer, EarlyStopping

class Distiller(BaseTrainer):
    """
    Knowledge distillation implementation for LightHART-TF
    Transfers knowledge from a teacher model (skeleton+accelerometer) to a student model (accelerometer only)
    """
    def __init__(self, arg):
        # Initialize base attributes
        super().__init__(arg)
        
        # Load teacher model
        logging.info("Loading teacher model...")
        self.teacher_model = self.load_teacher_model()
        
        # Initialize distillation loss
        self.load_distillation_loss()
        
        logging.info("Distiller initialized successfully")
    
    def load_teacher_model(self):
        """Load the teacher model using H5 weights"""
        try:
            if not hasattr(self.arg, 'teacher_model') or not hasattr(self.arg, 'teacher_args'):
                raise ValueError("Teacher model and args must be specified")
            
            # Import the teacher model class
            logging.info(f"Importing teacher model class: {self.arg.teacher_model}")
            teacher_class = self.import_class(self.arg.teacher_model)
            
            # Initialize model with args
            logging.info("Creating teacher model instance")
            teacher_model = teacher_class(**self.arg.teacher_args)
            
            # Check for teacher weights
            if not hasattr(self.arg, 'teacher_weight'):
                raise ValueError("Teacher weights path must be specified")
            
            teacher_weight_path = self.arg.teacher_weight
            
            # Find subject-specific weights if needed
            if hasattr(self, 'test_subject') and self.test_subject:
                subject_id = self.test_subject[0] if isinstance(self.test_subject, list) else self.test_subject
                
                # Try different weight formats for the specific subject
                weight_formats = [
                    f"{teacher_weight_path.rsplit('_', 1)[0]}_{subject_id}.h5",
                    f"{teacher_weight_path.rsplit('_', 1)[0]}_{subject_id}.weights.h5",
                    f"{teacher_weight_path.rsplit('.', 1)[0]}_{subject_id}.h5",
                    f"{teacher_weight_path.rsplit('.', 1)[0]}_{subject_id}.weights.h5"
                ]
                
                for weight_path in weight_formats:
                    if os.path.exists(weight_path):
                        teacher_weight_path = weight_path
                        logging.info(f"Found subject-specific weights: {teacher_weight_path}")
                        break
            
            # Initialize with dummy input to build the model
            try:
                logging.info("Building teacher model with dummy inputs")
                acc_frames = self.arg.teacher_args.get('acc_frames', 128)
                num_joints = self.arg.teacher_args.get('num_joints', 32)
                acc_coords = self.arg.teacher_args.get('acc_coords', 3)
                
                dummy_acc = tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                dummy_skl = tf.zeros((2, acc_frames, num_joints, 3), dtype=tf.float32)
                
                # Forward pass to initialize layers
                _ = teacher_model({'accelerometer': dummy_acc, 'skeleton': dummy_skl}, training=False)
                logging.info("Teacher model initialized with dummy inputs")
            except Exception as e:
                logging.warning(f"Error initializing teacher model with dummy inputs: {e}")
            
            # Try to load the weights (.h5 format preferred)
            if teacher_weight_path.endswith('.keras'):
                try:
                    # Force .h5 weight loading if .keras file exists
                    h5_path = teacher_weight_path.replace('.keras', '.h5')
                    weights_path = teacher_weight_path.replace('.keras', '.weights.h5')
                    
                    if os.path.exists(weights_path):
                        logging.info(f"Loading teacher weights from {weights_path}")
                        teacher_model.load_weights(weights_path)
                    elif os.path.exists(h5_path):
                        logging.info(f"Loading teacher weights from {h5_path}")
                        teacher_model.load_weights(h5_path)
                    else:
                        # Extract weights from .keras model
                        logging.info(f"Loading weights from .keras model: {teacher_weight_path}")
                        keras_model = tf.keras.models.load_model(teacher_weight_path, compile=False)
                        teacher_model.set_weights(keras_model.get_weights())
                except Exception as e:
                    logging.error(f"Error loading teacher keras model: {e}")
                    raise
            else:
                # Direct .h5 weight loading
                logging.info(f"Loading teacher weights from {teacher_weight_path}")
                teacher_model.load_weights(teacher_weight_path)
            
            # Mark teacher as non-trainable
            teacher_model.trainable = False
            
            # Count parameters
            try:
                num_params = sum(np.prod(v.get_shape().as_list()) for v in teacher_model.trainable_variables)
                logging.info(f"Teacher model parameters: {num_params:,}")
            except Exception as e:
                logging.warning(f"Failed to count parameters: {e}")
            
            return teacher_model
            
        except Exception as e:
            logging.error(f"Error loading teacher model: {e}")
            traceback.print_exc()
            raise
    
    def load_distillation_loss(self):
        """Load distillation loss function"""
        try:
            # Import loss module
            from utils.loss import DistillationLoss
            
            # Get parameters
            temperature = getattr(self.arg, 'temperature', 4.5)
            alpha = getattr(self.arg, 'alpha', 0.6)
            
            # Create loss function
            self.distillation_loss = DistillationLoss(
                temperature=temperature,
                alpha=alpha,
                pos_weight=self.pos_weights if hasattr(self, 'pos_weights') else None
            )
            
            logging.info(f"Distillation loss initialized with temperature={temperature}, alpha={alpha}")
            return True
        except Exception as e:
            logging.error(f"Error loading distillation loss: {e}")
            
            # Create custom distillation loss directly
            self._create_custom_distillation_loss()
            return False
    
    def _create_custom_distillation_loss(self):
        """Create distillation loss directly if module import fails"""
        temperature = getattr(self.arg, 'temperature', 4.5)
        alpha = getattr(self.arg, 'alpha', 0.6)
        
        def distillation_loss(student_logits, teacher_logits, labels, teacher_features, student_features):
            # Binary cross-entropy for hard labels
            if hasattr(self, 'pos_weights') and self.pos_weights is not None:
                hard_loss = tf.nn.weighted_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.float32),
                    logits=student_logits,
                    pos_weight=self.pos_weights
                )
            else:
                hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.float32),
                    logits=student_logits
                )
            
            # Weight based on teacher's correct predictions
            teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
            correct_mask = tf.cast(tf.equal(teacher_pred, tf.cast(labels, tf.float32)), tf.float32)
            
            # Higher weight for correct predictions
            weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
            weighted_hard_loss = weights * hard_loss
            
            # Feature distillation using KL divergence
            teacher_features_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
            student_features_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
            
            teacher_probs = tf.nn.softmax(teacher_features_flat / temperature, axis=-1)
            student_log_probs = tf.nn.log_softmax(student_features_flat / temperature, axis=-1)
            
            # KL divergence
            feature_loss = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs), axis=-1)
            feature_loss = feature_loss * (temperature ** 2)
            
            # Combine losses
            total_loss = alpha * tf.reduce_mean(feature_loss) + (1.0 - alpha) * tf.reduce_mean(weighted_hard_loss)
            return total_loss
        
        self.distillation_loss = distillation_loss
        logging.info(f"Custom distillation loss created with temperature={temperature}, alpha={alpha}")
    
    def viz_feature(self, teacher_features, student_features, epoch):
        """Visualize feature representations from teacher and student models"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create directory if needed
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Convert to numpy if needed
            if isinstance(teacher_features, tf.Tensor):
                teacher_features = teacher_features.numpy()
            if isinstance(student_features, tf.Tensor):
                student_features = student_features.numpy()
            
            # Flatten features
            if len(teacher_features.shape) > 2:
                teacher_features = np.reshape(teacher_features, (teacher_features.shape[0], -1))
            if len(student_features.shape) > 2:
                student_features = np.reshape(student_features, (student_features.shape[0], -1))
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            max_samples = min(8, teacher_features.shape[0])
            
            for i in range(max_samples):
                plt.subplot(2, 4, i+1)
                sns.kdeplot(teacher_features[i], label='Teacher', color='blue')
                sns.kdeplot(student_features[i], label='Student', color='red')
                if i == 0:
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'feature_distribution_epoch_{epoch}.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error visualizing features: {e}")
    
    def train(self, epoch):
        """Run training with knowledge distillation for one epoch"""
        try:
            logging.info(f"Starting distillation epoch {epoch+1}/{self.arg.num_epoch}")
            start_time = time.time()
            
            # Set models to appropriate modes
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
                    logging.info(f"Batch {batch_idx+1}/{total_batches}")
                
                try:
                    # Get batch data
                    inputs, targets, _ = loader[batch_idx]
                    targets = tf.cast(targets, tf.float32)
                    
                    # Run teacher and student models
                    with tf.GradientTape() as tape:
                        # Teacher predictions (no gradients)
                        teacher_outputs = self.teacher_model(inputs, training=False)
                        if isinstance(teacher_outputs, tuple) and len(teacher_outputs) > 1:
                            teacher_logits, teacher_features = teacher_outputs
                        else:
                            teacher_logits = teacher_outputs
                            # Create dummy features if teacher doesn't output them
                            teacher_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                        
                        # Student predictions (with gradients)
                        student_outputs = self.model(inputs, training=True)
                        if isinstance(student_outputs, tuple) and len(student_outputs) > 1:
                            student_logits, student_features = student_outputs
                        else:
                            student_logits = student_outputs
                            # Create dummy features if student doesn't output them
                            student_features = tf.zeros((tf.shape(targets)[0], self.arg.model_args['embed_dim']), dtype=tf.float32)
                        
                        # Calculate distillation loss
                        loss = self.distillation_loss(
                            student_logits, teacher_logits, targets, 
                            teacher_features, student_features
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
                        logging.warning(f"NaN gradients detected in batch {batch_idx}, skipping update")
                        continue
                    
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Visualize features periodically
                    if epoch % 10 == 0 and batch_idx == 0:
                        self.viz_feature(teacher_features, student_features, epoch)
                    
                    # Calculate predictions for metrics
                    if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] > 1:
                        # Multi-class
                        probabilities = tf.nn.softmax(student_logits)[:, 1]
                        predictions = tf.argmax(student_logits, axis=1)
                    else:
                        # Binary
                        student_logits = tf.squeeze(student_logits)
                        probabilities = tf.sigmoid(student_logits)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Collect statistics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            # Calculate and report metrics
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(
                    all_labels, all_preds, all_probs
                )
                
                self.train_loss_summary.append(float(train_loss))
                
                epoch_time = time.time() - start_time
                auc_str = f"{auc_score:.2f}%" if auc_score is not None else "N/A"
                
                logging.info(
                    f"Epoch {epoch+1} results: "
                    f"Loss={train_loss:.4f}, "
                    f"Acc={accuracy:.2f}%, "
                    f"F1={f1:.2f}%, "
                    f"Prec={precision:.2f}%, "
                    f"Rec={recall:.2f}%, "
                    f"AUC={auc_str} "
                    f"({epoch_time:.2f}s)"
                )
                
                # Run validation
                val_loss = self.eval(epoch, loader_name='val')
                
                self.val_loss_summary.append(float(val_loss))
                
                # Check for early stopping
                if self.early_stop(val_loss):
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    return True
                
                return False
            else:
                logging.warning(f"No steps completed in epoch {epoch+1}")
                return False
                
        except Exception as e:
            logging.error(f"Critical error in epoch {epoch+1}: {e}")
            traceback.print_exc()
            return False
    
    def start(self):
        """Run full distillation process"""
        try:
            if self.arg.phase == 'distill':
                logging.info("Starting knowledge distillation")
                
                results = []
                
                # Get subject splits
                val_subjects = getattr(self.arg, 'val_subjects', [38, 46])
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
                    
                    logging.info(f"\n=== Cross-validation fold {i+1}: Testing on subject {test_subject} ===")
                    logging.info(f"Train subjects: {self.train_subjects}")
                    logging.info(f"Val subjects: {self.val_subject}")
                    logging.info(f"Test subject: {test_subject}")
                    
                    # Reset models for each fold
                    tf.keras.backend.clear_session()
                    
                    # Load new teacher model with subject-specific weights
                    self.teacher_model = self.load_teacher_model()
                    
                    # Create new student model
                    self.model = self.load_model()
                    
                    # Load data for this fold
                    data_loaded = self.load_data()
                    if not data_loaded:
                        logging.warning(f"Skipping subject {test_subject} due to data loading issues")
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
                                logging.info(f"Early stopping at epoch {epoch+1}")
                                break
                        except Exception as epoch_error:
                            logging.error(f"Error in epoch {epoch+1}: {epoch_error}")
                            if epoch == 0:
                                logging.error(f"First epoch failed, skipping subject {test_subject}")
                                break
                            continue
                    
                    # Save best model weights
                    model_file = f"{self.model_path}_{test_subject}.h5"
                    self.model.save_weights(model_file)
                    logging.info(f"Model weights saved to {model_file}")
                    
                    # Final evaluation
                    logging.info(f"=== Final evaluation on subject {test_subject} ===")
                    
                    # Load best weights
                    weights_path = f"{self.model_path}_{test_subject}.h5"
                    if os.path.exists(weights_path):
                        self.model.load_weights(weights_path)
                        logging.info(f"Loaded best weights from {weights_path}")
                    
                    val_loss = self.eval(epoch=0, loader_name='test')
                    
                    # Visualize training progress
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary, subject_id=test_subject)
                    
                    subject_result = {
                        'test_subject': str(test_subject),
                        'accuracy': round(self.test_accuracy, 2),
                        'f1_score': round(self.test_f1, 2),
                        'precision': round(self.test_precision, 2),
                        'recall': round(self.test_recall, 2),
                        'auc': round(self.test_auc, 2) if self.test_auc is not None else None
                    }
                    results.append(subject_result)
                    
                    # Clean up for next fold
                    self.data_loader = {}
                    tf.keras.backend.clear_session()
                
                # Report final results
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)
                    
                    with open(os.path.join(self.arg.work_dir, 'distillation_scores.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Print final results
                    logging.info("\n=== Final Distillation Results ===")
                    for result in results:
                        subject = result['test_subject']
                        acc = result.get('accuracy', 'N/A')
                        f1 = result.get('f1_score', 'N/A')
                        precision = result.get('precision', 'N/A')
                        recall = result.get('recall', 'N/A')
                        auc = result.get('auc', 'N/A')
                        
                        logging.info(
                            f"Subject {subject}: "
                            f"Acc={acc}%, "
                            f"F1={f1}%, "
                            f"Prec={precision}%, "
                            f"Rec={recall}%, "
                            f"AUC={auc}%"
                        )
                    
                    # Add averages
                    avg_acc = sum(r.get('accuracy', 0) for r in results) / len(results)
                    avg_f1 = sum(r.get('f1_score', 0) for r in results) / len(results)
                    avg_prec = sum(r.get('precision', 0) for r in results) / len(results)
                    avg_rec = sum(r.get('recall', 0) for r in results) / len(results)
                    avg_auc = sum(r.get('auc', 0) for r in results if r.get('auc') is not None) / len(results)
                    
                    logging.info(
                        f"Average: "
                        f"Acc={avg_acc:.2f}%, "
                        f"F1={avg_f1:.2f}%, "
                        f"Prec={avg_prec:.2f}%, "
                        f"Rec={avg_rec:.2f}%, "
                        f"AUC={avg_auc:.2f}%"
                    )
                
                logging.info("Distillation completed successfully")
            else:
                logging.warning(f"Phase '{self.arg.phase}' not supported by distiller, use 'distill'")
        
        except Exception as e:
            logging.error(f"Fatal error in distillation process: {e}")
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
    # Parse arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                # Update args with config values if not already set
                for k, v in config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
            except yaml.YAMLError as e:
                logging.error(f"Error loading config file: {e}")
                return
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure GPU growth
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logging.info(f"Using GPU: {args.device}")
            logging.info(f"Found {len(physical_devices)} GPU(s): {[d.name for d in physical_devices]}")
            
            # Show GPU memory usage
            for i, device in enumerate(physical_devices):
                memory_info = tf.config.experimental.get_memory_info(f'/device:GPU:{i}')
                available_memory = memory_info.get('allocated_bytes', 0) / (1024 * 1024)
                total_memory = tf.config.experimental.get_device_details(device).get('memory_limit', 0) / (1024 * 1024)
                logging.info(f"GPU {i}: Memory {available_memory:.0f}/{total_memory:.0f} MB ({available_memory/total_memory*100:.1f}%)")
        else:
            logging.info("No GPU found, using CPU")
    except Exception as e:
        logging.warning(f"GPU configuration error: {e}")
    
    # Set random seeds
    try:
        import random
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        random.seed(args.seed)
        logging.info(f"Random seed set to {args.seed}")
    except Exception as e:
        logging.warning(f"Error setting random seeds: {e}")
    
    # Create and run distiller
    try:
        distiller = Distiller(args)
        distiller.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
