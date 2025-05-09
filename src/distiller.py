#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import json
import traceback
import time
from datetime import datetime
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from utils.loss_tf import DistillationLoss
from trainer.base_trainer import BaseTrainer, EarlyStopping
from utils.model_utils import import_class, load_model, count_parameters, save_model

logger = logging.getLogger('lightheart-tf')

class Distiller(BaseTrainer):
    """
    Knowledge Distillation trainer for transferring knowledge from teacher to student
    Designed to be compatible with the original LightHART PyTorch implementation
    """
    def __init__(self, arg):
        """Initialize distiller with provided arguments"""
        super().__init__(arg)
        
        # Override some attributes to match distillation setup
        self.teacher_model = None
        self.early_stop = EarlyStopping(patience=15, min_delta=0.001)
        
        # Load teacher model
        self.load_teacher()
        
        # Ensure we're using the student as our main model
        self.model = self.load_model(arg.model, arg.model_args)
        num_params = self.count_parameters(self.model)
        self.print_log(f"Student model parameters: {num_params:,}")
        
        # Initialize distillation loss
        self.load_distillation_loss()
    
    def load_teacher(self):
        """Load pre-trained teacher model"""
        try:
            if not hasattr(self.arg, 'teacher_model') or not hasattr(self.arg, 'teacher_args'):
                raise ValueError("Teacher model and args must be specified")
            
            model_class = self.import_class(self.arg.teacher_model)
            self.teacher_model = model_class(**self.arg.teacher_args)
            
            if not hasattr(self.arg, 'teacher_weight'):
                raise ValueError("Teacher weights path must be specified")
            
            # Try to load as full model first
            try:
                self.teacher_model = tf.keras.models.load_model(self.arg.teacher_weight)
                self.print_log(f"Loaded teacher model from {self.arg.teacher_weight}")
            except:
                # If that fails, try to load weights
                try:
                    self.teacher_model.load_weights(self.arg.teacher_weight)
                    self.print_log(f"Loaded teacher weights from {self.arg.teacher_weight}")
                except Exception as e:
                    # Get specific subject ID from test_subject if available
                    if hasattr(self, 'test_subject') and self.test_subject:
                        subject_id = self.test_subject[0] if isinstance(self.test_subject, list) else self.test_subject
                        weight_path = f"{self.arg.teacher_weight}_{subject_id}.weights.h5"
                        self.print_log(f"Trying to load teacher weights for subject {subject_id}: {weight_path}")
                        
                        if os.path.exists(weight_path):
                            self.teacher_model.load_weights(weight_path)
                            self.print_log(f"Loaded teacher weights from {weight_path}")
                        else:
                            raise ValueError(f"Teacher weights not found at {weight_path}")
                    else:
                        raise e
            
            # Verify teacher model loaded successfully by running inference on dummy data
            self.print_log("Testing teacher model with dummy data...")
            acc_frames = self.arg.teacher_args.get('acc_frames', 128)
            acc_coords = self.arg.teacher_args.get('acc_coords', 3)
            num_joints = self.arg.teacher_args.get('num_joints', 32)
            in_chans = self.arg.teacher_args.get('in_chans', 3)
            
            dummy_input = {
                'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32),
                'skeleton': tf.zeros((2, acc_frames, num_joints, in_chans), dtype=tf.float32)
            }
            
            _ = self.teacher_model(dummy_input, training=False)
            self.print_log("Teacher model loaded successfully")
            
            # Freeze teacher model
            self.teacher_model.trainable = False
            
            # Get teacher parameter count
            num_params = self.count_parameters(self.teacher_model)
            self.print_log(f"Teacher model parameters: {num_params:,}")
            
        except Exception as e:
            self.print_log(f"Error loading teacher model: {e}")
            traceback.print_exc()
            raise
    
    def load_distillation_loss(self):
        """Initialize distillation loss function"""
        try:
            if not hasattr(self, 'pos_weights') or self.pos_weights is None:
                self.pos_weights = tf.constant(1.0)
            
            # Get distillation parameters
            temperature = getattr(self.arg, 'temperature', 4.5)
            alpha = getattr(self.arg, 'alpha', 0.6)
            
            # If distiller_args are provided, use those
            if hasattr(self.arg, 'distiller_args'):
                temperature = self.arg.distiller_args.get('temperature', temperature)
                alpha = self.arg.distiller_args.get('alpha', alpha)
            
            self.distillation_loss = DistillationLoss(
                temperature=temperature, 
                alpha=alpha, 
                pos_weight=self.pos_weights
            )
            
            self.print_log(f"Distillation loss initialized with temperature={temperature}, alpha={alpha}")
            return True
        except Exception as e:
            self.print_log(f"Error loading distillation loss: {e}")
            return False
    
    def viz_feature(self, teacher_features, student_features, epoch, max_samples=8):
        """Visualize feature distributions for debugging (similar to PyTorch implementation)"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create directory if not exists
            viz_dir = os.path.join(self.arg.work_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Prepare features
            if isinstance(teacher_features, tf.Tensor):
                teacher_features = teacher_features.numpy()
            if isinstance(student_features, tf.Tensor):
                student_features = student_features.numpy()
            
            # Flatten features if needed
            if len(teacher_features.shape) > 2:
                teacher_features = teacher_features.reshape(teacher_features.shape[0], -1)
            if len(student_features.shape) > 2:
                student_features = student_features.reshape(student_features.shape[0], -1)
            
            # Limit samples for visualization
            num_samples = min(max_samples, teacher_features.shape[0])
            teacher_features = teacher_features[:num_samples]
            student_features = student_features[:num_samples]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            for i in range(num_samples):
                plt.subplot(2, 4, i+1)
                sns.kdeplot(teacher_features[i, :], bw_adjust=0.5, color='blue', label='Teacher')
                sns.kdeplot(student_features[i, :], bw_adjust=0.5, color='red', label='Student')
                if i == 0:
                    plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, f'Feature_KDE_{epoch}.png'))
            plt.close()
        except Exception as e:
            self.print_log(f"Error visualizing features: {e}")
    
    def train(self, epoch):
        """Train student model with knowledge distillation from teacher"""
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
                    
                    # Process through teacher model (no gradients)
                    with tf.GradientTape() as tape:
                        # Get teacher outputs with no gradient tracking
                        @tf.function(experimental_relax_shapes=True)
                        def get_teacher_outputs(inputs):
                            return self.teacher_model(inputs, training=False)
                        
                        teacher_logits, teacher_features = get_teacher_outputs(inputs)
                        
                        # Get student outputs
                        student_logits, student_features = self.model(inputs, training=True)
                        
                        # Apply distillation loss
                        loss = self.distillation_loss(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits, 
                            labels=targets,
                            teacher_features=teacher_features, 
                            student_features=student_features, 
                            training=True
                        )
                    
                    # Visualize features occasionally
                    if epoch % 10 == 0 and batch_idx == 0:
                        self.viz_feature(
                            teacher_features=teacher_features,
                            student_features=student_features,
                            epoch=epoch
                        )
                    
                    # Calculate gradients and update student model
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
                    
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Calculate predictions
                    if len(student_logits.shape) > 1 and student_logits.shape[-1] > 1:
                        probabilities = tf.nn.softmax(student_logits)[:, 1]
                        predictions = tf.argmax(student_logits, axis=-1)
                    else:
                        logits_squeezed = tf.squeeze(student_logits)
                        probabilities = tf.sigmoid(logits_squeezed)
                        predictions = tf.cast(probabilities > 0.5, tf.int32)
                    
                    # Track metrics
                    train_loss += loss.numpy()
                    all_labels.extend(targets.numpy().flatten())
                    all_preds.extend(predictions.numpy().flatten())
                    all_probs.extend(probabilities.numpy().flatten())
                    steps += 1
                    
                except Exception as e:
                    self.print_log(f"Error in distillation batch {batch_idx}: {e}")
                    continue
            
            if steps > 0:
                train_loss /= steps
                accuracy, f1, recall, precision, auc_score = self.calculate_metrics(
                    all_labels, all_preds, all_probs
                )
                
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
                
                self.print_log(f"Running validation for epoch {epoch+1}")
                val_loss = self.eval(epoch, loader_name='val')
                
                self.val_loss_summary.append(float(val_loss))
                
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
        """Main execution method"""
        try:
            if self.arg.phase == 'distill':
                self.print_log("Starting knowledge distillation with parameters:")
                for key, value in vars(self.arg).items():
                    if key not in ['teacher_args', 'student_args', 'model_args', 'dataset_args']:
                        self.print_log(f"  {key}: {value}")
                
                results = []
                
                # Loop through test subjects for cross-validation
                for i, test_subject in enumerate(self.test_eligible_subjects):
                    # Reset training state
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    self.data_loader = {}
                    
                    # Set up subjects for this fold
                    self.test_subject = [test_subject]
                    self.val_subject = self.val_subjects_fixed  # Always [38, 46]
                    
                    # Training subjects: all eligible except current test + fixed training subjects
                    self.train_subjects = [s for s in self.test_eligible_subjects if s != test_subject]
                    self.train_subjects.extend(self.train_subjects_fixed)  # Add [45, 36, 29]
                    
                    self.print_log(f"\n=== Cross-validation fold {i+1}: Testing on subject {test_subject} ===")
                    self.print_log(f"Train: {len(self.train_subjects)} subjects: {self.train_subjects}")
                    self.print_log(f"Val: {len(self.val_subject)} subjects: {self.val_subject}")
                    self.print_log(f"Test: Subject {test_subject}")
                    
                    # Reload models for each fold
                    tf.keras.backend.clear_session()
                    self.load_teacher()  # Load teacher with subject-specific weights if available
                    self.model = self.load_model(self.arg.model, self.arg.model_args)
                    
                    # Load data and initialize optimizer
                    if not self.load_data():
                        self.print_log(f"Skipping subject {test_subject} due to data loading issues")
                        continue
                    
                    self.load_optimizer()
                    self.load_distillation_loss()
                    self.early_stop.reset()
                    
                    # Train for specified epochs
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
                    
                    # Evaluate on test set
                    self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                    result = self.evaluate_test_set()
                    
                    # Visualize loss
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
                    
                    # Clean up
                    self.data_loader = {}
                    tf.keras.backend.clear_session()
                
                # Summarize results
                if results:
                    results = self.add_avg_df(results)
                    
                    # Save as CSV and JSON
                    import pandas as pd
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(self.arg.work_dir, 'distillation_scores.csv'), index=False)
                    
                    with open(os.path.join(self.arg.work_dir, 'distillation_scores.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Knowledge Distillation for Fall Detection')
    
    # Basic arguments
    parser.add_argument('--config', default='config/smartfallmm/distill.yaml',
                        help='Path to configuration file')
    parser.add_argument('--work-dir', type=str, default='../experiments/distill',
                        help='Working directory for outputs')
    parser.add_argument('--model-saved-name', type=str, default='student_model',
                        help='Base name for saving model')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--phase', type=str, default='distill',
                        choices=['distill'], help='Distillation phase')
    
    # Model arguments
    parser.add_argument('--teacher-model', type=str, default=None, 
                        help='Teacher model class path')
    parser.add_argument('--teacher-args', type=str, default=None, 
                        help='Teacher model arguments')
    parser.add_argument('--teacher-weight', type=str, default=None,
                        help='Path to teacher model weights')
    parser.add_argument('--model', type=str, default=None, 
                        help='Student model class path')
    parser.add_argument('--model-args', type=str, default=None, 
                        help='Student model arguments')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=4.5,
                        help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Weight between distillation and hard loss')
    parser.add_argument('--distiller-args', type=str, default=None,
                        help='Distillation arguments')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help='Testing batch size')
    parser.add_argument('--val-batch-size', type=int, default=16,
                        help='Validation batch size')
    parser.add_argument('--num-epoch', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch number')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--base-lr', type=float, default=0.001,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004,
                        help='Weight decay factor')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset to use')
    parser.add_argument('--dataset-args', type=str, default=None,
                        help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject IDs to use')
    parser.add_argument('--feeder', type=str, default=None,
                        help='Data feeder class path')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed for reproducibility')
    parser.add_argument('--result-file', type=str, default=None,
                        help='File to save testing results')
    parser.add_argument('--print-log', type=bool, default=True,
                        help='Whether to print logs')
    
    return parser

def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load configuration from YAML file
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                
                # Update arguments with values from config
                for k, v in config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
            except yaml.YAMLError as e:
                logger.error(f"Error loading config file: {e}")
                exit(1)
    
    # Configure TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging noise
    
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Using GPU(s): {[d.name for d in physical_devices]}")
        else:
            logger.info("No GPU found, using CPU")
    except Exception as e:
        logger.warning(f"GPU configuration error: {e}")
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Initialize distiller
    distiller = Distiller(args)
    
    # Start distillation
    distiller.start()

if __name__ == "__main__":
    main()
