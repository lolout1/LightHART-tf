#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Trainer for LightHART-TF 

Core trainer module with modular architecture
"""
import os
import logging
import json
import traceback
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd

# Import utility modules
from utils.model_utils import load_model, count_parameters, save_model
from utils.metrics import calculate_metrics, add_avg_df, calculate_class_weights
from utils.visualization import plot_distribution, plot_loss_curves, plot_confusion_matrix
from trainer.training_loop import train_epoch, evaluate_model
from trainer.evaluation import evaluate_test_set

logger = logging.getLogger('lightheart-tf')

class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.wait = 0
    
    def __call__(self, val_loss):
        """Check if training should stop"""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.wait = 0
            return False
        
        self.counter += 1
        self.wait += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.wait = 0
        self.best_loss = None
        self.early_stop = False

class BaseTrainer:
    """Base trainer for fall detection models"""
    
    def __init__(self, arg):
        """Initialize trainer with arguments"""
        self.arg = arg
        
        # Initialize metrics and state
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.test_accuracy = 0 
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0 
        self.test_auc = 0
        
        # Initialize dataset splits
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        
        # Initialize data variables
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = {}
        self.pos_weights = None
        
        # Initialize early stopping
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        
        # Setup directories and model paths
        self.setup_directories()
        
        # Import dynamic modules
        self.import_modules()
        
        # Load model
        self.model = self.load_model()
        
        # Report model information
        num_params = count_parameters(self.model)
        self.print_log(f"Model: {self.arg.model}")
        self.print_log(f"Parameters: {num_params:,}")
        self.print_log(f"Model size: {num_params * 4 / (1024**2):.2f} MB")
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        # Ensure work directory has timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if hasattr(self.arg, 'work_dir') and os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{timestamp}"
        
        # Create required directories
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'results'), exist_ok=True)
        
        # Set model path
        self.model_path = os.path.join(
            self.arg.work_dir, 
            'models', 
            self.arg.model_saved_name
        )
    
    def import_modules(self):
        """Import required modules dynamically"""
        try:
            # Import data handling modules
            from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
            self.prepare_dataset = prepare_smartfallmm_tf
            self.split_by_subjects = split_by_subjects_tf
            
            # Import model utilities
            from utils.model_utils import import_class
            self.import_class = import_class
            
        except ImportError as e:
            self.print_log(f"Error importing modules: {e}")
            raise
    
    def print_log(self, message):
        """Log message to console and file"""
        logger.info(message)
        
        if hasattr(self.arg, 'print_log') and self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(message, file=f)
    
    def load_model(self):
        """Load and initialize model based on configuration"""
        if self.arg.phase == 'train':
            model = load_model(
                self.arg.model, 
                self.arg.model_args, 
                self.arg.dataset_args,
                self.print_log
            )
        else:
            # For test/inference phase
            if hasattr(self.arg, 'weights') and self.arg.weights:
                try:
                    model = tf.keras.models.load_model(self.arg.weights)
                    self.print_log(f"Loaded model from {self.arg.weights}")
                except Exception:
                    # If loading full model fails, try loading weights
                    model = load_model(
                        self.arg.model, 
                        self.arg.model_args, 
                        self.arg.dataset_args,
                        self.print_log
                    )
                    model.load_weights(self.arg.weights)
                    self.print_log(f"Loaded weights from {self.arg.weights}")
            else:
                model = load_model(
                    self.arg.model, 
                    self.arg.model_args, 
                    self.arg.dataset_args,
                    self.print_log
                )
        
        return model
    
    def load_optimizer(self):
        """Initialize optimizer based on configuration"""
        if not hasattr(self.arg, 'optimizer'):
            self.arg.optimizer = 'adam'
            
        if not hasattr(self.arg, 'base_lr'):
            self.arg.base_lr = 0.001
            
        if not hasattr(self.arg, 'weight_decay'):
            self.arg.weight_decay = 0.0004
        
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.arg.base_lr
            )
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.arg.base_lr,
                momentum=0.9
            )
        else:
            self.print_log(f"Unknown optimizer: {self.arg.optimizer}, using Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
            
        self.print_log(f"Optimizer: {self.optimizer.__class__.__name__}, LR={self.arg.base_lr}")
    
    def load_loss(self):
        """Initialize loss function with class weights"""
        if not hasattr(self, 'pos_weights') or self.pos_weights is None:
            self.pos_weights = tf.constant(1.0)
        
        # Create weighted binary cross entropy loss
        def weighted_bce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            weights = y_true * (self.pos_weights - 1.0) + 1.0
            return tf.reduce_mean(weights * bce)
        
        self.criterion = weighted_bce
        self.print_log(f"Using BCE loss with pos_weight={self.pos_weights.numpy():.4f}")
    
    def load_data(self):
        """Load and prepare datasets"""
        try:
            # Import data feeder
            feeder_class_path = getattr(self.arg, 'feeder', 'utils.dataset_tf.UTD_MM_TF')
            Feeder = self.import_class(feeder_class_path)
            
            if self.arg.phase == 'train':
                # Create dataset builder
                builder = self.prepare_dataset(self.arg)
                
                # Check if we have valid subject lists
                if not self.train_subjects:
                    self.print_log("No training subjects specified")
                    return False
                
                # Prepare training data
                self.print_log(f"Processing training data for subjects: {self.train_subjects}")
                self.norm_train = self.split_by_subjects(builder, self.train_subjects, False)
                
                if any(len(x) == 0 for x in self.norm_train.values()):
                    self.print_log("Error: Training data is empty")
                    return False
                    
                # Create training data loader
                self.data_loader['train'] = Feeder(
                    dataset=self.norm_train,
                    batch_size=self.arg.batch_size
                )
                
                # Calculate class weights
                self.pos_weights = calculate_class_weights(self.norm_train['labels'], self.print_log)
                
                # Visualize training data distribution
                plot_distribution(
                    self.norm_train['labels'], 
                    self.arg.work_dir, 
                    'train',
                    self.print_log
                )
                
                # Prepare validation data if available
                if self.val_subject:
                    self.print_log(f"Processing validation data for subjects: {self.val_subject}")
                    self.norm_val = self.split_by_subjects(builder, self.val_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_val.values()):
                        self.print_log("Warning: Validation data is empty, using subset of training data")
                        # Use a subset of training data for validation
                        train_size = len(self.norm_train['labels'])
                        val_size = min(train_size // 5, 100)  # 20% or max 100 samples
                        
                        self.norm_val = {
                            k: v[-val_size:].copy() for k, v in self.norm_train.items()
                        }
                        self.norm_train = {
                            k: v[:-val_size].copy() for k, v in self.norm_train.items()
                        }
                    
                    # Create validation data loader
                    self.data_loader['val'] = Feeder(
                        dataset=self.norm_val,
                        batch_size=self.arg.val_batch_size
                    )
                    
                    # Visualize validation data distribution
                    plot_distribution(
                        self.norm_val['labels'], 
                        self.arg.work_dir, 
                        'val',
                        self.print_log
                    )
                
                # Prepare test data if available
                if self.test_subject:
                    self.print_log(f"Processing test data for subjects: {self.test_subject}")
                    self.norm_test = self.split_by_subjects(builder, self.test_subject, False)
                    
                    if any(len(x) == 0 for x in self.norm_test.values()):
                        self.print_log("Warning: Test data is empty")
                        return False
                        
                    # Create test data loader
                    self.data_loader['test'] = Feeder(
                        dataset=self.norm_test,
                        batch_size=self.arg.test_batch_size
                    )
                    
                    # Visualize test data distribution
                    subject_id = self.test_subject[0] if self.test_subject else 'unknown'
                    plot_distribution(
                        self.norm_test['labels'], 
                        self.arg.work_dir, 
                        f'test_{subject_id}',
                        self.print_log
                    )
                
                self.print_log("Data loading complete")
                return True
            elif self.arg.phase == 'test':
                # For test-only mode
                if not self.test_subject:
                    self.print_log("No test subjects specified")
                    return False
                
                builder = self.prepare_dataset(self.arg)
                self.norm_test = self.split_by_subjects(builder, self.test_subject, False)
                
                if any(len(x) == 0 for x in self.norm_test.values()):
                    self.print_log("Error: Test data is empty")
                    return False
                
                self.data_loader['test'] = Feeder(
                    dataset=self.norm_test,
                    batch_size=self.arg.test_batch_size
                )
                
                self.print_log("Test data loading complete")
                return True
                
        except Exception as e:
            self.print_log(f"Error loading data: {e}")
            traceback.print_exc()
            return False
    
    def train(self, epoch):
        """Train model for one epoch"""
        # Get training results
        train_loss, train_metrics, all_labels, all_preds = train_epoch(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            data_loader=self.data_loader['train'],
            epoch=epoch,
            num_epochs=self.arg.num_epoch,
            logger=self.print_log
        )
        
        # Save training loss
        self.train_loss_summary.append(train_loss)
        
        # Validate model
        val_loss, val_metrics = evaluate_model(
            model=self.model,
            criterion=self.criterion,
            data_loader=self.data_loader['val'],
            epoch=epoch,
            name='val',
            logger=self.print_log
        )
        
        # Save validation loss
        self.val_loss_summary.append(val_loss)
        
        # Check if this is the best model
        is_best = False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            is_best = True
            self.print_log(f"New best validation loss: {val_loss:.4f}")
        
        # Save best model
        if is_best:
            subject_id = self.test_subject[0] if self.test_subject else None
            save_model(
                model=self.model,
                base_filename=f"{self.model_path}_{subject_id}" if subject_id else f"{self.model_path}_epoch{epoch}",
                logger=self.print_log
            )
        
        # Check early stopping
        self.early_stop(val_loss)
        
        return train_loss, val_loss
    
    def start(self):
        """Main execution method for training or testing"""
        if self.arg.phase == 'train':
            # Log training parameters
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            
            # Create results list
            results = []
            
            # Define validation subjects
            val_subjects = [38, 46]  # Default validation subjects
            
            # Process each subject in leave-one-out cross-validation
            for test_subject in self.arg.subjects:
                # Skip validation subjects
                if test_subject in val_subjects:
                    continue
                
                # Reset metrics for this fold
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                # Define train/val/test split
                self.test_subject = [test_subject]
                self.val_subject = val_subjects
                self.train_subjects = [s for s in self.arg.subjects 
                                       if s != test_subject and s not in val_subjects]
                
                self.print_log(f"\n=== Cross-validation fold: Testing on subject {test_subject} ===")
                self.print_log(f"Train: {len(self.train_subjects)} subjects")
                self.print_log(f"Val: {len(self.val_subject)} subjects")
                self.print_log(f"Test: Subject {test_subject}")
                
                # Create new model instance
                self.model = self.load_model()
                
                # Load data
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data issues")
                    continue
                
                # Initialize optimizer and loss
                self.load_optimizer()
                self.load_loss()
                
                # Reset early stopping
                self.early_stop.reset()
                
                # Train for specified epochs
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    
                    # Check early stopping
                    if self.early_stop.early_stop:
                        self.print_log(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Load best model for testing
                best_weights = f"{self.model_path}_{test_subject}.weights.h5"
                if os.path.exists(best_weights):
                    self.model.load_weights(best_weights)
                    self.print_log(f"Loaded best weights from {best_weights}")
                
                # Final evaluation on test set
                self.print_log(f"=== Final evaluation on subject {test_subject} ===")
                
                test_metrics = evaluate_test_set(
                    model=self.model,
                    criterion=self.criterion,
                    data_loader=self.data_loader['test'],
                    subject_id=test_subject,
                    work_dir=self.arg.work_dir,
                    logger=self.print_log
                )
                
                # Store metrics
                self.test_accuracy = test_metrics['accuracy'] 
                self.test_f1 = test_metrics['f1']
                self.test_precision = test_metrics['precision']
                self.test_recall = test_metrics['recall']
                self.test_auc = test_metrics['auc']
                
                # Visualize loss curves
                plot_loss_curves(
                    self.train_loss_summary, 
                    self.val_loss_summary,
                    self.arg.work_dir,
                    test_subject,
                    self.print_log
                )
                
                # Store results
                subject_result = {
                    'test_subject': str(test_subject),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }
                
                results.append(subject_result)
                
                # Clear memory
                tf.keras.backend.clear_session()
            
            # Calculate and save average results
            if results:
                # Add average row
                results = add_avg_df(results)
                
                # Save results as CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(self.arg.work_dir, 'scores.csv'), index=False)
                
                # Save as JSON
                with open(os.path.join(self.arg.work_dir, 'scores.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Log final results
                self.print_log("\n=== Final Results ===")
                for result in results:
                    subject = result['test_subject']
                    accuracy = result['accuracy']
                    f1 = result['f1_score']
                    precision = result['precision']
                    recall = result['recall']
                    auc = result.get('auc', 'N/A')
                    
                    self.print_log(
                        f"Subject {subject}: "
                        f"Acc={accuracy:.2f}%, "
                        f"F1={f1:.2f}%, "
                        f"Prec={precision:.2f}%, "
                        f"Rec={recall:.2f}%, "
                        f"AUC={auc}"
                    )
            
            self.print_log("Training completed successfully")
            
        elif self.arg.phase == 'test':
            # Testing only mode
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for testing")
                return
            
            # Set up subject for testing
            if not hasattr(self, 'test_subject') or not self.test_subject:
                if hasattr(self.arg, 'subjects') and self.arg.subjects:
                    self.test_subject = [self.arg.subjects[0]]
                else:
                    self.print_log("No test subject specified")
                    return
            
            # Load data
            if not self.load_data():
                self.print_log("Failed to load test data")
                return
            
            # Initialize loss function
            self.load_loss()
            
            # Evaluate on test data
            self.print_log(f"Testing on subject {self.test_subject[0]}")
            
            test_metrics = evaluate_test_set(
                model=self.model,
                criterion=self.criterion,
                data_loader=self.data_loader['test'],
                subject_id=self.test_subject[0],
                work_dir=self.arg.work_dir,
                logger=self.print_log
            )
            
            # Store metrics
            self.test_accuracy = test_metrics['accuracy']
            self.test_f1 = test_metrics['f1']
            self.test_precision = test_metrics['precision']
            self.test_recall = test_metrics['recall']
            self.test_auc = test_metrics['auc']
            
            # Log results
            self.print_log(
                f"Test results: "
                f"Acc={self.test_accuracy:.2f}%, "
                f"F1={self.test_f1:.2f}%, "
                f"Prec={self.test_precision:.2f}%, "
                f"Rec={self.test_recall:.2f}%, "
                f"AUC={self.test_auc:.2f}%"
            )
            
        elif self.arg.phase == 'tflite':
            # TFLite export mode
            if not hasattr(self.arg, 'weights') or not self.arg.weights:
                self.print_log("No weights specified for TFLite export")
                return
            
            # Try to export TFLite model
            try:
                from utils.tflite_converter import convert_to_tflite
                
                # Generate output path
                if hasattr(self.arg, 'result_file') and self.arg.result_file:
                    tflite_path = self.arg.result_file
                else:
                    tflite_path = os.path.join(self.arg.work_dir, 'model.tflite')
                
                # Export model
                success = convert_to_tflite(
                    model=self.model,
                    save_path=tflite_path,
                    input_shape=(1, 128, 3),  # Default shape for accelerometer data
                    quantize=True
                )
                
                if success:
                    self.print_log(f"Successfully exported TFLite model to {tflite_path}")
                else:
                    self.print_log("Failed to export TFLite model")
                    
            except Exception as e:
                self.print_log(f"Error exporting TFLite model: {e}")
                traceback.print_exc()
        
        else:
            self.print_log(f"Unknown phase: {self.arg.phase}")
