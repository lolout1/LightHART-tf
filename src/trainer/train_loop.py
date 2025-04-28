# src/trainer/train_loop.py
import tensorflow as tf
import numpy as np
import logging
import os
import time
from ..utils.metrics import calculate_metrics
from ..utils.tflite_converter import convert_to_tflite

class FallDetectionTrainer:
    """Trainer for fall detection models with TFLite export"""
    
    def __init__(self, model, optimizer, loss_fn, train_data, val_data, test_data=None,
                patience=15, work_dir='./experiments', model_name='model',
                class_weights=None):
        """Initialize the trainer
        
        Args:
            model: TensorFlow model
            optimizer: TensorFlow optimizer
            loss_fn: Loss function
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Testing dataset (optional)
            patience: Patience for early stopping
            work_dir: Working directory for saving models
            model_name: Name for saved models
            class_weights: Class weights for imbalanced data
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.patience = patience
        self.work_dir = work_dir
        self.model_name = model_name
        self.class_weights = class_weights
        
        # Create directories
        self.model_dir = os.path.join(work_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    @tf.function
    def train_step(self, x, y):
        """Single training step with gradient updates"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits, _ = self.model(x, training=True)
            
            # Compute loss (handle multi-dimensional logits)
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                logits = tf.squeeze(logits)
            
            # Apply class weights if provided
            if self.class_weights is not None:
                sample_weights = tf.gather(
                    tf.constant(self.class_weights, dtype=tf.float32),
                    tf.cast(y, tf.int32)
                )
                loss = self.loss_fn(y, logits, sample_weight=sample_weights)
            else:
                loss = self.loss_fn(y, logits)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, logits
    
    @tf.function
    def test_step(self, x, y):
        """Single evaluation step"""
        # Forward pass
        logits, _ = self.model(x, training=False)
        
        # Compute loss (handle multi-dimensional logits)
        if len(logits.shape) > 1 and logits.shape[1] > 1:
            logits = tf.squeeze(logits)
        
        loss = self.loss_fn(y, logits)
        
        return loss, logits
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        # Initialize metrics
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Get total steps
        total_steps = tf.data.experimental.cardinality(self.train_data).numpy()
        if total_steps < 0:  # If cardinality is unknown
            total_steps = "?"
        
        # Progress logging
        start_time = time.time()
        logging.info(f"Epoch {epoch+1} training:")
        
        # Iterate over batches
        for step, (x, y) in enumerate(self.train_data):
            # Perform training step
            loss, logits = self.train_step(x, y)
            
            # Update metrics
            epoch_loss += loss
            all_preds.append(logits.numpy())
            all_labels.append(y.numpy())
            
            # Log progress
            if step % 10 == 0:
                logging.info(f"  Step {step}/{total_steps}, Loss: {loss:.4f}")
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / (step + 1)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = calculate_metrics(all_labels, all_preds)
        
        # Log results
        duration = time.time() - start_time
        logging.info(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, "
                    f"Accuracy: {metrics['accuracy']:.2f}%, "
                    f"F1: {metrics['f1']:.2f}%, "
                    f"Precision: {metrics['precision']:.2f}%, "
                    f"Recall: {metrics['recall']:.2f}%, "
                    f"AUC: {metrics['auc']:.2f}% "
                    f"({duration:.2f}s)")
        
        # Update history
        self.history['train_loss'].append(float(epoch_loss))
        self.history['train_metrics'].append(metrics)
        
        return epoch_loss, metrics
    
    def evaluate(self, dataset, name="val"):
        """Evaluate model on dataset"""
        # Initialize metrics
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Get total steps
        total_steps = tf.data.experimental.cardinality(dataset).numpy()
        if total_steps < 0:  # If cardinality is unknown
            total_steps = "?"
        
        # Progress logging
        start_time = time.time()
        logging.info(f"Evaluating on {name} set:")
        
        # Iterate over batches
        for step, (x, y) in enumerate(dataset):
            # Perform evaluation step
            loss, logits = self.test_step(x, y)
            
            # Update metrics
            total_loss += loss
            all_preds.append(logits.numpy())
            all_labels.append(y.numpy())
        
        # Calculate metrics
        avg_loss = total_loss / (step + 1)
        
        # Handle empty dataset case
        if not all_preds:
            logging.warning(f"No predictions made on {name} set")
            return avg_loss, {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': 50.0
            }
        
        # Concatenate all predictions and labels
        try:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            metrics = calculate_metrics(all_labels, all_preds)
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            metrics = {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': 50.0
            }
        
        # Log results
        duration = time.time() - start_time
        logging.info(f"{name.capitalize()}: Loss={avg_loss:.4f}, "
                    f"Acc={metrics['accuracy']:.2f}%, "
                    f"F1={metrics['f1']:.2f}%, "
                    f"Prec={metrics['precision']:.2f}%, "
                    f"Rec={metrics['recall']:.2f}%, "
                    f"AUC={metrics['auc']:.2f}% "
                    f"({duration:.2f}s)")
        
        return avg_loss, metrics
    
    def save_model(self, epoch, subject_id=None):
        """Save model weights and export to TFLite"""
        # Build model name
        if subject_id is not None:
            save_name = f"{self.model_name}_{subject_id}"
        else:
            save_name = f"{self.model_name}_{epoch}"
        
        # Save model weights
        weights_path = os.path.join(self.model_dir, f"{save_name}.weights.h5")
        self.model.save_weights(weights_path)
        logging.info(f"Model weights saved to {weights_path}")
        
        # Export to TFLite
        tflite_path = os.path.join(self.model_dir, f"{save_name}.tflite")
        success = convert_to_tflite(self.model, tflite_path)
        
        # Also save full model
        model_path = os.path.join(self.model_dir, save_name)
        self.model.save(model_path)
        logging.info(f"Full model saved to {model_path}")
        
        if success:
            logging.info("TFLite conversion successful!")
        else:
            logging.warning("TFLite conversion failed")
        
        return weights_path, tflite_path
    
    def check_early_stopping(self, val_loss, metrics, epoch):
        """Check for early stopping and save best model"""
        f1_score = metrics['f1']
        improved = False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
            logging.info(f"Validation loss improved to {val_loss:.4f}")
        
        # Check if F1 score improved
        if f1_score > self.best_val_f1:
            self.best_val_f1 = f1_score
            improved = True
            logging.info(f"Validation F1 score improved to {f1_score:.2f}%")
        
        # Reset counter if improved
        if improved:
            self.wait = 0
            self.best_epoch = epoch
            return self.save_model(epoch)
        
        # Increment counter if not improved
        self.wait += 1
        if self.wait >= self.patience:
            self.early_stop = True
            self.stopped_epoch = epoch
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
        
        return None, None
    
    def train(self, epochs, test_subject=None):
        """Train the model for specified number of epochs"""
        logging.info(f"Starting training for {epochs} epochs")
        
        # Training loop
        for epoch in range(epochs):
            # Train one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.evaluate(self.val_data, name="val")
            
            # Update history
            self.history['val_loss'].append(float(val_loss))
            self.history['val_metrics'].append(val_metrics)
            
            # Check early stopping and save best model
            weights_path, tflite_path = self.check_early_stopping(val_loss, val_metrics, epoch)
            
            # Stop if early stopping triggered
            if self.early_stop:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        if self.test_data is not None:
            logging.info("Evaluating on test set...")
            test_loss, test_metrics = self.evaluate(self.test_data, name="test")
            
            # Save model with test subject ID
            if test_subject is not None:
                self.save_model(self.best_epoch, subject_id=test_subject)
        
        return self.history
