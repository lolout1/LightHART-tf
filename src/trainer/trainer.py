# src/trainer/trainer.py
import tensorflow as tf
import os
import logging
import time
import numpy as np
from ..utils.metrics import calculate_metrics
from ..utils.tflite_converter import convert_to_tflite

class FallDetectionTrainer:
    """Trainer class with TFLite export support"""
    
    def __init__(self, model, optimizer, loss_fn, train_data, val_data, test_data=None,
                work_dir='./experiments', model_name='model',
                patience=15, pos_weight=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.work_dir = work_dir
        self.model_name = model_name
        self.patience = patience
        self.pos_weight = pos_weight
        
        # Create directories
        self.model_dir = os.path.join(work_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.wait = 0
        self.early_stop = False
        self.train_history = []
        self.val_history = []
    
    @tf.function
    def train_step(self, x, y):
        """Single training step with gradient updates"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits, _ = self.model(x, training=True)
            
            # Compute loss
            if self.pos_weight is not None:
                loss = self.loss_fn(y, logits, pos_weight=tf.constant(self.pos_weight))
            else:
                loss = self.loss_fn(y, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, logits
    
    @tf.function
    def test_step(self, x, y):
        """Single evaluation step"""
        # Forward pass
        logits, _ = self.model(x, training=False)
        
        # Compute loss
        loss = self.loss_fn(y, logits)
        
        return loss, logits
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        start_time = time.time()
        
        # Initialize metrics
        loss_sum = 0.0
        all_preds = []
        all_labels = []
        steps = 0
        
        # Create progress tracking
        logging.info(f"Epoch {epoch+1} training:")
        
        # Iterate through batches
        for step, (x, y, _) in enumerate(self.train_data):
            # Train step
            loss, logits = self.train_step(x, y)
            
            # Update metrics
            loss_sum += loss.numpy()
            all_preds.append(logits.numpy())
            all_labels.append(y.numpy())
            steps += 1
            
            # Log progress every 10 steps
            if step % 10 == 0:
                logging.info(f"  Step {step}, Loss: {loss.numpy():.4f}")
        
        # Calculate average loss
        avg_loss = loss_sum / steps
        
        # Calculate metrics
        try:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            metrics = calculate_metrics(all_labels, all_preds)
        except:
            logging.error("Error calculating training metrics")
            metrics = {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': 50.0
            }
        
        # Log results
        duration = time.time() - start_time
        logging.info(f"Train Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                    f"Acc={metrics['accuracy']:.2f}%, "
                    f"F1={metrics['f1']:.2f}%, "
                    f"Prec={metrics['precision']:.2f}%, "
                    f"Rec={metrics['recall']:.2f}%, "
                    f"AUC={metrics['auc']:.2f}% "
                    f"({duration:.2f}s)")
        
        # Store history
        self.train_history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'metrics': metrics
        })
        
        return avg_loss, metrics
    
    def evaluate(self, dataset, name="val"):
        """Evaluate model on dataset"""
        start_time = time.time()
        
        # Initialize metrics
        loss_sum = 0.0
        all_preds = []
        all_labels = []
        steps = 0
        
        # Iterate through batches
        for step, (x, y, _) in enumerate(dataset):
            # Evaluation step
            loss, logits = self.test_step(x, y)
            
            # Update metrics
            loss_sum += loss.numpy()
            all_preds.append(logits.numpy())
            all_labels.append(y.numpy())
            steps += 1
        
        # Calculate average loss
        avg_loss = loss_sum / steps
        
        # Calculate metrics
        try:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            metrics = calculate_metrics(all_labels, all_preds)
        except:
            logging.error(f"Error calculating {name} metrics")
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
        
        # Store validation history
        if name == "val":
            self.val_history.append({
                'epoch': len(self.train_history),
                'loss': float(avg_loss),
                'metrics': metrics
            })
        
        return avg_loss, metrics
    
    def save_model(self, epoch, subject_id=None):
        """Save model with TFLite export"""
        # Create model save name
        if subject_id is not None:
            save_name = f"{self.model_name}_{subject_id}"
        else:
            save_name = f"{self.model_name}_{epoch}"
        
        # Save model weights
        weights_path = os.path.join(self.model_dir, f"{save_name}.weights.h5")
        self.model.save_weights(weights_path)
        logging.info(f"Model weights saved to {weights_path}")
        
        # Save full model
        model_path = os.path.join(self.model_dir, save_name)
        self.model.save(model_path)
        logging.info(f"Full model saved to {model_path}")
        
        # Export to TFLite
        tflite_path = os.path.join(self.model_dir, f"{save_name}.tflite")
        convert_to_tflite(
            model=self.model,
            save_path=tflite_path,
            input_shape=(1, 128, 3),  # Accelerometer-only input shape
            use_lite_runtime=True
        )
        
        return model_path, tflite_path
    
    def check_early_stopping(self, val_loss, val_metrics):
        """Check for early stopping conditions"""
        improved = False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
            logging.info(f"New best validation loss: {val_loss:.4f}")
        
        # Check if F1 score improved
        if val_metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = val_metrics['f1']
            improved = True
            logging.info(f"New best validation F1: {val_metrics['f1']:.2f}%")
        
        # Reset or increment patience counter
        if improved:
            self.wait = 0
            return True
        else:
            self.wait += 1
            logging.info(f"Early stopping: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered!")
            
            return False
    
    def train(self, epochs, test_subject=None):
        """Train model for specified number of epochs"""
        logging.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(self.val_data, name="val")
            
            # Check early stopping and save model if improved
            if self.check_early_stopping(val_loss, val_metrics):
                self.best_epoch = epoch
                self.save_model(epoch, subject_id=test_subject)
            
            # Break if early stopping triggered
            if self.early_stop:
                logging.info(f"Early stopping at epoch {epoch+1}/{epochs}")
                break
        
        # Evaluate on test set
        if self.test_data is not None:
            logging.info(f"Evaluating best model (epoch {self.best_epoch+1}) on test set")
            test_loss, test_metrics = self.evaluate(self.test_data, name=f"test_{test_subject}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1
        }
