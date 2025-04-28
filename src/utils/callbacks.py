import tensorflow as tf
import numpy as np
import os
import logging

class EarlyStoppingTF:
    """Enhanced early stopping callback with better monitoring and logging"""
    def __init__(self, patience=15, min_delta=0.00001, verbose=True, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.verbose = verbose
        self.monitor = monitor
        self.mode = mode  # 'min' for metrics like loss, 'max' for metrics like accuracy
        self.wait = 0
        logging.info(f"Early stopping initialized: patience={patience}, min_delta={min_delta}")
    
    def is_improvement(self, current):
        """Check if the current value is an improvement"""
        if self.best_value is None:
            return True
            
        if self.mode == 'min':
            return current < (self.best_value - self.min_delta)
        else:
            return current > (self.best_value + self.min_delta)
    
    def __call__(self, current_value):
        if self.is_improvement(current_value):
            if self.verbose:
                improvement = "" if self.best_value is None else f" (improved from {self.best_value:.6f})"
                print(f"Early stopping: {self.monitor}={current_value:.6f}{improvement}")
            self.best_value = current_value
            self.counter = 0
            self.wait = 0
            return False
        else:
            self.counter += 1
            self.wait += 1
            if self.verbose:
                print(f"Early stopping: {self.monitor}={current_value:.6f} (no improvement, counter: {self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.wait} epochs without improvement")
                return True
            return False
    
    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.wait = 0


class ModelCheckpoint:
    """Model checkpoint callback that saves the best model based on a monitored metric"""
    def __init__(self, model, filepath, monitor='val_loss', mode='min', 
                 save_best_only=True, save_weights_only=True, verbose=True,
                 tflite_export=False, tflite_filepath=None):
        self.model = model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.best_value = None
        self.tflite_export = tflite_export
        self.tflite_filepath = tflite_filepath or f"{os.path.splitext(filepath)[0]}.tflite"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def is_improvement(self, current):
        """Check if the current value is an improvement"""
        if self.best_value is None:
            return True
            
        if self.mode == 'min':
            return current < self.best_value
        else:
            return current > self.best_value
    
    def export_to_tflite(self):
        """Export model to TFLite format"""
        if not self.tflite_export:
            return False
            
        try:
            # Create a clean model instance
            model_class = self.model.__class__
            model_config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
            new_model = model_class(**model_config)
            
            # Create input signature
            input_shape = next(iter(self.model.inputs)).shape
            if input_shape[0] is None:
                input_shape = (1,) + input_shape[1:]
                
            # Load weights
            new_model.build(input_shape)
            new_model.load_weights(self.filepath)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(self.tflite_filepath, 'wb') as f:
                f.write(tflite_model)
                
            if self.verbose:
                print(f"Model successfully exported to TFLite: {self.tflite_filepath}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error exporting to TFLite: {e}")
            return False
    
    def __call__(self, epoch, logs=None):
        current = logs.get(self.monitor) if logs else None
        if current is None:
            print(f"Warning: {self.monitor} not available in logs")
            return False
            
        if self.save_best_only and not self.is_improvement(current):
            return False
            
        self.best_value = current
        
        try:
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
                
            if self.verbose:
                print(f"Model {'weights' if self.save_weights_only else ''} saved to {self.filepath}")
            
            # Export to TFLite if enabled
            if self.tflite_export:
                self.export_to_tflite()
                
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


class LearningRateScheduler:
    """Learning rate scheduler with logging and visualization"""
    def __init__(self, schedule, verbose=True):
        self.schedule = schedule
        self.verbose = verbose
        self.history = []
        
    def __call__(self, epoch, lr):
        new_lr = self.schedule(epoch, lr)
        self.history.append((epoch, new_lr))
        if self.verbose:
            print(f"Epoch {epoch+1}: Learning rate adjusted to {new_lr:.6f}")
        return new_lr
