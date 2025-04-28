# src/utils/callbacks.py
"""Callbacks for early stopping and model checkpointing"""
import numpy as np
import logging

class EarlyStoppingTF:
    """Early stopping callback for TensorFlow"""
    def __init__(self, patience=15, min_delta=0.00001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.wait = 0
    
    def __call__(self, current_value):
        """Check if training should stop"""
        if current_value < (self.best_value - self.min_delta):
            # Improvement found
            self.best_value = current_value
            self.counter = 0
            self.wait = 0
            return False
        else:
            # No improvement
            self.counter += 1
            self.wait += 1
            
            if self.verbose:
                logging.info(f"Early stopping: val_loss={current_value:.6f} "
                            f"(no improvement, counter: {self.counter}/{self.patience})")
                
            if self.counter >= self.patience:
                # Trigger early stopping
                self.early_stop = True
                
                if self.verbose:
                    logging.info(f"Early stopping triggered after {self.wait} epochs "
                                f"without improvement")
                return True
                
            return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.wait = 0
