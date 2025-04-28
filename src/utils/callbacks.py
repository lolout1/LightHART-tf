import tensorflow as tf
import numpy as np

class EarlyStoppingTF:
    def __init__(self, patience=15, min_delta=0.00001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.wait = 0
    
    def __call__(self, current_value):
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
            self.wait = 0
            return False
        else:
            self.counter += 1
            self.wait += 1
            if self.verbose:
                print(f"Early stopping: val_loss={current_value:.6f} (no improvement, counter: {self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.wait} epochs without improvement")
                return True
            return False
    
    def reset(self):
        self.counter = 0
        self.best_value = float('inf')
        self.early_stop = False
        self.wait = 0
