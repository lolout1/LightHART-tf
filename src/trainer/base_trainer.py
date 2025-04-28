"""
Base Trainer Module for LightHART-tf
Implements core training functionality
"""
import os
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
import importlib
import json

class BaseTrainer:
    """
    Base trainer class with core functionality
    To be extended by specific trainers
    """
    def __init__(self, args):
        """Initialize base trainer with configuration"""
        self.args = args
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self):
        """Set up experiment directories"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if hasattr(self.args, 'work_dir') and os.path.exists(self.args.work_dir):
            self.args.work_dir = f"{self.args.work_dir}_{timestamp}"
        
        # Create required directories
        os.makedirs(self.args.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.args.work_dir, 'visualizations'), exist_ok=True)
        
        # Store model path
        self.model_path = os.path.join(
            self.args.work_dir, 
            'models', 
            self.args.model_saved_name
        )

    def setup_logging(self):
        """Configure logging"""
        log_file = os.path.join(self.args.work_dir, 'training.log')
        
        # Configure root logger
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LightHART-tf')

    def log(self, message):
        """Log a message"""
        self.logger.info(message)
        
        # Also write to separate log file if needed
        if hasattr(self.args, 'print_log') and self.args.print_log:
            log_path = os.path.join(self.args.work_dir, 'log.txt')
            with open(log_path, 'a') as f:
                print(message, file=f)

    def import_class(self, class_path):
        """Dynamically import a class from string path"""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.log(f"Error importing {class_path}: {e}")
            raise

    def save_config(self):
        """Save configuration to the experiment directory"""
        config_path = os.path.join(self.args.work_dir, 'config.json')
        with open(config_path, 'w') as f:
            # Convert args to dict and handle non-serializable types
            config_dict = {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) 
                          else v for k, v in vars(self.args).items()}
            json.dump(config_dict, f, indent=2)
        self.log(f"Configuration saved to {config_path}")

    def load_model(self):
        """Load model based on configuration"""
        raise NotImplementedError("Subclasses must implement load_model()")

    def load_data(self):
        """Load and prepare datasets"""
        raise NotImplementedError("Subclasses must implement load_data()")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        raise NotImplementedError("Subclasses must implement train_epoch()")

    def validate(self, epoch):
        """Validate model"""
        raise NotImplementedError("Subclasses must implement validate()")

    def test(self):
        """Test model"""
        raise NotImplementedError("Subclasses must implement test()")

    def save_model(self, identifier=None):
        """Save model checkpoint"""
        raise NotImplementedError("Subclasses must implement save_model()")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        raise NotImplementedError("Subclasses must implement load_checkpoint()")

    def fit(self):
        """Execute full training procedure"""
        self.log("Beginning training")
        
        # Train for the specified number of epochs
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.log(f"Epoch {epoch+1}/{self.args.num_epoch}")
            
            # Train and validate for one epoch
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(epoch)
            
            # Log progress
            self.log(f"Epoch {epoch+1} - Train: {train_stats}, Val: {val_stats}")
            
            # Check for early stopping
            if hasattr(self, 'early_stop') and self.early_stop(val_stats['loss']):
                self.log(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        self.log("Training completed")
    
    def run(self):
        """Main execution method"""
        if self.args.phase == 'train':
            self.fit()
        elif self.args.phase == 'test':
            self.test()
        else:
            self.log(f"Unknown phase: {self.args.phase}")
