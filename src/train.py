#!/usr/bin/env python3
"""
Main training script for LightHART-tf
Initializes the training process and executes the base_trainer
"""
import os
import argparse
import sys
import yaml
import json
import logging
import tensorflow as tf
import numpy as np
import random
from datetime import datetime

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def str2bool(v):
    """Convert string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def init_seed(seed):
    """Initialize random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Set deterministic operations if available
    if hasattr(tf, 'config'):
        try:
            tf.config.experimental.enable_op_determinism()
        except:
            pass

def import_class(import_str):
    """Import class dynamically from string"""
    import importlib
    mod_str, _sep, class_str = import_str.rpartition('.')
    try:
        mod = importlib.import_module(mod_str)
        return getattr(mod, class_str)
    except (ImportError, AttributeError) as e:
        logging.error(f"Error importing {import_str}: {e}")
        raise ImportError(f'Class {class_str} cannot be found ({e})')

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fall Detection Training')
    
    # Configuration
    parser.add_argument('--config', default='./config/smartfallmm/student.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset name')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        help='Testing batch size')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        help='Validation batch size')
    parser.add_argument('--num-epoch', type=int, default=70,
                        help='Number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use')
    parser.add_argument('--base-lr', type=float, default=0.001,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--model', default=None,
                        help='Model class name')
    parser.add_argument('--device', default='0',
                        help='GPU device ID(s)')
    parser.add_argument('--model-args', default=None, type=str,
                        help='Model initialization arguments')
    parser.add_argument('--weights', type=str,
                        help='Path to pre-trained weights')
    parser.add_argument('--model-saved-name', type=str, default='model',
                        help='Name for saving model')
    
    # Loss parameters
    parser.add_argument('--loss', default='bce',
                        help='Loss function to use')
    parser.add_argument('--loss-args', default='{}', type=str,
                        help='Loss function arguments')
    
    # Dataset parameters
    parser.add_argument('--dataset-args', default=None, type=str,
                        help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int,
                        help='Subject IDs to use')
    parser.add_argument('--feeder', default=None,
                        help='Data feeder class')
    parser.add_argument('--train-feeder-args', default='{}', type=str,
                        help='Training data feeder arguments')
    parser.add_argument('--val-feeder-args', default='{}', type=str,
                        help='Validation data feeder arguments')
    parser.add_argument('--test-feeder-args', default='{}', type=str,
                        help='Testing data feeder arguments')
    parser.add_argument('--include-val', type=str2bool, default=True,
                        help='Whether to include validation set')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed')
    parser.add_argument('--work-dir', type=str, default='experiments/student',
                        help='Working directory for outputs')
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print logs')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'],
                        help='Training or testing phase')
    parser.add_argument('--num-worker', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--result-file', type=str,
                        help='File to save results')
    
    # TensorFlow-specific parameters
    parser.add_argument('--mixed-precision', type=str2bool, default=False,
                        help='Whether to use mixed precision training')
    
    return parser

def main():
    """Main function"""
    # Parse arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load configuration from YAML if provided
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            
            # Update arguments from yaml
            for k, v in yaml_cfg.items():
                if k in vars(args) and getattr(args, k) is None:
                    setattr(args, k, v)
                    
    # Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Configure dynamic memory growth
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found, using CPU")
    
    # Set seeds for reproducibility
    init_seed(args.seed)
    
    # Create timestamp-based work directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(args.work_dir):
        args.work_dir = f"{args.work_dir}_{timestamp}"
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Save configuration
    with open(f"{args.work_dir}/config.yaml", 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Initialize logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.work_dir}/train.log"),
            logging.StreamHandler()
        ]
    )
    
    # Log startup information
    logging.info(f"Starting {args.phase} with configuration from {args.config}")
    logging.info(f"Working directory: {args.work_dir}")
    logging.info(f"Using TensorFlow {tf.__version__}")
    logging.info(f"GPU devices: {physical_devices}")
    
    # Import trainer class and start training/testing
    try:
        from base_trainer import BaseTrainer
        trainer = BaseTrainer(args)
        trainer.start()
    except ImportError:
        # Try alternate import path
        try:
            sys.path.append(os.path.join(current_dir, 'src/trainer'))
            from base_trainer import BaseTrainer
            trainer = BaseTrainer(args)
            trainer.start()
        except Exception as e:
            logging.error(f"Error starting trainer: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    logging.info("Training/testing completed successfully")
    
if __name__ == "__main__":
    main()
