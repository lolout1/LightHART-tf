#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main training script for LightHART-TF

This script provides command line interface for training and evaluating
fall detection models based on the TensorFlow implementation of LightHART.
"""
import os
import argparse
import yaml
import json
import sys
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lightheart-tf')

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
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Set deterministic operations if available
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fall Detection Training')
    
    # Basic arguments
    parser.add_argument('--config', default='./config/smartfallmm/optimized.yaml',
                        help='Path to configuration file')
    parser.add_argument('--work-dir', type=str, default='./experiments',
                        help='Working directory for outputs')
    parser.add_argument('--model-saved-name', type=str, default='model',
                        help='Base name for saving model')
    parser.add_argument('--device', default='0',
                        help='GPU device ID')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test', 'tflite'],
                        help='Training, testing, or TFLite export phase')
    
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
    
    # Model parameters
    parser.add_argument('--model', default=None, 
                        help='Model class path')
    parser.add_argument('--model-args', default=None, 
                        help='Model arguments')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset to use')
    parser.add_argument('--dataset-args', default=None,
                        help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject IDs to use')
    parser.add_argument('--feeder', default=None,
                        help='Data feeder class path')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed for reproducibility')
    parser.add_argument('--result-file', type=str, default=None,
                        help='File to save testing results')
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print logs')
    parser.add_argument('--mixed-precision', type=str2bool, default=False,
                        help='Use mixed precision training')
    
    return parser

def setup_gpu(device_id):
    """Configure GPU settings"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Using GPU: {device_id}")
            return True
        except Exception as e:
            logger.warning(f"Error configuring GPU: {e}")
    
    logger.info("No GPU available, using CPU")
    return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load configuration from YAML file
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update arguments with values from config
        for k, v in config.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if hasattr(args, 'work_dir') and os.path.exists(args.work_dir):
        args.work_dir = f"{args.work_dir}_{timestamp}"
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(os.path.join(args.work_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save configuration
    with open(os.path.join(args.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Set up GPU
    has_gpu = setup_gpu(args.device)
    
    # Set random seed
    init_seed(args.seed)
    
    # Enable mixed precision if requested
    if args.mixed_precision and has_gpu:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision enabled with policy: {policy}")
    
    # Import base trainer
    try:
        from trainer.base_trainer import BaseTrainer
        trainer = BaseTrainer(args)
        trainer.start()
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
