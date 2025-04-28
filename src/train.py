# train.py
#!/usr/bin/env python
import os
import argparse
import yaml
import json
import sys
import logging
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lightheart-tf')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def init_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                pass

def get_args():
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
    parser.add_argument('--model', type=str, default=None, 
                        help='Model class path')
    parser.add_argument('--model-args', type=dict, default=None, 
                        help='Model arguments')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Distillation parameters
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Teacher model class path for distillation')
    parser.add_argument('--teacher-args', type=dict, default=None,
                        help='Teacher model arguments')
    parser.add_argument('--teacher-weight', type=str, default=None,
                        help='Path to teacher model weights')
    parser.add_argument('--distill-args', type=dict, default=None,
                        help='Distillation arguments')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset to use')
    parser.add_argument('--dataset-args', type=dict, default=None,
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
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print logs')
    parser.add_argument('--mixed-precision', type=str2bool, default=False,
                        help='Use mixed precision training')
    
    return parser

def setup_gpu(device_id):
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
    # Parse command line arguments
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
                sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not hasattr(args, 'work_dir') or args.work_dir is None:
        args.work_dir = f"./experiments/run_{timestamp}"
    elif os.path.exists(args.work_dir):
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
    setup_gpu(args.device)
    
    # Set random seed
    init_seed(args.seed)
    
    # Initialize distill_args if not present
    if not hasattr(args, 'distill_args') or args.distill_args is None:
        args.distill_args = {}
    
    # Determine if we should use distillation
    is_distillation = hasattr(args, 'teacher_model') and args.teacher_model is not None
    
    # Import appropriate trainer
    try:
        if is_distillation:
            from trainer.distiller import DistillationTrainer
            logger.info("Using knowledge distillation training")
            trainer = DistillationTrainer(args)
        else:
            from trainer.base_trainer import BaseTrainer
            logger.info("Using standard training")
            trainer = BaseTrainer(args)
        
        # Start training
        trainer.start()
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
