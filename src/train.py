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
import traceback

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
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                pass

def log_best_results(results_dir, model_metrics, timestamp=None):
    import pandas as pd
    avg_metrics = {}
    subject_count = 0
    for subject, metrics in model_metrics.items():
        subject_count += 1
        for metric, value in metrics.items():
            if metric not in avg_metrics:
                avg_metrics[metric] = 0
            avg_metrics[metric] += value
    for metric in avg_metrics:
        avg_metrics[metric] /= max(1, subject_count)
    if timestamp is None:
        now = datetime.now()
    else:
        now = datetime.fromtimestamp(timestamp)
    month_names = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    month_name = month_names[now.month - 1]
    day_names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first"]
    day_name = day_names[now.day - 1]
    f1_acc = f"f1_{avg_metrics['f1_score']:.2f}_acc_{avg_metrics['accuracy']:.2f}"
    filename = f"{month_name}-{day_name}-{f1_acc}.txt"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), 'w') as f:
        f.write(f"Average Test Metrics (across {subject_count} subjects):\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.2f}%\n")
        f.write("\nPer-Subject Metrics:\n")
        for subject, metrics in model_metrics.items():
            f.write(f"Subject {subject}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.2f}%\n")
    logger.info(f"Results saved to {os.path.join(results_dir, filename)}")
    return os.path.join(results_dir, filename)

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection Training')
    parser.add_argument('--config', default='./config/smartfallmm/optimized.yaml', help='Path to configuration file')
    parser.add_argument('--num-worker', type=int, default=0, help='Number of worker processes for data loading')
    parser.add_argument('--work-dir', type=str, default='./experiments', help='Working directory for outputs')
    parser.add_argument('--model-saved-name', type=str, default='model', help='Base name for saving model')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'distill', 'tflite'], help='Training phase')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, help='Testing batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--num-epoch', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch number')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--base-lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay factor')
    parser.add_argument('--model', type=str, default=None, help='Model class path')
    parser.add_argument('--model-args', type=str, default=None, help='Model arguments')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset to use')
    parser.add_argument('--dataset-args', type=str, default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=None, help='Subject IDs to use')
    parser.add_argument('--feeder', type=str, default=None, help='Data feeder class path')
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility')
    parser.add_argument('--result-file', type=str, default=None, help='File to save testing results')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Whether to print logs')
    parser.add_argument('--mixed-precision', type=str2bool, default=False, help='Use mixed precision training')
    parser.add_argument('--use-smv', type=str2bool, default=False, help='Use Signal Magnitude Vector calculation')
    parser.add_argument('--train-subjects-fixed', nargs='+', type=int, default=[45, 36, 29], help='Always include these in training')
    parser.add_argument('--val-subjects-fixed', nargs='+', type=int, default=[38, 46], help='Always use these for validation')
    parser.add_argument('--test-eligible-subjects', nargs='+', type=int, default=[32, 39, 30, 31, 33, 34, 35, 37, 43, 44], help='Eligible for test split')
    return parser

def setup_gpu(device_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Using GPU: {device_id}")
            devices_str = ", ".join([f"'{d.name}'" for d in physical_devices])
            logger.info(f"Found {len(physical_devices)} GPU(s): [{devices_str}]")
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                                                '--format=csv,nounits,noheader'])
                memory_info = result.decode('utf-8').strip().split('\n')
                for i, info in enumerate(memory_info):
                    total, used = map(int, info.split(','))
                    logger.info(f"GPU {i}: Memory {used}/{total} MB ({used/total*100:.1f}%)")
            except:
                pass
            return True
        except Exception as e:
            logger.warning(f"Error configuring GPU: {e}")
    logger.info("No GPU available, using CPU")
    return False

def import_class(import_str):
    if import_str is None:
        raise ValueError("Import path cannot be None")
    mod_str, _sep, class_str = import_str.rpartition('.')
    for prefix in ['', 'src.']:
        try:
            import importlib
            module = importlib.import_module(f"{prefix}{mod_str}")
            return getattr(module, class_str)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Cannot import {class_str} from {mod_str}")

def main():
    parser = get_args()
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                for k, v in config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
            except yaml.YAMLError as e:
                logger.error(f"Error loading config file: {e}")
                sys.exit(1)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not hasattr(args, 'work_dir') or args.work_dir is None:
        args.work_dir = f"./experiments/run_{timestamp}"
    elif os.path.exists(args.work_dir) and not args.work_dir.endswith(timestamp):
        args.work_dir = f"{args.work_dir}_{timestamp}"
    
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'visualizations'), exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(args.work_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    with open(os.path.join(args.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    has_gpu = setup_gpu(args.device)
    init_seed(args.seed)
    
    if args.mixed_precision and has_gpu:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info(f"Mixed precision enabled with policy: {policy}")
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    
    try:
        if args.phase == 'distill':
            from distiller import Distiller
            trainer = Distiller(args)
        else:
            from trainer.base_trainer import BaseTrainer
            trainer = BaseTrainer(args)
        
        trainer.start()
        
        if args.phase in ['train', 'distill']:
            try:
                if os.path.islink('experiments/latest'):
                    os.unlink('experiments/latest')
                os.symlink(os.path.basename(args.work_dir), 'experiments/latest')
                logger.info(f"Created symlink: experiments/latest -> {os.path.basename(args.work_dir)}")
            except:
                logger.warning("Failed to create symlink to latest experiment")
    
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
