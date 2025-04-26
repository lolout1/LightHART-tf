import os
import random
import numpy as np
import tensorflow as tf
import yaml

def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Unsupported boolean value.')

def init_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def import_class(import_str):
    """Dynamically import a class."""
    import sys
    import traceback
    
    mod_str, _sep, class_str = import_str.rpartition('.')
    try:
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except (ImportError, AttributeError):
        if mod_str.startswith('src.'):
            alt_mod_str = mod_str[4:]
            try:
                __import__(alt_mod_str)
                return getattr(sys.modules[alt_mod_str], class_str)
            except: pass
        print(f"Error importing {import_str}")
        print(traceback.format_exc())
        raise ImportError(f"Cannot import {import_str}")

def save_config(src_path, dest_dir):
    """Save configuration file to destination directory."""
    config_filename = os.path.basename(src_path)
    with open(src_path, 'r') as f_src:
        with open(f'{dest_dir}/{config_filename}', 'w') as f_dst:
            f_dst.write(f_src.read())

def create_fold_splits(subjects, fold_index=None, validation_subjects=[38, 46]):
    """Create train/val/test splits with leave-one-out cross-validation.
    
    If fold_index is provided, returns splits for that specific fold.
    Otherwise, returns a list of all possible fold splits.
    """
    all_splits = []
    
    for i, test_subject in enumerate(subjects):
        # Skip validation subjects
        if test_subject in validation_subjects:
            continue
            
        # Create train subjects (everyone except test and validation)
        train_subjects = [s for s in subjects if s != test_subject and s not in validation_subjects]
        
        # Create fold
        fold = {
            'train': train_subjects,
            'val': validation_subjects,
            'test': [test_subject],
            'fold': i
        }
        
        all_splits.append(fold)
    
    if fold_index is not None:
        if 0 <= fold_index < len(all_splits):
            return all_splits[fold_index]
        else:
            raise ValueError(f"Fold index {fold_index} out of range (0-{len(all_splits)-1})")
    
    return all_splits
