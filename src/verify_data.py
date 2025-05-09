#!/usr/bin/env python
# verify_data.py - Verifies dataset loading and preprocessing

import os
import logging
import numpy as np
from utils.dataset_tf import prepare_smartfallmm_tf, SmartFallMM, split_by_subjects_tf

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data-verification')

class DummyArgs:
    def __init__(self):
        self.dataset_args = {
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd',
            'modalities': ['accelerometer','skeleton'],  # Test with accelerometer only
            'age_group': ['young'],
            'sensors': ['watch'],
            'use_dtw': True,
            'verbose': True
        }
        self.subjects = [32, 39, 30, 31, 33]  # Sample subjects

def main():
    """Verify data loading and preprocessing."""
    logger.info("Starting data verification...")
    
    # Find data directory
    possible_paths = [
        os.path.join(os.getcwd(), 'data/smartfallmm'),
        os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm')
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
            logger.info(f"Found data directory at: {data_dir}")
            break
    
    if data_dir is None:
        logger.error("Data directory not found")
        return
    
    # List data directory contents
    logger.info(f"Data directory contents: {os.listdir(data_dir)}")
    
    # Test direct loading with SmartFallMM
    logger.info("Testing direct loading with SmartFallMM...")
    dataset = SmartFallMM(root_dir=data_dir)
    dataset.pipeline(
        age_group=['young'], 
        modalities=['accelerometer'], 
        sensors=['watch']
    )
    
    logger.info(f"Matched trials: {len(dataset.matched_trials)}")
    if dataset.matched_trials:
        for i, trial in enumerate(dataset.matched_trials[:5]):
            logger.info(f"Trial {i}: subject={trial.subject_id}, action={trial.action_id}, seq={trial.sequence_number}")
            logger.info(f"  Files: {trial.files}")
    
    # Test with dataset builder and args
    logger.info("Testing with dataset builder...")
    dummy_args = DummyArgs()
    builder = prepare_smartfallmm_tf(dummy_args)
    
    logger.info("Testing data splitting...")
    data = split_by_subjects_tf(builder, [32, 39], False)
    
    logger.info("Data split results:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    logger.info("Data verification complete")

if __name__ == "__main__":
    main()
