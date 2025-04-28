# src/utils/data_processor.py
import tensorflow as tf
import numpy as np
import logging
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
import matplotlib.pyplot as plt

def filter_data_by_ids(data, ids):
    """Filter data by selected IDs"""
    return data[ids, :]

def filter_repeated_ids(path):
    """Filter repeated IDs in DTW path"""
    seen_first = set()
    seen_second = set()

    for (first, second) in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def align_sequences_with_dtw(acc_data, skel_data, joint_id=9):
    """Align accelerometer and skeleton data using DTW
    
    Args:
        acc_data: Accelerometer data [frames, channels]
        skel_data: Skeleton data [frames, joints, channels] or [frames, joints*channels]
        joint_id: Joint ID to use for alignment (default: 9 - left wrist)
        
    Returns:
        tuple: Aligned accelerometer and skeleton data
    """
    try:
        # Extract left wrist joint data from skeleton
        if len(skel_data.shape) == 3:  # [frames, joints, channels]
            skeleton_joint_data = skel_data[:, joint_id-1, :]
        else:  # [frames, joints*channels]
            skeleton_joint_data = skel_data[:, (joint_id-1)*3:joint_id*3]
        
        # Calculate Frobenius norm
        skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        acc_frob_norm = np.linalg.norm(acc_data, axis=1)
        
        # Apply DTW
        distance, path = fastdtw(
            acc_frob_norm[:, np.newaxis], 
            skeleton_frob_norm[:, np.newaxis],
            dist=euclidean
        )
        
        # Filter repeated IDs
        acc_ids, skel_ids = filter_repeated_ids(path)
        
        # Apply filtering
        aligned_acc = filter_data_by_ids(acc_data, list(acc_ids))
        aligned_skel = filter_data_by_ids(skel_data, list(skel_ids))
        
        return aligned_acc, aligned_skel
    except Exception as e:
        logging.error(f"DTW alignment failed: {e}")
        # Return original data if alignment fails
        return acc_data, skel_data

def preprocess_data(data, add_smv=True, align=True):
    """Preprocess accelerometer and skeleton data
    
    Args:
        data: Dictionary containing 'accelerometer' and 'skeleton' data
        add_smv: Whether to add Signal Magnitude Vector
        align: Whether to align data using DTW
        
    Returns:
        dict: Preprocessed data
    """
    result = {}
    
    # Check if required data is present
    if 'accelerometer' not in data:
        logging.error("Accelerometer data is required for preprocessing")
        return data
    
    acc_data = data['accelerometer']
    
    # Align data if requested and skeleton data is available
    if align and 'skeleton' in data:
        skel_data = data['skeleton']
        acc_data, skel_data = align_sequences_with_dtw(acc_data, skel_data)
        result['skeleton'] = skel_data
    
    # Add Signal Magnitude Vector if requested
    if add_smv:
        # Calculate mean
        mean = np.mean(acc_data, axis=1, keepdims=True)
        zero_mean = acc_data - mean
        
        # Calculate sum of squares
        sum_squared = np.sum(np.square(zero_mean), axis=-1, keepdims=True)
        
        # Calculate SMV
        smv = np.sqrt(sum_squared)
        
        # Concatenate with original data
        acc_with_smv = np.concatenate([smv, acc_data], axis=-1)
        result['accelerometer'] = acc_with_smv
    else:
        result['accelerometer'] = acc_data
    
    # Copy labels if present
    if 'labels' in data:
        result['labels'] = data['labels']
    
    return result
