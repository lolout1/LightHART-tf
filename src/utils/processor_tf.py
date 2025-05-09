#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
processor_tf.py - TensorFlow implementation of signal processing functions
Exact match to PyTorch implementation with robust error handling
"""

import tensorflow as tf
import numpy as np
import logging
from scipy.signal import butter, filtfilt, find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

logger = logging.getLogger('processor-tf')

def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    """
    Apply Butterworth low-pass filter identical to PyTorch implementation
    
    Args:
        data: Input signal data
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered data
    """
    try:
        # Check if data is valid for filtering
        if len(data) < 10:
            return data
            
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logger.warning(f"Butterworth filter error: {e}")
        return data

def avg_pool_tf(sequence, max_length=128, window_size=5):
    """
    Apply average pooling to sequence data for length normalization
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
        window_size: Size of pooling window
    
    Returns:
        Pooled sequence
    """
    try:
        shape = sequence.shape
        
        # Skip pooling if sequence is already shorter than max_length
        if shape[0] <= max_length:
            return sequence
        
        # Reshape for 1D convolution
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        
        # Convert to TensorFlow tensor
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        
        # Calculate appropriate stride to achieve desired length
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        
        # Apply pooling
        sequence = tf.nn.avg_pool1d(
            sequence, 
            ksize=window_size, 
            strides=stride, 
            padding='VALID'
        )
        
        # Convert back to numpy and reshape
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool_tf: {e}")
        # Return truncated original sequence as fallback
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence

def pad_sequence_tf(sequence, max_length):
    """
    Pad sequence to fixed length with pooling for longer sequences
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
    
    Returns:
        Padded sequence
    """
    try:
        shape = list(sequence.shape)
        
        # If sequence is already the right length
        if shape[0] == max_length:
            return sequence
            
        # If sequence is longer, apply pooling
        if shape[0] > max_length:
            pooled_sequence = avg_pool_tf(sequence, max_length)
        else:
            # If sequence is shorter, use original
            pooled_sequence = sequence
        
        # Create output array with target shape
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        
        # Copy pooled data (preserves original sequence ordering)
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_tf: {e}")
        
        # Create a valid fallback padded sequence
        shape = list(sequence.shape)
        shape[0] = max_length
        padded = np.zeros(shape, dtype=sequence.dtype)
        copy_length = min(len(sequence), max_length)
        padded[:copy_length] = sequence[:copy_length]
        return padded

def filter_data_by_ids(data, ids):
    """
    Filter data by specific indices
    
    Args:
        data: Input data array
        ids: List of indices to keep
    
    Returns:
        Filtered data
    """
    if len(ids) == 0:
        return data[:0]  # Return empty array with same shape
    return data[ids]

def filter_repeated_ids(path):
    """
    Filter DTW path to get unique mappings
    
    Args:
        path: DTW path as list of index pairs
    
    Returns:
        Two sets of unique indices
    """
    seen_first = set()
    seen_second = set()
    
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second
def align_sequence_dtw(data, joint_id=9, use_dtw=True):
    """
    Align skeleton and accelerometer data using DTW
    EXACT match to PyTorch implementation with improved error handling
    
    Args:
        data: Dictionary with 'skeleton' and 'accelerometer' arrays
        joint_id: Joint ID to use for alignment (default: 9 = left wrist)
        use_dtw: Whether to use DTW alignment
        
    Returns:
        Dictionary with aligned data
    """
    try:
        # Skip if missing required modalities
        if 'skeleton' not in data or 'accelerometer' not in data:
            logger.warning("Missing required modalities for DTW alignment")
            return data
        
        # Skip DTW if disabled or if only one modality is present
        if not use_dtw or len(data.keys()) == 1:
            logger.info("DTW alignment skipped (disabled or single modality)")
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        # Get dynamic keys (non-skeleton modalities)
        dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
        
        # Extract specific joint data - handle different skeleton formats
        skeleton_data = data['skeleton']
        
        # Handle different skeleton formats - needs to extract joint coordinates
        if len(skeleton_data.shape) == 3:  # [frames, joints*3]
            start_idx = (joint_id - 1) * 3
            end_idx = joint_id * 3
            if skeleton_data.shape[1] >= end_idx:
                skeleton_joint_data = skeleton_data[:, start_idx:end_idx]
            else:
                logger.warning(f"Skeleton data has insufficient joints: {skeleton_data.shape}")
                min_length = min(len(data['accelerometer']), len(skeleton_data))
                data['accelerometer'] = data['accelerometer'][:min_length]
                data['skeleton'] = skeleton_data[:min_length]
                return data
        elif len(skeleton_data.shape) == 4:  # [frames, joints, 3]
            if skeleton_data.shape[1] >= joint_id:
                skeleton_joint_data = skeleton_data[:, joint_id-1, :]
            else:
                logger.warning(f"Skeleton data has insufficient joints: {skeleton_data.shape}")
                min_length = min(len(data['accelerometer']), len(skeleton_data))
                data['accelerometer'] = data['accelerometer'][:min_length]
                data['skeleton'] = skeleton_data[:min_length]
                return data
        else:
            logger.warning(f"Unsupported skeleton shape: {skeleton_data.shape}")
            return data
        
        # Get accelerometer data from first dynamic key
        inertial_data = data[dynamic_keys[0]]
        
        # Handle multiple inertial sensors if present
        if len(dynamic_keys) > 1:
            gyroscope_data = data[dynamic_keys[1]]
            min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
            inertial_data = inertial_data[:min_len, :]
            data[dynamic_keys[1]] = gyroscope_data[:min_len, :]
        
        # Safety check for empty data
        if len(skeleton_joint_data) == 0 or len(inertial_data) == 0:
            logger.warning("Empty data for DTW alignment, skipping")
            min_length = min(len(data['accelerometer']), len(skeleton_data))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = skeleton_data[:min_length]
            return data
        
        # Calculate magnitude vectors for alignment - EXACT match to PyTorch implementation
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)
        
        # Safety check for empty norms
        if len(skeleton_frob_norm) == 0 or len(inertial_frob_norm) == 0:
            logger.warning("Empty norm vectors for DTW alignment, skipping")
            min_length = min(len(data['accelerometer']), len(skeleton_data))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = skeleton_data[:min_length]
            return data
        
        # IMPORTANT FIX: Make sure vectors are properly shaped for fastdtw
        # FastDTW expects 2D arrays with shape (n_samples, n_features)
        skeleton_frob_norm_2d = skeleton_frob_norm.reshape(-1, 1)
        inertial_frob_norm_2d = inertial_frob_norm.reshape(-1, 1)
        
        # Apply DTW with euclidean distance - EXACT match to PyTorch implementation
        try:
            distance, path = fastdtw(
                inertial_frob_norm_2d, 
                skeleton_frob_norm_2d,
                dist=euclidean,
                radius=10  # Use smaller radius for faster processing
            )
        except Exception as e:
            logger.error(f"DTW calculation error: {e}")
            # Simple alignment fallback
            min_length = min(len(inertial_data), len(skeleton_data))
            data['accelerometer'] = inertial_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
            return data
        
        # Filter repeated indices - EXACT match to PyTorch implementation
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        
        # Convert to lists and sort
        inertial_ids = sorted(list(inertial_ids))
        skeleton_ids = sorted(list(skeleton_ids))
        
        # CRITICAL FIX: Ensure indices are within bounds
        inertial_ids = [idx for idx in inertial_ids if idx < len(inertial_data)]
        skeleton_ids = [idx for idx in skeleton_ids if idx < len(skeleton_data)]
        
        # Sort again after filtering
        inertial_ids.sort()
        skeleton_ids.sort()
        
        # Apply alignment if we have enough indices
        if len(inertial_ids) > 10 and len(skeleton_ids) > 10:
            # Safety check: ensure all indices are valid
            try:
                data['skeleton'] = filter_data_by_ids(data['skeleton'], skeleton_ids)
                for key in dynamic_keys:
                    data[key] = filter_data_by_ids(data[key], inertial_ids)
                
                logger.info(f"DTW alignment successful: mapped {len(inertial_ids)} accelerometer frames to "
                           f"{len(skeleton_ids)} skeleton frames")
            except Exception as e:
                logger.error(f"Error applying DTW indices: {e}")
                min_length = min(len(data['accelerometer']), len(data['skeleton']))
                data['accelerometer'] = data['accelerometer'][:min_length]
                data['skeleton'] = data['skeleton'][:min_length]
        else:
            logger.warning(f"DTW alignment yielded too few matched indices")
            
            # Fall back to simple length matching
            min_length = min(len(inertial_data), len(skeleton_data))
            data['accelerometer'] = inertial_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
        
        return data
        
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        
        # Simple fallback alignment
        if 'accelerometer' in data and 'skeleton' in data:
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
        
        return data
def selective_windowing(data, window_size, label):
    """
    Apply selective windowing around detected peaks in accelerometer data
    EXACT match to PyTorch implementation
    
    Args:
        data: Dictionary of modality data
        window_size: Window size (usually 128)
        label: Class label (0=ADL, 1=fall)
    
    Returns:
        Dictionary with windowed data
    """
    try:
        # Verify we have accelerometer data
        if 'accelerometer' not in data or len(data['accelerometer']) < window_size:
            logger.warning(f"Insufficient accelerometer data for windowing")
            result = {'labels': np.array([label])}
            
            # Add dummy data for each modality
            for key in data:
                if key != 'labels':
                    if len(data[key]) == 0:
                        continue
                        
                    # Create appropriate dummy shape based on modality
                    if key == 'accelerometer':
                        result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
                    elif key == 'skeleton':
                        if len(data[key].shape) == 3:  # [frames, joints*3]
                            result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
                        else:  # [frames, joints, 3]
                            result[key] = np.zeros((1, window_size, data[key].shape[1], data[key].shape[2]), dtype=np.float32)
                    else:
                        result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
            
            return result
        
        # Calculate signal magnitude
        acc_data = data['accelerometer']
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Set parameters based on label - EXACT match to PyTorch implementation
        if label == 1:  # Fall
            height, distance = 1.4, 50
        else:  # Non-fall
            height, distance = 1.2, 100
        
        # Find peaks
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # Create windows around each peak
        if len(peaks) == 0:
            # Create a window at the middle point if no peaks
            peaks = [len(sqrt_sum) // 2]
            logger.info(f"No peaks found for label {label}, using center point")
        
        # Create windows for each modality
        windows_dict = {}
        
        for key in data:
            if key == 'labels':
                continue
                
            # Skip missing modalities
            if key not in data or len(data[key]) == 0:
                continue
            
            windows = []
            
            for peak in peaks:
                # Calculate window boundaries centered on peak
                start = max(0, peak - window_size // 2)
                end = min(len(data[key]), start + window_size)
                
                # Skip if window would be too small
                if end - start < window_size * 0.75:  # Require at least 75% of window size
                    continue
                
                # Create window with proper shape
                window = np.zeros((window_size,) + data[key].shape[1:], dtype=data[key].dtype)
                
                # Fill with data
                copy_length = end - start
                window[:copy_length] = data[key][start:end]
                windows.append(window)
            
            # Add windows to result if any were created
            if windows:
                try:
                    # Ensure all windows have the same shape
                    first_shape = windows[0].shape
                    consistent_windows = [w for w in windows if w.shape == first_shape]
                    
                    if not consistent_windows:
                        # No consistent windows - create one dummy window
                        logger.warning(f"No consistent window shapes for {key}")
                        windows_dict[key] = np.zeros((1, window_size) + first_shape[1:], dtype=windows[0].dtype)
                    else:
                        # Stack consistent windows
                        windows_dict[key] = np.stack(consistent_windows)
                        
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
                    # Create single dummy window as fallback
                    windows_dict[key] = np.zeros((1, window_size) + data[key].shape[1:], dtype=data[key].dtype)
            else:
                # Create dummy window if none were valid
                windows_dict[key] = np.zeros((1, window_size) + data[key].shape[1:], dtype=data[key].dtype)
        
        # Add labels
        if windows_dict:
            # Use length of first modality for label count
            first_key = next(iter(windows_dict))
            windows_dict['labels'] = np.full(len(windows_dict[first_key]), label, dtype=np.int32)
            logger.info(f"Created {len(windows_dict['labels'])} windows for label {label}")
        else:
            # Fallback with single label
            windows_dict['labels'] = np.array([label], dtype=np.int32)
            
            # Ensure all modalities have dummy data
            for key in data:
                if key != 'labels' and key not in windows_dict:
                    if key == 'accelerometer':
                        windows_dict[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
                    elif key == 'skeleton':
                        if len(data[key].shape) == 3:  # [frames, joints*3]
                            windows_dict[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
                        else:  # [frames, joints, 3]
                            windows_dict[key] = np.zeros((1, window_size, data[key].shape[1], data[key].shape[2]), dtype=np.float32)
        
        return windows_dict
        
    except Exception as e:
        logger.error(f"Error in selective_windowing: {e}")
        
        # Create minimal valid output as fallback
        result = {'labels': np.array([label])}
        for key in data:
            if key != 'labels':
                if key == 'accelerometer':
                    result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
                elif key == 'skeleton':
                    result[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
                else:
                    result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
        return result
