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
    if not isinstance(ids, list):
        ids = list(ids)
    
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
    Align skeleton and accelerometer data using DTW.
    Works with single or multiple modalities.
    
    Args:
        data: Dictionary with modality data
        joint_id: Joint ID to use for alignment
        use_dtw: Whether to use DTW alignment
        
    Returns:
        Dictionary with aligned data
    """
    try:
        # Check for single modality scenario
        has_skeleton = 'skeleton' in data and data['skeleton'] is not None and len(data['skeleton']) > 0
        has_accelerometer = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
        
        # If we only have one modality or DTW is disabled, no alignment needed
        if (not has_skeleton or not has_accelerometer) or not use_dtw:
            logger.info(f"DTW alignment skipped (use_dtw={use_dtw}, has_skeleton={has_skeleton}, has_accelerometer={has_accelerometer})")
            return data
        
        logger.info(f"Performing DTW alignment between skeleton and accelerometer data")
        
        # Extract joint data for alignment
        if len(data['skeleton'].shape) == 4:  # [frames, joints, 3]
            skeleton_joint_data = data['skeleton'][:, joint_id-1, :]
        elif len(data['skeleton'].shape) == 3 and data['skeleton'].shape[1] >= joint_id * 3:
            start_idx = (joint_id - 1) * 3
            end_idx = joint_id * 3
            skeleton_joint_data = data['skeleton'][:, start_idx:end_idx]
        else:
            logger.warning(f"Skeleton data shape not compatible for joint extraction: {data['skeleton'].shape}")
            return data
        
        # Get accelerometer data and other inertial data if present
        inertial_data = data['accelerometer']
        dynamic_keys = [k for k in data.keys() if k != 'skeleton' and k != 'labels']
        
        # Handle multiple inertial sensors if present (e.g., accelerometer and gyroscope)
        if len(dynamic_keys) > 1:
            # Make sure all inertial data has same length
            min_length = min(len(data[key]) for key in dynamic_keys)
            for key in dynamic_keys:
                data[key] = data[key][:min_length]
        
        # Calculate magnitude vectors for alignment - EXACT match to PyTorch implementation
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)
        
        # Safety check for empty norms
        if len(skeleton_frob_norm) == 0 or len(inertial_frob_norm) == 0:
            logger.warning(f"Empty norm vectors for DTW alignment, skipping")
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        # Reshape for fastdtw (requires 2D arrays)
        skeleton_frob_norm_2d = skeleton_frob_norm.reshape(-1, 1)
        inertial_frob_norm_2d = inertial_frob_norm.reshape(-1, 1)
        
        # Apply DTW with euclidean distance - EXACT match to PyTorch implementation
        try:
            distance, path = fastdtw(
                inertial_frob_norm_2d, 
                skeleton_frob_norm_2d,
                dist=euclidean,
                radius=20  # Default radius for accuracy
            )
            logger.info(f"DTW completed with distance: {distance:.2f}")
        except Exception as e:
            logger.error(f"DTW calculation error: {e}")
            # Simple alignment fallback
            min_length = min(len(inertial_data), len(data['skeleton']))
            data['accelerometer'] = inertial_data[:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        # Filter repeated indices - EXACT match to PyTorch implementation
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        
        # Convert to lists and sort
        inertial_ids = sorted(list(inertial_ids))
        skeleton_ids = sorted(list(skeleton_ids))
        
        # CRITICAL FIX: Ensure indices are within bounds
        inertial_ids = [idx for idx in inertial_ids if idx < len(inertial_data)]
        skeleton_ids = [idx for idx in skeleton_ids if idx < len(data['skeleton'])]
        
        # Apply alignment if we have enough matched indices
        if len(inertial_ids) > 10 and len(skeleton_ids) > 10:
            logger.info(f"Applying DTW alignment: {len(inertial_ids)} accelerometer frames mapped to {len(skeleton_ids)} skeleton frames")
            data['skeleton'] = filter_data_by_ids(data['skeleton'], skeleton_ids)
            for key in dynamic_keys:
                data[key] = filter_data_by_ids(data[key], inertial_ids)
        else:
            logger.warning(f"DTW alignment produced too few matches ({len(inertial_ids)}/{len(skeleton_ids)}), using simple truncation")
            min_length = min(len(inertial_data), len(data['skeleton']))
            data['accelerometer'] = inertial_data[:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
        
        return data
        
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        
        # Simple fallback alignment
        if 'accelerometer' in data and 'skeleton' in data:
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
        
        return data

def create_dummy_windows(data, label, window_size):
    """Create minimal valid windows when processing fails."""
    result = {'labels': np.array([label])}
    
    for key in data:
        if key != 'labels':
            if key == 'accelerometer':
                result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
            elif key == 'skeleton':
                result[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
            elif key == 'gyroscope':
                result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
    
    return result

def selective_windowing(data, window_size, label):
    """
    Apply selective windowing around detected peaks with support for single modality.
    
    Args:
        data: Dictionary of modality data
        window_size: Window size (usually 128)
        label: Class label (0=ADL, 1=fall)
    
    Returns:
        Dictionary with windowed data
    """
    try:
        # Identify which modalities are available
        available_modalities = [k for k in data.keys() if k != 'labels' and len(data[k]) > 0]
        logger.info(f"Available modalities for windowing: {available_modalities}")
        
        # Require at least one valid modality
        if not available_modalities:
            logger.warning("No valid modalities for windowing")
            return {'labels': np.array([label])}
        
        # Use accelerometer for peak detection if available, otherwise use first available modality
        peak_modality = 'accelerometer' if 'accelerometer' in available_modalities else available_modalities[0]
        peak_data = data[peak_modality]
        
        # Verify data is sufficient
        if len(peak_data) < window_size:
            logger.warning(f"Insufficient {peak_modality} data for windowing: {len(peak_data)} < {window_size}")
            return create_dummy_windows(data, label, window_size)
        
        # Calculate signal magnitude for peak detection
        sqrt_sum = np.sqrt(np.sum(peak_data**2, axis=1))
        
        # Set parameters based on label - EXACT match to PyTorch implementation
        if label == 1:  # Fall
            height, distance = 1.4, 50
        else:  # Non-fall
            height, distance = 1.2, 100
        
        # Find peaks
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # If no peaks found, use middle point
        if len(peaks) == 0:
            peaks = [len(sqrt_sum) // 2]
            logger.info(f"No peaks found for label {label}, using center point")
        
        # Create windows for each modality
        windows_dict = {}
        
        for key in available_modalities:
            windows = []
            
            for peak in peaks:
                # Calculate window boundaries centered on peak
                start = max(0, peak - window_size // 2)
                end = min(len(data[key]), start + window_size)
                
                # Skip if window would be too small
                if end - start < window_size * 0.75:  # Require at least 75% of window size
                    continue
                
                # Create properly sized window
                window = np.zeros((window_size,) + data[key].shape[1:], dtype=data[key].dtype)
                
                # Fill with data
                copy_length = end - start
                window[:copy_length] = data[key][start:end]
                windows.append(window)
            
            # Add windows to result
            if windows:
                try:
                    windows_dict[key] = np.stack(windows)
                    logger.info(f"Created {len(windows)} windows for {key}, shape: {windows_dict[key].shape}")
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
            else:
                logger.warning(f"No valid windows created for {key}")
        
        # Add labels
        if windows_dict:
            # Use length of first modality for label count
            first_key = next(iter(windows_dict))
            windows_dict['labels'] = np.full(len(windows_dict[first_key]), label, dtype=np.int32)
            logger.info(f"Created {len(windows_dict['labels'])} windows for label {label}")
            return windows_dict
        else:
            # Fallback to dummy windows
            return create_dummy_windows(data, label, window_size)
        
    except Exception as e:
        logger.error(f"Error in selective_windowing: {e}")
        return create_dummy_windows(data, label, window_size)

def sliding_window(data, max_length, stride=10, label=0):
    """
    Simple sliding window implementation that works for both single and multi-modal data.
    
    Args:
        data: Dictionary of modality data
        max_length: Window size
        stride: Stride between windows
        label: Class label
        
    Returns:
        Dictionary with windowed data
    """
    try:
        # Identify which modalities are available
        available_modalities = [k for k in data.keys() if k != 'labels' and len(data[k]) > 0]
        logger.info(f"Sliding window for modalities: {available_modalities}")
        
        # Require at least one valid modality
        if not available_modalities:
            logger.warning("No valid modalities for sliding window")
            return {'labels': np.array([label])}
        
        # Use first available modality to determine sequence length
        first_modality = available_modalities[0]
        seq_length = len(data[first_modality])
        
        # Verify data is sufficient
        if seq_length < max_length:
            logger.warning(f"Insufficient {first_modality} data for windowing: {seq_length} < {max_length}")
            return create_dummy_windows(data, label, max_length)
        
        # Calculate window start indices
        window_starts = range(0, seq_length - max_length + 1, stride)
        
        # Create windows for each modality
        windows_dict = {}
        
        for key in available_modalities:
            windows = []
            
            for start in window_starts:
                # Calculate window boundaries
                end = start + max_length
                
                # Extract window
                window = data[key][start:end]
                
                # Verify window is correct size
                if len(window) == max_length:
                    windows.append(window)
            
            # Add windows to result
            if windows:
                try:
                    windows_dict[key] = np.stack(windows)
                    logger.info(f"Created {len(windows)} sliding windows for {key}, shape: {windows_dict[key].shape}")
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
            else:
                logger.warning(f"No valid windows created for {key}")
        
        # Add labels
        if windows_dict:
            # Use length of first modality for label count
            first_key = next(iter(windows_dict))
            windows_dict['labels'] = np.full(len(windows_dict[first_key]), label, dtype=np.int32)
            return windows_dict
        else:
            # Fallback to dummy windows
            return create_dummy_windows(data, label, max_length)
        
    except Exception as e:
        logger.error(f"Error in sliding_window: {e}")
        return create_dummy_windows(data, label, max_length)

# Utility for testing the processor
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create test data
    test_acc = np.random.randn(256, 3).astype(np.float32)
    test_skl = np.random.randn(256, 32, 3).astype(np.float32)
    test_data = {'accelerometer': test_acc, 'skeleton': test_skl}
    
    # Test filtering
    filtered_acc = butterworth_filter(test_acc)
    logger.info(f"Filtered accelerometer data shape: {filtered_acc.shape}")
    
    # Test padding
    padded_acc = pad_sequence_tf(test_acc, 128)
    logger.info(f"Padded accelerometer data shape: {padded_acc.shape}")
    
    # Test alignment
    aligned_data = align_sequence_dtw(test_data)
    for key, value in aligned_data.items():
        logger.info(f"Aligned {key} shape: {value.shape}")
    
    # Test windowing
    windowed_data = selective_windowing(test_data, 128, 1)
    for key, value in windowed_data.items():
        logger.info(f"Windowed {key} shape: {value.shape}")
