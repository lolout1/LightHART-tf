#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
processor_tf.py - TensorFlow implementation of signal processing functions
Matches PyTorch implementation exactly with robust error handling
"""

import tensorflow as tf
import numpy as np
import logging
from scipy.signal import butter, filtfilt, find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

def align_sequence_dtw(data, joint_id=9, use_dtw=True):
    """
    Align skeleton and accelerometer data using DTW
    
    Args:
        data: Dictionary with 'skeleton' and 'accelerometer' arrays
        joint_id: Joint ID to use for alignment (default: 9 = left wrist)
        use_dtw: Whether to use DTW alignment
        
    Returns:
        Dictionary with aligned data
    """
    try:
        # Check if we have both required modalities
        if 'skeleton' not in data or 'accelerometer' not in data:
            logger.warning("Missing required modalities for DTW alignment")
            return data
        
        # Skip DTW if disabled
        if not use_dtw:
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        # Extract joint data - handle different skeleton formats
        skeleton_data = data['skeleton']
        
        # Handle different skeleton formats
        if len(skeleton_data.shape) == 3:  # [frames, joints*3]
            start_idx = (joint_id - 1) * 3
            end_idx = joint_id * 3
            skeleton_joint_data = skeleton_data[:, start_idx:end_idx]
        elif len(skeleton_data.shape) == 4:  # [frames, joints, 3]
            skeleton_joint_data = skeleton_data[:, joint_id-1, :]
        else:
            logger.warning(f"Unexpected skeleton shape: {skeleton_data.shape}")
            return data
        
        # Get accelerometer data
        acc_data = data['accelerometer']
        
        # Calculate magnitude vectors for alignment
        skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        # Apply DTW - exactly like PyTorch implementation
        try:
            distance, path = fastdtw(
                acc_norm[:, np.newaxis],
                skeleton_norm[:, np.newaxis],
                dist=euclidean,
                radius=15  # Same as PyTorch implementation
            )
        except Exception as e:
            logger.error(f"DTW calculation error: {e}")
            # Simple alignment fallback
            min_length = min(len(acc_data), len(skeleton_data))
            data['accelerometer'] = acc_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
            return data
        
        # Extract unique indices for mapping - exactly like PyTorch implementation
        acc_indices = set()
        skeleton_indices = set()
        
        for acc_idx, skl_idx in path:
            if acc_idx not in acc_indices and skl_idx not in skeleton_indices:
                acc_indices.add(acc_idx)
                skeleton_indices.add(skl_idx)
        
        # Convert to sorted lists for temporal ordering
        acc_indices = sorted(list(acc_indices))
        skeleton_indices = sorted(list(skeleton_indices))
        
        # Apply alignment if we have enough indices
        if len(acc_indices) > 10 and len(skeleton_indices) > 10:
            data['accelerometer'] = data['accelerometer'][acc_indices]
            data['skeleton'] = data['skeleton'][skeleton_indices]
            
            logger.info(f"DTW alignment successful: mapped {len(acc_indices)} accelerometer frames to "
                        f"{len(skeleton_indices)} skeleton frames")
        else:
            logger.warning(f"DTW alignment yielded too few matched indices")
            
            # Fall back to simple length matching
            min_length = min(len(acc_data), len(skeleton_data))
            data['accelerometer'] = acc_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
        
        # Ensure consistent lengths
        min_length = min(len(data['accelerometer']), len(data['skeleton']))
        data['accelerometer'] = data['accelerometer'][:min_length]
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

def selective_windowing(data, window_size, label):
    """
    Apply selective windowing around detected peaks in accelerometer data
    
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
                        result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
                    elif key == 'skeleton':
                        if len(data[key].shape) == 3:  # [frames, joints*3]
                            result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
                        else:  # [frames, joints, 3]
                            result[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
                    else:
                        result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
            
            return result
        
        # Calculate signal magnitude
        acc_data = data['accelerometer']
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Set parameters based on label
        # Identical parameters to PyTorch implementation
        if label == 1:  # Fall
            height, distance = 1.4, 50
        else:  # Non-fall
            height, distance = 1.2, 100
        
        # Find peaks
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # Create windows around each peak
        if len(peaks) == 0:
            # Create a window at the middle point
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
                
                # Create and prepare window
                window = np.zeros((window_size, *data[key].shape[1:]), dtype=data[key].dtype)
                
                # Fill with data
                window[:end-start] = data[key][start:end]
                windows.append(window)
            
            # Add windows to result if any were created
            if windows:
                try:
                    windows_dict[key] = np.stack(windows)
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
                    # Create single dummy window as fallback
                    windows_dict[key] = np.zeros((1, window_size, *data[key].shape[1:]), dtype=data[key].dtype)
            else:
                # Create dummy window if none were valid
                windows_dict[key] = np.zeros((1, window_size, *data[key].shape[1:]), dtype=data[key].dtype)
        
        # Add labels
        if windows_dict:
            # Use length of first modality for label count
            first_key = next(iter(windows_dict))
            windows_dict['labels'] = np.full(len(windows_dict[first_key]), label, dtype=np.int32)
        else:
            # Fallback with single label
            windows_dict['labels'] = np.array([label], dtype=np.int32)
            
            # Ensure all modalities have dummy data
            for key in data:
                if key != 'labels' and key not in windows_dict:
                    if key == 'accelerometer':
                        windows_dict[key] = np.zeros((1, window_size, 3), dtype=np.float32)
                    elif key == 'skeleton':
                        windows_dict[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
        
        logger.info(f"Created {len(windows_dict['labels'])} windows for label {label}")
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
