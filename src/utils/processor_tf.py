# src/utils/processor_tf.py
import numpy as np
import logging
from collections import defaultdict
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm
import tensorflow as tf

logger = logging.getLogger(__name__)


def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    """Apply Butterworth filter - matches PyTorch implementation"""
    try:
        if len(data) < 10:
            return data
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        logger.error(f"Error in butterworth filter: {e}")
        return data


def avg_pool_tf(sequence, max_length=128, window_size=5):
    """Average pooling using TensorFlow"""
    try:
        shape = sequence.shape
        if shape[0] <= max_length:
            return sequence
        
        # Reshape and transpose for pooling
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        
        # Calculate stride
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        
        # Apply average pooling
        sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
        
        # Convert back to numpy and reshape
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool_tf: {e}")
        return sequence[:max_length] if len(sequence) > max_length else sequence


def pad_sequence_tf(sequence, max_length):
    """Pad sequence to fixed length"""
    try:
        shape = list(sequence.shape)
        if shape[0] == max_length:
            return sequence
        
        if shape[0] > max_length:
            pooled_sequence = avg_pool_tf(sequence, max_length)
        else:
            pooled_sequence = sequence
        
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_tf: {e}")
        return sequence


def align_sequence_dtw(data, joint_id=9):
    """Align sequences using DTW - matches PyTorch implementation"""
    try:
        # Extract skeleton and accelerometer data
        if 'skeleton' not in data or 'accelerometer' not in data:
            logger.warning("Missing skeleton or accelerometer data for DTW")
            return data
        
        skeleton_joint_data = data['skeleton'][:, (joint_id-1)*3:joint_id*3]
        inertial_data = data['accelerometer']
        
        # Get other modalities
        dynamic_keys = sorted([key for key in data.keys() if key not in ["skeleton", "labels"]])
        
        # Handle multiple inertial sensors
        if len(dynamic_keys) > 1:
            gyroscope_data = data[dynamic_keys[1]]
            min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
            inertial_data = inertial_data[:min_len, :]
            data[dynamic_keys[1]] = gyroscope_data[:min_len, :]
        
        # Calculate Frobenius norms
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)
        
        # Run DTW
        distance, path = fastdtw(
            inertial_frob_norm[:, np.newaxis], 
            skeleton_frob_norm[:, np.newaxis], 
            dist=euclidean
        )
        
        # Filter repeated indices
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        
        # Apply alignment
        data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_ids))
        for key in dynamic_keys:
            data[key] = filter_data_by_ids(data[key], list(inertial_ids))
        
        return data
    
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        return data


def filter_repeated_ids(path):
    """Filter repeated indices from DTW path"""
    seen_first = set()
    seen_second = set()
    
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second


def filter_data_by_ids(data, ids):
    """Filter data by indices"""
    return data[ids, :]


def selective_sliding_window(data, window_size, label):
    """Selective sliding window using peak detection - matches PyTorch"""
    try:
        if 'accelerometer' not in data:
            return {'labels': np.array([])}
        
        # Calculate signal magnitude
        sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))
        
        # Set peak detection parameters based on label
        if label == 1:  # Fall
            height, distance = 1.4, 50  # Parameters from PyTorch
        else:  # Non-fall
            height, distance = 1.2, 100
        
        # Find peaks
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        if len(peaks) == 0:
            return {'labels': np.array([])}
        
        windowed_data = defaultdict(list)
        
        # Extract windows around peaks
        for modality, modality_data in data.items():
            if modality == 'labels':
                continue
                
            windows = []
            for peak in peaks:
                start = max(0, peak - window_size // 2)
                end = min(len(modality_data), start + window_size)
                
                # Adjust if window is too small
                if end - start < window_size:
                    start = max(0, end - window_size)
                
                if modality_data[start:end].shape[0] == window_size:
                    windows.append(modality_data[start:end])
            
            if windows:
                windowed_data[modality] = windows
        
        # Convert to numpy arrays
        for key in windowed_data:
            windowed_data[key] = np.array(windowed_data[key])
        
        # Add labels
        if windowed_data:
            num_windows = len(windowed_data[next(iter(windowed_data))])
            windowed_data['labels'] = np.repeat(label, num_windows)
        else:
            windowed_data['labels'] = np.array([])
        
        return dict(windowed_data)
    
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {e}")
        return {'labels': np.array([])}


def sliding_window(data, max_length, stride, label):
    """Standard sliding window implementation"""
    try:
        windowed_data = {}
        
        # Get reference length from first available modality
        reference_key = next((k for k in data.keys() if k != 'labels'), None)
        if not reference_key:
            return {'labels': np.array([])}
        
        # Process each modality
        for key, value in data.items():
            if key == 'labels':
                continue
            
            windows = []
            for start in range(0, len(value) - max_length + 1, stride):
                end = start + max_length
                window = value[start:end]
                if len(window) == max_length:
                    windows.append(window)
            
            if windows:
                windowed_data[key] = np.array(windows)
        
        # Add labels
        if windowed_data:
            num_windows = len(windowed_data[next(iter(windowed_data))])
            windowed_data['labels'] = np.repeat(label, num_windows)
        else:
            windowed_data['labels'] = np.array([])
        
        return windowed_data
    
    except Exception as e:
        logger.error(f"Error in sliding_window: {e}")
        return {'labels': np.array([])}
