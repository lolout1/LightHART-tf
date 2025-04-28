import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import tensorflow as tf
import logging

def csvloader(file_path, verbose=False):
    try:
        if not os.path.exists(file_path):
            if verbose: logging.warning(f"File not found: {file_path}")
            return None
        
        file_data = pd.read_csv(file_path, index_col=False, header=0)
        if file_data.shape[0] <= 2:
            return None
            
        if 'skeleton' in file_path:
            cols = 96  # 32 joints * 3 coordinates
        else:
            cols = 3  # x, y, z accelerometer
            
        if file_data.shape[1] < cols:
            return None
        
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        
        if np.isnan(activity_data).any():
            activity_data = np.nan_to_num(activity_data)
        
        if len(activity_data) < 16:
            return None
            
        return activity_data
    except Exception as e:
        if verbose: logging.error(f"Error loading {file_path}: {e}")
        return None

def butterworth_filter(data, cutoff=7.5, fs=25, order=4, filter_type='low'):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        return data

def reshape_skeleton(skeleton_data):
    """Reshape skeleton data to match the original PyTorch implementation"""
    if len(skeleton_data.shape) == 2:
        # Reshape from (frames, 96) to (frames, 32, 3)
        frames, features = skeleton_data.shape
        if features == 96:
            return skeleton_data.reshape(frames, 32, 3)
    return skeleton_data

def dtw_alignment(acc_data, skl_data, joint_id=9, max_warping_window=50):
    """DTW alignment between accelerometer and skeleton data"""
    try:
        # Reshape skeleton data if needed
        if len(skl_data.shape) == 2:
            skl_data = reshape_skeleton(skl_data)
        
        # Extract joint data - match original implementation
        if len(skl_data.shape) == 3:
            joint_data = skl_data[:, joint_id-1, :]
        else:
            return None, None
        
        # Calculate norms
        skl_norm = np.linalg.norm(joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        if len(skl_norm) < 10 or len(acc_norm) < 10:
            return None, None
        
        # Fast DTW implementation (simplified)
        n, m = len(acc_norm), len(skl_norm)
        dist = np.zeros((n+1, m+1))
        dist[1:, 0] = np.inf
        dist[0, 1:] = np.inf
        
        for i in range(1, n+1):
            window_start = max(1, i-max_warping_window) if max_warping_window else 1
            window_end = min(m+1, i+max_warping_window+1) if max_warping_window else m+1
            
            for j in range(window_start, window_end):
                cost = abs(acc_norm[i-1] - skl_norm[j-1])
                dist[i, j] = cost + min(dist[i-1, j], dist[i, j-1], dist[i-1, j-1])
        
        # Backtrack
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            options = [dist[i-1, j-1], dist[i-1, j], dist[i, j-1]]
            min_idx = np.argmin(options)
            if min_idx == 0: i -= 1; j -= 1
            elif min_idx == 1: i -= 1
            else: j -= 1
        
        path.reverse()
        
        # Filter unique indices
        acc_indices, skl_indices = [], []
        seen_acc, seen_skl = set(), set()
        
        for i, j in path:
            if i not in seen_acc and j not in seen_skl:
                acc_indices.append(i)
                seen_acc.add(i)
                skl_indices.append(j)
                seen_skl.add(j)
        
        if len(acc_indices) < 10 or len(skl_indices) < 10:
            return None, None
            
        return np.array(acc_indices), np.array(skl_indices)
    except Exception as e:
        return None, None

def align_sequences(data):
    """Align accelerometer and skeleton data using DTW"""
    if 'skeleton' not in data or 'accelerometer' not in data:
        return data
    
    acc_data = data['accelerometer']
    skl_data = data['skeleton']
    
    # Get alignment indices
    acc_indices, skl_indices = dtw_alignment(acc_data, skl_data)
    
    if acc_indices is None or skl_indices is None:
        return data
    
    # Apply indices to data
    aligned_data = {}
    for key in data:
        if key == 'accelerometer':
            aligned_data[key] = data[key][acc_indices]
        elif key == 'skeleton':
            aligned_data[key] = data[key][skl_indices]
        else:
            aligned_data[key] = data[key]
    
    # Ensure same length
    min_len = min(len(aligned_data['accelerometer']), len(aligned_data['skeleton']))
    for key in aligned_data:
        if key in ['accelerometer', 'skeleton']:
            aligned_data[key] = aligned_data[key][:min_len]
    
    return aligned_data

def sliding_window(data, window_size, stride=None, label=None):
    """Apply sliding window to data"""
    if not data:
        return None
        
    stride = stride or max(1, window_size // 4)
    windowed_data = {}
    
    # Process each modality
    for key, value in data.items():
        if key != 'labels' and len(value) >= window_size:
            windows = []
            for i in range(0, max(1, len(value) - window_size + 1), stride):
                window = value[i:i+window_size]
                if len(window) == window_size:
                    windows.append(window)
            
            if windows:
                windowed_data[key] = np.stack(windows)
    
    # Ensure consistent window counts across modalities
    window_counts = {k: len(v) for k, v in windowed_data.items()}
    if window_counts and len(set(window_counts.values())) > 1:
        min_count = min(window_counts.values())
        for key in windowed_data:
            windowed_data[key] = windowed_data[key][:min_count]
    
    # Add labels
    if label is not None and windowed_data:
        window_count = len(next(iter(windowed_data.values())))
        windowed_data['labels'] = np.full(window_count, label, dtype=np.int32)
    
    return windowed_data

def normalize_data(data):
    """Normalize data using StandardScaler"""
    normalized_data = {}
    
    # Preserve labels
    if 'labels' in data:
        normalized_data['labels'] = data['labels']
    
    # Normalize each modality
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            orig_shape = value.shape
            reshaped = value.reshape(-1, value.shape[-1])
            
            # Replace NaN/Inf values
            if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                reshaped = np.nan_to_num(reshaped)
            
            # Apply StandardScaler
            scaler = StandardScaler()
            normalized = scaler.fit_transform(reshaped)
            normalized_data[key] = normalized.reshape(orig_shape)
    
    return normalized_data

def process_trial(trial, subjects, window_size=64, stride=None):
    """Process a single trial from raw data to windowed, normalized data"""
    if trial["subject_id"] not in subjects:
        return None
    
    # Determine label based on action_id
    label = int(trial["action_id"] > 9)  # For fall detection
    
    # Load data for each modality
    trial_data = {}
    
    for modality, file_path in trial["files"].items():
        unimodal_data = csvloader(file_path)
        
        if unimodal_data is not None and len(unimodal_data) > 0:
            if modality == 'accelerometer':
                # Apply Butterworth filter
                unimodal_data = butterworth_filter(unimodal_data)
            elif modality == 'skeleton':
                # Reshape skeleton data
                unimodal_data = reshape_skeleton(unimodal_data)
            
            trial_data[modality] = unimodal_data
    
    # Check if we have both modalities
    if 'accelerometer' not in trial_data:
        return None
    
    # Align sequences if possible
    if 'skeleton' in trial_data:
        trial_data = align_sequences(trial_data)
    
    # Apply sliding window
    windowed_data = sliding_window(trial_data, window_size, stride, label)
    
    return windowed_data
