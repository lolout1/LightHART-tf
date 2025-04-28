import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import tensorflow as tf
import logging

def csvloader_tf(file_path, verbose=False):
    try:
        if not os.path.exists(file_path):
            if verbose: logging.warning(f"File not found: {file_path}")
            return None
        
        # Match LightHART's approach - similar to utils/processor/base.py
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        
        # Determine columns to select
        if 'skeleton' in file_path:
            cols = 96  # For skeleton
        else:
            cols = 3   # For accelerometer/gyroscope
        
        # Skip if insufficient data
        if file_data.shape[0] <= 2:
            if verbose: logging.warning(f"Insufficient data rows in {file_path}: {file_data.shape}")
            return None
            
        if file_data.shape[1] < cols:
            if verbose: logging.warning(f"Invalid data shape in {file_path}: {file_data.shape}")
            return None
        
        # Extract activity data - matching LightHART implementation
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float64)
        
        # Handle NaN values
        if np.isnan(activity_data).any():
            if verbose: logging.warning(f"NaN values in {file_path}")
            activity_data = np.nan_to_num(activity_data)
        
        if len(activity_data) < 16:
            if verbose: logging.warning(f"Insufficient data in {file_path}: {len(activity_data)} rows")
            return None
            
        return activity_data
    except Exception as e:
        if verbose: logging.error(f"Error loading {file_path}: {e}")
        return None

def butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4, filter_type='low', verbose=False):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        if verbose: logging.error(f"Filtering error: {e}")
        return data

def dtw_tf(x, y, verbose=True, max_warping_window=50):
    if len(x) < 10 or len(y) < 10:
        min_len = min(len(x), len(y))
        return [(i, i) for i in range(min_len)]
    
    try:
        # Subsample for large sequences
        max_length = 1000
        x_step = max(1, len(x) // max_length) if len(x) > max_length else 1
        y_step = max(1, len(y) // max_length) if len(y) > max_length else 1
        
        x_subsampled = x[::x_step] if x_step > 1 else x
        y_subsampled = y[::y_step] if y_step > 1 else y
        
        x_len, y_len = len(x_subsampled), len(y_subsampled)
        cost = np.ones((x_len+1, y_len+1)) * np.inf
        cost[0, 0] = 0
        
        for i in range(1, x_len+1):
            window_start = max(1, i - max_warping_window) if max_warping_window else 1
            window_end = min(y_len+1, i + max_warping_window + 1) if max_warping_window else y_len+1
            
            for j in range(window_start, window_end):
                dist = np.linalg.norm(x_subsampled[i-1] - y_subsampled[j-1])
                cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        i, j = x_len, y_len
        path = []
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            options = [cost[i-1, j-1], cost[i-1, j], cost[i, j-1]]
            min_idx = np.argmin(options)
            if min_idx == 0: i -= 1; j -= 1
            elif min_idx == 1: i -= 1
            else: j -= 1
        
        path.reverse()
        
        if x_step > 1 or y_step > 1:
            path = [(min(i*x_step, len(x)-1), min(j*y_step, len(y)-1)) for i, j in path]
        
        return path
    except Exception as e:
        if verbose: logging.error(f"DTW error: {e}")
        min_len = min(len(x), len(y))
        scale = len(y) / len(x) if len(x) > 0 else 1
        return [(i, min(int(i * scale), len(y)-1)) for i in range(min_len)]

def align_sequence_tf(data, verbose=False, use_dtw=True):
    if 'skeleton' not in data or 'accelerometer' not in data or not use_dtw:
        return data
    
    try:
        joint_id = 9  # Same as in LightHART
        skl_data = data['skeleton']
        acc_data = data['accelerometer']
        
        if len(skl_data) == 0 or len(acc_data) == 0:
            return data
        
        try:
            if len(skl_data.shape) == 3:
                joint_data = skl_data[:, (joint_id-1)*3:joint_id*3]
            elif len(skl_data.shape) == 4:
                joint_data = skl_data[:, joint_id-1, :]
            else:
                return data
        except Exception:
            joint_data = skl_data[:, :3] if len(skl_data.shape) == 3 else skl_data[:, 0, :]
        
        # Calculate norms (matching LightHART implementation)
        skl_norm = np.linalg.norm(joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        # Perform DTW
        path = dtw_tf(acc_norm[:, np.newaxis], skl_norm[:, np.newaxis], verbose=verbose)
        
        # Extract unique indices (matching filter_repeated_ids in LightHART)
        acc_indices, skl_indices = [], []
        seen_acc, seen_skl = set(), set()
        
        for i, j in path:
            if i not in seen_acc and j not in seen_skl:
                acc_indices.append(i)
                seen_acc.add(i)
                skl_indices.append(j)
                seen_skl.add(j)
        
        if len(acc_indices) < 10 or len(skl_indices) < 10:
            return data
        
        # Filter data
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
    except Exception as e:
        if verbose: logging.error(f"Alignment error: {e}")
        return data

def sliding_window_tf(data, window_size, label=None, stride=None, verbose=False, min_windows=1):
    try:
        if not data or not any(k in data for k in ['accelerometer', 'skeleton']):
            if verbose: logging.warning("No valid data for sliding window")
            return None
        
        if stride is None:
            stride = max(1, window_size // 4)  # Default stride
        
        windowed_data = {}
        window_counts = {}
        
        for key, value in data.items():
            if key != 'labels' and len(value) >= window_size:
                windows = []
                valid_windows = 0
                total_attempts = 0
                
                # Create sliding windows (matching LightHART implementation)
                for i in range(0, max(1, len(value) - window_size + 1), stride):
                    window = value[i:i+window_size]
                    total_attempts += 1
                    
                    if len(window) == window_size:
                        # Replace NaN/Inf with zeros
                        if np.isnan(window).any() or np.isinf(window).any():
                            window = np.nan_to_num(window)
                        
                        windows.append(window)
                        valid_windows += 1
                
                if len(windows) >= min_windows:
                    try:
                        windowed_data[key] = np.stack(windows)
                        window_counts[key] = valid_windows
                    except Exception as e:
                        if verbose: logging.error(f"Error stacking windows for {key}: {e}")
                        continue
                elif verbose:
                    logging.warning(f"No valid windows for {key}")
        
        if not windowed_data:
            if verbose: logging.warning("No windowed data created")
            return None
        
        # Ensure all modalities have same number of windows
        window_sizes = set(window_counts.values())
        if len(window_sizes) > 1:
            min_count = min(window_counts.values())
            for key in windowed_data:
                windowed_data[key] = windowed_data[key][:min_count]
        
        # Add labels (using int64)
        if label is not None:
            window_count = len(next(iter(windowed_data.values())))
            windowed_data['labels'] = np.full(window_count, label, dtype=np.int64)
        
        return windowed_data
    except Exception as e:
        if verbose: logging.error(f"Windowing error: {e}")
        return None

def normalize_data_tf(data, verbose=False):
    normalized_data = {}
    
    if 'labels' in data:
        normalized_data['labels'] = data['labels'].astype(np.int64)  # Ensure int64
    
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            try:
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                
                # Replace NaN/Inf values
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    reshaped = np.nan_to_num(reshaped)
                
                # Apply StandardScaler (matching LightHART)
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                normalized_data[key] = normalized.reshape(orig_shape)
                
                if verbose: 
                    logging.info(f"Normalized {key}: orig mean={np.mean(reshaped):.4f}, norm mean={np.mean(normalized):.4f}")
            except Exception as e:
                if verbose: logging.error(f"Error normalizing {key}: {e}")
                normalized_data[key] = value
    
    return normalized_data

def _process_trial(args):
    from utils.loaders import csvloader_tf, butterworth_filter_tf, align_sequence_tf, sliding_window_tf
    
    trial, subjects, builder_args, verbose, use_dtw = args
    
    if trial["subject_id"] not in subjects:
        return None
    
    try:
        # Determine label based on task (matching LightHART)
        if builder_args and 'task' in builder_args:
            task = builder_args['task']
            if task == 'fd':
                label = int(trial["action_id"] > 9)
            elif task == 'age':
                label = int(trial["subject_id"] < 29 or trial["subject_id"] > 46)
            else:
                label = trial["action_id"] - 1
        else:
            label = trial["action_id"] - 1
        
        # Load data for each modality
        trial_data = defaultdict(np.ndarray)
        required_modalities = builder_args.get('modalities', []) if builder_args else []
        loaded_modalities = set()
        
        for modality, file_path in trial["files"].items():
            if modality in required_modalities or not required_modalities:
                try:
                    unimodal_data = csvloader_tf(file_path, verbose=verbose)
                    
                    if unimodal_data is not None and len(unimodal_data) > 0:
                        if modality == 'accelerometer':
                            # Apply Butterworth filter with parameters matching LightHART
                            unimodal_data = butterworth_filter_tf(unimodal_data, cutoff=7.5, fs=25, order=4, verbose=verbose)
                        
                        trial_data[modality] = unimodal_data
                        loaded_modalities.add(modality)
                    else:
                        if verbose: logging.warning(f"No valid data loaded from {file_path}")
                except Exception as e:
                    if verbose: logging.error(f"Error loading {modality} file {file_path}: {e}")
                    continue
        
        # Check required modalities
        missing = set(required_modalities) - loaded_modalities
        if missing and verbose:
            logging.warning(f"Missing modalities for trial {trial['subject_id']}-{trial['action_id']}-{trial['sequence_number']}: {missing}")
            if 'accelerometer' not in loaded_modalities:
                return None
        
        # Apply DTW alignment
        if use_dtw and 'skeleton' in trial_data and 'accelerometer' in trial_data:
            trial_data = align_sequence_tf(trial_data, verbose=verbose, use_dtw=use_dtw)
        
        if 'accelerometer' not in trial_data or len(trial_data['accelerometer']) < 10:
            if verbose: logging.warning(f"Insufficient accelerometer data after alignment")
            return None
        
        # Apply windowing based on mode
        if builder_args and 'mode' in builder_args:
            mode = builder_args['mode']
            max_length = builder_args.get('max_length', 64)
            
            if mode == 'sliding_window':
                stride = max(1, max_length // 4)  # Match LightHART default stride
                processed_data = sliding_window_tf(trial_data, max_length, label, stride=stride, verbose=verbose)
            else:
                # For avg_pool mode
                processed_data = {}
                for k, v in trial_data.items():
                    if len(v) > max_length:
                        processed_data[k] = v[:max_length]
                    else:
                        # Pad shorter sequences
                        padded = np.zeros((max_length,) + v.shape[1:], dtype=v.dtype)
                        padded[:len(v)] = v
                        processed_data[k] = padded
                processed_data['labels'] = np.array([label], dtype=np.int64)
        else:
            processed_data = sliding_window_tf(trial_data, 64, label, verbose=verbose)
        
        # Final validation
        if not processed_data or 'labels' not in processed_data or len(processed_data['labels']) == 0:
            if verbose: logging.warning(f"No processed data created for trial {trial['subject_id']}-{trial['action_id']}-{trial['sequence_number']}")
            return None
        
        return processed_data
    except Exception as e:
        if verbose: logging.error(f"Error processing trial for subject {trial['subject_id']}: {e}")
        return None
