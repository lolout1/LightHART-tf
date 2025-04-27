import numpy as np
from scipy.signal import butter, filtfilt
import os
import pandas as pd

def csvloader_tf(file_path, verbose=False):
    """Load CSV data with error handling"""
    if not os.path.exists(file_path):
        return None
        
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size < 100:
            return None
            
        # Read file data
        file_data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        
        if file_data.empty:
            return None
        
        # Determine columns to extract
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
        
        # Validate shape
        if file_data.shape[0] <= 2 or file_data.shape[1] < cols:
            return None
            
        # Extract data
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        
        # Handle NaN values
        if np.isnan(activity_data).any():
            activity_data = np.nan_to_num(activity_data, nan=0.0)
        
        # Check minimum length
        if len(activity_data) < 16:
            return None
            
        return activity_data
        
    except Exception as e:
        if verbose:
            print(f"Error loading {file_path}: {e}")
        return None

def butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4, verbose=False):
    """Apply Butterworth filter with error handling"""
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        if verbose:
            print(f"Filtering error: {e}")
        return data

def dtw_tf(x, y, verbose=False):
    """Dynamic Time Warping with error handling"""
    if len(x) < 10 or len(y) < 10:
        min_len = min(len(x), len(y))
        return [(i, i) for i in range(min_len)]
    
    try:
        # Subsample for large sequences
        max_length = 1000
        x_step = max(1, len(x) // max_length) if len(x) > max_length else 1
        y_step = max(1, len(y) // max_length) if len(y) > max_length else 1
        
        if x_step > 1 or y_step > 1:
            x_subsampled = x[::x_step]
            y_subsampled = y[::y_step]
        else:
            x_subsampled, y_subsampled = x, y
            
        # Calculate cost matrix
        x_len, y_len = len(x_subsampled), len(y_subsampled)
        cost = np.zeros((x_len+1, y_len+1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        for i in range(1, x_len+1):
            for j in range(1, y_len+1):
                dist = np.linalg.norm(x_subsampled[i-1] - y_subsampled[j-1])
                cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        # Backtracking
        i, j = x_len, y_len
        path = []
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            options = [cost[i-1, j-1], cost[i-1, j], cost[i, j-1]]
            min_idx = np.argmin(options)
            if min_idx == 0:
                i -= 1
                j -= 1
            elif min_idx == 1:
                i -= 1
            else:
                j -= 1
        
        path = path[::-1]
        
        # Scale back if subsampled
        if x_step > 1 or y_step > 1:
            path = [(min(i*x_step, len(x)-1), min(j*y_step, len(y)-1)) for i, j in path]
        
        return path
        
    except Exception as e:
        if verbose:
            print(f"DTW error: {e}")
        min_len = min(len(x), len(y))
        return [(i, i) for i in range(min_len)]

def align_sequence_tf(data, verbose=False):
    """Align sequences with error handling"""
    if 'skeleton' not in data or 'accelerometer' not in data:
        return data
    
    try:
        skl_data = data['skeleton']
        acc_data = data['accelerometer']
        
        if len(skl_data) == 0 or len(acc_data) == 0:
            return data
        
        # Extract joint data
        joint_id = 9
        try:
            if len(skl_data.shape) == 3:
                joint_data = skl_data[:, (joint_id-1)*3:joint_id*3]
            elif len(skl_data.shape) == 4:
                joint_data = skl_data[:, joint_id-1, :]
            else:
                return data
        except Exception:
            return data
        
        # Calculate norms
        skl_norm = np.linalg.norm(joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        # Perform DTW
        path = dtw_tf(acc_norm[:, np.newaxis], skl_norm[:, np.newaxis], verbose=verbose)
        
        # Extract indices
        acc_indices, skl_indices = [], []
        seen_acc, seen_skl = set(), set()
        
        for i, j in path:
            if i not in seen_acc:
                acc_indices.append(i)
                seen_acc.add(i)
            if j not in seen_skl:
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
        if verbose:
            print(f"Alignment error: {e}")
        return data

def sliding_window_tf(data, window_size, label, stride=None, verbose=False):
    """Extract sliding windows with error handling"""
    try:
        if not data or not any(k in data for k in ['accelerometer', 'skeleton']):
            return None
            
        if stride is None:
            stride = max(1, window_size // 4)
            
        windowed_data = {}
        
        # Process each modality
        for key, value in data.items():
            if key != 'labels' and len(value) >= window_size:
                windows = []
                # Use a safer approach to windows
                for i in range(0, max(1, len(value) - window_size + 1), stride):
                    windows.append(value[i:i+window_size])
                
                if windows:
                    windowed_data[key] = np.stack(windows)
                    if 'labels' not in windowed_data:
                        windowed_data['labels'] = np.full(len(windows), label, dtype=np.int32)
        
        if not windowed_data or 'labels' not in windowed_data:
            return None
            
        return windowed_data
        
    except Exception as e:
        if verbose:
            print(f"Windowing error: {e}")
        return None

def normalize_data_tf(data, verbose=False):
    """Normalize data with error handling"""
    from sklearn.preprocessing import StandardScaler
    
    normalized_data = {}
    
    # Keep labels as is
    if 'labels' in data:
        normalized_data['labels'] = data['labels']
    
    # Normalize each modality
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            try:
                # Save original shape
                orig_shape = value.shape
                
                # Reshape for normalization
                reshaped = value.reshape(-1, value.shape[-1])
                
                # Handle NaN/Inf values
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    reshaped = np.nan_to_num(reshaped)
                
                # Apply normalization
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                
                # Reshape back
                normalized_data[key] = normalized.reshape(orig_shape)
                
            except Exception as e:
                if verbose:
                    print(f"Error normalizing {key}: {e}")
                normalized_data[key] = value
    
    return normalized_data
