import time
import numpy as np
from scipy.signal import butter, filtfilt
import os
import pandas as pd

def csvloader_tf(file_path, verbose=False):
    """Load data from CSV files with robust error handling."""
    if not os.path.exists(file_path):
        if verbose:
            print(f"File not found: {file_path}")
        return None
        
    try:
        # First, check if file is empty or corrupted
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # Very small files are likely corrupted
            if verbose:
                print(f"File too small, likely corrupted: {file_path} ({file_size} bytes)")
            return None
            
        # Try to read the file
        file_data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        
        if file_data.empty:
            if verbose:
                print(f"Empty file: {file_path}")
            return None
        
        # Extract data based on file type
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
        
        # Check if file has enough rows and columns
        if file_data.shape[0] <= 2 or file_data.shape[1] < cols:
            if verbose:
                print(f"Invalid data format: {file_path} - shape {file_data.shape}")
            return None
            
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        
        # Check for NaN values
        if np.isnan(activity_data).any():
            if verbose:
                print(f"NaN values in file: {file_path}")
            # Try to fill NaN values
            activity_data = np.nan_to_num(activity_data, nan=0.0)
        
        # Verify minimum length requirement
        if len(activity_data) < 16:
            if verbose:
                print(f"Data too short: {file_path} - length {len(activity_data)}")
            return None
            
        return activity_data
        
    except Exception as e:
        if verbose:
            print(f"Error loading {file_path}: {e}")
        return None

def butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4, verbose=False):
    """Apply Butterworth filter with error handling."""
    try:
        if verbose:
            start_time = time.time()
            print(f"Filtering data of shape {data.shape} with cutoff={cutoff}Hz")
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Filtering completed in {elapsed:.3f}s")
        
        return filtered_data
    except Exception as e:
        if verbose:
            print(f"Filtering error: {e}")
        return data  # Return original data if filtering fails

def dtw_tf(x, y, verbose=False):
    """DTW alignment with robust error handling."""
    if len(x) < 10 or len(y) < 10:
        if verbose:
            print(f"Sequences too short for DTW: {len(x)}, {len(y)}")
        return [(i, i) for i in range(min(len(x), len(y)))]
    
    try:
        if verbose:
            start_time = time.time()
            print(f"DTW alignment starting: sequences of shapes {x.shape} and {y.shape}")
        
        x_len = len(x)
        y_len = len(y)
        
        # For very large sequences, subsample to speed up DTW
        max_length = 1000
        if x_len > max_length or y_len > max_length:
            x_step = max(1, x_len // max_length)
            y_step = max(1, y_len // max_length)
            x = x[::x_step]
            y = y[::y_step]
            x_len = len(x)
            y_len = len(y)
            if verbose:
                print(f"Subsampled sequences to {x_len}, {y_len}")
        
        cost = np.zeros((x_len+1, y_len+1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        # DTW cost matrix calculation
        for i in range(1, x_len+1):
            for j in range(1, y_len+1):
                dist = np.linalg.norm(x[i-1] - y[j-1])
                cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        # Backtracking to find optimal path
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
        
        # If subsampled, scale indices back to original
        if x_step > 1 or y_step > 1:
            path = [(i*x_step, j*y_step) for i, j in path]
            # Ensure path doesn't exceed original sequence lengths
            path = [(min(i, x_len-1), min(j, y_len-1)) for i, j in path]
        
        if verbose:
            elapsed = time.time() - start_time
            compression = 100 * (1 - len(path) / max(x_len, y_len))
            print(f"DTW completed in {elapsed:.3f}s - path length: {len(path)}, compression: {compression:.1f}%")
        
        return path
        
    except Exception as e:
        if verbose:
            print(f"DTW error: {e}")
        # Return identity mapping as fallback
        min_len = min(len(x), len(y))
        return [(i, i) for i in range(min_len)]

def align_sequence_tf(data, verbose=False):
    """Align sequences with robust error handling."""
    if 'skeleton' not in data or 'accelerometer' not in data:
        return data
    
    try:
        if verbose:
            start_time = time.time()
            print(f"Starting sequence alignment for {list(data.keys())} modalities")
        
        skl_data = data['skeleton']
        acc_data = data['accelerometer']
        
        if len(skl_data) == 0 or len(acc_data) == 0:
            if verbose:
                print("Empty modality data, skipping alignment")
            return data
        
        # Extract joint data (left wrist, joint 9)
        joint_id = 9
        try:
            if len(skl_data.shape) == 3:  # Shape: [time, joints*3]
                joint_data = skl_data[:, (joint_id-1)*3:joint_id*3]
            elif len(skl_data.shape) == 4:  # Shape: [time, joints, 3]
                joint_data = skl_data[:, joint_id-1, :]
            else:
                if verbose:
                    print(f"Unexpected skeleton shape: {skl_data.shape}, skipping alignment")
                return data
        except Exception as e:
            if verbose:
                print(f"Error extracting joint data: {e}")
            return data
        
        # Calculate norms for DTW
        skl_norm = np.linalg.norm(joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        # Perform DTW
        path = dtw_tf(acc_norm[:, np.newaxis], skl_norm[:, np.newaxis], verbose=verbose)
        
        # Extract aligned indices
        acc_indices = [i for i, _ in path]
        skl_indices = [j for _, j in path]
        
        # Remove duplicates while preserving order
        acc_unique = []
        skl_unique = []
        seen_acc = set()
        seen_skl = set()
        
        for i, j in path:
            if i not in seen_acc:
                acc_unique.append(i)
                seen_acc.add(i)
            if j not in seen_skl:
                skl_unique.append(j)
                seen_skl.add(j)
        
        if len(acc_unique) < 10 or len(skl_unique) < 10:
            if verbose:
                print(f"Not enough unique indices after alignment: {len(acc_unique)}, {len(skl_unique)}")
            return data
        
        # Filter data
        aligned_data = {}
        for key in data:
            if key == 'accelerometer':
                aligned_data[key] = data[key][acc_unique]
            elif key == 'skeleton':
                aligned_data[key] = data[key][skl_unique]
            else:
                aligned_data[key] = data[key]
        
        # Ensure same length
        min_len = min(len(aligned_data['accelerometer']), len(aligned_data['skeleton']))
        for key in aligned_data:
            if key in ['accelerometer', 'skeleton']:
                aligned_data[key] = aligned_data[key][:min_len]
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Alignment completed in {elapsed:.3f}s, new length: {min_len}")
        
        return aligned_data
        
    except Exception as e:
        if verbose:
            print(f"Alignment error: {e}")
        return data

def sliding_window_tf(data, window_size, label, stride=None, verbose=False):
    """Extract sliding windows with error handling."""
    try:
        if not data or not any(k in data for k in ['accelerometer', 'skeleton']):
            return None
            
        if stride is None:
            stride = window_size // 4  # Default stride is 1/4 window size
            
        windowed_data = {}
        
        # Process each modality
        for key, value in data.items():
            if key != 'labels' and len(value) >= window_size:
                windows = []
                for i in range(0, max(1, len(value) - window_size + 1), stride):
                    windows.append(value[i:i+window_size])
                
                if windows:
                    windowed_data[key] = np.stack(windows)
                    if 'labels' not in windowed_data:
                        windowed_data['labels'] = np.full(len(windows), label, dtype=np.int32)
        
        if not windowed_data or len(windowed_data) < 2:  # Need at least one modality + labels
            if verbose:
                print(f"Not enough data after windowing, got {list(windowed_data.keys())}")
            return None
            
        return windowed_data
        
    except Exception as e:
        if verbose:
            print(f"Windowing error: {e}")
        return None

def normalize_data_tf(data, verbose=False):
    """Normalize data with error handling."""
    from sklearn.preprocessing import StandardScaler
    
    normalized_data = {}
    
    # Keep labels as is
    if 'labels' in data:
        normalized_data['labels'] = data['labels']
    
    # Normalize each modality
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            try:
                if verbose:
                    print(f"Normalizing {key} data of shape {value.shape}")
                
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                
                # Avoid NaN/Inf values
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    if verbose:
                        print(f"Warning: NaN/Inf values in {key}, replacing with zeros")
                    reshaped = np.nan_to_num(reshaped)
                
                # Normalize data
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                
                # Reshape back to original shape
                normalized_data[key] = normalized.reshape(orig_shape)
                
            except Exception as e:
                if verbose:
                    print(f"Error normalizing {key}: {e}")
                normalized_data[key] = value
    
    return normalized_data
