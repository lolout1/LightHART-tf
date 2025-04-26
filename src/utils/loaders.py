import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def csvloader_tf(file_path):
    """Load data from CSV files with various formats."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        if ';' in first_line:
            # Format with timestamp: "2022-08-04 15:39:30.932;-9.712467;-0.8350951;0.5192425"
            file_data = pd.read_csv(file_path, sep=';', header=None)
            # Skip the timestamp column (first column)
            activity_data = file_data.iloc[:, 1:4].to_numpy(dtype=np.float32)
        else:
            # Standard CSV format
            file_data = pd.read_csv(file_path, header=None)
            if 'skeleton' in file_path:
                cols = 96
            else:
                cols = 3
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                
        # Skip files that are too short
        if len(activity_data) < 16:  # Minimum length check
            print(f"Warning: File {file_path} is too short ({len(activity_data)} samples)")
            return None
            
        return activity_data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4):
    """Apply Butterworth filter to smooth data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def align_sequence_tf(data):
    """Align sequences across modalities to ensure same length."""
    min_length = float('inf')
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            min_length = min(min_length, len(value))
    
    if min_length < float('inf'):
        for key in data:
            if key != 'labels' and len(data[key]) > min_length:
                data[key] = data[key][:min_length]
    
    return data

def sliding_window_tf(data, window_size, label):
    """Extract sliding windows from time series data."""
    from collections import defaultdict
    windowed_data = defaultdict(list)
    stride = max(1, window_size // 4)
    
    for key, value in data.items():
        if key != 'labels' and len(value) >= window_size:
            windows = []
            for i in range(0, max(1, len(value) - window_size + 1), stride):
                windows.append(value[i:i+window_size])
            
            if windows:
                windowed_data[key] = np.stack(windows)
                if 'labels' not in windowed_data:
                    windowed_data['labels'] = np.full(len(windows), label, dtype=np.int32)
    
    return windowed_data

def normalize_data_tf(data):
    """Normalize data using StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    normalized_data = {}
    
    if 'labels' in data:
        normalized_data['labels'] = data['labels']
    
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            orig_shape = value.shape
            reshaped = value.reshape(-1, value.shape[-1])
            
            scaler = StandardScaler()
            normalized = scaler.fit_transform(reshaped)
            
            normalized_data[key] = normalized.reshape(orig_shape)
    
    return normalized_data
