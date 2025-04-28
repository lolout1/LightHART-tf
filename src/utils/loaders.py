import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging

def csvloader_tf(file_path, verbose=False):
    try:
        if not os.path.exists(file_path):
            return None
        
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
        
        if file_data.shape[0] <= 2:
            return None
            
        if file_data.shape[1] < cols:
            return None
        
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float64)
        
        if np.isnan(activity_data).any():
            activity_data = np.nan_to_num(activity_data)
        
        if len(activity_data) < 10:
            return None
            
        return activity_data
    except Exception as e:
        return None

def butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4, filter_type='low', verbose=False):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        return data

def dtw_tf(x, y, verbose=False, max_warping_window=50):
    if len(x) < 10 or len(y) < 10:
        min_len = min(len(x), len(y))
        return [(i, i) for i in range(min_len)]
    
    try:
        max_length = 500
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
        min_len = min(len(x), len(y))
        scale = len(y) / len(x) if len(x) > 0 else 1
        return [(i, min(int(i * scale), len(y)-1)) for i in range(min_len)]

def align_sequence_tf(data, verbose=False, use_dtw=True):
    if 'skeleton' not in data or 'accelerometer' not in data or not use_dtw:
        return data
    
    try:
        joint_id = 9
        skl_data = data['skeleton']
        acc_data = data['accelerometer']
        
        if len(skl_data) == 0 or len(acc_data) == 0:
            return data
        
        try:
            if len(skl_data.shape) == 3:
                joint_data = skl_data[:, (joint_id-1)*3:joint_id*3]
            elif len(skl_data.shape) == 4:
                joint_data = skl_data[:, :, joint_id-1, :]
            else:
                return data
        except Exception:
            joint_data = skl_data[:, :3] if len(skl_data.shape) == 3 else skl_data[:, 0, :]
        
        skl_norm = np.linalg.norm(joint_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        
        path = dtw_tf(acc_norm[:, np.newaxis], skl_norm[:, np.newaxis], verbose=verbose)
        
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
        
        aligned_data = {}
        for key in data:
            if key == 'accelerometer':
                aligned_data[key] = data[key][acc_indices]
            elif key == 'skeleton':
                aligned_data[key] = data[key][skl_indices]
            else:
                aligned_data[key] = data[key]
        
        min_len = min(len(aligned_data['accelerometer']), len(aligned_data['skeleton']))
        for key in aligned_data:
            if key in ['accelerometer', 'skeleton']:
                aligned_data[key] = aligned_data[key][:min_len]
        
        return aligned_data
    except Exception as e:
        return data

def normalize_data_tf(data, verbose=False):
    normalized_data = {}
    
    if 'labels' in data:
        normalized_data['labels'] = data['labels'].astype(np.int64)
    
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            try:
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    reshaped = np.nan_to_num(reshaped)
                
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                normalized_data[key] = normalized.reshape(orig_shape)
                
                if verbose: 
                    logging.info(f"Normalized {key}: orig mean={np.mean(reshaped):.4f}, norm mean={np.mean(normalized):.4f}")
            except Exception as e:
                normalized_data[key] = value
    
    return normalized_data
