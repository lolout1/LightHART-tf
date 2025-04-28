# utils/loaders.py
'''
Dataset Builder integrations - exact match to PyTorch implementation
'''
import os
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from utils.dataset_tf import (
    csvloader, matloader, butterworth_filter, align_sequence,
    selective_sliding_window, sliding_window, pad_sequence_numpy
)

# Re-export key functions for compatibility
LOADER_MAP = {
    'csv': csvloader, 
    'mat': matloader
}

def process_trial(trial: Dict[str, Any], subjects: List[int], window_size: int = 64, 
                 stride: int = None, use_dtw: bool = True) -> Dict[str, np.ndarray]:
    '''
    Process a single trial - compatible with PyTorch implementation
    '''
    if trial["subject_id"] not in subjects:
        return None
    
    # Determine label based on action_id (for fall detection)
    label = int(trial["action_id"] > 9)
    
    # Load data for each modality
    trial_data = {}
    
    for modality, file_path in trial["files"].items():
        try:
            unimodal_data = csvloader(file_path)
            
            if unimodal_data is None or len(unimodal_data) == 0:
                continue
                
            if modality == 'accelerometer':
                # Apply Butterworth filter
                unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
            elif modality == 'skeleton':
                # Reshape skeleton data if needed
                if len(unimodal_data.shape) == 2 and unimodal_data.shape[1] == 96:
                    frames = unimodal_data.shape[0]
                    unimodal_data = unimodal_data.reshape(frames, 32, 3)
            
            trial_data[modality] = unimodal_data
        except Exception as e:
            import logging
            logging.error(f"Error loading {file_path}: {e}")
    
    # Check if we have all required modalities
    if not all(modality in trial_data for modality in ['accelerometer', 'skeleton']):
        return None
    
    # Apply DTW alignment if enabled
    if use_dtw:
        try:
            trial_data = align_sequence(trial_data)
        except Exception as e:
            import logging
            logging.error(f"Error in DTW alignment: {e}")
            return None
    
    # Apply selective sliding window based on label
    if label == 1:  # Fall
        windowed_data = selective_sliding_window(
            data=trial_data,
            length=trial_data['skeleton'].shape[0],
            window_size=window_size,
            stride_size=10,
            height=1.4,
            distance=50,
            label=label,
            fuse=False  # Set to True if using gyroscope
        )
    else:  # Non-fall
        windowed_data = selective_sliding_window(
            data=trial_data,
            length=trial_data['skeleton'].shape[0],
            window_size=window_size,
            stride_size=10,
            height=1.2,
            distance=100,
            label=label,
            fuse=False  # Set to True if using gyroscope
        )
    
    return windowed_data

def normalize_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Normalize data - exact match to PyTorch implementation
    '''
    from sklearn.preprocessing import StandardScaler
    
    normalized_data = {}
    
    # Preserve labels
    if 'labels' in data:
        normalized_data['labels'] = data['labels']
    
    # Normalize each modality
    for key, value in data.items():
        if key != 'labels' and len(value) > 0:
            try:
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                
                # Handle NaN and Inf values
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    reshaped = np.nan_to_num(reshaped)
                
                # Apply StandardScaler
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                normalized_data[key] = normalized.reshape(orig_shape)
            except Exception as e:
                import logging
                logging.error(f"Error normalizing {key}: {e}")
                normalized_data[key] = value  # Use original data if normalization fails
    
    return normalized_data

