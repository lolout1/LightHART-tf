'''
Dataset Builder integrations - matches PyTorch implementation
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
                 stride: int = 10, use_dtw: bool = True, modalities: List[str] = ['skeleton']) -> Dict[str, np.ndarray]:
    '''
    Process a single trial - matches PyTorch implementation
    '''
    import logging
    if trial["subject_id"] not in subjects:
        return None

    # Determine label based on action_id (for fall detection)
    label = int(trial["action_id"] > 9)

    # Load data for requested modalities
    trial_data = {}
    available_modalities = []

    for modality in modalities:
        file_path = trial["files"].get(modality)
        if not file_path:
            logging.warning(f"No file path for {modality} in trial {trial['subject_id']}")
            continue
        try:
            unimodal_data = csvloader(file_path)
            if unimodal_data is None or len(unimodal_data) == 0:
                logging.warning(f"Empty data loaded for {modality} at {file_path}")
                continue
            if unimodal_data.shape[0] < 10:
                logging.warning(f"Sequence too short ({unimodal_data.shape[0]} frames) for {modality} at {file_path}")
                continue
            if modality == 'accelerometer':
                unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                if unimodal_data.shape[0] > 250:
                    unimodal_data = select_subwindow_pandas(unimodal_data)
            elif modality == 'skeleton':
                if len(unimodal_data.shape) == 2 and unimodal_data.shape[1] == 96:
                    frames = unimodal_data.shape[0]
                    unimodal_data = unimodal_data.reshape(frames, 32, 3)
            trial_data[modality] = unimodal_data
            available_modalities.append(modality)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")

    # Check if we have any required modalities
    if not available_modalities:
        logging.error(f"No valid modalities loaded for trial {trial['subject_id']}")
        return None

    # Apply DTW alignment if enabled and multiple modalities are present
    if use_dtw and len(available_modalities) > 1 and 'skeleton' in trial_data:
        try:
            trial_data = align_sequence(trial_data)
        except Exception as e:
            logging.error(f"Error in DTW alignment for trial {trial['subject_id']}: {e}")
            return None

    # Process each modality with appropriate windowing
    windowed_data = defaultdict(list)
    for modality in available_modalities:
        modality_data = {modality: trial_data[modality], 'labels': np.array([label])}
        if modality == 'skeleton':
            processed = sliding_window(
                data=modality_data,
                clearing_time_index=window_size-1,
                max_time=trial_data[modality].shape[0],
                sub_window_size=window_size,
                stride_size=stride,
                label=label
            )
        elif modality == 'accelerometer':
            processed = selective_sliding_window(
                data=modality_data,
                length=trial_data[modality].shape[0],
                window_size=window_size,
                stride_size=stride,
                height=1.4 if label == 1 else 1.2,
                distance=50 if label == 1 else 100,
                label=label,
                fuse=False
            )
        else:
            logging.warning(f"Unsupported modality {modality}, using sliding window")
            processed = sliding_window(
                data=modality_data,
                clearing_time_index=window_size-1,
                max_time=trial_data[modality].shape[0],
                sub_window_size=window_size,
                stride_size=stride,
                label=label
            )
        if not any(v.size > 0 for v in processed.values()):
            logging.warning(f"No valid windows generated for {modality} in trial {trial['subject_id']}")
            continue
        for key, value in processed.items():
            windowed_data[key].append(value)

    if not windowed_data:
        logging.error(f"No valid windows generated for trial {trial['subject_id']}")
        return None

    # Concatenate windows
    result = {}
    for key in windowed_data:
        try:
            result[key] = np.concatenate(windowed_data[key], axis=0)
        except Exception as e:
            logging.error(f"Error concatenating {key} for trial {trial['subject_id']}: {e}")
            result[key] = np.array([])

    return result

def normalize_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Normalize data - matches PyTorch implementation
    '''
    from sklearn.preprocessing import StandardScaler

    normalized_data = {}
    if 'labels' in data:
        normalized_data['labels'] = data['labels']

    for key, value in data.items():
        if key != 'labels' and value.size > 0:
            try:
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    reshaped = np.nan_to_num(reshaped)
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                normalized_data[key] = normalized.reshape(orig_shape)
            except Exception as e:
                logging.error(f"Error normalizing {key}: {e}")
                normalized_data[key] = value
    return normalized_data

def select_subwindow_pandas(unimodal_data: np.ndarray) -> np.ndarray:
    '''
    Select subwindow with highest variance - matches PyTorch implementation
    '''
    import pandas as pd
    n = len(unimodal_data)
    if n <= 250:
        return unimodal_data
    magnitude = np.linalg.norm(unimodal_data, axis=1)
    df = pd.DataFrame({"values": magnitude})
    df["variance"] = df["values"].rolling(window=125).var()
    max_idx = df["variance"].idxmax()
    final_start = max(0, max_idx - 100)
    final_end = min(n, max_idx + 100)
    return unimodal_data[final_start:final_end, :]
