#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_tf.py - TensorFlow dataset loader for SmartFallMM
Matches the PyTorch implementation exactly with robust error handling
"""

import os
import logging
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dataset-tf')

# Constants that match PyTorch implementation exactly
SAMPLING_RATE = 25  # Hz - Matching PyTorch version
JOINT_ID_FOR_ALIGNMENT = 9  # Left wrist joint for alignment
BUTTERWORTH_CUTOFF = 7.5  # Hz for low-pass filter
BUTTERWORTH_ORDER = 4  # Filter order

# ===== File Loading Functions =====

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load CSV sensor data with format matching PyTorch implementation.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments
    
    Returns:
        np.ndarray: Loaded data array
    """
    try:
        import pandas as pd
        
        # Determine if this is skeleton data
        is_skeleton = 'skeleton' in file_path.lower()
        
        # Try reading file
        try:
            # First attempt with default settings
            df = pd.read_csv(file_path, index_col=False, on_bad_lines='skip')
            
            # Check and clean up header rows
            if any(col.lower() in ['time', 'timestamp'] for col in df.columns) or df.iloc[0].astype(str).str.contains('time|timestamp', case=False).any():
                # Skip header rows
                df = df.iloc[1:]
            
            # Extract correct columns
            if is_skeleton:
                # For skeleton data, take the last 96 columns (32 joints x 3 coordinates)
                num_cols = df.shape[1]
                data_cols = 96 if num_cols >= 96 else num_cols
                activity_data = df.iloc[2:, -data_cols:].to_numpy(dtype=np.float32)
            else:
                # For accelerometer/gyroscope data (last 3 columns: x, y, z)
                activity_data = df.iloc[2:, -3:].to_numpy(dtype=np.float32)
        
        except Exception as e:
            # First approach failed, try alternative parsing
            logger.warning(f"Standard CSV parsing failed for {file_path}: {e}. Trying alternative approach.")
            
            # Check for semicolon delimiter
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    # Parse with semicolon delimiter
                    df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip')
                    
                    # Extract data - semicolon format typically has timestamp in column 0
                    if df.shape[1] >= 4:
                        activity_data = df.iloc[:, 1:4].astype(np.float32).to_numpy()
                    else:
                        logger.warning(f"Not enough columns in CSV: {df.shape[1]}")
                        # Return empty array to trigger fallback
                        return np.array([])
                else:
                    # Last attempt - try more aggressive skipping of initial rows
                    try:
                        df = pd.read_csv(file_path, index_col=False, header=None, on_bad_lines='skip')
                        df = df.iloc[2:]  # Skip first two rows regardless
                        
                        if is_skeleton:
                            if df.shape[1] >= 96:
                                activity_data = df.iloc[:, -96:].to_numpy(dtype=np.float32)
                            else:
                                activity_data = df.iloc[:, :].to_numpy(dtype=np.float32)
                        else:
                            if df.shape[1] >= 3:
                                activity_data = df.iloc[:, -3:].to_numpy(dtype=np.float32)
                            else:
                                # Not enough columns
                                logger.warning(f"CSV file has insufficient columns: {df.shape[1]}")
                                return np.array([])
                    except Exception as inner_e:
                        logger.error(f"All CSV parsing attempts failed for {file_path}: {inner_e}")
                        return np.array([])
        
        # Handle empty result
        if activity_data.size == 0 or len(activity_data) < 5:
            logger.warning(f"Empty or very small dataset from {file_path}")
            return np.array([])
        
        # Apply Butterworth filter to accelerometer/gyroscope data
        if not is_skeleton and len(activity_data) > 30:
            try:
                activity_data = butterworth_filter(activity_data, 
                                                  cutoff=BUTTERWORTH_CUTOFF, 
                                                  fs=SAMPLING_RATE)
            except Exception as e:
                logger.warning(f"Butterworth filtering failed for {file_path}: {e}")
        
        logger.info(f"Loaded {file_path}: shape={activity_data.shape}")
        return activity_data
            
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        return np.array([])

def matloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load MatLab files with same parameters as PyTorch implementation.
    
    Args:
        file_path: Path to MAT file
        **kwargs: Additional arguments including key for data extraction
        
    Returns:
        np.ndarray: Loaded data array
    """
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        
        if key not in ['d_iner', 'd_skel']:
            raise ValueError(f'Unsupported key {key} for matlab file')
        
        data = loadmat(file_path)[key]
        
        # Apply filtering for inertial data
        if key == 'd_iner' and data.shape[0] > 30:
            data = butterworth_filter(data, 
                                     cutoff=BUTTERWORTH_CUTOFF, 
                                     fs=SAMPLING_RATE)
        
        logger.info(f"Loaded MAT file {file_path}: shape={data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading MAT file {file_path}: {e}")
        return np.array([])

# Map file extensions to loader functions
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

# ===== Data Processing Functions =====

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to sensor data for noise reduction.
    Matches PyTorch implementation exactly with fs=25.
    
    Args:
        data: Input data array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz (MUST BE 25 to match PyTorch)
        order: Filter order
        
    Returns:
        np.ndarray: Filtered data
    """
    try:
        # Check if data is long enough for filtering
        min_samples = max(3 * order, 15)
        if len(data) <= min_samples:
            logger.warning(f"Sequence too short for filtering: {len(data)} samples")
            return data
        
        # Calculate Nyquist frequency and normalize cutoff
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Design and apply filter - Exactly like PyTorch implementation
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logger.warning(f"Butterworth filter error: {e}")
        return data  # Return original data if filtering fails

def avg_pool(sequence: np.ndarray, max_length: int, window_size: int = 5) -> np.ndarray:
    """
    Apply average pooling to sequence for length normalization.
    Matches PyTorch implementation's logic.
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
        window_size: Size of pooling window
        
    Returns:
        np.ndarray: Pooled sequence
    """
    try:
        shape = sequence.shape
        
        # Skip pooling if sequence is already shorter than max_length
        if shape[0] <= max_length:
            return sequence
        
        # Reshape for 1D convolution - match PyTorch implementation
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        
        # Convert to TensorFlow tensor
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        
        # Calculate appropriate stride to achieve desired length
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        
        # Apply pooling with same window size as PyTorch
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
        logger.error(f"Error in avg_pool: {e}")
        # Return truncated original sequence as fallback
        return sequence[:min(len(sequence), max_length)]

def pad_sequence_numpy(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """
    Pad sequence to a fixed length with pooling for longer sequences.
    Matches PyTorch implementation exactly.
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
        
    Returns:
        np.ndarray: Padded sequence
    """
    try:
        shape = list(sequence.shape)
        
        # If sequence is already the right length
        if shape[0] == max_length:
            return sequence
            
        # If sequence is longer, apply pooling
        if shape[0] > max_length:
            pooled_sequence = avg_pool(sequence=sequence, max_length=max_length)
        else:
            # If sequence is shorter, use original
            pooled_sequence = sequence
        
        # Create output array with target shape
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        
        # Copy pooled data (this preserves original sequence ordering)
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_numpy: {e}")
        
        # Create a valid fallback padded sequence
        dummy = np.zeros((max_length, *sequence.shape[1:]), dtype=sequence.dtype)
        dummy[:min(len(sequence), max_length)] = sequence[:min(len(sequence), max_length)]
        return dummy

def align_sequence(data: Dict[str, np.ndarray], use_dtw: bool = True) -> Dict[str, np.ndarray]:
    """
    Align accelerometer and skeleton data using joint 9 (left wrist).
    Uses Dynamic Time Warping for precise alignment exactly like PyTorch version.
    
    Args:
        data: Dictionary of modality data
        use_dtw: Whether to use DTW for alignment
        
    Returns:
        Dict[str, np.ndarray]: Aligned data
    """
    try:
        # Check if we have both required modalities
        has_acc = 'accelerometer' in data and len(data['accelerometer']) > 10
        has_skeleton = 'skeleton' in data and len(data['skeleton']) > 10
        
        if not (has_acc and has_skeleton):
            logger.warning(f"Missing required modalities for DTW alignment. "
                          f"accelerometer: {has_acc}, skeleton: {has_skeleton}")
            return data
            
        # Skip DTW if disabled
        if not use_dtw:
            # Simple length matching
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            
            # Truncate both to same length
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            
            logger.info(f"Basic alignment: truncated to {min_length} frames")
            return data

        # Extract joint 9 (left wrist) data from skeleton
        joint_id = JOINT_ID_FOR_ALIGNMENT
        skeleton_data = data['skeleton']
        
        # Handling skeleton format based on its shape
        if len(skeleton_data.shape) == 3:  # [frames, joints, coords]
            if skeleton_data.shape[1] >= joint_id and skeleton_data.shape[2] == 3:
                skeleton_joint_data = skeleton_data[:, joint_id-1, :]
            else:
                logger.warning(f"Skeleton joint {joint_id} not available in shape {skeleton_data.shape}")
                return data
        elif len(skeleton_data.shape) == 2:  # [frames, joints*coords]
            # This is the format in your logs: (140, 96)
            # We need to extract the joint_id coordinates
            if skeleton_data.shape[1] >= joint_id * 3:
                # Extract coordinates for joint_id (each joint has 3 coordinates)
                start_idx = (joint_id - 1) * 3
                end_idx = joint_id * 3
                skeleton_joint_data = skeleton_data[:, start_idx:end_idx]
            else:
                logger.warning(f"Skeleton data shape {skeleton_data.shape} doesn't have enough joints")
                return data
        else:
            logger.warning(f"Unexpected skeleton shape: {skeleton_data.shape}")
            return data
        
        # Get accelerometer data
        acc_data = data['accelerometer']
        
        # Ensure minimum length for DTW
        if len(skeleton_joint_data) < 20 or len(acc_data) < 20:
            logger.warning(f"Sequences too short for reliable DTW")
            
            # Fall back to simple alignment
            min_length = min(len(acc_data), len(skeleton_data))
            data['accelerometer'] = acc_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
            return data
        
        # Calculate magnitude vectors for alignment - exactly like PyTorch
        try:
            skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
            acc_norm = np.linalg.norm(acc_data, axis=1)
            
            # Apply DTW - matching PyTorch exactly
            distance, path = fastdtw(
                acc_norm[:, np.newaxis],  # Reshape for fastdtw
                skeleton_norm[:, np.newaxis],
                dist=euclidean,
                radius=15  # Same radius as PyTorch
            )
            
            # Extract unique indices for mapping - same logic as PyTorch
            acc_indices = set()
            skeleton_indices = set()
            
            for acc_idx, skeleton_idx in path:
                if acc_idx not in acc_indices and skeleton_idx not in skeleton_indices:
                    acc_indices.add(acc_idx)
                    skeleton_indices.add(skeleton_idx)
            
            # Convert to sorted lists for temporal ordering
            acc_indices = sorted(list(acc_indices))
            skeleton_indices = sorted(list(skeleton_indices))
            
            # Apply alignment if we have enough indices
            if len(acc_indices) > 20 and len(skeleton_indices) > 20:
                # Update data with aligned values
                data['accelerometer'] = data['accelerometer'][acc_indices]
                data['skeleton'] = data['skeleton'][skeleton_indices]
                
                logger.info(f"DTW alignment successful: mapped {len(acc_indices)} frames "
                           f"from {len(acc_data)} accelerometer to {len(skeleton_indices)} "
                           f"from {len(skeleton_data)} skeleton frames")
            else:
                logger.warning(f"DTW yielded too few matched indices")
                
                # Fall back to simple length matching
                min_length = min(len(acc_data), len(skeleton_data))
                data['accelerometer'] = acc_data[:min_length]
                data['skeleton'] = skeleton_data[:min_length]
        
        except Exception as e:
            logger.error(f"Error during DTW calculation: {e}")
            
            # Fall back to simple length matching
            min_length = min(len(acc_data), len(skeleton_data))
            data['accelerometer'] = acc_data[:min_length]
            data['skeleton'] = skeleton_data[:min_length]
        
        # Final check for consistency across modalities
        min_length = min(len(data['accelerometer']), len(data['skeleton']))
        data['accelerometer'] = data['accelerometer'][:min_length]
        data['skeleton'] = data['skeleton'][:min_length]
        
        return data
        
    except Exception as e:
        logger.error(f"Critical error in align_sequence: {e}")
        traceback.print_exc()
        
        # Return original data if alignment completely fails
        return data

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, 
                           height: float, distance: int, label: int) -> Dict[str, np.ndarray]:
    """
    Apply sliding window around detected peaks in signal.
    Matches PyTorch implementation exactly.
    
    Args:
        data: Dictionary of modality data
        window_size: Size of window
        height: Peak height threshold (1.4 for falls, 1.2 for non-falls)
        distance: Minimum distance between peaks (50 for falls, 100 for non-falls)
        label: Class label
        
    Returns:
        Dict[str, np.ndarray]: Windowed data
    """
    try:
        # Verify we have accelerometer data
        if 'accelerometer' not in data or len(data['accelerometer']) < window_size:
            logger.warning(f"Insufficient accelerometer data for windowing")
            
            # Create minimal valid output
            result = {'labels': np.array([label])}
            
            # Create dummy windows for each modality
            for key in data:
                if key != 'labels':
                    if key == 'accelerometer':
                        result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
                    elif key == 'skeleton':
                        result[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
                    else:
                        # Default shape for other modalities
                        result[key] = np.zeros((1, window_size, data[key].shape[1]), dtype=np.float32)
            
            return result
        
        # Calculate signal magnitude exactly as in PyTorch version
        acc_data = data['accelerometer']
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Find peaks with parameters matching PyTorch exactly
        # Using height and distance parameters passed from the calling function
        # For falls (label=1): height=1.4, distance=50
        # For non-falls (label=0): height=1.2, distance=100
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # If no peaks found, create a fallback window
        if len(peaks) == 0:
            logger.warning(f"No peaks found for label {label}. Creating central window.")
            
            # Create a window in the middle of the sequence
            mid_point = len(sqrt_sum) // 2
            peaks = [mid_point]
        
        # Create windows around each peak
        windowed_data = {}
        
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
                
                # Handle differently based on modality and shape
                if key == 'skeleton':
                    # Handle skeleton specifically
                    skeleton_data = data[key][start:end]
                    
                    # Check the shape and reshape if needed
                    if len(skeleton_data.shape) == 2 and skeleton_data.shape[1] == 96:
                        # This is in format [frames, joints*coords] - reshape to [frames, joints, coords]
                        # 96 = 32 joints * 3 coordinates
                        reshaped_data = skeleton_data.reshape(skeleton_data.shape[0], 32, 3)
                        window[:len(reshaped_data)] = reshaped_data
                    elif len(skeleton_data.shape) == 3:
                        # Already in format [frames, joints, coords]
                        window[:len(skeleton_data)] = skeleton_data
                    else:
                        logger.warning(f"Unexpected skeleton shape for windowing: {skeleton_data.shape}")
                        continue
                else:
                    # For other modalities (like accelerometer)
                    window_data = data[key][start:end]
                    window[:len(window_data)] = window_data
                
                windows.append(window)
            
            # Add windows to result if any were created
            if windows:
                try:
                    windowed_data[key] = np.stack(windows)
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
                    # Check for consistent shapes
                    shapes = [w.shape for w in windows]
                    if len(set(shapes)) > 1:
                        logger.error(f"Inconsistent window shapes: {shapes}")
                    continue
            else:
                # Create dummy window if none were valid
                if key == 'accelerometer':
                    shape = (1, window_size, 3)
                elif key == 'skeleton':
                    shape = (1, window_size, 32, 3)
                else:
                    shape = (1, window_size, data[key].shape[1])
                    
                windowed_data[key] = np.zeros(shape, dtype=np.float32)
                logger.warning(f"No valid windows for {key}, created dummy")
        
        # Add labels
        if windowed_data:
            # Use length of first modality for label count
            first_modality = next(iter(windowed_data.values()))
            windowed_data['labels'] = np.repeat(label, len(first_modality))
        else:
            # Fallback with single label
            windowed_data['labels'] = np.array([label])
            
            # Ensure all modalities have dummy data
            for key in data:
                if key != 'labels' and key not in windowed_data:
                    if key == 'accelerometer':
                        windowed_data[key] = np.zeros((1, window_size, 3), dtype=np.float32)
                    elif key == 'skeleton':
                        windowed_data[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
                    else:
                        windowed_data[key] = np.zeros((1, window_size, 3), dtype=np.float32)
        
        logger.info(f"Created {len(windowed_data['labels'])} windows for label {label}")
        return windowed_data
        
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {e}")
        traceback.print_exc()
        
        # Return minimal valid output
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

# ===== Dataset Classes =====

class ModalityFile:
    """Represents a file for a specific modality."""
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    """Container for files of a specific modality."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))

class MatchedTrial:
    """Container for matched files across modalities."""
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

class SmartFallMM:
    """Manager for SmartFall multimodal dataset."""
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {"old": {}, "young": {}}
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}
    
    def add_modality(self, age_group: str, modality_name: str) -> None:
        """Add a modality to track."""
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        self.age_groups[age_group][modality_name] = Modality(modality_name)
    
    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        """Select which sensor to use for each modality."""
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self) -> None:
        """Load files for all selected modalities."""
        total_files = 0
        
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                # Determine directory path based on modality
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name:
                        continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                
                if not os.path.exists(modality_dir):
                    logger.warning(f"Directory not found: {modality_dir}")
                    continue
                
                # Find and process files
                files_loaded = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith(('.csv', '.mat')):
                            try:
                                # Parse file name - format: S##A##T##.csv/mat
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                                files_loaded += 1
                            except Exception as e:
                                logger.warning(f"Error parsing file {file}: {e}")
                
                total_files += files_loaded
                logger.info(f"Loaded {files_loaded} files for {modality_name} in {age_group}")
        
        logger.info(f"Total files loaded: {total_files}")
    
    def match_trials(self, required_modalities=None) -> None:
        """Match files across modalities."""
        trial_dict = {}
        
        # Group files by (subject_id, action_id, sequence_number)
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path
        
        # Determine required modalities
        if required_modalities is None:
            # Default to requiring only accelerometer if available
            required_modalities = ['accelerometer']
        
        # Create matched trials
        complete_trials = []
        partial_trials = 0
        
        for key, files_dict in trial_dict.items():
            # Check if required modalities are present
            has_required = all(modality in files_dict for modality in required_modalities)
            
            if has_required:
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                
                complete_trials.append(matched_trial)
            else:
                partial_trials += 1
        
        self.matched_trials = complete_trials
        logger.info(f"Matched {len(complete_trials)} complete trials across modalities")
        logger.info(f"Skipped {partial_trials} incomplete trials lacking required modalities")
    
    def pipeline(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        """Run the full data pipeline."""
        # Add modalities for each age group
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                
                # Select sensors for each modality
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        # Load and match files
        self.load_files()
        self.match_trials(modalities)

class DatasetBuilder:
    """Builds datasets for training/validation/testing."""
    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.verbose = kwargs.get('verbose', False)
        self.use_dtw = kwargs.get('use_dtw', True)
        self.num_workers = kwargs.get('num_workers', 0)
        
        logger.info(f"DatasetBuilder initialized: mode={mode}, max_length={max_length}, use_dtw={self.use_dtw}")
    
    def load_file(self, file_path, **kwargs):
        """Load a file with the appropriate loader."""
        try:
            file_type = file_path.split('.')[-1].lower()
            
            if file_type not in LOADER_MAP:
                logger.error(f"Unsupported file type: {file_type}")
                return None
                
            loader = LOADER_MAP[file_type]
            return loader(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        """
        Build dataset for specified subjects with exact PyTorch logic.
        
        Args:
            subjects: List of subject IDs to include
            fuse: Whether to fuse inertial data (unused but kept for compatibility)
        """
        # Reset data
        self.data = defaultdict(list)
        
        # Get all matching trials for the specified subjects
        matching_trials = [t for t in self.dataset.matched_trials if t.subject_id in subjects]
        logger.info(f"Found {len(matching_trials)} trials for {len(subjects)} subjects")
        
        if not matching_trials:
            # Create empty dataset with minimal structure
            logger.warning("No matching trials found, creating dummy dataset")
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            return
        
        # Process each trial
        processed_count = 0
        for i, trial in enumerate(matching_trials):
            if i % 20 == 0 or i == len(matching_trials) - 1:
                logger.info(f"Processing trial {i+1}/{len(matching_trials)}")
            
            try:
                # Determine label based on task
                if self.task == 'fd':
                    label = int(trial.action_id > 9)  # Falls are actions 10+
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)  # Age classification
                else:
                    label = trial.action_id - 1  # Action recognition
                
                # Load data for each modality
                trial_data = defaultdict(np.ndarray)
                
                for modality, file_path in trial.files.items():
                    # Get key for MATLAB files
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys and modality.lower() in keys:
                        key = keys[modality.lower()]
                    
                    # Load the file
                    data = self.load_file(file_path, key=key)
                    
                    if data is not None and len(data) > 5:  # Skip very short sequences
                        trial_data[modality] = data
                
                # Skip if missing key modalities
                req_modalities = self.kwargs.get('required_modalities', ['accelerometer'])
                if not all(m in trial_data for m in req_modalities):
                    missing = [m for m in req_modalities if m not in trial_data]
                    logger.warning(f"Trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number} "
                                  f"missing required modalities: {missing}")
                    continue
                
                # Handle skeleton data format - ensure it's in the right shape
                if 'skeleton' in trial_data:
                    skeleton_data = trial_data['skeleton']
                    
                    # Check the shape and reshape if needed
                    if len(skeleton_data.shape) == 2 and skeleton_data.shape[1] == 96:
                        # This is in format [frames, joints*coords]
                        # Reshaped to [frames, joints, coords] format expected by the rest of the pipeline
                        trial_data['skeleton'] = skeleton_data.reshape(skeleton_data.shape[0], 32, 3)
                        logger.info(f"Reshaped skeleton data from {skeleton_data.shape} to {trial_data['skeleton'].shape}")
                
                # Apply DTW alignment if configured and multiple modalities present
                multi_modal = 'skeleton' in trial_data and 'accelerometer' in trial_data
                
                if multi_modal:
                    trial_data = align_sequence(trial_data, use_dtw=self.use_dtw)
                
                # Process data based on mode
                if self.mode == 'avg_pool':
                    # Simple padding/pooling
                    result = {}
                    for key, value in trial_data.items():
                        result[key] = pad_sequence_numpy(value, self.max_length)
                    result['labels'] = np.array([label])
                else:
                    # Selective sliding window with PyTorch parameters
                    if label == 1:  # Fall
                        result = selective_sliding_window(
                            trial_data, 
                            self.max_length, 
                            height=1.4,  # Exactly like PyTorch
                            distance=50,  # Exactly like PyTorch
                            label=label
                        )
                    else:  # Non-fall
                        result = selective_sliding_window(
                            trial_data, 
                            self.max_length, 
                            height=1.2,  # Exactly like PyTorch
                            distance=100,  # Exactly like PyTorch
                            label=label
                        )
                
                # Add to dataset if valid
                if result and len(result.get('labels', [])) > 0:
                    for key, value in result.items():
                        if len(value) > 0:
                            self.data[key].append(value)
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {e}")
                traceback.print_exc()
        
        # Concatenate data for each modality
        empty_keys = []
        for key in list(self.data.keys()):
            if len(self.data[key]) > 0:
                try:
                    # Check for consistent shapes before concatenation
                    shapes = [arr.shape for arr in self.data[key]]
                    if len(set(str(s) for s in shapes)) > 1:
                        logger.error(f"Inconsistent shapes for {key}: {shapes}")
                        
                        # Try to standardize shapes for skeleton
                        if key == 'skeleton':
                            standardized = []
                            for arr in self.data[key]:
                                if len(arr.shape) == 3 and arr.shape[1:] == (self.max_length, 96):
                                    # Convert from (N, max_length, 96) to (N, max_length, 32, 3)
                                    standardized.append(arr.reshape(arr.shape[0], arr.shape[1], 32, 3))
                                elif len(arr.shape) == 4 and arr.shape[1:] == (self.max_length, 32, 3):
                                    # Already in correct format
                                    standardized.append(arr)
                                else:
                                    logger.warning(f"Skipping incompatible array shape: {arr.shape}")
                            
                            if standardized:
                                self.data[key] = np.concatenate(standardized, axis=0)
                                logger.info(f"Standardized {key} shape: {self.data[key].shape}")
                            else:
                                empty_keys.append(key)
                        else:
                            # For other modalities, try best-effort concatenation
                            try:
                                self.data[key] = np.concatenate(self.data[key], axis=0)
                                logger.info(f"{key} shape: {self.data[key].shape}")
                            except Exception as concat_error:
                                logger.error(f"Concatenation failed for {key}: {concat_error}")
                                empty_keys.append(key)
                    else:
                        # Shapes are consistent, concatenate normally
                        self.data[key] = np.concatenate(self.data[key], axis=0)
                        logger.info(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    empty_keys.append(key)
            else:
                empty_keys.append(key)
        
        # Remove empty modalities
        for key in empty_keys:
            if key in self.data:
                del self.data[key]
        
        if processed_count == 0 or 'labels' not in self.data or len(self.data['labels']) == 0:
            logger.warning("No valid data processed, creating dummy dataset")
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            
            # Add skeleton if required
            if 'skeleton' in req_modalities:
                self.data['skeleton'] = np.zeros((1, self.max_length, 32, 3), dtype=np.float32)
    
    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Normalize data for each modality using StandardScaler.
        Matches PyTorch implementation exactly.
        
        Returns:
            Dict[str, np.ndarray]: Normalized data
        """
        try:
            has_data = False
            
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    has_data = True
                    
                    if len(value.shape) >= 2:  # Must be at least 2D
                        # Standardize each modality
                        num_samples = value.shape[0]
                        
                        # Reshape based on modality
                        if key == 'skeleton' and len(value.shape) == 4:
                            # Special handling for skeleton data [samples, frames, joints, coords]
                            orig_shape = value.shape
                            reshaped = value.reshape(num_samples * value.shape[1], -1)
                        else:
                            # Standard handling for other modalities
                            orig_shape = value.shape
                            reshaped = value.reshape(num_samples * value.shape[1], -1)
                        
                        # Apply StandardScaler
                        try:
                            scaler = StandardScaler()
                            norm_data = scaler.fit_transform(reshaped)
                            
                            # Reshape back to original dimensions
                            self.data[key] = norm_data.reshape(orig_shape)
                            
                            logger.info(f"Normalized {key}: shape={self.data[key].shape}, "
                                       f"range=[{self.data[key].min():.2f}, {self.data[key].max():.2f}]")
                        except Exception as e:
                            logger.warning(f"Normalization failed for {key}: {e}. Using original data.")
            
            if not has_data:
                logger.warning("No data to normalize")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            traceback.print_exc()
            # Return unnormalized data
            return self.data

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow dataset for multimodal data with batching support."""
    
    def __init__(self, dataset, batch_size, use_smv=False):
        """
        Initialize the dataset.
        
        Args:
            dataset: Dictionary containing modality data and labels
            batch_size: Batch size for training
            use_smv: Whether to use Signal Magnitude Vector
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        
        # Extract data from dataset dictionary
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        # Handle missing data with appropriate logging
        if self.acc_data is None or len(self.acc_data) == 0:
            logger.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
        
        # Process skeleton data
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                # Shape: [frames, joints*3]
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    # Reshape to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) == 4:
                # Already in format [batch, frames, joints, 3]
                self.skl_seq, self.skl_length, self.skl_joints, self.skl_dims = self.skl_data.shape
        else:
            # Create dummy skeleton data if needed
            logger.warning("No skeleton data in dataset, creating dummy data")
            self.skl_data = np.zeros((self.num_samples, self.acc_seq, 32, 3), dtype=np.float32)
        
        # Handle missing labels
        if self.labels is None or len(self.labels) == 0:
            logger.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Prepare data for TensorFlow
        self._prepare_data()
        self.indices = np.arange(self.num_samples)
        
        logger.info(f"Initialized UTD_MM_TF with {self.num_samples} samples")
    
    def _prepare_data(self):
        """Prepare data for TensorFlow processing."""
        try:
            # Convert to TensorFlow tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate SMV if requested
            if self.use_smv:
                self._calc_smv()
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
    
    def _calc_smv(self):
        """Calculate Signal Magnitude Vector for accelerometer data."""
        try:
            # Calculate and save SMV as in PyTorch implementation
            mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
            zero_mean = self.acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            self.smv = tf.sqrt(sum_squared)
            logger.info(f"SMV calculated with shape: {self.smv.shape}")
        except Exception as e:
            logger.error(f"Error calculating SMV: {e}")
            self.smv = None
    
    def __len__(self):
        """Return the number of batches."""
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        """Get a batch of data."""
        try:
            # Get indices for this batch
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            # Gather data for this batch
            batch_data = {}
            batch_acc = tf.gather(self.acc_data, tf_indices)
            
            # Add SMV if requested
            if self.use_smv:
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, tf_indices)
                else:
                    # Calculate on the fly if needed
                    mean = tf.reduce_mean(batch_acc, axis=1, keepdims=True)
                    zero_mean = batch_acc - mean
                    sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                    batch_smv = tf.sqrt(sum_squared)
                
                # Concatenate SMV with accelerometer data
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            # Add skeleton data
            batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            
            # Get labels
            batch_labels = tf.gather(self.labels, tf_indices)
            
            # Log shapes for the first batch
            if idx == 0:
                for key, value in batch_data.items():
                    logger.info(f"First batch {key} shape: {value.shape}")
                logger.info(f"First batch labels shape: {batch_labels.shape}")
                logger.info(f"First batch indices shape: {tf_indices.shape}")
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logger.error(f"Error in batch generation {idx}: {e}")
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.acc_seq, 32, 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = tf.range(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        """Called at the end of each epoch to shuffle data."""
        np.random.shuffle(self.indices)

# ===== Helper Functions for SmartFallMM Dataset =====

def prepare_smartfallmm_tf(arg) -> DatasetBuilder:
    """
    Prepare SmartFall dataset builder using command line arguments.
    
    Args:
        arg: Arguments containing dataset configuration
        
    Returns:
        DatasetBuilder: Configured dataset builder
    """
    # Find data directory
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    if not os.path.exists(data_dir):
        # Try relative path from parent directory
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm')
    
    if not os.path.exists(data_dir):
        logger.warning(f"SmartFall data directory not found at {data_dir}")
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    # Get configuration parameters with defaults
    age_group = arg.dataset_args.get('age_group', ['young'])
    modalities = arg.dataset_args.get('modalities', ['accelerometer'])
    sensors = arg.dataset_args.get('sensors', ['watch'])
    
    # Initialize dataset
    sm_dataset = SmartFallMM(root_dir=data_dir)
    
    # Run data pipeline
    sm_dataset.pipeline(age_group=age_group, modalities=modalities, sensors=sensors)
    
    # Create dataset builder
    builder_kwargs = {
        'verbose': arg.dataset_args.get('verbose', False),
        'use_dtw': arg.dataset_args.get('use_dtw', True),
        'num_workers': getattr(arg, 'num_worker', 0),
        'required_modalities': ['accelerometer']  # Only require accelerometer for flexibility
    }
    
    builder = DatasetBuilder(
        sm_dataset, 
        arg.dataset_args.get('mode', 'sliding_window'), 
        arg.dataset_args.get('max_length', 128),
        arg.dataset_args.get('task', 'fd'),
        **builder_kwargs
    )
    
    return builder

def split_by_subjects_tf(builder: DatasetBuilder, subjects: List[int], fuse: bool) -> Dict[str, np.ndarray]:
    """
    Split dataset by subjects.
    
    Args:
        builder: Dataset builder
        subjects: List of subject IDs to include
        fuse: Whether to fuse inertial data
        
    Returns:
        Dict[str, np.ndarray]: Dataset split by subjects
    """
    try:
        # Build dataset for specified subjects
        builder.make_dataset(subjects, fuse)
        
        # Normalize data
        norm_data = builder.normalization()
        
        # Validate data and ensure consistency
        modalities_to_check = ['accelerometer', 'skeleton']
        has_valid_data = False
        
        for key in modalities_to_check:
            if key in norm_data and norm_data[key] is not None and len(norm_data[key]) > 0:
                has_valid_data = True
                if np.isnan(norm_data[key]).any():
                    logger.warning(f"NaN values detected in {key}, replacing with zeros")
                    norm_data[key] = np.nan_to_num(norm_data[key])
        
        # Ensure labels exist and match data length
        if 'labels' not in norm_data or len(norm_data['labels']) == 0:
            # Use acceleration data to determine label count if available
            if 'accelerometer' in norm_data and len(norm_data['accelerometer']) > 0:
                logger.warning("Creating dummy labels based on accelerometer data length")
                norm_data['labels'] = np.zeros(len(norm_data['accelerometer']), dtype=np.int32)
            else:
                # Fallback to minimal labels
                norm_data['labels'] = np.array([0], dtype=np.int32)
        
        # Create dummy skeleton if needed
        if 'skeleton' not in norm_data and 'accelerometer' in norm_data:
            logger.warning("No skeleton data, creating dummy")
            acc_shape = norm_data['accelerometer'].shape
            norm_data['skeleton'] = np.zeros((acc_shape[0], acc_shape[1], 32, 3), dtype=np.float32)
        
        # Final check for empty datasets
        if not has_valid_data:
            logger.warning("No valid data, creating dummy dataset")
            norm_data = {
                'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, builder.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
        
        return norm_data
        
    except Exception as e:
        logger.error(f"Error in split_by_subjects_tf: {e}")
        traceback.print_exc()
        
        # Return minimal valid dataset as fallback
        return {
            'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32),
            'skeleton': np.zeros((1, builder.max_length, 32, 3), dtype=np.float32),
            'labels': np.zeros(1, dtype=np.int32)
        }
