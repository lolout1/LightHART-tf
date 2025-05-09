#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_tf.py - TensorFlow dataset loader for SmartFallMM
Handles loading, preprocessing, aligning, and windowing multimodal sensor data
"""

import os
import logging
import traceback
from typing import List, Dict, Tuple, Any
import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dataset-tf')

# Constants for data processing
SAMPLING_RATE = {
    'skeleton': 30,          # 30 Hz for skeleton data
    'accelerometer': 50,     # Variable, but approximately 50 Hz
    'gyroscope': 50,         # Variable, but approximately 50 Hz
    'meta': 50               # 50 Hz for meta sensors
}
JOINT_ID_FOR_ALIGNMENT = 9  # Left wrist joint for alignment

# ===== File Loading Functions =====

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load CSV sensor data with support for different formats.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments
    
    Returns:
        np.ndarray: Loaded data array
    """
    import pandas as pd
    try:
        # Determine file type
        is_skeleton = 'skeleton' in file_path
        
        # Read first few lines to determine format
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            has_timestamp = first_line.startswith('202') or ';' in first_line
            has_semicolon = ';' in first_line
        
        if has_semicolon:
            # Watch/phone format: timestamp;x;y;z
            df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip')
            
            # Skip timestamp column (first column)
            # Check if the file is empty or has too few columns
            if df.shape[1] < 4:
                logger.warning(f"CSV file has fewer than 4 columns: {file_path}")
                return np.zeros((10, 3), dtype=np.float32)
                
            acc_data = df.iloc[:, 1:4].astype(float).to_numpy(dtype=np.float32)
        else:
            # Standard CSV format with different column counts
            df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
            
            if is_skeleton:
                # Skeleton data should have 96 columns (32 joints * 3 coordinates)
                cols = 96
            else:
                # Inertial sensor data has 3 columns (x, y, z)
                cols = 3
            
            # Handle different data structures and skip header rows
            if df.shape[1] >= cols:
                # Skip first 2 rows (headers) and take the last 'cols' columns
                acc_data = df.iloc[2:, -cols:].astype(float).to_numpy(dtype=np.float32)
            else:
                logger.warning(f"CSV file has fewer columns than expected: {file_path}")
                acc_data = np.zeros((10, cols), dtype=np.float32)
        
        # Apply Butterworth filter for inertial data
        if not is_skeleton and acc_data.shape[0] > 30:
            try:
                acc_data = butterworth_filter(acc_data, cutoff=7.5, fs=50, order=4)
            except Exception as e:
                logger.warning(f"Filtering failed: {e}. Using unfiltered data.")
        
        return acc_data
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        # Return dummy data instead of None to avoid breaking the pipeline
        return np.zeros((10, 3 if not is_skeleton else 96), dtype=np.float32)

def matloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load MatLab files with support for different data types.
    
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
            try:
                data = butterworth_filter(data, cutoff=7.5, fs=50, order=4)
            except Exception as e:
                logger.warning(f"Filtering failed: {e}. Using unfiltered data.")
        
        return data
    except Exception as e:
        logger.error(f"Error loading MAT file {file_path}: {e}")
        # Return dummy data
        return np.zeros((10, 3), dtype=np.float32)

# Map file extensions to loader functions
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

# ===== Data Processing Functions =====

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 50, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to sensor data for noise reduction.
    
    Args:
        data: Input data array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        np.ndarray: Filtered data
    """
    try:
        # Check if data is long enough for filtering
        min_samples = max(3 * order, 15)
        if len(data) <= min_samples:
            logger.warning(f"Sequence too short for filtering: {len(data)} samples, need > {min_samples}")
            return data
        
        # Calculate Nyquist frequency and normalize cutoff
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Design and apply filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logger.warning(f"Butterworth filter error: {e}")
        return data  # Return original data if filtering fails

def avg_pool(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """
    Apply average pooling to sequence for length normalization.
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
        
    Returns:
        np.ndarray: Pooled sequence
    """
    try:
        shape = sequence.shape
        
        # Skip pooling if sequence is already shorter than max_length
        if shape[0] <= max_length:
            return sequence
        
        # Reshape for 1D convolution
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        
        # Convert to TensorFlow tensor
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        
        # Calculate appropriate stride to achieve desired length
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        
        # Apply pooling
        sequence = tf.nn.avg_pool1d(sequence, ksize=5, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        
        # Reshape to original format
        sequence = sequence.reshape(-1, *shape[1:])
        
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool: {e}")
        # Return truncated original sequence as fallback
        return sequence[:min(len(sequence), max_length)]

def pad_sequence_numpy(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """
    Pad sequence to a fixed length with pooling for longer sequences.
    
    Args:
        sequence: Input sequence data
        max_length: Target sequence length
        
    Returns:
        np.ndarray: Padded sequence
    """
    try:
        shape = list(sequence.shape)
        
        # If sequence is already longer than max_length, truncate it
        if shape[0] > max_length:
            return sequence[:max_length]
        
        # Apply pooling if needed
        pooled_sequence = avg_pool(sequence=sequence, max_length=max_length)
        
        # Create output array
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        
        # Copy pooled data
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_numpy: {e}")
        # Create an empty padded sequence as fallback
        dummy = np.zeros((max_length, *sequence.shape[1:]), dtype=sequence.dtype)
        dummy[:min(len(sequence), max_length)] = sequence[:min(len(sequence), max_length)]
        return dummy

def align_sequence(data: Dict[str, np.ndarray], use_dtw: bool = True) -> Dict[str, np.ndarray]:
    """
    Align accelerometer and skeleton data using joint 9 (left wrist).
    Uses Dynamic Time Warping for more accurate alignment.
    
    Args:
        data: Dictionary of modality data
        use_dtw: Whether to use DTW for alignment
        
    Returns:
        Dict[str, np.ndarray]: Aligned data
    """
    try:
        # Check if we have both skeleton and inertial data
        if "skeleton" not in data or not any(k in data for k in ["accelerometer", "gyroscope"]):
            return data

        # Skip DTW if disabled
        if not use_dtw:
            # Simple length matching
            min_length = float('inf')
            for key, value in data.items():
                if key != 'labels' and len(value) > 0:
                    min_length = min(min_length, len(value))
            
            if min_length < float('inf'):
                for key in data:
                    if key != 'labels' and len(data[key]) > min_length:
                        data[key] = data[key][:min_length]
            return data

        # Proceed with DTW alignment using joint 9 (left wrist)
        joint_id = JOINT_ID_FOR_ALIGNMENT  # Using joint 9 as specified
        inertial_key = "accelerometer" if "accelerometer" in data else "gyroscope"
        
        # Extract skeleton joint data
        skeleton_data = data['skeleton']
        if len(skeleton_data.shape) == 4:  # [batch, frames, joints, coords]
            skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
        elif len(skeleton_data.shape) == 3:  # [frames, joints, coords] or [frames, joints*coords]
            if skeleton_data.shape[2] >= joint_id * 3:
                # Handle flattened joint data (x,y,z for each joint consecutively)
                skeleton_joint_data = skeleton_data[:, :, (joint_id-1)*3:joint_id*3]
            else:
                # Fallback to available data
                skeleton_joint_data = skeleton_data[:, :, min(joint_id-1, skeleton_data.shape[2]-1)]
        else:
            # Invalid shape, return original data
            logger.warning(f"Invalid skeleton data shape: {skeleton_data.shape}")
            return data
        
        # Get inertial data
        inertial_data = data[inertial_key]
        
        # Handle multiple inertial sensor data (if both accelerometer and gyroscope are present)
        if "gyroscope" in data and inertial_key == "accelerometer":
            gyroscope_data = data["gyroscope"]
            min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
            inertial_data = inertial_data[:min_len, :]
            data["gyroscope"] = gyroscope_data[:min_len, :]
        
        # Ensure data is long enough for alignment
        if len(skeleton_joint_data) < 5 or len(inertial_data) < 5:
            logger.warning("Sequences too short for DTW alignment")
            return data
        
        # Calculate norms for alignment
        skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        inertial_norm = np.linalg.norm(inertial_data, axis=1)
        
        try:
            # Apply FastDTW for sequence alignment
            distance, path = fastdtw(
                inertial_norm[:, np.newaxis], 
                skeleton_norm[:, np.newaxis],
                dist=euclidean,
                radius=15  # Use larger radius for better alignment
            )
            
            # Extract matched indices
            inertial_ids = set()
            skeleton_ids = set()
            
            for i, j in path:
                if i not in inertial_ids and j not in skeleton_ids:
                    inertial_ids.add(i)
                    skeleton_ids.add(j)
            
            # Convert to sorted lists
            inertial_ids = sorted(list(inertial_ids))
            skeleton_ids = sorted(list(skeleton_ids))
            
            # Apply alignment if we have enough matched indices
            if len(inertial_ids) > 5 and len(skeleton_ids) > 5:
                data['skeleton'] = data['skeleton'][skeleton_ids]
                for key in [k for k in data.keys() if k != 'skeleton' and k != 'labels']:
                    data[key] = data[key][inertial_ids]
                
                # Ensure consistent lengths after DTW
                min_length = min(len(data['skeleton']), len(data[inertial_key]))
                for key in data:
                    if key != 'labels' and len(data[key]) > min_length:
                        data[key] = data[key][:min_length]
                
                logger.info(f"DTW alignment successful: {len(inertial_ids)} inertial frames matched to {len(skeleton_ids)} skeleton frames")
            
        except Exception as e:
            logger.warning(f"DTW alignment failed: {e}. Using simple length matching.")
            # Fallback to simple length matching
            min_length = min(skeleton_data.shape[0], inertial_data.shape[0])
            for key in data:
                if key != 'labels' and len(data[key]) > min_length:
                    data[key] = data[key][:min_length]
        
        return data
    except Exception as e:
        logger.error(f"Error in align_sequence: {e}")
        # Return original data if alignment fails
        return data

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, 
                             height: float, distance: int, label: int) -> Dict[str, np.ndarray]:
    """
    Apply sliding window around peaks in the signal for segmentation.
    
    Args:
        data: Dictionary of modality data
        window_size: Size of window
        height: Peak height threshold
        distance: Minimum distance between peaks
        label: Class label
        
    Returns:
        Dict[str, np.ndarray]: Windowed data
    """
    try:
        # Extract accelerometer data for peak detection
        if 'accelerometer' not in data:
            # Return empty result if no accelerometer data
            return {k: np.array([]) for k in data.keys()}
        
        acc_data = data['accelerometer']
        
        # Calculate signal magnitude
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Find peaks
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # Create sliding windows around peaks
        windowed_data = defaultdict(list)
        
        for modality, modality_data in data.items():
            if modality == 'labels':
                continue
            
            windows = []
            for peak in peaks:
                start = max(0, peak - window_size//2)
                end = min(len(modality_data), start + window_size)
                
                # Skip if window is too small
                if end - start < window_size:
                    continue
                
                windows.append(modality_data[start:end])
            
            if windows:
                windowed_data[modality] = np.stack(windows)
                
        # Add labels
        if windowed_data:
            sample_modality = next(iter(windowed_data.keys()))
            windowed_data['labels'] = np.repeat(label, len(windowed_data[sample_modality]))
            logger.info(f"Created {len(windowed_data['labels'])} windows for label {label}")
        else:
            # Ensure we return something
            windowed_data['labels'] = np.array([label])
            for key in data:
                if key != 'labels':
                    windowed_data[key] = np.zeros((1, window_size, *data[key].shape[1:]), dtype=data[key].dtype)
            
        return windowed_data
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {e}")
        traceback.print_exc()
        
        # Return empty dict with label
        result = {'labels': np.array([label])}
        for key in data:
            if key != 'labels':
                result[key] = np.zeros((1, window_size, *data[key].shape[1:]), dtype=np.float32)
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
                        if file.endswith('.csv') or file.endswith('.mat'):
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
        
        # Set default required modalities if not specified
        if required_modalities is None:
            required_modalities = ['accelerometer']  # Only require accelerometer as minimum
        
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
        
        logger.info(f"DatasetBuilder initialized with mode={mode}, max_length={max_length}, task={task}")
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        """
        Build dataset for specified subjects.
        
        Args:
            subjects: List of subject IDs to include
            fuse: Whether to fuse inertial data
        """
        self.data = defaultdict(list)
        
        # Get all matching trials for the specified subjects
        matching_trials = [t for t in self.dataset.matched_trials if t.subject_id in subjects]
        logger.info(f"Found {len(matching_trials)} matching trials for {len(subjects)} subjects")
        
        if not matching_trials:
            # Create empty dataset with minimal structure
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            return
        
        # Process trials
        if self.num_workers > 1:
            self._process_trials_parallel(matching_trials, fuse)
        else:
            self._process_trials_sequential(matching_trials, fuse)
        
        # Concatenate data for each modality
        self._concatenate_modalities()
    
    def _process_trials_parallel(self, trials: List[MatchedTrial], fuse: bool) -> None:
        """Process trials in parallel using multiprocessing."""
        # Prepare arguments for parallel processing
        args_list = [(t, self.mode, self.max_length, self.task, self.use_dtw, self.kwargs) for t in trials]
        
        # Use multiprocessing pool
        with multiprocessing.Pool(processes=min(self.num_workers, multiprocessing.cpu_count())) as pool:
            results = pool.map(process_trial_file, args_list)
        
        # Filter out None results and process valid ones
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully processed {len(valid_results)} trials out of {len(trials)}")
        
        # Collect all modalities
        all_modalities = set()
        for result in valid_results:
            all_modalities.update(result.keys())
        all_modalities.discard('labels')
        
        # Initialize data containers for each modality
        for modality in all_modalities:
            self.data[modality] = []
        
        self.data['labels'] = []
        
        # Collect data from all valid results
        for result in valid_results:
            # Add all available modalities
            for modality in all_modalities:
                if modality in result and modality != 'labels':
                    self.data[modality].append(result[modality])
            
            # Add labels
            if 'labels' in result:
                self.data['labels'].append(result['labels'])
    
    def _process_trials_sequential(self, trials: List[MatchedTrial], fuse: bool) -> None:
        """Process trials sequentially."""
        processed_count = 0
        
        for i, trial in enumerate(trials):
            if i % 10 == 0:
                logger.info(f"Processing trial {i+1}/{len(trials)}")
            
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
                    file_type = file_path.split('.')[-1]
                    if file_type not in ['csv', 'mat']:
                        continue
                    
                    loader = LOADER_MAP[file_type]
                    
                    # Get key for MATLAB files
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys and modality.lower() in keys:
                        key = keys[modality.lower()]
                    
                    # Load the file
                    data = loader(file_path, key=key)
                    
                    if data is not None and len(data) > 5:  # Skip very short sequences
                        trial_data[modality] = data
                
                # Skip trial if missing accelerometer
                if 'accelerometer' not in trial_data or len(trial_data['accelerometer']) == 0:
                    continue
                
                # Apply DTW alignment if both modalities present
                has_skeleton = 'skeleton' in trial_data and len(trial_data['skeleton']) > 0
                
                if has_skeleton and self.use_dtw:
                    trial_data = align_sequence(trial_data, use_dtw=True)
                else:
                    # Simple alignment for accelerometer-only
                    trial_data = align_sequence(trial_data, use_dtw=False)
                
                # Process the data based on mode
                result = None
                
                if self.mode == 'avg_pool':
                    # Use average pooling
                    result = {}
                    for key, value in trial_data.items():
                        if key != 'labels':
                            result[key] = pad_sequence_numpy(value, self.max_length)
                    result['labels'] = np.array([label])
                else:
                    # Apply selective sliding window
                    if label == 1:  # Fall
                        result = selective_sliding_window(
                            data=trial_data,
                            window_size=self.max_length,
                            height=1.4,  # Higher threshold for falls
                            distance=50,  # Smaller distance for falls
                            label=label
                        )
                    else:  # Non-fall
                        result = selective_sliding_window(
                            data=trial_data,
                            window_size=self.max_length,
                            height=1.2,  # Lower threshold for non-falls
                            distance=100,  # Larger distance for non-falls
                            label=label
                        )
                
                # Add processed data to dataset
                if result and all(len(v) > 0 for k, v in result.items() if k != 'labels'):
                    for key, value in result.items():
                        self.data[key].append(value)
                    processed_count += 1
            
            except Exception as e:
                logger.error(f"Error processing trial {trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {e}")
                traceback.print_exc()
        
        logger.info(f"Successfully processed {processed_count} trials out of {len(trials)}")
    
    def _concatenate_modalities(self) -> None:
        """Concatenate data for all modalities."""
        for key in list(self.data.keys()):
            if len(self.data[key]) > 0:
                try:
                    # Concatenate data arrays
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                    
                    if self.verbose:
                        logger.info(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    del self.data[key]
        
        # Check if we have labels
        if 'labels' not in self.data or len(self.data['labels']) == 0:
            # Create labels based on first modality
            primary_key = next((k for k in self.data if k != 'labels'), None)
            if primary_key:
                self.data['labels'] = np.zeros(len(self.data[primary_key]), dtype=np.int32)
                logger.warning(f"Created dummy labels for {len(self.data[primary_key])} samples")
    
    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Normalize data for each modality.
        
        Returns:
            Dict[str, np.ndarray]: Normalized data
        """
        try:
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    if len(value.shape) >= 2:  # Must be at least 2D
                        num_samples = value.shape[0]
                        
                        # Reshape for normalization
                        orig_shape = value.shape
                        reshaped = value.reshape(num_samples * value.shape[1], -1)
                        
                        # Apply StandardScaler
                        scaler = StandardScaler()
                        try:
                            norm_data = scaler.fit_transform(reshaped)
                            
                            # Reshape back to original dimensions
                            self.data[key] = norm_data.reshape(orig_shape)
                            
                            logger.info(f"Normalized {key}: min={self.data[key].min():.2f}, max={self.data[key].max():.2f}")
                        except Exception as e:
                            logger.warning(f"Normalization failed for {key}: {e}. Using original data.")
            
            return self.data
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            # Return unnormalized data
            return self.data

def process_trial_file(args):
    """
    Process a single trial - used for parallel processing.
    
    Args:
        args: Tuple of arguments (trial, mode, max_length, task, use_dtw, kwargs)
    
    Returns:
        Dict[str, np.ndarray]: Processed trial data
    """
    trial, mode, max_length, task, use_dtw, kwargs = args
    
    try:
        # Determine label based on task
        if task == 'fd':
            label = int(trial.action_id > 9)  # Fall detection (>9 = fall)
        elif task == 'age':
            label = int(trial.subject_id < 29 or trial.subject_id > 46)  # Age classification
        else:
            label = trial.action_id - 1  # Action recognition
        
        # Load data for each modality
        trial_data = defaultdict(np.ndarray)
        
        for modality, file_path in trial.files.items():
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                continue
            
            loader = LOADER_MAP[file_type]
            
            # Get key for MATLAB files
            keys = kwargs.get('keys', None)
            key = None
            if keys and modality.lower() in keys:
                key = keys[modality.lower()]
            
            # Load the file
            data = loader(file_path, key=key)
            
            if data is not None and len(data) > 5:  # Skip very short sequences
                trial_data[modality] = data
                
                # Apply Butterworth filter to accelerometer
                if modality == 'accelerometer' and len(data) > 30:
                    try:
                        trial_data[modality] = butterworth_filter(data, cutoff=7.5, fs=50)
                    except Exception:
                        pass  # Keep original if filtering fails
        
        # Skip trial if missing accelerometer
        if 'accelerometer' not in trial_data or len(trial_data['accelerometer']) == 0:
            return None
        
        # Apply DTW alignment if both modalities present
        has_skeleton = 'skeleton' in trial_data and len(trial_data['skeleton']) > 0
        
        if has_skeleton and use_dtw:
            trial_data = align_sequence(trial_data, use_dtw=True)
        else:
            # Simple alignment for accelerometer-only
            trial_data = align_sequence(trial_data, use_dtw=False)
        
        # Process the data based on mode
        if mode == 'avg_pool':
            # Use average pooling
            result = {}
            for key, value in trial_data.items():
                if key != 'labels':
                    result[key] = pad_sequence_numpy(value, max_length)
            result['labels'] = np.array([label])
        else:
            # Apply selective sliding window
            if label == 1:  # Fall
                result = selective_sliding_window(
                    data=trial_data,
                    window_size=max_length,
                    height=1.4,  # Higher threshold for falls
                    distance=50,  # Smaller distance for falls
                    label=label
                )
            else:  # Non-fall
                result = selective_sliding_window(
                    data=trial_data,
                    window_size=max_length,
                    height=1.2,  # Lower threshold for non-falls
                    distance=100,  # Larger distance for non-falls
                    label=label
                )
        
        # Verify we have valid data
        if any(key != 'labels' and (result[key] is None or len(result[key]) == 0) for key in result):
            return None
        
        return result
    except Exception as e:
        logger.error(f"Error processing trial: {e}")
        return None

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
        'num_workers': getattr(arg, 'num_worker', 0)
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
        
        # Validate data
        for key, value in norm_data.items():
            if key != 'labels':
                if value is None or len(value) == 0:
                    logger.warning(f"No data for {key} after processing")
        
        # Ensure labels exist
        if 'labels' not in norm_data or len(norm_data['labels']) == 0:
            # Find primary modality for label count
            primary_key = next((k for k in norm_data if k != 'labels' and len(norm_data[k]) > 0), None)
            if primary_key:
                logger.warning(f"Creating dummy labels based on {primary_key} shape")
                norm_data['labels'] = np.zeros(len(norm_data[primary_key]), dtype=np.int32)
        
        # Ensure we have at least accelerometer data
        if 'accelerometer' not in norm_data or len(norm_data['accelerometer']) == 0:
            logger.warning("No accelerometer data, creating dummy data")
            if 'labels' in norm_data and len(norm_data['labels']) > 0:
                # Create dummy accelerometer data
                dummy_shape = (len(norm_data['labels']), 128, 3)
                norm_data['accelerometer'] = np.zeros(dummy_shape, dtype=np.float32)
            else:
                # Complete fallback
                norm_data['accelerometer'] = np.zeros((1, 128, 3), dtype=np.float32)
                norm_data['labels'] = np.zeros(1, dtype=np.int32)
        
        return norm_data
    except Exception as e:
        logger.error(f"Error in split_by_subjects_tf: {e}")
        
        # Return minimal valid dataset
        return {
            'accelerometer': np.zeros((1, 128, 3), dtype=np.float32),
            'skeleton': np.zeros((1, 128, 32, 3), dtype=np.float32),
            'labels': np.zeros(1, dtype=np.int32)
        }

# ===== TensorFlow Dataset Class =====

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow dataset for SmartFallMM with batching support."""
    
    def __init__(self, dataset, batch_size, use_smv=False):
        """
        Initialize the dataset.
        
        Args:
            dataset: Dictionary containing modality data and labels
            batch_size: Batch size for training
            use_smv: Whether to use Signal Magnitude Vector (currently disabled)
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = False  # Force to False as requested
        
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
                # Handle flattened skeleton data
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    # Reshape to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) == 4:
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
            
            # No SMV calculation as requested
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
    
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
            
            # Add accelerometer data without SMV
            batch_data['accelerometer'] = batch_acc
            
            # Add skeleton if available
            if hasattr(self, 'skl_data') and self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            
            # Get labels
            batch_labels = tf.gather(self.labels, tf_indices)
            
            # For debugging first batch
            if idx == 0:
                for key, value in batch_data.items():
                    logger.info(f"First batch {key} shape: {value.shape}")
                logger.info(f"First batch labels shape: {batch_labels.shape}")
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logger.error(f"Error in batch generation {idx}: {e}")
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = tf.range(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        """Called at the end of each epoch to shuffle data."""
        np.random.shuffle(self.indices)
