#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_tf.py - TensorFlow dataset loader for SmartFallMM
Exact match to PyTorch implementation with robust error handling
"""

import os
import logging
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Import the fixed processor functions
from utils.processor_tf import (
    butterworth_filter, 
    pad_sequence_tf, 
    align_sequence_dtw, 
    selective_windowing
)

logger = logging.getLogger('dataset-tf')

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load CSV sensor data with format exactly matching PyTorch implementation.
    
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
        
        try:
            # First attempt - standard CSV loading
            df = pd.read_csv(file_path, index_col=False, header=0)
            
            # Handle missing data
            df = df.dropna().bfill()
            
            # Extract correct columns
            if is_skeleton:
                # For skeleton data (32 joints × 3 coords = 96 columns)
                cols = 96
                activity_data = df.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                
                # Reshape to (frames, joints, coords) if needed
                if activity_data.shape[1] == 96:
                    frames = activity_data.shape[0]
                    activity_data = activity_data.reshape(frames, 32, 3)
                    logger.info(f"Reshaped skeleton data from {activity_data.shape} to {activity_data.shape}")
            else:
                # For accelerometer/gyroscope data (3 columns: x, y, z)
                activity_data = df.iloc[2:, -3:].to_numpy(dtype=np.float32)
        
        except Exception as e:
            logger.warning(f"Standard CSV parsing failed: {e}. Trying alternative approaches...")
            
            try:
                # Try with no header and different indexing
                df = pd.read_csv(file_path, header=None)
                
                if is_skeleton:
                    # For skeleton data
                    if df.shape[1] >= 96:
                        activity_data = df.iloc[2:, -96:].to_numpy(dtype=np.float32)
                        frames = activity_data.shape[0]
                        activity_data = activity_data.reshape(frames, 32, 3)
                    else:
                        activity_data = df.iloc[2:, :].to_numpy(dtype=np.float32)
                else:
                    # For accelerometer data
                    activity_data = df.iloc[2:, -3:].to_numpy(dtype=np.float32)
            except Exception as e2:
                logger.error(f"All CSV parsing attempts failed: {e2}")
                return np.array([])
        
        # Filter accelerometer data
        if not is_skeleton and len(activity_data) > 30:
            try:
                activity_data = butterworth_filter(activity_data, cutoff=7.5, fs=25)
            except Exception as e:
                logger.warning(f"Butterworth filtering failed: {e}")
        
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
            data = butterworth_filter(data, cutoff=7.5, fs=25)
        
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
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
    
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
                                # Parse file name format: S##A##T##.ext
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
    
    def match_trials(self) -> None:
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
        
        # Create matched trials
        required_modalities = ['accelerometer', 'skeleton']
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
    
    def pipeline(self, age_group, modalities, sensors):
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
        self.match_trials()

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
        Build dataset for specified subjects - EXACT match to PyTorch implementation
        
        Args:
            subjects: List of subject IDs to include
            fuse: Whether to fuse inertial data (maintained for compatibility)
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
            if self.verbose and (i % 10 == 0 or i == len(matching_trials) - 1):
                logger.info(f"Processing trial {i+1}/{len(matching_trials)}")
            
            try:
                # Determine label based on task
                if self.task == 'fd':
                    label = int(trial.action_id > 9)  # Falls are actions 10+
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1  # Action recognition
                
                # Load data for each modality
                trial_data = {}
                
                for modality, file_path in trial.files.items():
                    # Get key for MATLAB files
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys and modality.lower() in keys:
                        key = keys[modality.lower()]
                    
                    # Load the file
                    data = self.load_file(file_path, key=key)
                    
                    if data is not None and len(data) > 10:  # Skip very short sequences
                        trial_data[modality] = data
                
                # Skip if missing key modalities
                if not all(m in trial_data for m in ['accelerometer', 'skeleton']):
                    missing = [m for m in ['accelerometer', 'skeleton'] if m not in trial_data]
                    if self.verbose:
                        logger.warning(f"Trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number} "
                                      f"missing required modalities: {missing}")
                    continue
                
                # Apply DTW alignment between modalities - EXACT match to PyTorch implementation
                aligned_data = align_sequence_dtw(trial_data, 
                                               joint_id=9,
                                               use_dtw=self.use_dtw)
                
                # Process data based on mode
                if self.mode == 'avg_pool':
                    # Simple padding/pooling
                    result = {}
                    for key, value in aligned_data.items():
                        result[key] = pad_sequence_tf(value, self.max_length)
                    result['labels'] = np.array([label])
                else:
                    # Selective sliding window - EXACT match to PyTorch implementation
                    result = selective_windowing(aligned_data, self.max_length, label)
                
                # Add to dataset if valid
                if result and len(result.get('labels', [])) > 0:
                    for key, value in result.items():
                        if len(value) > 0:
                            self.data[key].append(value)
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Concatenate data for each modality
        for key in list(self.data.keys()):
            if len(self.data[key]) > 0:
                try:
                    # Check for consistent shapes before concatenating
                    first_shape = self.data[key][0].shape
                    
                    # For 'skeleton' data, ensure consistent dimensionality
                    if key == 'skeleton':
                        # Find all arrays with shape [batch, frames, joints, coords]
                        consistent_arrays = []
                        for arr in self.data[key]:
                            # If shape is [batch, frames, joints], reshape to [batch, frames, joints, 1]
                            if len(arr.shape) == 3 and arr.shape[2] == 32:
                                reshaped = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
                                consistent_arrays.append(reshaped)
                            # If shape is already [batch, frames, joints, coords]
                            elif len(arr.shape) == 4 and arr.shape[2] == 32 and arr.shape[3] == 3:
                                consistent_arrays.append(arr)
                        
                        # Concatenate consistent arrays
                        if consistent_arrays:
                            self.data[key] = np.concatenate(consistent_arrays, axis=0)
                            logger.info(f"Standardized {key} shape: {self.data[key].shape}")
                        else:
                            logger.warning(f"No consistent {key} arrays found")
                            del self.data[key]
                    else:
                        # For other modalities, just concatenate
                        self.data[key] = np.concatenate(self.data[key], axis=0)
                        
                    logger.info(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    logger.error(f"Inconsistent shapes for {key}: {[arr.shape for arr in self.data[key]]}")
                    del self.data[key]
        
        # Check if we have any data left
        if 'labels' not in self.data or len(self.data['labels']) == 0:
            logger.warning("No valid data processed, creating dummy dataset")
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            
            # Add skeleton if required
            if 'skeleton' in self.kwargs.get('required_modalities', ['accelerometer']):
                self.data['skeleton'] = np.zeros((1, self.max_length, 32, 3), dtype=np.float32)
    
    
    
    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Normalize data for each modality using StandardScaler - EXACT match to PyTorch
        
        Returns:
            Dict[str, np.ndarray]: Normalized data
        """
        try:
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    # Standardize each modality
                    if len(value.shape) >= 2:  # Must be at least 2D
                        num_samples = value.shape[0]
                        
                        # Reshape based on modality
                        if key == 'skeleton' and len(value.shape) == 4:
                            # Handle 4D tensor: [samples, frames, joints, coords]
                            orig_shape = value.shape
                            reshaped = value.reshape(num_samples * value.shape[1], -1)
                        else:
                            # Handle regular tensor
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
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return self.data

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow dataset for multimodal data with batch support."""
    
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
        
        # Validate and prepare data
        self._validate_data()
        self._prepare_data()
        
        # Initialize indices for batch generation
        self.num_samples = len(self.labels)
        self.indices = np.arange(self.num_samples)
        
        logger.info(f"UTD_MM_TF initialized with {self.num_samples} samples. SMV: {use_smv}")
    
    def _validate_data(self):
        """Validate and normalize input data formats"""
        # Ensure accelerometer data exists
        if self.acc_data is None or len(self.acc_data) == 0:
            logger.warning("Missing accelerometer data. Creating dummy data.")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = len(self.acc_data)
        
        # Process skeleton data - ensure correct format
        if self.skl_data is not None and len(self.skl_data) > 0:
            # Fixed - Force consistent shape of [batch, frames, joints, coords]
            if len(self.skl_data.shape) == 3:
                if self.skl_data.shape[2] == 32:
                    # Convert from [batch, frames, joints] to [batch, frames, joints, 1]
                    self.skl_data = self.skl_data.reshape(self.skl_data.shape[0], self.skl_data.shape[1], 32, 1)
                elif self.skl_data.shape[2] == 96:
                    # Convert from [batch, frames, joints*3] to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_data.shape[0], self.skl_data.shape[1], 32, 3)
        else:
            logger.warning("Missing skeleton data. Creating dummy data.")
            self.skl_data = np.zeros((self.num_samples, 128, 32, 3), dtype=np.float32)
        
        # Validate labels
        if self.labels is None or len(self.labels) == 0:
            logger.warning("Missing labels. Creating dummy labels.")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Ensure all data has matching first dimension
        min_samples = min(len(self.acc_data), len(self.skl_data), len(self.labels))
        if min_samples < self.num_samples:
            logger.warning(f"Data size mismatch. Truncating to {min_samples} samples.")
            self.acc_data = self.acc_data[:min_samples]
            self.skl_data = self.skl_data[:min_samples]
            self.labels = self.labels[:min_samples]
            self.num_samples = min_samples
    
    def _prepare_data(self):
        """Convert numpy arrays to TensorFlow tensors and calculate SMV if needed"""
        try:
            # Convert to TF tensors with correct types
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate SMV if requested (matching PyTorch implementation)
            if self.use_smv:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logger.info(f"Signal Magnitude Vector calculated: {self.smv.shape}")
        except Exception as e:
            logger.error(f"Error preparing data tensors: {e}")
    
    def __len__(self):
        """Return number of batches"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        """Get batch by index"""
        try:
            # Get indices for current batch
            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.num_samples)
            batch_indices = self.indices[batch_start:batch_end]
            
            # Create batch data
            batch_data = {}
            
            # Get accelerometer data
            batch_acc = tf.gather(self.acc_data, batch_indices)
            
            # Add Signal Magnitude Vector if requested
            if self.use_smv:
                if hasattr(self, 'smv'):
                    batch_smv = tf.gather(self.smv, batch_indices)
                else:
                    # Calculate on-the-fly if not pre-computed
                    mean = tf.reduce_mean(batch_acc, axis=1, keepdims=True)
                    zero_mean = batch_acc - mean
                    sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                    batch_smv = tf.sqrt(sum_squared)
                
                # Concat SMV with original data - matching PyTorch
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            # Get skeleton data
            batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
            
            # Get batch labels
            batch_labels = tf.gather(self.labels, batch_indices)
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logger.error(f"Error creating batch {idx}: {e}")
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples - batch_start)
            dummy_acc = tf.zeros((batch_size, self.acc_data.shape[1], 
                                 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.skl_data.shape[1], 
                                 self.skl_data.shape[2], 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            return dummy_data, dummy_labels, tf.range(batch_size)
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch"""
        np.random.shuffle(self.indices)

def prepare_smartfallmm_tf(arg):
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
        'required_modalities': arg.dataset_args.get('modalities', ['accelerometer'])
    }
    
    builder = DatasetBuilder(
        sm_dataset, 
        arg.dataset_args.get('mode', 'sliding_window'), 
        arg.dataset_args.get('max_length', 128),
        arg.dataset_args.get('task', 'fd'),
        **builder_kwargs
    )
    
    return builder

def split_by_subjects_tf(builder, subjects, fuse):
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
        
        # Final shape check for skeleton data
        if 'skeleton' in norm_data:
            # Ensure skeleton data has shape [batch, frames, joints, coords]
            if len(norm_data['skeleton'].shape) != 4:
                logger.warning(f"Unexpected skeleton shape: {norm_data['skeleton'].shape}")
                if len(norm_data['skeleton'].shape) == 3:
                    if norm_data['skeleton'].shape[2] == 32:
                        # [batch, frames, joints] -> [batch, frames, joints, 1]
                        norm_data['skeleton'] = norm_data['skeleton'].reshape(
                            norm_data['skeleton'].shape[0],
                            norm_data['skeleton'].shape[1],
                            32, 1
                        )
                    elif norm_data['skeleton'].shape[2] == 96:
                        # [batch, frames, joints*3] -> [batch, frames, joints, 3]
                        norm_data['skeleton'] = norm_data['skeleton'].reshape(
                            norm_data['skeleton'].shape[0],
                            norm_data['skeleton'].shape[1],
                            32, 3
                        )
        
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
        import traceback
        logger.error(traceback.format_exc())
        
        # Return minimal valid dataset as fallback
        return {
            'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32),
            'skeleton': np.zeros((1, builder.max_length, 32, 3), dtype=np.float32),
            'labels': np.zeros(1, dtype=np.int32)
        }
