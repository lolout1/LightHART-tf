# utils/dataset_tf.py
import os
import logging
import traceback
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing

# Configure logging
logger = logging.getLogger('dataset-tf')

# Constants
SAMPLING_RATE = {'skeleton': 30, 'accelerometer': 25, 'gyroscope': 50}
JOINT_ID_FOR_ALIGNMENT = 9  # Left wrist joint for alignment

def csvloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    """Load CSV sensor data with support for different formats."""
    import pandas as pd
    try:
        is_skeleton = 'skeleton' in file_path
        
        # Detect file format 
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        has_semicolon = ';' in first_line
        
        if has_semicolon:  # Watch/phone format with timestamp
            df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip')
            if df.shape[1] < 4:
                logger.debug(f"Invalid CSV format in {file_path}")
                return None
            data = df.iloc[:, 1:4].astype(float).to_numpy(dtype=np.float32)
        else:  # Standard CSV format
            df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
            cols = 96 if is_skeleton else 3
            if df.shape[1] >= cols:
                data = df.iloc[2:, -cols:].astype(float).to_numpy(dtype=np.float32)
            else:
                logger.debug(f"Invalid column count in {file_path}")
                return None
        
        # Validate data shape and content
        if data.shape[0] < 10:  # Too short sequences
            return None
            
        if np.isnan(data).any() or np.isinf(data).any():
            logger.debug(f"NaN or Inf values in {file_path}")
            return None
            
        # Apply Butterworth filter for inertial data
        if not is_skeleton and data.shape[0] > 30:
            data = butterworth_filter(data, cutoff=7.5, fs=50)
            
        # Reshape skeleton data if needed
        if is_skeleton and data.shape[1] == 96:
            num_frames = data.shape[0]
            data = data.reshape(num_frames, 32, 3)
            
        return data
    except Exception as e:
        logger.debug(f"Error loading CSV {file_path}: {e}")
        return None

def matloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    """Load MatLab files with support for different data types."""
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        
        if key not in ['d_iner', 'd_skel']:
            logger.debug(f"Invalid key {key} for matlab file")
            return None
        
        data = loadmat(file_path)[key]
        
        # Validate data
        if np.isnan(data).any() or np.isinf(data).any():
            logger.debug(f"NaN or Inf values in {file_path}")
            return None
            
        # Apply filtering for inertial data
        if key == 'd_iner' and data.shape[0] > 30:
            data = butterworth_filter(data, cutoff=7.5, fs=25)
        
        return data
    except Exception as e:
        logger.debug(f"Error loading MAT {file_path}: {e}")
        return None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter to sensor data for noise reduction."""
    try:
        min_samples = max(3 * order, 15)
        if len(data) <= min_samples:
            return data
        
        # Check for NaN or Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            # Clean data by replacing NaN/Inf with mean
            cleaned_data = data.copy()
            col_means = np.nanmean(cleaned_data, axis=0)
            # Replace NaN/Inf with column means
            for i in range(cleaned_data.shape[1]):
                mask = np.isnan(cleaned_data[:, i]) | np.isinf(cleaned_data[:, i])
                cleaned_data[mask, i] = col_means[i]
            data = cleaned_data
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        
        # Final check for NaN/Inf in result
        if np.isnan(filtered_data).any() or np.isinf(filtered_data).any():
            return data  # Return original if filtering failed
        
        return filtered_data
    except Exception as e:
        logger.debug(f"Butterworth filter error: {e}")
        return data

def avg_pool(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """Apply average pooling to sequence for length normalization."""
    try:
        shape = sequence.shape
        if shape[0] <= max_length:
            return sequence
        
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        sequence = tf.nn.avg_pool1d(sequence, ksize=5, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        
        return sequence
    except Exception as e:
        logger.debug(f"Error in avg_pool: {e}")
        return sequence[:min(len(sequence), max_length)]

def pad_sequence_numpy(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """Pad sequence to a fixed length with pooling for longer sequences."""
    try:
        shape = list(sequence.shape)
        if shape[0] > max_length:
            return sequence[:max_length]
        
        pooled_sequence = avg_pool(sequence=sequence, max_length=max_length)
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        
        return new_sequence
    except Exception as e:
        logger.debug(f"Error in pad_sequence_numpy: {e}")
        return None

def align_sequence(data: Dict[str, np.ndarray], use_dtw: bool) -> Dict[str, np.ndarray]:
    """Align accelerometer and skeleton data using dynamic time warping."""
    try:
        # If DTW is disabled or required modalities are missing, return original data
        if not use_dtw or "skeleton" not in data or not any(k in data for k in ["accelerometer", "gyroscope"]):
            return data

        # Extract data for alignment
        joint_id = JOINT_ID_FOR_ALIGNMENT
        inertial_key = "accelerometer" if "accelerometer" in data else "gyroscope"
        
        # Extract skeleton joint data
        skeleton_data = data['skeleton']
        if len(skeleton_data.shape) == 4:  # [batch, frames, joints, coords]
            skeleton_joint_data = skeleton_data[:, (joint_id-1), :]
        elif len(skeleton_data.shape) == 3:  # [frames, joints, coords]
            skeleton_joint_data = skeleton_data[:, (joint_id-1), :]
        elif len(skeleton_data.shape) == 2 and skeleton_data.shape[1] >= joint_id * 3:
            skeleton_joint_data = skeleton_data[:, (joint_id-1)*3:joint_id*3]
        else:
            logger.debug(f"Invalid skeleton data shape: {skeleton_data.shape}")
            return {}  # Return empty to skip this trial
        
        inertial_data = data[inertial_key]
        
        # Check for minimum sequence length
        if len(skeleton_joint_data) < 10 or len(inertial_data) < 10:
            logger.debug("Sequences too short for DTW alignment")
            return {}  # Return empty to skip this trial
        
        # Check for NaN/Inf
        if (np.isnan(skeleton_joint_data).any() or np.isinf(skeleton_joint_data).any() or
            np.isnan(inertial_data).any() or np.isinf(inertial_data).any()):
            logger.debug("NaN or Inf values detected in data")
            return {}  # Return empty to skip this trial
        
        # Calculate norms for alignment
        skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        inertial_norm = np.linalg.norm(inertial_data, axis=1)
        
        # Apply DTW
        try:
            distance, path = fastdtw(
                inertial_norm[:, np.newaxis], 
                skeleton_norm[:, np.newaxis],
                dist=euclidean,
                radius=15
            )
            
            # Extract matched indices
            inertial_ids = set()
            skeleton_ids = set()
            
            for i, j in path:
                if i not in inertial_ids and j not in skeleton_ids:
                    inertial_ids.add(i)
                    skeleton_ids.add(j)
            
            inertial_ids = sorted(list(inertial_ids))
            skeleton_ids = sorted(list(skeleton_ids))
            
            # Apply alignment if we have enough matched indices
            if len(inertial_ids) > 5 and len(skeleton_ids) > 5:
                data['skeleton'] = data['skeleton'][skeleton_ids]
                for key in [k for k in data.keys() if k != 'skeleton' and k != 'labels']:
                    data[key] = data[key][inertial_ids]
                
                # Ensure all modalities have consistent lengths
                min_length = min(len(data['skeleton']), len(data[inertial_key]))
                for key in list(data.keys()):
                    if key != 'labels' and len(data[key]) > min_length:
                        data[key] = data[key][:min_length]
                
                return data
            else:
                logger.debug("Not enough matched indices from DTW")
                return {}  # Return empty to skip this trial
            
        except Exception as e:
            logger.debug(f"DTW alignment error: {e}")
            return {}  # Return empty to skip this trial
            
    except Exception as e:
        logger.debug(f"General error in align_sequence: {e}")
        return {}  # Return empty to skip this trial

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, height: float, distance: int, label: int) -> Dict[str, np.ndarray]:
    """Apply sliding window around peaks in the signal for segmentation."""
    try:
        if 'accelerometer' not in data:
            return {}  # Skip if no accelerometer data
        
        acc_data = data['accelerometer']
        
        # Check for NaN/Inf
        if np.isnan(acc_data).any() or np.isinf(acc_data).any():
            logger.debug("NaN or Inf values in accelerometer data")
            return {}
        
        # Find peaks in signal magnitude
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        # Skip if no peaks found
        if len(peaks) == 0:
            logger.debug("No peaks found in signal")
            return {}
        
        windowed_data = defaultdict(list)
        
        # Create windows around each peak
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
            
            # Skip if no valid windows
            if not windows:
                return {}
                
            windowed_data[modality] = np.stack(windows)
        
        # Add labels if we have data
        if windowed_data and len(next(iter(windowed_data.values()))) > 0:
            sample_modality = next(iter(windowed_data.keys()))
            windowed_data['labels'] = np.repeat(label, len(windowed_data[sample_modality]))
            return windowed_data
        else:
            return {}
            
    except Exception as e:
        logger.debug(f"Error in selective_sliding_window: {e}")
        return {}

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
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        self.age_groups[age_group][modality_name] = Modality(modality_name)
    
    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self) -> None:
        total_files = 0
        
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
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
                
                files_loaded = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith(('.csv', '.mat')):
                            try:
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                                files_loaded += 1
                            except Exception as e:
                                logger.debug(f"Error parsing file {file}: {e}")
                
                total_files += files_loaded
        
        logger.info(f"Loaded {total_files} files")
    
    def match_trials(self, required_modalities=None) -> None:
        trial_dict = {}
        
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path
        
        if required_modalities is None:
            required_modalities = ['accelerometer']
        
        complete_trials = []
        partial_trials = 0
        
        for key, files_dict in trial_dict.items():
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
        logger.info(f"Matched {len(complete_trials)} trials, skipped {partial_trials} incomplete trials")
    
    def pipeline(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
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
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        self.data = defaultdict(list)
        
        matching_trials = [t for t in self.dataset.matched_trials if t.subject_id in subjects]
        logger.info(f"Found {len(matching_trials)} trials for {len(subjects)} subjects")
        
        if not matching_trials:
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            return
        
        if self.num_workers > 1:
            self._process_trials_parallel(matching_trials, fuse)
        else:
            self._process_trials_sequential(matching_trials, fuse)
        
        self._concatenate_modalities()
    
    def _process_trials_parallel(self, trials: List[MatchedTrial], fuse: bool) -> None:
        args_list = [(t, self.mode, self.max_length, self.task, self.use_dtw, self.kwargs) for t in trials]
        
        with multiprocessing.Pool(processes=min(self.num_workers, multiprocessing.cpu_count())) as pool:
            results = pool.map(process_trial_file, args_list)
        
        valid_results = [r for r in results if r is not None and len(r) > 0]
        
        all_modalities = set()
        for result in valid_results:
            all_modalities.update(result.keys())
        all_modalities.discard('labels')
        
        for modality in all_modalities:
            self.data[modality] = []
        
        self.data['labels'] = []
        
        for result in valid_results:
            for modality in all_modalities:
                if modality in result and modality != 'labels':
                    self.data[modality].append(result[modality])
            
            if 'labels' in result:
                self.data['labels'].append(result['labels'])
    
    def _process_trials_sequential(self, trials: List[MatchedTrial], fuse: bool) -> None:
        processed_count = 0
        skipped_count = 0
        
        for i, trial in enumerate(trials):
            if i % 50 == 0:
                logger.info(f"Processing trial {i+1}/{len(trials)}")
            
            try:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                # Load data for each modality
                trial_data = {}
                
                for modality, file_path in trial.files.items():
                    file_type = file_path.split('.')[-1]
                    if file_type not in ['csv', 'mat']:
                        continue
                    
                    loader = LOADER_MAP[file_type]
                    
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys and modality.lower() in keys:
                        key = keys[modality.lower()]
                    
                    # Load data and check for None (which indicates errors)
                    data = loader(file_path, key=key)
                    
                    if data is not None and len(data) > 5:
                        trial_data[modality] = data
                
                # Skip if missing required data
                if 'accelerometer' not in trial_data or len(trial_data['accelerometer']) == 0:
                    skipped_count += 1
                    continue
                
                # Handle DTW alignment based on flag
                if self.use_dtw:
                    # Skip if missing skeleton with use_dtw=True
                    if 'skeleton' not in trial_data or len(trial_data['skeleton']) == 0:
                        skipped_count += 1
                        continue
                    
                    # Apply DTW alignment
                    aligned_data = align_sequence(trial_data, use_dtw=True)
                    
                    # Skip if alignment failed (empty dict)
                    if not aligned_data:
                        skipped_count += 1
                        continue
                    
                    trial_data = aligned_data
                elif 'skeleton' not in trial_data:
                    # Create dummy skeleton data when use_dtw=False and skeleton is missing
                    acc_frames = trial_data['accelerometer'].shape[0]
                    trial_data['skeleton'] = np.zeros((acc_frames, 32, 3), dtype=np.float32)
                
                # Process based on mode
                result = None
                
                if self.mode == 'avg_pool':
                    result = {}
                    for key, value in trial_data.items():
                        if key != 'labels':
                            padded = pad_sequence_numpy(value, self.max_length)
                            if padded is None:
                                result = None
                                break
                            result[key] = padded
                    
                    if result:
                        result['labels'] = np.array([label])
                else:
                    # Selective sliding window
                    if label == 1:  # Fall
                        result = selective_sliding_window(
                            data=trial_data,
                            window_size=self.max_length,
                            height=1.4,  # Higher threshold for falls
                            distance=50,
                            label=label
                        )
                    else:  # Non-fall
                        result = selective_sliding_window(
                            data=trial_data,
                            window_size=self.max_length,
                            height=1.2,  # Lower threshold for non-falls
                            distance=100,
                            label=label
                        )
                
                # Add processed data if valid
                if result and len(result) > 0 and all(len(v) > 0 for k, v in result.items() if k != 'labels'):
                    for key, value in result.items():
                        if key not in self.data:
                            self.data[key] = []
                        self.data[key].append(value)
                    processed_count += 1
                else:
                    skipped_count += 1
            
            except Exception as e:
                logger.debug(f"Error processing trial {trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {e}")
                skipped_count += 1
        
        logger.info(f"Processed {processed_count}/{len(trials)} trials, skipped {skipped_count} trials")
    
    def _concatenate_modalities(self) -> None:
        for key in list(self.data.keys()):
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                    
                    if self.verbose:
                        logger.info(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    del self.data[key]
        
        # Handle empty data case
        if not self.data or 'accelerometer' not in self.data or len(self.data['accelerometer']) == 0:
            logger.warning("No valid data processed, creating dummy dataset")
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
    
    def normalization(self) -> Dict[str, np.ndarray]:
        try:
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    if len(value.shape) >= 2:
                        num_samples = value.shape[0]
                        
                        orig_shape = value.shape
                        reshaped = value.reshape(num_samples * value.shape[1], -1)
                        
                        scaler = StandardScaler()
                        try:
                            norm_data = scaler.fit_transform(reshaped)
                            self.data[key] = norm_data.reshape(orig_shape)
                        except Exception as e:
                            logger.warning(f"Normalization failed for {key}: {e}")
            
            return self.data
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return self.data

def process_trial_file(args):
    """Process a single trial - used for parallel processing."""
    trial, mode, max_length, task, use_dtw, kwargs = args
    
    try:
        if task == 'fd':
            label = int(trial.action_id > 9)
        elif task == 'age':
            label = int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            label = trial.action_id - 1
        
        # Load data for each modality
        trial_data = {}
        
        for modality, file_path in trial.files.items():
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                continue
            
            loader = LOADER_MAP[file_type]
            
            keys = kwargs.get('keys', None)
            key = None
            if keys and modality.lower() in keys:
                key = keys[modality.lower()]
            
            data = loader(file_path, key=key)
            
            if data is not None and len(data) > 5:
                trial_data[modality] = data
        
        # Skip if missing required data
        if 'accelerometer' not in trial_data or len(trial_data['accelerometer']) == 0:
            return None
        
        # Handle DTW alignment
        if use_dtw:
            # Skip if missing skeleton with use_dtw=True
            if 'skeleton' not in trial_data or len(trial_data['skeleton']) == 0:
                return None
            
            # Apply DTW alignment
            aligned_data = align_sequence(trial_data, use_dtw=True)
            
            # Skip if alignment failed
            if not aligned_data:
                return None
            
            trial_data = aligned_data
        elif 'skeleton' not in trial_data:
            # Create dummy skeleton data when use_dtw=False and skeleton is missing
            acc_frames = trial_data['accelerometer'].shape[0]
            trial_data['skeleton'] = np.zeros((acc_frames, 32, 3), dtype=np.float32)
        
        # Process based on mode
        result = None
        
        if mode == 'avg_pool':
            result = {}
            for key, value in trial_data.items():
                if key != 'labels':
                    padded = pad_sequence_numpy(value, max_length)
                    if padded is None:
                        return None
                    result[key] = padded
            
            if result:
                result['labels'] = np.array([label])
        else:
            # Selective sliding window
            if label == 1:  # Fall
                result = selective_sliding_window(
                    data=trial_data,
                    window_size=max_length,
                    height=1.4,
                    distance=50,
                    label=label
                )
            else:  # Non-fall
                result = selective_sliding_window(
                    data=trial_data,
                    window_size=max_length,
                    height=1.2,
                    distance=100,
                    label=label
                )
        
        # Return valid results
        if result and len(result) > 0 and all(len(v) > 0 for k, v in result.items() if k != 'labels'):
            return result
        
        return None
    except Exception as e:
        logger.debug(f"Error in process_trial_file: {e}")
        return None

def prepare_smartfallmm_tf(arg) -> DatasetBuilder:
    """Prepare SmartFall dataset builder using command line arguments."""
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm')
    
    if not os.path.exists(data_dir):
        logger.warning(f"SmartFall data directory not found at {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    age_group = arg.dataset_args.get('age_group', ['young'])
    modalities = arg.dataset_args.get('modalities', ['accelerometer'])
    sensors = arg.dataset_args.get('sensors', ['watch'])
    use_dtw = arg.dataset_args.get('use_dtw', True)
    
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(age_group=age_group, modalities=modalities, sensors=sensors)
    
    builder_kwargs = {
        'verbose': arg.dataset_args.get('verbose', False),
        'use_dtw': use_dtw,
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
    """Split dataset by subjects."""
    try:
        builder.make_dataset(subjects, fuse)
        norm_data = builder.normalization()
        
        # Validate that we have data
        if (not norm_data or 
            'accelerometer' not in norm_data or len(norm_data['accelerometer']) == 0 or
            'labels' not in norm_data or len(norm_data['labels']) == 0):
            
            logger.warning("No valid data found, creating dummy dataset")
            return {
                'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, builder.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
        
        # Make sure skeleton exists if use_dtw is False
        if 'skeleton' not in norm_data and not builder.use_dtw:
            frames = norm_data['accelerometer'].shape[0]
            max_length = norm_data['accelerometer'].shape[1]
            norm_data['skeleton'] = np.zeros((frames, max_length, 32, 3), dtype=np.float32)
        
        return norm_data
    except Exception as e:
        logger.error(f"Error in split_by_subjects_tf: {e}")
        
        return {
            'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32),
            'skeleton': np.zeros((1, builder.max_length, 32, 3), dtype=np.float32),
            'labels': np.zeros(1, dtype=np.int32)
        }

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow dataset for SmartFallMM with batching support."""
    
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        
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
            
            # Calculate SMV if requested
            if self.use_smv:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logger.info(f"SMV calculated with shape: {self.smv.shape}")
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector (SMV) for a batch of samples."""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
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
                    batch_smv = self.cal_smv(batch_acc)
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            # Add skeleton data
            batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            
            # Get labels
            batch_labels = tf.gather(self.labels, tf_indices)
            
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
