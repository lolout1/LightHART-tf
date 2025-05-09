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
import multiprocessing

# Global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Loads CSV data with robust handling of different formats
    """
    import pandas as pd
    try:
        # Check file format by reading first line
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            
        # Determine delimiter based on file content
        if ';' in first_line:
            # Format: timestamp;x;y;z - SmartFall watch/phone data format
            df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip')
            # Skip timestamp column (first column)
            acc_data = df.iloc[:, 1:4].astype(float).to_numpy(dtype=np.float32)
        else:
            # Standard CSV format
            df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
            # Handle different column counts
            if 'skeleton' in file_path:
                cols = 96
            else:
                cols = 3
                
            # Handle different data structures
            if df.shape[1] >= cols:
                acc_data = df.iloc[2:, -cols:].astype(float).to_numpy(dtype=np.float32)
            else:
                logging.warning(f"CSV file has fewer columns than expected: {file_path}")
                # Create dummy data with the right shape
                acc_data = np.zeros((10, cols), dtype=np.float32)
            
        return acc_data
    except Exception as e:
        logging.error(f"Error loading CSV file {file_path}: {e}")
        # Return small dummy data instead of None to allow processing to continue
        return np.zeros((10, 3 if 'skeleton' not in file_path else 96), dtype=np.float32)

def matloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Loads MatLab files with robust error handling
    """
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            raise ValueError(f'Unsupported key {key} for matlab file')
        
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logging.error(f"Error loading MAT file {file_path}: {e}")
        # Return dummy data
        return np.zeros((10, 3), dtype=np.float32)

# Map file extensions to loader functions
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth filter to accelerometer data with robust error handling
    """
    try:
        # Check if data is long enough for filtering - prevent padlen errors
        min_samples = max(3 * order, 15)
        if len(data) <= min_samples:
            logging.warning(f"Sequence too short for filtering: {len(data)} samples, need > {min_samples}")
            return data
            
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logging.warning(f"Butterworth filter error: {e}")
        return data  # Return original data if filtering fails

def avg_pool(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """
    Apply average pooling to data for sequence length normalization
    """
    try:
        shape = sequence.shape
        
        # Skip pooling if sequence is already shorter than max_length
        if shape[0] <= max_length:
            return sequence
            
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        
        # Calculate stride to achieve desired length
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        
        # Apply pooling
        sequence = tf.nn.avg_pool1d(sequence, ksize=5, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        
        # Reshape to original format
        sequence = sequence.reshape(-1, *shape[1:])
        
        return sequence
    except Exception as e:
        logging.error(f"Error in avg_pool: {e}")
        # Return truncated original sequence
        return sequence[:min(len(sequence), max_length)]

def pad_sequence_numpy(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """
    Pad sequence to a fixed length with robust error handling
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
        logging.error(f"Error in pad_sequence_numpy: {e}")
        # Create an empty padded sequence as fallback
        dummy = np.zeros((max_length, *sequence.shape[1:]), dtype=sequence.dtype)
        dummy[:min(len(sequence), max_length)] = sequence[:min(len(sequence), max_length)]
        return dummy

def ensure_3d(array, expected_shape=None):
    """
    Ensure array is 3D with consistent shapes
    """
    if array is None:
        return None
        
    try:
        # Handle different dimensions
        if len(array.shape) == 2:
            # Convert 2D to 3D by adding a batch dimension
            array = np.expand_dims(array, axis=0)
        elif len(array.shape) > 3:
            # Reshape higher dimensions appropriately
            if array.shape[-1] == 3 and array.shape[-2] == 32:  # skeleton data
                batch, frames, joints, coords = array.shape
                array = array.reshape(batch, frames, joints*coords)
        
        # Validate shape if expected
        if expected_shape is not None:
            if array.shape[1:] != expected_shape[1:]:
                # Reshape to match expected shape
                try:
                    array = array.reshape(array.shape[0], *expected_shape[1:])
                except ValueError:
                    # If reshape fails, pad or truncate
                    temp = np.zeros((array.shape[0], *expected_shape[1:]), dtype=array.dtype)
                    min_frames = min(array.shape[1], expected_shape[1])
                    min_features = min(array.shape[2], expected_shape[2])
                    temp[:, :min_frames, :min_features] = array[:, :min_frames, :min_features]
                    array = temp
        
        return array
    except Exception as e:
        logging.error(f"Error in ensure_3d: {e}")
        if expected_shape is not None:
            return np.zeros((1, *expected_shape[1:]), dtype=np.float32)
        else:
            # Default shape if none provided
            return np.zeros((1, 128, 3), dtype=np.float32)

def align_sequence(data: Dict[str, np.ndarray], use_dtw: bool = True) -> Dict[str, np.ndarray]:
    """
    Align accelerometer and skeleton data with robust error handling
    """
    try:
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

        # Proceed with DTW alignment
        joint_id = 9  # Left wrist joint
        inertial_key = "accelerometer" if "accelerometer" in data else "gyroscope"
        
        # Extract skeleton joint data
        skeleton_data = data['skeleton']
        if len(skeleton_data.shape) == 4:
            skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
        elif len(skeleton_data.shape) == 3:
            if skeleton_data.shape[2] >= joint_id * 3:
                skeleton_joint_data = skeleton_data[:, :, (joint_id-1)*3:joint_id*3]
            else:
                skeleton_joint_data = skeleton_data[:, :, min(joint_id-1, skeleton_data.shape[2]-1)]
        else:
            return data
        
        inertial_data = data[inertial_key]
        
        # Handle multiple inertial sensor data
        if "gyroscope" in data and inertial_key == "accelerometer":
            gyroscope_data = data["gyroscope"]
            min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
            inertial_data = inertial_data[:min_len, :]
            data["gyroscope"] = gyroscope_data[:min_len, :]
        
        # Ensure data is long enough for alignment
        if len(skeleton_joint_data) < 5 or len(inertial_data) < 5:
            logging.warning("Sequences too short for DTW alignment")
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
                radius=1  # Use smaller radius for faster processing
            )
            
            # Extract matched indices
            inertial_ids = set()
            skeleton_ids = set()
            
            for i, j in path:
                if i not in inertial_ids and j not in skeleton_ids:
                    inertial_ids.add(i)
                    skeleton_ids.add(j)
            
            # Convert to sorted lists and filter data
            if inertial_ids and skeleton_ids:
                inertial_ids = sorted(list(inertial_ids))
                skeleton_ids = sorted(list(skeleton_ids))
                
                if len(inertial_ids) > 5 and len(skeleton_ids) > 5:
                    data['skeleton'] = data['skeleton'][skeleton_ids]
                    for key in [k for k in data.keys() if k != 'skeleton' and k != 'labels']:
                        data[key] = data[key][inertial_ids]
                    
                    # Ensure consistent lengths
                    min_length = min(len(data['skeleton']), len(data[inertial_key]))
                    for key in data:
                        if key != 'labels' and len(data[key]) > min_length:
                            data[key] = data[key][:min_length]
                
        except Exception as e:
            logging.warning(f"DTW alignment failed: {e}. Using simple length matching.")
            # Fallback to simple length matching
            min_length = min(skeleton_data.shape[0], inertial_data.shape[0])
            for key in data:
                if key != 'labels' and len(data[key]) > min_length:
                    data[key] = data[key][:min_length]
        
        return data
    except Exception as e:
        logging.error(f"Error in align_sequence: {e}")
        # Return original data if alignment fails
        return data

def sliding_window(data: Dict[str, np.ndarray], window_size: int, stride_size: int, label: int) -> Dict[str, np.ndarray]:
    """
    Apply sliding window to data for segmentation with robust error handling
    """
    try:
        # Determine max time from available modalities
        if 'skeleton' not in data:
            key = 'accelerometer' if 'accelerometer' in data else 'gyroscope'
            max_time = data[key].shape[0]
        else:
            max_time = data['skeleton'].shape[0]
        
        # Skip windowing if sequence is too short
        if max_time <= window_size:
            result = {}
            for k, v in data.items():
                if k != 'labels':
                    # Pad if needed
                    if len(v) < window_size:
                        padded = np.zeros((window_size, *v.shape[1:]), dtype=v.dtype)
                        padded[:len(v)] = v
                        result[k] = np.expand_dims(padded, axis=0)  # Add batch dimension
                    else:
                        result[k] = np.expand_dims(v[:window_size], axis=0)  # Add batch dimension
            
            result['labels'] = np.array([label])
            return result
        
        # Generate windows
        stride_size = max(1, stride_size)  # Ensure positive stride
        windows = range(0, max_time - window_size + 1, stride_size)
        result = defaultdict(list)
        
        for start in windows:
            end = start + window_size
            for key, value in data.items():
                if key != 'labels':
                    if start < len(value) and end <= len(value):
                        result[key].append(value[start:end])
        
        # Stack arrays for each modality
        for key in result:
            if result[key]:
                result[key] = np.stack(result[key])
            else:
                # Ensure there's at least one window
                dummy_shape = (1, window_size, *data[key].shape[1:])
                result[key] = np.zeros(dummy_shape, dtype=data[key].dtype)
        
        # Add labels if we have data
        if result:
            sample_key = next(iter(result.keys()))
            result['labels'] = np.repeat(label, len(result[sample_key]))
        else:
            # Ensure we return something
            result['labels'] = np.array([label])
            
        return result
    except Exception as e:
        logging.error(f"Error in sliding_window: {e}")
        # Return minimal valid result
        result = {}
        for k, v in data.items():
            if k != 'labels':
                dummy_shape = (1, window_size, *v.shape[1:])
                result[k] = np.zeros(dummy_shape, dtype=v.dtype)
        result['labels'] = np.array([label])
        return result

class ModalityFile:
    """Represents a file for a specific modality"""
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    """Container for files of a specific modality"""
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))

class MatchedTrial:
    """Container for matched files across modalities"""
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

class SmartFallMM:
    """Manager for SmartFall multimodal dataset"""
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {"old": {}, "young": {}}
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}
    
    def add_modality(self, age_group: str, modality_name: str) -> None:
        """Add a modality to track"""
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        self.age_groups[age_group][modality_name] = Modality(modality_name)
    
    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        """Select which sensor to use for each modality"""
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self) -> None:
        """Load files for all selected modalities"""
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
                    logging.warning(f"Directory not found: {modality_dir}")
                    continue
                    
                # Find and process files
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                # Parse file name - format: S##A##T##.csv (Subject, Action, Trial)
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                            except Exception as e:
                                logging.warning(f"Error parsing file {file}: {e}")
    
    def match_trials(self, required_modalities=None) -> None:
        """Match files across modalities"""
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
        for key, files_dict in trial_dict.items():
            # Check if required modalities are present
            has_required = all(modality in files_dict for modality in required_modalities)
            
            if has_required:
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                
                complete_trials.append(matched_trial)
        
        self.matched_trials = complete_trials
        logging.info(f"Matched {len(complete_trials)} trials across modalities")
    
    def pipeline(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        """Run the full data pipeline"""
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

# Process a single file - for parallel processing
def process_trial_file(args):
    """Process a single trial - used for multiprocessing"""
    trial, mode, max_length, task, use_dtw, kwargs = args
    try:
        label = 0
        if task == 'fd':
            label = int(trial.action_id > 9)  # Fall detection (>9 = fall)
        elif task == 'age':
            label = int(trial.subject_id < 29 or trial.subject_id > 46)  # Age classification
        else:
            label = trial.action_id - 1  # Action recognition
            
        # Load data
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
            try:
                data = loader(file_path, key=key)
                if data is not None and len(data) > 5:  # Skip very short sequences
                    trial_data[modality] = data
                    
                    # Apply Butterworth filter to accelerometer
                    if modality == 'accelerometer' and len(data) > 30:
                        try:
                            trial_data[modality] = butterworth_filter(data, cutoff=7.5, fs=25)
                        except Exception:
                            pass  # Keep original if filtering fails
            except Exception:
                continue
                
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
            
        # Process the data (windowing, etc.)
        if mode == 'avg_pool':
            result = {}
            for key, value in trial_data.items():
                if key != 'labels':
                    result[key] = pad_sequence_numpy(value, max_length)
            result['labels'] = np.array([label])
        else:
            # Apply sliding window
            window_size = max_length
            stride = 32
            result = sliding_window(trial_data, window_size, stride, label)
            
        # Verify we have valid data
        for key in list(result.keys()):
            if key != 'labels' and (result[key] is None or len(result[key]) == 0):
                return None
                
        return result
    except Exception:
        return None

class DatasetBuilder:
    """Builds datasets for training/validation/testing with parallel processing support"""
    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.verbose = kwargs.get('verbose', False)
        self.use_dtw = kwargs.get('use_dtw', True)
        self.num_workers = kwargs.get('num_workers', 0)
        
        logging.info(f"DatasetBuilder initialized with mode={mode}, max_length={max_length}, task={task}, num_workers={self.num_workers}")
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        """Build dataset for specified subjects using parallel processing"""
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Get all matching trials
        matching_trials = [t for t in self.dataset.matched_trials if t.subject_id in subjects]
        logging.info(f"Found {len(matching_trials)} matching trials for {len(subjects)} subjects")
        
        if not matching_trials:
            # Create empty dataset
            self.data = {
                'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32),
                'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32),
                'labels': np.zeros(1, dtype=np.int32)
            }
            return
            
        # Process trials in parallel if requested
        if self.num_workers > 1:
            # Prepare arguments for parallel processing
            args_list = [(t, self.mode, self.max_length, self.task, self.use_dtw, self.kwargs) for t in matching_trials]
            
            # Use multiprocessing pool
            with multiprocessing.Pool(processes=min(self.num_workers, multiprocessing.cpu_count())) as pool:
                results = pool.map(process_trial_file, args_list)
                
            # Filter out None results and process valid ones
            valid_results = [r for r in results if r is not None]
            logging.info(f"Successfully processed {len(valid_results)} trials out of {len(matching_trials)}")
            
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
                    
        else:
            # Process trials sequentially
            processed_count = 0
            for trial in matching_trials:
                args = (trial, self.mode, self.max_length, self.task, self.use_dtw, self.kwargs)
                result = process_trial_file(args)
                
                if result is not None:
                    # Add all modalities
                    for modality, data in result.items():
                        if modality != 'labels':
                            self.data[modality].append(data)
                    
                    # Add labels        
                    if 'labels' in result:
                        self.data['labels'].append(result['labels'])
                        
                    processed_count += 1
                    
            logging.info(f"Successfully processed {processed_count} trials out of {len(matching_trials)}")
            
        # Concatenate data for each modality
        for key in list(self.data.keys()):
            if key != 'labels' and len(self.data[key]) > 0:
                try:
                    # Normalize dimensions
                    normalized_arrays = []
                    expected_ndim = 3  # (batch, seq_len, features)
                    
                    for arr in self.data[key]:
                        # Add missing dimensions
                        if len(arr.shape) < expected_ndim:
                            while len(arr.shape) < expected_ndim:
                                arr = np.expand_dims(arr, axis=0)
                        normalized_arrays.append(arr)
                    
                    # Concatenate if we have any data
                    if normalized_arrays:
                        self.data[key] = np.concatenate(normalized_arrays, axis=0)
                        
                        if self.verbose:
                            logging.info(f"{key} shape: {self.data[key].shape}")
                    else:
                        # Create dummy data if no valid arrays
                        del self.data[key]
                except Exception as e:
                    logging.error(f"Error concatenating {key}: {e}")
                    del self.data[key]
        
        # Handle labels
        if 'labels' in self.data and len(self.data['labels']) > 0:
            try:
                # Flatten and normalize labels
                flat_labels = []
                for label_array in self.data['labels']:
                    if isinstance(label_array, np.ndarray):
                        flat_labels.extend(label_array.flatten())
                    else:
                        flat_labels.append(label_array)
                
                self.data['labels'] = np.array(flat_labels)
                
                if self.verbose:
                    logging.info(f"labels shape: {self.data['labels'].shape}")
            except Exception as e:
                logging.error(f"Error concatenating labels: {e}")
                # Fallback to dummy labels
                if 'accelerometer' in self.data:
                    self.data['labels'] = np.zeros(len(self.data['accelerometer']), dtype=np.int32)
    
    def normalization(self) -> Dict[str, np.ndarray]:
        """Normalize data for each modality"""
        try:
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    if len(value.shape) >= 2:  # Must be at least 2D to normalize
                        num_samples = value.shape[0]
                        
                        # Reshape for normalization
                        orig_shape = value.shape
                        reshaped = value.reshape(num_samples * value.shape[1], -1)
                        
                        # Apply StandardScaler
                        scaler = StandardScaler()
                        norm_data = scaler.fit_transform(reshaped)
                        
                        # Reshape back to original dimensions
                        self.data[key] = norm_data.reshape(orig_shape)
                        
                        logging.info(f"Normalized {key}: min={self.data[key].min():.2f}, max={self.data[key].max():.2f}")
            
            return self.data
        except Exception as e:
            logging.error(f"Error in normalization: {e}")
            # Return unnormalized data
            return self.data

def prepare_smartfallmm_tf(arg) -> DatasetBuilder:
    """Prepare SmartFall dataset builder with comprehensive error handling"""
    # Find data directory with fallbacks
    possible_data_dirs = [
        os.path.join(os.getcwd(), 'data/smartfallmm'),
        os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm')
    ]
    
    data_dir = None
    for path in possible_data_dirs:
        if os.path.exists(path):
            data_dir = path
            break
    
    if data_dir is None:
        logging.error("SmartFall data directory not found, searching in: " + ", ".join(possible_data_dirs))
        # Create a dummy directory for testing
        data_dir = possible_data_dirs[0]
        os.makedirs(data_dir, exist_ok=True)
    
    # Get configuration parameters with defaults
    age_group = arg.dataset_args.get('age_group', ['young'])
    modalities = arg.dataset_args.get('modalities', ['accelerometer'])
    sensors = arg.dataset_args.get('sensors', ['watch'])
    
    # Initialize dataset
    sm_dataset = SmartFallMM(root_dir=data_dir)
    
    # Run data pipeline
    sm_dataset.pipeline(age_group=age_group, modalities=modalities, sensors=sensors)
    
    # Create dataset builder with multi-worker support
    builder_kwargs = {
        'verbose': arg.dataset_args.get('verbose', False),
        'use_dtw': arg.dataset_args.get('use_dtw', True),
        'num_workers': getattr(arg, 'num_worker', 0)  # Use num_worker parameter for parallel processing
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
    """Split dataset by subjects with comprehensive error handling"""
    try:
        builder.make_dataset(subjects, fuse)
        norm_data = builder.normalization()
        
        # Validate output data
        for key, value in norm_data.items():
            if key != 'labels':
                if value is None or len(value) == 0:
                    logging.warning(f"No data for {key} after processing")
        
        # Ensure labels exist
        if 'labels' not in norm_data or len(norm_data['labels']) == 0:
            # Find primary modality to match label count
            primary_key = next((k for k in norm_data if k != 'labels' and len(norm_data[k]) > 0), None)
            if primary_key:
                logging.warning(f"Creating dummy labels based on {primary_key} shape")
                norm_data['labels'] = np.zeros(len(norm_data[primary_key]), dtype=np.int32)
        
        # Ensure we have at least accelerometer data
        if 'accelerometer' not in norm_data or len(norm_data['accelerometer']) == 0:
            logging.warning("No accelerometer data, creating dummy data")
            if 'labels' in norm_data and len(norm_data['labels']) > 0:
                # Create dummy accelerometer data based on label count
                dummy_shape = (len(norm_data['labels']), 128, 3)
                norm_data['accelerometer'] = np.zeros(dummy_shape, dtype=np.float32)
            else:
                # Complete fallback
                norm_data['accelerometer'] = np.zeros((1, 128, 3), dtype=np.float32)
                norm_data['labels'] = np.zeros(1, dtype=np.int32)
        
        return norm_data
    except Exception as e:
        logging.error(f"Error in split_by_subjects_tf: {e}")
        logging.error(traceback.format_exc())
        
        # Return minimal valid dataset
        return {
            'accelerometer': np.zeros((1, 128, 3), dtype=np.float32),
            'skeleton': np.zeros((1, 128, 32, 3), dtype=np.float32),
            'labels': np.zeros(1, dtype=np.int32)
        }

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow implementation of the UTD-MM dataset with optimized data loading"""
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        
        # Extract data from dataset with robust error handling
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        # Handle missing accelerometer data
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset, using dummy data")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
            self.acc_seq = 128  # Default value
            self.channels = 3   # Default value
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
        
        # Handle missing skeleton data
        if self.skl_data is None or len(self.skl_data) == 0:
            logging.warning("No skeleton data in dataset, using dummy data")
            self.skl_data = np.zeros((self.num_samples, self.acc_seq, 32, 3), dtype=np.float32)
        else:
            # Normalize skeleton data shape
            if len(self.skl_data.shape) == 3:
                # Handle flattened skeleton data
                joints = self.skl_data.shape[2] // 3
                if joints * 3 == self.skl_data.shape[2]:
                    # Reshape to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_data.shape[0], self.skl_data.shape[1], joints, 3)
        
        # Handle missing labels
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
            
        # Convert to TensorFlow tensors and prepare dataset for high-performance loading
        self._prepare_data()
        
        # Initialize dataset indexing
        self.indices = np.arange(self.num_samples)
        
        logging.info(f"Initialized UTD_MM_TF with {self.num_samples} samples")
    
    def _prepare_data(self):
        """Prepare data for TensorFlow with optimized data loading"""
        try:
            # Handle potential NaNs
            self.acc_data = np.nan_to_num(self.acc_data)
            self.skl_data = np.nan_to_num(self.skl_data)
            
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
                logging.info(f"SMV calculated with shape: {self.smv.shape}")
                
            # Create TensorFlow dataset for efficient loading
            if hasattr(self, 'smv') and self.use_smv:
                # Dataset with SMV
                acc_with_smv = tf.concat([self.smv, self.acc_data], axis=-1)
                self.tf_dataset = tf.data.Dataset.from_tensor_slices(({
                    'accelerometer': acc_with_smv,
                    'skeleton': self.skl_data
                }, self.labels, tf.range(self.num_samples)))
            else:
                # Dataset without SMV
                self.tf_dataset = tf.data.Dataset.from_tensor_slices(({
                    'accelerometer': self.acc_data,
                    'skeleton': self.skl_data
                }, self.labels, tf.range(self.num_samples)))
                
            # Cache for performance
            self.tf_dataset = self.tf_dataset.cache()
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            logging.error(traceback.format_exc())
            
            # Create fallback tensors
            self.acc_data = tf.zeros((self.num_samples, self.acc_seq, 3), dtype=tf.float32)
            self.skl_data = tf.zeros((self.num_samples, self.acc_seq, 32, 3), dtype=tf.float32)
            self.labels = tf.zeros(self.num_samples, dtype=tf.int32)
            
            if self.use_smv:
                self.smv = tf.zeros((self.num_samples, self.acc_seq, 1), dtype=tf.float32)
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch"""
        np.random.shuffle(self.indices)
        
        # Create shuffled dataset based on new indices
        if hasattr(self, 'tf_dataset'):
            # Create fresh shuffled dataset
            shuffled_indices = tf.convert_to_tensor(self.indices)
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=self.num_samples)
    
    def __len__(self):
        """Return number of batches"""
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        """Get a batch of data with optimized implementation"""
        try:
            # Get indices for this batch
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Convert to tensor
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            # Gather data efficiently
            batch_data = {}
            batch_acc = tf.gather(self.acc_data, tf_indices)
            
            # Add SMV if requested
            if self.use_smv:
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, tf_indices)
                else:
                    # Calculate SMV on-the-fly
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
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logging.error(f"Error in batch generation {idx}: {e}")
            logging.error(traceback.format_exc())
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.acc_seq, 32, 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = tf.range(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
