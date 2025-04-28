import os
import logging
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, butter, filtfilt
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lightheart-tf')

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    import pandas as pd
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        
        # Determine columns based on the file path
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
            
        # Handle potential column count mismatches
        if file_data.shape[1] >= cols:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            logger.warning(f"File {file_path} has fewer columns than expected: {file_data.shape[1]} < {cols}")
            # Use all available columns and pad with zeros if needed
            actual_cols = file_data.shape[1]
            activity_data = np.zeros((file_data.shape[0]-2, cols), dtype=np.float32)
            activity_data[:, :actual_cols] = file_data.iloc[2:, :].to_numpy(dtype=np.float32)
            
        return activity_data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return np.array([])

def matloader(file_path: str, **kwargs) -> np.ndarray:
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        key = 'd_iner' if 'iner' in file_path else 'd_skel'
        logger.warning(f"Using default key {key} for {file_path}")
    try:
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logger.error(f"Error loading mat file {file_path}: {e}")
        return np.array([])

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
             max_length: int = 512, shape=None) -> np.ndarray:
    if sequence.size == 0:
        return np.array([])
        
    if shape is None:
        shape = sequence.shape
        
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
    
    stride = ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    
    try:
        sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool: {e}")
        # Return padded/truncated array as fallback
        result = np.zeros((max_length, *shape[1:]), dtype=sequence.dtype)
        copy_len = min(max_length, shape[0])
        result[:copy_len] = sequence[:copy_len]
        return result

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape=None) -> np.ndarray:
    if sequence.size == 0:
        return np.array([])
        
    if input_shape is None:
        input_shape = sequence.shape
        
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    # If sequence is longer than max_length, use avg_pool
    if sequence.shape[0] >= max_sequence_length:
        return avg_pool(sequence, max_length=max_sequence_length, shape=input_shape)
    
    # Otherwise, create a zero-filled array and copy the sequence
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(sequence)] = sequence
    return new_sequence

def sliding_window(data, clearing_time_index, max_time, sub_window_size, stride_size, label):
    result = defaultdict(list)
    
    # Handle missing data modalities gracefully
    for key, value in data.items():
        if key == 'labels':
            continue
            
        if value is None or value.size == 0:
            logger.warning(f"Empty data for {key}, skipping in sliding window")
            continue
            
        # Calculate valid window indices
        try:
            seq_length = value.shape[0]
            if seq_length < sub_window_size:
                logger.warning(f"Sequence length {seq_length} < window size {sub_window_size} for {key}")
                continue
                
            # Adjust max_time based on sequence length
            adjusted_max_time = min(max_time, seq_length - sub_window_size + 1)
            
            # Generate window indices
            windows = []
            for start_idx in range(0, adjusted_max_time, stride_size):
                end_idx = start_idx + sub_window_size
                if end_idx <= seq_length:
                    windows.append(value[start_idx:end_idx])
                    
            if windows:
                result[key] = np.stack(windows)
        except Exception as e:
            logger.error(f"Error in sliding_window for {key}: {e}")
    
    # Create labels array if we have any valid windows
    if result and any(v.shape[0] > 0 for v in result.values()):
        # Use the first modality's window count
        first_key = next(iter(result.keys()))
        window_count = result[first_key].shape[0]
        result['labels'] = np.full(window_count, label)
    
    return result

def selective_sliding_window(data, length, window_size, stride_size, height, distance, label, fuse=False):
    if not data or all(v.size == 0 for k, v in data.items() if k != 'labels'):
        logger.warning("No valid data for selective sliding window")
        return {k: np.array([]) for k in data.keys()}
    
    # Find peaks in accelerometer data if available
    peaks = []
    if 'accelerometer' in data and data['accelerometer'].size > 0:
        acc_data = data['accelerometer']
        try:
            # Calculate magnitude
            sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
            peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
            
            # If no peaks found, use regular sliding window
            if len(peaks) == 0:
                logger.warning(f"No peaks found with height={height}, distance={distance}")
                return sliding_window(data, window_size-1, length, window_size, stride_size, label)
        except Exception as e:
            logger.error(f"Error detecting peaks: {e}")
            return sliding_window(data, window_size-1, length, window_size, stride_size, label)
    else:
        # Without accelerometer data, fall back to regular sliding window
        return sliding_window(data, window_size-1, length, window_size, stride_size, label)
    
    # Extract windows around peaks
    windowed_data = defaultdict(list)
    for key, value in data.items():
        if key == 'labels' or value.size == 0:
            continue
            
        windows = []
        for peak in peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(value), start + window_size)
            
            # Ensure window is complete
            if end - start >= window_size:
                windows.append(value[start:end])
                
        if windows:
            windowed_data[key] = np.stack(windows)
    
    # Create labels if we have any windows
    if windowed_data and any(v.shape[0] > 0 for v in windowed_data.values()):
        # Use the first modality's window count
        first_key = next(iter(windowed_data.keys()))
        window_count = windowed_data[first_key].shape[0]
        windowed_data['labels'] = np.full(window_count, label)
    
    # Apply sensor fusion if requested
    if fuse and set(("accelerometer", "gyroscope")).issubset(windowed_data):
        try:
            windowed_data = fuse_inertial_data(windowed_data, window_size)
        except Exception as e:
            logger.error(f"Error in sensor fusion: {e}")
    
    return windowed_data

def quaternion_to_euler(q):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(q)
    return rot.as_euler('xyz', degrees=True)

def fuse_inertial_data(data, window_size):
    try:
        from ahrs.filters import Madgwick
        
        q = np.array([1, 0, 0, 0], dtype=np.float64)
        quaternions = []
        length = len(data['accelerometer'])
        madgwick = Madgwick()
        
        for i in range(length):
            transformed_windows = []
            for j in range(window_size):
                gyro_data = data['gyroscope'][i][j,:]
                acc_data = data['accelerometer'][i][j,:]
                q = madgwick.updateIMU(q, acc=acc_data, gyr=gyro_data)
                euler_angels = quaternion_to_euler(q)
                transformed_windows.append(euler_angels)
            quaternions.append(np.array(transformed_windows))
            
        data['fused'] = np.array(quaternions)
        return data
    except Exception as e:
        logger.error(f"Error in sensor fusion: {e}")
        return data

def filter_data_by_ids(data, ids):
    if data.size == 0 or len(ids) == 0:
        return np.array([])
    return data[ids, :]

def filter_repeated_ids(path):
    seen_first = set()
    seen_second = set()
    
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def align_sequence(data):
    if 'skeleton' not in data or data['skeleton'].size == 0:
        logger.warning("No skeleton data for DTW alignment")
        return data
        
    # Find accelerometer or gyroscope data
    dynamic_keys = [k for k in data.keys() if k != 'skeleton' and k != 'labels']
    if not dynamic_keys:
        logger.warning("No inertial data for DTW alignment")
        return data
    
    # Extract joint data
    joint_id = 9  # Use consistent joint for alignment
    skeleton_data = data['skeleton']
    
    # Handle different skeleton data shapes
    if len(skeleton_data.shape) == 4:  # [frames, joints, coords, dim]
        skeleton_joint_data = skeleton_data[:, joint_id-1, :]
    elif len(skeleton_data.shape) == 3:  # [frames, joints, coords]
        skeleton_joint_data = skeleton_data[:, joint_id-1, :]
    else:  # [frames, flattened_joints*coords]
        skeleton_joint_data = skeleton_data[:, (joint_id-1)*3:(joint_id)*3]
    
    # Get the first inertial modality
    inertial_key = dynamic_keys[0]
    inertial_data = data[inertial_key]
    
    # If multiple inertial modalities, align them first
    if len(dynamic_keys) > 1:
        for key in dynamic_keys[1:]:
            if key in data and data[key].size > 0:
                # Take the minimum length
                min_len = min(inertial_data.shape[0], data[key].shape[0])
                inertial_data = inertial_data[:min_len]
                data[key] = data[key][:min_len]
    
    try:
        # Calculate norms for DTW
        skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        inertial_norm = np.linalg.norm(inertial_data, axis=1)
        
        # Apply DTW
        distance, path = fastdtw(
            inertial_norm[:, np.newaxis], 
            skeleton_norm[:, np.newaxis],
            dist=euclidean
        )
        
        # Filter repeated indices
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        
        # Apply the filtered indices
        data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_ids))
        for key in dynamic_keys:
            if key in data and data[key].size > 0:
                data[key] = filter_data_by_ids(data[key], list(inertial_ids))
                
        logger.info(f"DTW alignment: skeleton {len(skeleton_ids)}, inertial {len(inertial_ids)}")
        
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
    
    return data

def butterworth_filter(data, cutoff=7.5, fs=25, order=4, filter_type='low'):
    if data.size == 0:
        return np.array([])
        
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logger.error(f"Error in Butterworth filter: {e}")
        return data

class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path
        
    def __repr__(self):
        return f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, file_path='{self.file_path}')"

class Modality:
    def __init__(self, name):
        self.name = name
        self.files = []
        
    def add_file(self, subject_id, action_id, sequence_number, file_path):
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))
        
    def __repr__(self):
        return f"Modality(name='{self.name}', files={len(self.files)} files)"

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
        
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path
        
    def __repr__(self):
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"

class SmartFallMM:
    def __init__(self, root_dir=None):
        self.root_dir = root_dir or os.path.join(os.getcwd(), 'data/smartfallmm')
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
        
    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        self.age_groups[age_group][modality_name] = Modality(modality_name)
        
    def select_sensor(self, modality_name, sensor_name=None):
        self.selected_sensors[modality_name] = sensor_name
        
    def load_files(self):
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                # Handle skeleton data (no sensor required)
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    # Only load from selected sensor
                    if modality_name not in self.selected_sensors:
                        continue
                    sensor_name = self.selected_sensors[modality_name]
                    if not sensor_name:
                        continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                
                if not os.path.exists(modality_dir):
                    logger.warning(f"Directory not found: {modality_dir}")
                    continue
                
                # Load files
                loaded_count = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                # Try various file name patterns
                                # Pattern 1: S01A01T01.csv
                                if len(file) >= 9 and file[0] == 'S' and file[3] == 'A' and file[6] == 'T':
                                    subject_id = int(file[1:3])
                                    action_id = int(file[4:6])
                                    sequence_number = int(file[7:9])
                                    file_path = os.path.join(root, file)
                                    modality.add_file(subject_id, action_id, sequence_number, file_path)
                                    loaded_count += 1
                                    continue
                                
                                # Pattern 2: S01_A01_T01.csv
                                parts = file.split('_')
                                if len(parts) >= 3:
                                    s_part = next((p for p in parts if p.startswith('S')), None)
                                    a_part = next((p for p in parts if p.startswith('A')), None)
                                    t_part = next((p for p in parts if p.startswith('T')), None)
                                    
                                    if s_part and a_part and t_part:
                                        subject_id = int(s_part[1:])
                                        action_id = int(a_part[1:])
                                        sequence_number = int(t_part[1:])
                                        file_path = os.path.join(root, file)
                                        modality.add_file(subject_id, action_id, sequence_number, file_path)
                                        loaded_count += 1
                                        continue
                                
                                # Pattern 3: Extract numbers from filename
                                import re
                                numbers = re.findall(r'\d+', file)
                                if len(numbers) >= 3:
                                    subject_id = int(numbers[0])
                                    action_id = int(numbers[1])
                                    sequence_number = int(numbers[2])
                                    file_path = os.path.join(root, file)
                                    modality.add_file(subject_id, action_id, sequence_number, file_path)
                                    loaded_count += 1
                                    continue
                                    
                                logger.warning(f"Unrecognized file name format: {file}")
                            except Exception as e:
                                logger.warning(f"Error parsing file {file}: {e}")
                
                logger.info(f"Loaded {loaded_count} files for {modality_name} in {age_group}")
    
    def match_trials(self):
        # Create a mapping of (subject_id, action_id, sequence_number) to files
        trial_map = {}
        
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for file in modality.files:
                    key = (file.subject_id, file.action_id, file.sequence_number)
                    
                    if key not in trial_map:
                        trial_map[key] = {}
                    
                    trial_map[key][modality_name] = file.file_path
        
        # Create matched trials
        for key, files in trial_map.items():
            subject_id, action_id, sequence_number = key
            
            # Check if we have at least one modality
            if files:
                trial = MatchedTrial(subject_id, action_id, sequence_number)
                
                for modality_name, file_path in files.items():
                    trial.add_file(modality_name, file_path)
                
                self.matched_trials.append(trial)
        
        logger.info(f"Created {len(self.matched_trials)} matched trials")
    
    def pipe_line(self, age_group, modalities, sensors):
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                
                if modality == 'skeleton':
                    self.select_sensor(modality, None)
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        self.load_files()
        self.match_trials()

class DatasetBuilder:
    def __init__(self, dataset, mode='sliding_window', max_length=128, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = kwargs.get('fuse', False)
        self.diff = []
        
    def load_file(self, file_path):
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                logger.warning(f"Unsupported file type: {file_type}")
                return np.array([])
                
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return np.array([])
    
    def process(self, data, label):
        if not data or all(v.size == 0 for k, v in data.items() if k != 'labels'):
            logger.warning(f"No valid data to process for label {label}")
            return {k: np.array([]) for k in data.keys()}
        
        if self.mode == 'avg_pool':
            result = {}
            for key, value in data.items():
                if key == 'labels':
                    result[key] = value
                    continue
                
                if value.size > 0:
                    result[key] = pad_sequence_numpy(
                        value, 
                        max_sequence_length=self.max_length,
                        input_shape=value.shape
                    )
                else:
                    result[key] = np.array([])
            return result
        else:
            # Use selective sliding window for accelerometer
            if 'accelerometer' in data and data['accelerometer'].size > 0:
                if label == 1:  # Fall
                    return selective_sliding_window(
                        data, 
                        length=data['accelerometer'].shape[0] if 'accelerometer' in data else 0,
                        window_size=self.max_length, 
                        stride_size=10, 
                        height=1.4, 
                        distance=50, 
                        label=label,
                        fuse=self.fuse
                    )
                else:  # Non-fall
                    return selective_sliding_window(
                        data, 
                        length=data['accelerometer'].shape[0] if 'accelerometer' in data else 0,
                        window_size=self.max_length, 
                        stride_size=10, 
                        height=1.2, 
                        distance=100, 
                        label=label,
                        fuse=self.fuse
                    )
            else:
                # Use regular sliding window for skeleton-only data
                return sliding_window(
                    data, 
                    clearing_time_index=self.max_length-1, 
                    max_time=data['skeleton'].shape[0] if 'skeleton' in data else 0, 
                    sub_window_size=self.max_length, 
                    stride_size=32, 
                    label=label
                )
    
    def select_subwindow_pandas(self, unimodal_data):
        if unimodal_data.size == 0 or unimodal_data.shape[0] <= 250:
            return unimodal_data
            
        try:
            import pandas as pd
            n = len(unimodal_data)
            magnitude = np.linalg.norm(unimodal_data, axis=1)
            df = pd.DataFrame({"values": magnitude})
            df["variance"] = df["values"].rolling(window=125).var()
            
            max_idx = df["variance"].idxmax()
            if pd.isna(max_idx):
                return unimodal_data[:250]  # Return first 250 frames if no valid max
                
            final_start = max(0, int(max_idx) - 100)
            final_end = min(n, final_start + 200)
            return unimodal_data[final_start:final_end]
        except Exception as e:
            logger.error(f"Error in select_subwindow_pandas: {e}")
            return unimodal_data[:250]  # Return first 250 frames as fallback
    
    def make_dataset(self, subjects, fuse=False):
        self.data = defaultdict(list)
        self.fuse = fuse
        
        processed_count = 0
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
                
            # Determine label based on task
            if self.task == 'fd':
                label = int(trial.action_id > 9)
            elif self.task == 'age':
                label = int(trial.subject_id < 29 or trial.subject_id > 46)
            else:
                label = trial.action_id - 1
            
            # Load data for each modality
            trial_data = {}
            for modality, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    if unimodal_data.size == 0:
                        logger.warning(f"Empty data for {modality} in trial {trial.subject_id}")
                        continue
                        
                    # Apply pre-processing
                    if modality == 'accelerometer':
                        unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                        if unimodal_data.shape[0] > 250:
                            unimodal_data = self.select_subwindow_pandas(unimodal_data)
                    elif modality == 'skeleton' and len(unimodal_data.shape) == 2 and unimodal_data.shape[1] == 96:
                        # Reshape 2D skeleton data to 3D
                        frames = unimodal_data.shape[0]
                        unimodal_data = unimodal_data.reshape(frames, 32, 3)
                        
                    trial_data[modality] = unimodal_data
                except Exception as e:
                    logger.error(f"Error processing {modality} in trial {trial.subject_id}: {e}")
            
            # Skip trials with no data
            if not trial_data:
                logger.warning(f"No valid data for trial {trial.subject_id}")
                continue
                
            # Apply DTW alignment if multiple modalities
            if len(trial_data) > 1 and 'skeleton' in trial_data:
                try:
                    trial_data = align_sequence(trial_data)
                except Exception as e:
                    logger.error(f"Error in alignment for trial {trial.subject_id}: {e}")
            
            # Process data (windowing, etc.)
            try:
                processed_data = self.process(trial_data, label)
                
                # Add to dataset if valid
                if processed_data and any(v.shape[0] > 0 for k, v in processed_data.items() if k != 'labels'):
                    for key, value in processed_data.items():
                        if value.shape[0] > 0:  # Skip empty arrays
                            self.data[key].append(value)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Error processing trial {trial.subject_id}: {e}")
        
        logger.info(f"Processed {processed_count} trials")
        
        # Concatenate data
        for key in list(self.data.keys()):
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    self.data[key] = np.array([])
        
        # Ensure all modalities have the same number of samples
        self._align_sample_counts()
        
        return self.data
    
    def _align_sample_counts(self):
        # Find the modality with the minimum number of samples
        sample_counts = {k: v.shape[0] for k, v in self.data.items() if k != 'labels' and v.size > 0}
        if not sample_counts:
            return
            
        min_count = min(sample_counts.values())
        
        # Truncate all modalities to this size
        for key in list(self.data.keys()):
            if key != 'labels' and self.data[key].size > 0 and self.data[key].shape[0] > min_count:
                self.data[key] = self.data[key][:min_count]
        
        # Ensure labels match
        if 'labels' in self.data and self.data['labels'].size > 0:
            self.data['labels'] = self.data['labels'][:min_count]
    
    def random_resampling(self):
        if 'labels' not in self.data or self.data['labels'].size == 0:
            logger.warning("No labels for resampling")
            return
            
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            # Check if we have data to resample
            if not any(k != 'labels' and v.size > 0 for k, v in self.data.items()):
                logger.warning("No data for resampling")
                return
                
            # Get labels
            y = self.data['labels']
            
            # Flatten data for resampling
            X_flat = {}
            orig_shapes = {}
            for key, value in self.data.items():
                if key != 'labels' and value.size > 0:
                    orig_shapes[key] = value.shape[1:]
                    X_flat[key] = value.reshape(value.shape[0], -1)
            
            # Initialize resampler
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            
            # Resample first modality and get indices
            first_key = next(iter(X_flat.keys()))
            X_resampled, y_resampled = rus.fit_resample(X_flat[first_key], y)
            
            # Update data with resampled values
            for key, value in X_flat.items():
                self.data[key] = X_resampled.reshape(-1, *orig_shapes[key])
                
            self.data['labels'] = y_resampled
            
            logger.info(f"Resampled data: {Counter(y_resampled)}")
        except Exception as e:
            logger.error(f"Error in random resampling: {e}")
    
    def normalization(self):
        if not self.data:
            logger.warning("No data to normalize")
            return self.data
            
        normalized_data = {}
        
        # Copy labels directly
        if 'labels' in self.data:
            normalized_data['labels'] = self.data['labels']
        
        # Normalize each modality
        for key, value in self.data.items():
            if key == 'labels' or value.size == 0:
                continue
                
            try:
                # Reshape for normalization
                orig_shape = value.shape
                reshaped = value.reshape(-1, value.shape[-1])
                
                # Handle NaN values
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    logger.warning(f"NaN or Inf values in {key}, replacing with zeros")
                    reshaped = np.nan_to_num(reshaped)
                
                # Apply standardization
                scaler = StandardScaler()
                normalized = scaler.fit_transform(reshaped)
                
                # Reshape back
                normalized_data[key] = normalized.reshape(orig_shape)
                
            except Exception as e:
                logger.error(f"Error normalizing {key}: {e}")
                normalized_data[key] = value  # Use original if normalization fails
        
        return normalized_data

# Data loader compatible with TensorFlow
class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, modalities=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get data modalities
        self.modalities = modalities or ['accelerometer']
        
        # Validate dataset
        if not isinstance(dataset, dict) or 'labels' not in dataset:
            raise ValueError("Dataset must be a dictionary with 'labels' key")
            
        self.labels = dataset['labels']
        if self.labels.size == 0:
            raise ValueError("Empty labels in dataset")
            
        self.num_samples = len(self.labels)
        
        # Initialize indices
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Process input data
        self.data = {}
        for modality in self.modalities:
            modality_data = dataset.get(modality)
            if modality_data is None or modality_data.size == 0:
                logger.warning(f"Missing or empty {modality} data, creating dummy data")
                # Create dummy data with correct first dimension
                dummy_shape = (self.num_samples, 128, 3)  # Default shape
                self.data[modality] = np.zeros(dummy_shape, dtype=np.float32)
                continue
                
            # Convert to tensor
            tensor = tf.convert_to_tensor(modality_data, dtype=tf.float32)
            
            # Handle different shapes
            if modality == 'skeleton':
                if len(tensor.shape) == 3 and tensor.shape[2] == 96:
                    # Reshape flat skeleton data
                    tensor = tf.reshape(tensor, [-1, tensor.shape[1], 32, 3])
            
            self.data[modality] = tensor
            
        # Add signal magnitude vector for accelerometer
        if 'accelerometer' in self.data:
            self._compute_smv()
    
    def _compute_smv(self):
        # Skip if no accelerometer data or already has 4 channels (SMV included)
        if 'accelerometer' not in self.data or self.data['accelerometer'].shape[-1] >= 4:
            return
            
        # Calculate SMV
        acc_data = self.data['accelerometer']
        mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
        zero_mean = acc_data - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        
        # Concatenate with original data
        self.data['accelerometer'] = tf.concat([acc_data, smv], axis=-1)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        # Calculate batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        
        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = min(self.batch_size, self.num_samples)
        
        # Get indices for this batch
        batch_indices = self.indices[start_idx:end_idx]
        
        # Gather data for each modality
        batch_data = {}
        for modality in self.modalities:
            if modality in self.data:
                batch_data[modality] = tf.gather(self.data[modality], batch_indices)
        
        # Get labels
        batch_labels = tf.gather(self.labels, batch_indices)
        
        return batch_data, batch_labels, batch_indices
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def prepare_smartfallmm_tf(arg):
    # Get root directory
    root_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    
    # Create dataset
    dataset = SmartFallMM(root_dir=root_dir)
    
    # Configure pipeline
    dataset.pipe_line(
        age_group=arg.dataset_args.get('age_group', ['young']),
        modalities=arg.dataset_args.get('modalities', ['accelerometer']),
        sensors=arg.dataset_args.get('sensors', ['watch'])
    )
    
    # Create dataset builder
    builder = DatasetBuilder(
        dataset=dataset,
        mode=arg.dataset_args.get('mode', 'sliding_window'),
        max_length=arg.dataset_args.get('max_length', 128),
        task=arg.dataset_args.get('task', 'fd'),
        fuse=arg.distill_args.get('fuse', False) if hasattr(arg, 'distill_args') and arg.distill_args else False
    )
    
    return builder

def split_by_subjects(builder, subjects, fuse=False):
    if not subjects:
        logger.warning("No subjects specified for split")
        return {}
        
    # Make dataset
    data = builder.make_dataset(subjects, fuse=fuse)
    
    # Apply normalization
    normalized_data = builder.normalization()
    
    return normalized_data
