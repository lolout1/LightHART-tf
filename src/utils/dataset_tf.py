# utils/dataset_tf.py
from typing import List, Dict, Tuple, Any
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Constants matching PyTorch implementation
SAMPLING_RATE = 200  # Hz (samples per second)
TARGET_DURATION = 12  # Target seconds for all instances
TARGET_SAMPLES = TARGET_DURATION * SAMPLING_RATE  # 12s * 200Hz = 2400 samples
TOLERANCE = 50 
TEST_YOUNG = ["SA03", "SA10", "SA15", "SA20"]
TEST_ELDERLY = ["SE02", "SE06", "SE10", "SE14"]
TEST_SUBJECTS = TEST_YOUNG + TEST_ELDERLY

# Helper functions for file loading - exact match to PyTorch
def csvloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads csv data - exact match to PyTorch implementation
    '''
    import pandas as pd
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        if 'skeleton' in file_path: 
            cols = 96
        else: 
            cols = 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def matloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads MatLab files - exact match to PyTorch implementation
    '''
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv': csvloader, 
    'mat': matloader
}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1, 
             max_length: int = 512, shape: int = None) -> np.ndarray:
    '''
    Executes average pooling to smoothen out the data - exact match to PyTorch implementation
    '''
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
    stride = ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
    sequence = sequence.numpy().squeeze(0).transpose(1, 0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    '''
    Pools and pads the sequence to uniform length - exact match to PyTorch implementation
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def sliding_window(data: Dict[str, np.ndarray], clearing_time_index: int, max_time: int, 
                   sub_window_size: int, stride_size: int, label: int) -> Dict[str, np.ndarray]:
    '''
    Sliding Window - exact match to PyTorch implementation
    '''
    assert clearing_time_index >= sub_window_size - 1, "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data['skeleton'].shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1

    sub_windows = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    
    result = {}
    for key in data.keys():
        if key != 'labels':
            result[key] = data[key][sub_windows]
    
    result['labels'] = np.repeat(label, len(result[list(result.keys())[0]]))
    return result

def selective_sliding_window(data: Dict[str, np.ndarray], length: int, window_size: int, 
                            stride_size: int, height: float, distance: int, 
                            label: int, fuse: bool) -> Dict[str, np.ndarray]:
    '''
    Selective sliding window - exact match to PyTorch implementation
    '''
    # Extract accelerometer data for peak detection
    if 'accelerometer' in data:
        acc_data = data['accelerometer']
    else:
        # Return empty result if no accelerometer data
        return {k: np.array([]) for k in data.keys()}
    
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
    
    # Fuse inertial data if requested
    if fuse and set(("accelerometer", "gyroscope")).issubset(windowed_data):
        windowed_data = fuse_inertial_data(windowed_data, window_size)
    
    # Add labels
    if windowed_data:
        sample_modality = next(iter(windowed_data.keys()))
        windowed_data['labels'] = np.repeat(label, len(windowed_data[sample_modality]))
    
    return windowed_data

def fuse_inertial_data(data: Dict[str, np.ndarray], window_size: int) -> Dict[str, np.ndarray]:
    '''
    Fusion of inertial data - exact match to PyTorch implementation
    '''
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

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    '''
    Convert quaternion to Euler angles - exact match to PyTorch implementation
    '''
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(q)
    return rot.as_euler('xyz', degrees=True)

def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    '''
    Index the different modalities with only selected ids - exact match to PyTorch
    '''
    return data[ids, :]

def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filtering indices those match with multiple other indices - exact match to PyTorch
    '''
    seen_first = set()
    seen_second = set()

    for (first, second) in path: 
        if first not in seen_first and second not in seen_second: 
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def align_sequence(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]: 
    '''
    Matching the skeleton and phone data using dynamic time warping - exact match to PyTorch
    '''
    joint_id = 9
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
    
    # Handle different skeleton data shapes
    skeleton_data = data['skeleton']
    if len(skeleton_data.shape) == 4:  # (batch, frames, joints, coords)
        skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
    elif len(skeleton_data.shape) == 3:  # (frames, joints, coords)
        skeleton_joint_data = skeleton_data[:, joint_id-1, :]
    else:  # (frames, flattened_features)
        skeleton_joint_data = skeleton_data[:, (joint_id-1)*3:joint_id*3]
    
    inertial_data = data[dynamic_keys[0]]
    
    if len(dynamic_keys) > 1: 
        gyroscope_data = data[dynamic_keys[1]]
        min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
        inertial_data = inertial_data[:min_len, :]
        data[dynamic_keys[1]] = gyroscope_data[:min_len, :]

    # Calculate Frobenius norm
    skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
    inertial_frob_norm = np.linalg.norm(inertial_data, axis=1)
    
    # DTW alignment
    distance, path = fastdtw(inertial_frob_norm[:, np.newaxis], 
                             skeleton_frob_norm[:, np.newaxis], 
                             dist=euclidean)

    # Filter repeated IDs
    inertial_ids, skeleton_idx = filter_repeated_ids(path)
    
    # Apply filtering
    data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
    for key in dynamic_keys: 
        data[key] = filter_data_by_ids(data[key], list(inertial_ids))
    
    return data

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, 
                       order: int = 4, filter_type: str = 'low') -> np.ndarray:
    '''
    Function to filter noise - exact match to PyTorch implementation
    '''
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

class DatasetBuilder:
    '''
    Builds a numpy file for the data and labels - exact match to PyTorch implementation
    '''
    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.diff = []
    
    def load_file(self, file_path: str) -> np.ndarray:
        '''Loads a file - exact match to PyTorch implementation'''
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        return data
    
    def _import_loader(self, file_path: str) -> callable:
        '''Imports the correct loader for the file type - exact match to PyTorch'''
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def process(self, data: Dict[str, np.ndarray], label: int) -> Dict[str, np.ndarray]:
        '''Process data - exact match to PyTorch implementation'''
        if self.mode == 'avg_pool':
            # Use average pooling
            result = {}
            for key, value in data.items():
                if key != 'labels':
                    result[key] = pad_sequence_numpy(
                        sequence=value, 
                        max_sequence_length=self.max_length,
                        input_shape=value.shape
                    )
            result['labels'] = data.get('labels', np.array([label]))
            return result
        else:
            # Use selective sliding window for fall detection
            if label == 1:  # Fall
                return selective_sliding_window(
                    data=data, 
                    length=data['skeleton'].shape[0],
                    window_size=self.max_length, 
                    stride_size=10, 
                    height=1.4, 
                    distance=50,
                    label=label,
                    fuse=self.fuse
                )
            else:  # Non-fall
                return selective_sliding_window(
                    data=data, 
                    length=data['skeleton'].shape[0],
                    window_size=self.max_length, 
                    stride_size=10, 
                    height=1.2, 
                    distance=100,
                    label=label,
                    fuse=self.fuse
                )
    
    def _add_trial_data(self, trial_data: Dict[str, np.ndarray]) -> None:
        '''Add processed trial data to the dataset - exact match to PyTorch'''
        for modality, modality_data in trial_data.items():
            self.data[modality].append(modality_data)
    
    def _len_check(self, d: Dict[str, np.ndarray]) -> bool:
        '''Check if all arrays have length > 1 - exact match to PyTorch'''
        return all(len(v) > 1 for v in d.values())
    
    def get_size_diff(self, trial_data: Dict[str, np.ndarray]) -> int:
        '''Get difference in size between accelerometer and skeleton - exact match to PyTorch'''
        return trial_data['accelerometer'].shape[0] - trial_data['skeleton'].shape[0]
    
    def store_trial_diff(self, difference: int) -> None:
        '''Store difference in trial sizes - exact match to PyTorch'''
        self.diff.append(difference)
    
    def viz_trial_diff(self) -> None:
        '''Visualize trial differences - exact match to PyTorch'''
        plt.hist(self.diff, bins=range(min(self.diff), max(self.diff) + 2, 200), 
                edgecolor='black', alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Numbers")
        plt.savefig("Distribution.png")
    
    def select_subwindow_pandas(self, unimodal_data: np.ndarray) -> np.ndarray:
        '''Select subwindow with pandas - exact match to PyTorch'''
        import pandas as pd
        n = len(unimodal_data)
        magnitude = np.linalg.norm(unimodal_data, axis=1)
        df = pd.DataFrame({"values": magnitude})
        df["variance"] = df["values"].rolling(window=125).var()
        
        # Get index of highest variance
        max_idx = df["variance"].idxmax()
        
        # Get segment
        final_start = max(0, max_idx-100)
        final_end = min(n, max_idx + 100)
        return unimodal_data[final_start:final_end, :]
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        '''Reads all files and makes a numpy array - exact match to PyTorch'''
        self.data = defaultdict(list)
        self.fuse = fuse
        count = 0
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                # Determine label based on task
                if self.task == 'fd': 
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                trial_data = defaultdict(np.ndarray)
                executed = False
                
                # Load data for each modality
                for modality, file_path in trial.files.items():
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    
                    try: 
                        executed = True
                        unimodal_data = self.load_file(file_path)
                        if unimodal_data is None:
                            executed = False
                            break
                            
                        trial_data[modality] = unimodal_data
                        
                        # Apply filters and preprocessing
                        if modality == 'accelerometer':
                            trial_data[modality] = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                            if unimodal_data.shape[0] > 250:
                                trial_data[modality] = self.select_subwindow_pandas(trial_data[modality])
                    except Exception as e:
                        executed = False
                        logging.error(f"Error processing {modality} from {file_path}: {e}")
                
                # Only continue if data loading was successful
                if executed and len(trial_data) >= 2:  # Need at least 2 modalities
                    # Apply DTW alignment
                    try:
                        trial_data = align_sequence(trial_data)
                        # Apply sliding window or avg pool
                        trial_data = self.process(trial_data, label)
                        
                        # Add to dataset if valid
                        if self._len_check(trial_data):
                            self._add_trial_data(trial_data)
                    except Exception as e:
                        logging.error(f"Error in alignment/processing: {e}")
                
                count += 1
        
        # Concatenate data
        for key in self.data:
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logging.error(f"Error concatenating {key}: {e}")
    
    def random_resampling(self) -> None:
        '''Random resampling for class balance - exact match to PyTorch'''
        from imblearn.under_sampling import RandomUnderSampler
        
        ros = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        
        # Flatten arrays for resampling
        num_samples = len(self.data['labels'])
        flattened_data = {}
        original_shapes = {}
        
        for key, value in self.data.items():
            if key != 'labels':
                original_shapes[key] = value.shape[1:]
                flattened_data[key] = value.reshape(num_samples, -1)
        
        # Apply resampling
        resampled_data = {}
        resampled_labels = None
        
        for key, value in flattened_data.items():
            if resampled_labels is None:
                resampled_value, resampled_labels = ros.fit_resample(value, self.data['labels'])
            else:
                resampled_value, _ = ros.fit_resample(value, self.data['labels'])
            
            # Reshape back to original shape
            resampled_data[key] = resampled_value.reshape(-1, *original_shapes[key])
        
        # Update data
        for key, value in resampled_data.items():
            self.data[key] = value
        
        self.data['labels'] = resampled_labels
    
    def normalization(self) -> Dict[str, np.ndarray]:
        '''Normalize data - exact match to PyTorch'''
        for key, value in self.data.items():
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)
        return self.data

# Main dataset classes
class ModalityFile:
    '''
    Represents an individual file in a modality - exact match to PyTorch implementation
    '''
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str:
        return (
            f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, file_path='{self.file_path}')"
        )

class Modality:
    '''
    Represents a modality - exact match to PyTorch implementation
    '''
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)
    
    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={self.files})"

class MatchedTrial:
    """
    Represents a matched trial - exact match to PyTorch implementation
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return (
            f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, files={self.files})"
        )

class SmartFallMM:
    """
    Represents the SmartFallMM dataset - exact match to PyTorch implementation
    """
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}
    
    def add_modality(self, age_group: str, modality_name: str) -> None:
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        
        self.age_groups[age_group][modality_name] = Modality(modality_name)
    
    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self) -> None:
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
                    logging.warning(f"Directory not found: {modality_dir}")
                    continue
                
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                            except Exception as e:
                                logging.warning(f"Error parsing file {file}: {e}")
                
                logging.info(f"Loaded {len(modality.files)} files for {modality_name} in {age_group}")
    
    def match_trials(self) -> None:
        '''Matches files from different modalities - exact match to PyTorch'''
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    matched_trial = self._find_or_create_matched_trial(
                        modality_file.subject_id,
                        modality_file.action_id,
                        modality_file.sequence_number
                    )
                    matched_trial.add_file(modality_name, modality_file.file_path)
        
        # Count trials with all required modalities
        required_modalities = set()
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities.keys():
                required_modalities.add(modality_name)
        
        complete_trials = []
        for trial in self.matched_trials:
            if all(modality in trial.files for modality in required_modalities):
                complete_trials.append(trial)
        
        self.matched_trials = complete_trials
        logging.info(f"Found {len(self.matched_trials)} complete matched trials")
    
    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and trial.action_id == action_id 
                    and trial.sequence_number == sequence_number):
                return trial
        
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial
    
    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]) -> None:
        '''Complete data pipeline - exact match to PyTorch'''
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        self.load_files()
        self.match_trials()

# Helper functions for SmartFallMM dataset
def prepare_smartfallmm(arg) -> DatasetBuilder:
    '''Function for dataset preparation - exact match to PyTorch'''
    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'))
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    builder = DatasetBuilder(
        sm_dataset, 
        arg.dataset_args['mode'], 
        arg.dataset_args['max_length'],
        arg.dataset_args['task']
    )
    return builder

def split_by_subjects(builder: DatasetBuilder, subjects: List[int], fuse: bool) -> Dict[str, np.ndarray]:
    '''Function to filter by subjects - exact match to PyTorch'''
    builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    return norm_data

# TensorFlow-specific dataset class for batching
class UTD_MM_TF(tf.keras.utils.Sequence):
    '''TensorFlow data feeder compatible with PyTorch UTD_MM'''
    def __init__(self, dataset: Dict[str, np.ndarray], batch_size: int):
        self.batch_size = batch_size
        self.dataset = dataset
        
        # Extract data
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        # Validate data
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 64, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Process and prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        '''Prepare data for TensorFlow - compatible with PyTorch'''
        try:
            # Convert to TensorFlow tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate signal magnitude vector (SMV)
            mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
            zero_mean = self.acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            self.smv = tf.sqrt(sum_squared)
            
            # Concatenate SMV with accelerometer data
            self.acc_data_with_smv = tf.concat([self.smv, self.acc_data], axis=-1)
            
            # Convert skeleton data if available
            if self.skl_data is not None and len(self.skl_data) > 0:
                self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            # Create fallback tensors
            shape = self.acc_data.shape if hasattr(self, 'acc_data') else (self.num_samples, 64, 3)
            self.acc_data = tf.zeros(shape, dtype=tf.float32)
            self.smv = tf.zeros((*shape[:-1], 1), dtype=tf.float32)
            self.acc_data_with_smv = tf.zeros((*shape[:-1], shape[-1]+1), dtype=tf.float32)
    
    def __len__(self):
        '''Number of batches - compatible with PyTorch'''
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        '''Get a batch - compatible with PyTorch'''
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # Create batch data
        batch_data = {}
        
        # Add accelerometer with SMV
        batch_data['accelerometer'] = tf.gather(self.acc_data_with_smv, tf.range(start_idx, end_idx))
        
        # Add skeleton if available
        if hasattr(self, 'skl_data') and self.skl_data is not None:
            batch_data['skeleton'] = tf.gather(self.skl_data, tf.range(start_idx, end_idx))
        
        # Get labels and indices
        batch_labels = tf.gather(self.labels, tf.range(start_idx, end_idx))
        batch_indices = tf.range(start_idx, end_idx).numpy()
        
        return batch_data, batch_labels, batch_indices
