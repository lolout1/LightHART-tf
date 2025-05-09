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
import traceback

SAMPLING_RATE = 30
TARGET_DURATION = 12
TARGET_SAMPLES = TARGET_DURATION * SAMPLING_RATE
TOLERANCE = 50 
TEST_YOUNG = ["SA03", "SA10", "SA15", "SA20"]
TEST_ELDERLY = ["SE02", "SE06", "SE10", "SE14"]
TEST_SUBJECTS = TEST_YOUNG + TEST_ELDERLY

def csvloader(file_path: str, **kwargs) -> np.ndarray:
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
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1, max_length: int = 512, shape: int = None) -> np.ndarray:
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
    stride = ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
    sequence = sequence.numpy().squeeze(0).transpose(1, 0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.array) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    # Simple quaternion to euler conversion without requiring ahrs
    x, y, z, w = q[1], q[2], q[3], q[0]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return np.array([roll, pitch, yaw]) * 180.0 / np.pi  # Convert to degrees

def simple_fuse_inertial_data(data: Dict[str, np.ndarray], window_size: int) -> Dict[str, np.ndarray]:
    """Simplified fusion that doesn't require ahrs"""
    length = len(data['accelerometer'])
    fused_data = []
    for i in range(length):
        transformed_windows = []
        for j in range(window_size):
            # Just use normalized values of accelerometer as a simple orientation estimation
            acc_sample = data['accelerometer'][i][j,:]
            # Simple normalization to get direction
            norm = np.linalg.norm(acc_sample)
            if norm > 0:
                acc_norm = acc_sample / norm
            else:
                acc_norm = np.zeros_like(acc_sample)
            # Convert to pseudo euler angles
            roll = np.arctan2(acc_norm[1], acc_norm[2]) * 180 / np.pi
            pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180 / np.pi
            yaw = 0  # Can't determine yaw from accelerometer alone
            transformed_windows.append(np.array([roll, pitch, yaw]))
        fused_data.append(np.array(transformed_windows))
    data['fused'] = np.array(fused_data)
    return data

def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    return data[ids, :]

def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    seen_first = set()
    seen_second = set()
    for (first, second) in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    return seen_first, seen_second

def align_sequence(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    joint_id = 9
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
    if "skeleton" not in data or not dynamic_keys:
        return data
    skeleton_data = data['skeleton']
    if len(skeleton_data.shape) == 4:
        skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
    elif len(skeleton_data.shape) == 3:
        skeleton_joint_data = skeleton_data[:, joint_id-1, :]
    else:
        skeleton_joint_data = skeleton_data[:, (joint_id-1)*3:joint_id*3]
    inertial_data = data[dynamic_keys[0]]
    if len(dynamic_keys) > 1:
        gyroscope_data = data[dynamic_keys[1]]
        min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
        inertial_data = inertial_data[:min_len, :]
        data[dynamic_keys[1]] = gyroscope_data[:min_len, :]
    skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
    interial_frob_norm = np.linalg.norm(inertial_data, axis=1)
    distance, path = fastdtw(interial_frob_norm[:, np.newaxis], skeleton_frob_norm[:, np.newaxis], dist=euclidean)
    interial_ids, skeleton_idx = filter_repeated_ids(path)
    data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
    for key in dynamic_keys:
        data[key] = filter_data_by_ids(data[key], list(interial_ids))
    return data

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, order: int = 4, filter_type: str = 'low') -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

def sliding_window(data: Dict[str, np.ndarray], clearing_time_index: int, max_time: int, sub_window_size: int, stride_size: int, label: int) -> Dict[str, np.ndarray]:
    assert clearing_time_index >= sub_window_size - 1, "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1
    if max_time >= data['skeleton'].shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
    sub_windows = (start + np.expand_dims(np.arange(sub_window_size), 0) + np.expand_dims(np.arange(max_time, step=stride_size), 0).T)
    result = {}
    for key in data.keys():
        if key != 'labels':
            result[key] = data[key][sub_windows]
    result['labels'] = np.repeat(label, len(result[list(result.keys())[0]]))
    return result

def selective_sliding_window(data: Dict[str, np.ndarray], length: int, window_size: int, stride_size: int, height: float, distance: int, label: int, fuse: bool) -> Dict[str, np.ndarray]:
    if 'accelerometer' not in data:
        return {k: np.array([]) for k in data.keys()}
    acc_data = data['accelerometer']
    sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
    peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
    windowed_data = defaultdict(list)
    for modality, modality_data in data.items():
        if modality == 'labels':
            continue
        windows = []
        for peak in peaks:
            start = max(0, peak - window_size//2)
            end = min(len(modality_data), start + window_size)
            if end - start < window_size:
                continue
            windows.append(modality_data[start:end])
        if windows:
            windowed_data[modality] = np.stack(windows)
    if fuse and set(("accelerometer", "gyroscope")).issubset(windowed_data):
        windowed_data = simple_fuse_inertial_data(windowed_data, window_size)
    if windowed_data:
        sample_modality = next(iter(windowed_data.keys()))
        windowed_data['labels'] = np.repeat(label, len(windowed_data[sample_modality]))
    return windowed_data

class DatasetBuilder:
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
    
    def load_file(self, file_path):
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        return data
    
    def _import_loader(self, file_path:str) -> callable:
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def process(self, data: Dict[str, np.ndarray], label: int) -> Dict[str, np.ndarray]:
        if self.mode == 'avg_pool':
            result = {}
            for key, value in data.items():
                if key != 'labels':
                    result[key] = pad_sequence_numpy(sequence=value, max_sequence_length=self.max_length, input_shape=value.shape)
            result['labels'] = data.get('labels', np.array([label]))
            return result
        else:
            if label == 1:
                return selective_sliding_window(data=data, length=data['skeleton'].shape[0], window_size=self.max_length,
                                              stride_size=10, height=1.4, distance=50, label=label, fuse=self.fuse)
            else:
                return selective_sliding_window(data=data, length=data['skeleton'].shape[0], window_size=self.max_length,
                                              stride_size=10, height=1.2, distance=100, label=label, fuse=self.fuse)
    
    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            self.data[modality].append(modality_data)
    
    def _len_check(self, d):
        return all(len(v) > 1 for v in d.values())
    
    def get_size_diff(self, trial_data):
        return trial_data['accelerometer'].shape[0] - trial_data['skeleton'].shape[0]
    
    def store_trial_diff(self, difference):
        self.diff.append(difference)
    
    def select_subwindow_pandas(self, unimodal_data):
        import pandas as pd
        n = len(unimodal_data)
        magnitude = np.linalg.norm(unimodal_data, axis=1)
        df = pd.DataFrame({"values": magnitude})
        df["variance"] = df["values"].rolling(window=125).var()
        max_idx = df["variance"].idxmax()
        final_start = max(0, max_idx-100)
        final_end = min(n, max_idx + 100)
        return unimodal_data[final_start:final_end, :]
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        self.data = defaultdict(list)
        self.fuse = fuse
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                trial_data = defaultdict(np.ndarray)
                executed = True
                for modality, file_path in trial.files.items():
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    try:
                        unimodal_data = self.load_file(file_path)
                        if unimodal_data is None:
                            executed = False
                            break
                        trial_data[modality] = unimodal_data
                        if modality == 'accelerometer':
                            unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                        if modality == 'accelerometer' and unimodal_data.shape[0] > 250:
                            trial_data[modality] = self.select_subwindow_pandas(unimodal_data)
                    except Exception as e:
                        executed = False
                        logging.error(f"Error processing {modality} from {file_path}: {e}")
                if executed and len(trial_data) >= 2:
                    try:
                        trial_data = align_sequence(trial_data)
                        trial_data = self.process(trial_data, label)
                        if self._len_check(trial_data):
                            self._add_trial_data(trial_data)
                    except Exception as e:
                        logging.error(f"Error in alignment/processing: {e}")
        for key in self.data:
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logging.error(f"Error concatenating {key}: {e}")
    
    def random_resampling(self):
        try:
            from imblearn.under_sampling import RandomUnderSampler
            ros = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            num_samples = len(self.data['labels'])
            flattened_data = {}
            original_shapes = {}
            for key, value in self.data.items():
                if key != 'labels':
                    original_shapes[key] = value.shape[1:]
                    flattened_data[key] = value.reshape(num_samples, -1)
            resampled_data = {}
            resampled_labels = None
            for key, value in flattened_data.items():
                if resampled_labels is None:
                    resampled_value, resampled_labels = ros.fit_resample(value, self.data['labels'])
                else:
                    resampled_value, _ = ros.fit_resample(value, self.data['labels'])
                resampled_data[key] = resampled_value.reshape(-1, *original_shapes[key])
            for key, value in resampled_data.items():
                self.data[key] = value
            self.data['labels'] = resampled_labels
        except ImportError:
            logging.warning("imblearn not available, skipping resampling")
    
    def normalization(self) -> Dict[str, np.ndarray]:
        for key, value in self.data.items():
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)
        return self.data

class ModalityFile:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)

class MatchedTrial:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

class SmartFallMM:
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
    
    def match_trials(self) -> None:
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path
        required_modalities = list(self.age_groups['young'].keys())
        complete_trials = []
        for key, files_dict in trial_dict.items():
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                complete_trials.append(matched_trial)
        self.matched_trials = complete_trials
    
    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]):
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

def prepare_smartfallmm_tf(arg) -> DatasetBuilder:
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm')
    sm_dataset = SmartFallMM(root_dir=data_dir)
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

def split_by_subjects_tf(builder: DatasetBuilder, subjects: List[int], fuse: bool) -> Dict[str, np.ndarray]:
    builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    return norm_data
