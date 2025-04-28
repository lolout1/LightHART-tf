from typing import List, Dict, Tuple, Any
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions for file loading
def csvloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads csv data - matches PyTorch implementation
    '''
    import pandas as pd
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        cols = 96 if 'skeleton' in file_path else 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return np.array([])

def matloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads MatLab files - matches PyTorch implementation
    '''
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
             max_length: int = 512, shape: int = None) -> np.ndarray:
    '''
    Executes average pooling to smoothen out the data - matches PyTorch implementation
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
    Pads or repeats sequence to uniform length - matches PyTorch implementation
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    seq_length = sequence.shape[0]
    if seq_length >= max_sequence_length:
        return avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    # Repeat sequence to reach or exceed max_length
    padded_sequence = np.zeros(shape, sequence.dtype)
    repeat_count = (max_sequence_length + seq_length - 1) // seq_length
    repeated = np.tile(sequence, (repeat_count, 1))[:max_sequence_length]
    padded_sequence[:len(repeated)] = repeated
    return padded_sequence

def sliding_window(data: Dict[str, np.ndarray], clearing_time_index: int, max_time: int,
                   sub_window_size: int, stride_size: int, label: int) -> Dict[str, np.ndarray]:
    '''
    Regular sliding window - matches PyTorch implementation
    '''
    result = defaultdict(list)
    for key in data.keys():
        if key != 'labels' and data[key].size > 0:
            modality_data = data[key]
            seq_length = modality_data.shape[0]
            logging.debug(f"Processing {key} with sequence length {seq_length}, window size {sub_window_size}")
            if seq_length < 10:  # Skip very short sequences
                logging.warning(f"Sequence length {seq_length} for {key} too short, skipping")
                continue
            if seq_length < sub_window_size:
                logging.info(f"Repeating {key} sequence from {seq_length} to {sub_window_size}")
                modality_data = pad_sequence_numpy(modality_data, sub_window_size, modality_data.shape)
                seq_length = sub_window_size
            max_time = max(1, seq_length - sub_window_size + 1)
            sub_windows = (
                np.expand_dims(np.arange(sub_window_size), 0) +
                np.expand_dims(np.arange(0, max_time, stride_size), 0).T
            )
            try:
                windows = modality_data[sub_windows]
                result[key] = windows
            except Exception as e:
                logging.error(f"Error generating windows for {key}: {e}")
                result[key] = np.array([])
    if not result or all(v.size == 0 for v in result.values()):
        logging.error("No valid windows generated in sliding_window")
        return {k: np.array([]) for k in data.keys()}
    result['labels'] = np.repeat(label, len(result[list(result.keys())[0]]))
    return result

def selective_sliding_window(data: Dict[str, np.ndarray], length: int, window_size: int,
                            stride_size: int, height: float, distance: int,
                            label: int, fuse: bool) -> Dict[str, np.ndarray]:
    '''
    Selective sliding window with peak detection - matches PyTorch implementation
    '''
    if 'accelerometer' not in data or data['accelerometer'].size == 0:
        logging.warning("No accelerometer data available, falling back to sliding window")
        return sliding_window(data, window_size-1, length, window_size, stride_size, label)
    acc_data = data['accelerometer']
    sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
    peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
    windowed_data = defaultdict(list)
    if len(peaks) == 0:
        logging.warning(f"No peaks detected with height={height}, distance={distance}, falling back to sliding window")
        return sliding_window(data, window_size-1, length, window_size, stride_size, label)
    for modality, modality_data in data.items():
        if modality == 'labels' or modality_data.size == 0:
            continue
        windows = []
        for peak in peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(modality_data), start + window_size)
            if end - start >= window_size:
                windows.append(modality_data[start:end])
        if windows:
            windowed_data[modality] = np.stack(windows)
    if fuse and set(("accelerometer", "gyroscope")).issubset(windowed_data):
        windowed_data = fuse_inertial_data(windowed_data, window_size)
    if windowed_data:
        sample_modality = next(iter(windowed_data.keys()))
        windowed_data['labels'] = np.repeat(label, len(windowed_data[sample_modality]))
    else:
        logging.warning("No windows generated, falling back to sliding window")
        return sliding_window(data, window_size-1, length, window_size, stride_size, label)
    return windowed_data

def fuse_inertial_data(data: Dict[str, np.ndarray], window_size: int) -> Dict[str, np.ndarray]:
    '''
    Fusion of inertial data - matches PyTorch implementation
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
            euler_angles = quaternion_to_euler(q)
            transformed_windows.append(euler_angles)
        quaternions.append(np.array(transformed_windows))
    data['fused'] = np.array(quaternions)
    return data

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    '''
    Convert quaternion to Euler angles - matches PyTorch implementation
    '''
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(q)
    return rot.as_euler('xyz', degrees=True)

def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    return data[ids, :]

def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    seen_first = set()
    seen_second = set()
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    return seen_first, seen_second

def align_sequence(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    joint_id = 9
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
    skeleton_data = data.get('skeleton')
    if skeleton_data is None or skeleton_data.size == 0:
        logging.error("Skeleton data missing or empty")
        return {k: np.array([]) for k in data.keys()}
    if len(skeleton_data.shape) == 4:
        skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
    elif len(skeleton_data.shape) == 3:
        skeleton_joint_data = skeleton_data[:, joint_id-1, :]
    else:
        skeleton_joint_data = skeleton_data[:, (joint_id-1)*3:joint_id*3]
    if not dynamic_keys:
        return data
    inertial_data = data[dynamic_keys[0]]
    if len(dynamic_keys) > 1:
        gyroscope_data = data[dynamic_keys[1]]
        min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
        inertial_data = inertial_data[:min_len, :]
        data[dynamic_keys[1]] = gyroscope_data[:min_len, :]
    skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
    inertial_frob_norm = np.linalg.norm(inertial_data, axis=1)
    distance, path = fastdtw(inertial_frob_norm[:, np.newaxis], skeleton_frob_norm[:, np.newaxis], dist=euclidean)
    inertial_ids, skeleton_idx = filter_repeated_ids(path)
    data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
    for key in dynamic_keys:
        data[key] = filter_data_by_ids(data[key], list(inertial_ids))
    return data

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25,
                       order: int = 4, filter_type: str = 'low') -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

class DatasetBuilder:
    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = kwargs.get('fuse', False)
        self.diff = []

    def load_file(self, file_path: str) -> np.ndarray:
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        if data is None or data.size == 0:
            logging.error(f"Failed to load file: {file_path}")
            return np.array([])
        return data

    def _import_loader(self, file_path: str) -> callable:
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]

    def process(self, data: Dict[str, np.ndarray], label: int) -> Dict[str, np.ndarray]:
        if not data or all(v.size == 0 for v in data.values() if v is not None):
            logging.error("Empty or invalid input data")
            return {k: np.array([]) for k in data.keys()}
        modalities = [k for k in data.keys() if k != 'labels']
        result = defaultdict(list)
        for modality in modalities:
            modality_data = {modality: data[modality], 'labels': data.get('labels', np.array([label]))}
            if modality == 'skeleton':
                processed = sliding_window(
                    data=modality_data,
                    clearing_time_index=self.max_length-1,
                    max_time=data[modality].shape[0],
                    sub_window_size=self.max_length,
                    stride_size=10,
                    label=label
                )
            elif modality == 'accelerometer':
                processed = selective_sliding_window(
                    data=modality_data,
                    length=data[modality].shape[0],
                    window_size=self.max_length,
                    stride_size=10,
                    height=1.4 if label == 1 else 1.2,
                    distance=50 if label == 1 else 100,
                    label=label,
                    fuse=self.fuse
                )
            else:
                logging.warning(f"Unsupported modality {modality}, using sliding window")
                processed = sliding_window(
                    data=modality_data,
                    clearing_time_index=self.max_length-1,
                    max_time=data[modality].shape[0],
                    sub_window_size=self.max_length,
                    stride_size=10,
                    label=label
                )
            if not any(v.size > 0 for v in processed.values()):
                logging.warning(f"No valid windows generated for {modality}")
                continue
            for key, value in processed.items():
                result[key].append(value)
        if not result:
            logging.error("No valid windows generated for any modality")
            return {k: np.array([]) for k in data.keys()}
        for key in result:
            result[key] = np.concatenate(result[key], axis=0)
        return result

    def _add_trial_data(self, trial_data: Dict[str, np.ndarray]) -> None:
        for modality, modality_data in trial_data.items():
            if modality_data.size > 0:
                self.data[modality].append(modality_data)
            else:
                logging.warning(f"Empty data for {modality}, skipping")

    def _len_check(self, d: Dict[str, np.ndarray]) -> bool:
        return all(len(v) > 0 for v in d.values() if v.size > 0)

    def get_size_diff(self, trial_data: Dict[str, np.ndarray]) -> int:
        acc_len = trial_data.get('accelerometer', np.array([])).shape[0]
        skl_len = trial_data.get('skeleton', np.array([])).shape[0]
        return acc_len - skl_len

    def store_trial_diff(self, difference: int) -> None:
        self.diff.append(difference)

    def viz_trial_diff(self) -> None:
        if not self.diff:
            logging.warning("No trial differences to visualize")
            return
        plt.hist(self.diff, bins=range(min(self.diff), max(self.diff) + 2, 200),
                 edgecolor='black', alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Numbers")
        plt.savefig("Distribution.png")
        plt.close()

    def select_subwindow_pandas(self, unimodal_data: np.ndarray) -> np.ndarray:
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

    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        self.data = defaultdict(list)
        self.fuse = fuse
        count = 0
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                trial_data = defaultdict(np.ndarray)
                executed = False
                for modality, file_path in trial.files.items():
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys.get(modality.lower())
                    try:
                        executed = True
                        unimodal_data = self.load_file(file_path)
                        if unimodal_data is None or unimodal_data.size == 0:
                            logging.warning(f"Empty data loaded for {modality} at {file_path}")
                            executed = False
                            break
                        if unimodal_data.shape[0] < 10:
                            logging.warning(f"Sequence too short ({unimodal_data.shape[0]} frames) for {modality} at {file_path}")
                            executed = False
                            break
                        trial_data[modality] = unimodal_data
                        if modality == 'accelerometer':
                            trial_data[modality] = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                            if unimodal_data.shape[0] > 250:
                                trial_data[modality] = self.select_subwindow_pandas(trial_data[modality])
                    except Exception as e:
                        logging.error(f"Error processing {modality} from {file_path}: {e}")
                        executed = False
                        break
                if executed and len(trial_data) >= 1:
                    try:
                        if 'skeleton' in trial_data and any(k in trial_data for k in ['accelerometer', 'gyroscope']):
                            trial_data = align_sequence(trial_data)
                        processed_data = self.process(trial_data, label)
                        if self._len_check(processed_data):
                            self._add_trial_data(processed_data)
                            count += 1
                        else:
                            logging.warning(f"Empty processed data for trial {trial.subject_id}")
                    except Exception as e:
                        logging.error(f"Error in alignment/processing for trial {trial.subject_id}: {e}")
        logging.info(f"Processed {count} trials")
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logging.error(f"Error concatenating {key}: {e}")
                    self.data[key] = np.array([])

    def random_resampling(self) -> None:
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

    def normalization(self) -> Dict[str, np.ndarray]:
        for key, value in self.data.items():
            if key != 'labels' and value.size > 0:
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

    def __repr__(self) -> str:
        return f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, file_path='{self.file_path}')"

class Modality:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []

    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))

    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={self.files})"

class MatchedTrial:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}

    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"

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
                logging.info(f"Loaded {len(modality.files)} files for {modality_name} in {age_group}")

    def match_trials(self) -> None:
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    matched_trial = self._find_or_create_matched_trial(
                        modality_file.subject_id,
                        modality_file.action_id,
                        modality_file.sequence_number
                    )
                    matched_trial.add_file(modality_name, modality_file.file_path)
        required_modalities = set()
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities.keys():
                required_modalities.add(modality_name)
        complete_trials = []
        for trial in self.matched_trials:
            if any(modality in trial.files for modality in required_modalities):
                complete_trials.append(trial)
        self.matched_trials = complete_trials
        logging.info(f"Found {len(self.matched_trials)} complete matched trials")

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        for trial in self.matched_trials:
            if trial.subject_id == subject_id and trial.action_id == action_id and trial.sequence_number == sequence_number:
                return trial
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial

    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]) -> None:
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
    builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    return norm_data

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, modalities=['skeleton'], shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.modalities = modalities
        self.shuffle = shuffle
        self.labels = dataset.get('labels')
        if self.labels is None or len(self.labels) == 0:
            logging.error("No labels found in dataset")
            raise ValueError("Dataset must contain non-empty labels")
        self.num_samples = len(self.labels)
        self.indices = tf.range(self.num_samples)
        if self.shuffle:
            self.indices = tf.random.shuffle(self.indices)
        self.data = {}
        for modality in self.modalities:
            modality_data = dataset.get(modality, dataset.get(f'{modality}_data'))
            if modality_data is None or modality_data.size == 0:
                logging.error(f"{modality} data is required but not found or empty")
                raise ValueError(f"{modality} data is required but not found or empty")
            tensor = tf.convert_to_tensor(modality_data, dtype=tf.float32)
            if modality == 'skeleton' and len(tensor.shape) == 3 and tensor.shape[2] == 96:
                tensor = tf.reshape(tensor, [-1, tensor.shape[1], 32, 3])
            self.data[modality] = tensor
        if 'accelerometer' in self.modalities:
            self._compute_smv()

    def _compute_smv(self):
        mean = tf.reduce_mean(self.data['accelerometer'], axis=1, keepdims=True)
        zero_mean = self.data['accelerometer'] - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        self.data['accelerometer'] = tf.concat([self.data['accelerometer'], smv], axis=-1)

    def __len__(self):
        return int(tf.math.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            start_idx = 0
            end_idx = min(self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        data = {modality: tf.gather(self.data[modality], batch_indices) for modality in self.modalities}
        batch_labels = tf.gather(self.labels, batch_indices)
        return data, batch_labels, batch_indices

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = tf.random.shuffle(self.indices)
