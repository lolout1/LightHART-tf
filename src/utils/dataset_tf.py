import os
import logging
import traceback
from typing import List, Dict, Tuple, Any
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler

SAMPLING_RATE = 30
TARGET_SAMPLES = 360  # 12s * 30Hz
TEST_YOUNG = ["SA03", "SA10", "SA15", "SA20"]
TEST_ELDERLY = ["SE02", "SE06", "SE10", "SE14"]
TEST_SUBJECTS = TEST_YOUNG + TEST_ELDERLY

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    import pandas as pd
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        cols = 96 if 'skeleton' in file_path else 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def matloader(file_path: str, **kwargs) -> np.ndarray:
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    return loadmat(file_path)[key]

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def butterworth_filter(data: np.ndarray, cutoff: float = 7.5, fs: float = 25, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def avg_pool(sequence: np.ndarray, max_length: int) -> np.ndarray:
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
    stride = ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = tf.nn.avg_pool1d(sequence, ksize=5, strides=stride, padding='VALID')
    sequence = sequence.numpy().squeeze(0).transpose(1, 0)
    return sequence.reshape(-1, *shape[1:])

def pad_sequence_numpy(sequence: np.ndarray, max_length: int) -> np.ndarray:
    shape = list(sequence.shape)
    shape[0] = max_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_length)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def ensure_3d(array, expected_shape=None):
    """Ensure array is 3D with consistent shapes"""
    if array is None:
        return None
        
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

def align_sequence(data: Dict[str, np.ndarray], use_dtw: bool = True) -> Dict[str, np.ndarray]:
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

    # Proceed with DTW
    joint_id = 9
    inertial_key = "accelerometer" if "accelerometer" in data else "gyroscope"
    
    skeleton_data = data['skeleton']
    if len(skeleton_data.shape) == 4:
        skeleton_joint_data = skeleton_data[:, :, joint_id-1, :]
    elif len(skeleton_data.shape) == 3:
        if skeleton_data.shape[2] >= joint_id * 3:
            skeleton_joint_data = skeleton_data[:, :, (joint_id-1)*3:joint_id*3]
        else:
            skeleton_joint_data = skeleton_data[:, :, joint_id-1]
    else:
        return data
    
    inertial_data = data[inertial_key]
    
    if "gyroscope" in data and inertial_key == "accelerometer":
        gyroscope_data = data["gyroscope"]
        min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
        inertial_data = inertial_data[:min_len, :]
        data["gyroscope"] = gyroscope_data[:min_len, :]
    
    skeleton_norm = np.linalg.norm(skeleton_joint_data, axis=1)
    inertial_norm = np.linalg.norm(inertial_data, axis=1)
    
    try:
        distance, path = fastdtw(
            inertial_norm[:, np.newaxis], 
            skeleton_norm[:, np.newaxis],
            dist=euclidean
        )
        
        inertial_ids = set()
        skeleton_ids = set()
        
        for i, j in path:
            if i not in inertial_ids and j not in skeleton_ids:
                inertial_ids.add(i)
                skeleton_ids.add(j)
        
        inertial_ids = sorted(list(inertial_ids))
        skeleton_ids = sorted(list(skeleton_ids))
        
        # Filter data
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
        min_length = min(skeleton_data.shape[0], inertial_data.shape[0])
        for key in data:
            if key != 'labels' and len(data[key]) > min_length:
                data[key] = data[key][:min_length]
    
    return data

def sliding_window(data: Dict[str, np.ndarray], window_size: int, stride_size: int, label: int) -> Dict[str, np.ndarray]:
    # Determine max time from available modalities
    if 'skeleton' not in data:
        key = 'accelerometer' if 'accelerometer' in data else 'gyroscope'
        max_time = data[key].shape[0]
    else:
        max_time = data['skeleton'].shape[0]
    
    # Handle short sequences
    if max_time <= window_size:
        return {k: v[:window_size] if k != 'labels' else np.array([label]) for k, v in data.items()}
    
    # Generate windows
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
    
    # Add labels if we have data
    if result:
        sample_key = next(iter(result.keys()))
        result['labels'] = np.repeat(label, len(result[sample_key]))
    
    return result

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
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))

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
            required_modalities = list(self.age_groups['young'].keys())
        
        complete_trials = []
        for key, files_dict in trial_dict.items():
            if 'accelerometer' in files_dict:  # Always require accelerometer
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                complete_trials.append(matched_trial)
        
        self.matched_trials = complete_trials
        logging.info(f"Matched {len(complete_trials)} trials across modalities")
    
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
        self.match_trials(modalities)

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
        self.verbose = kwargs.get('verbose', False)
        self.use_dtw = kwargs.get('use_dtw', True)
    
    def load_file(self, file_path):
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        loader = LOADER_MAP[file_type]
        return loader(file_path, **self.kwargs)
    
    def process(self, data: Dict[str, np.ndarray], label: int) -> Dict[str, np.ndarray]:
        if self.mode == 'avg_pool':
            result = {}
            for key, value in data.items():
                if key != 'labels':
                    result[key] = pad_sequence_numpy(sequence=value, max_length=self.max_length)
            result['labels'] = data.get('labels', np.array([label]))
            return result
        else:
            window_size = self.max_length
            stride = 32
            if 'accelerometer' not in data:
                return {k: np.array([]) for k in data.keys()}
            
            return sliding_window(data, window_size, stride, label)
    
    def make_dataset(self, subjects: List[int], fuse: bool) -> None:
        self.data = defaultdict(list)
        self.fuse = fuse
        processed_count = 0
        
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
                            trial_data[modality] = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                    except Exception as e:
                        executed = False
                        logging.error(f"Error processing {modality} from {file_path}: {e}")
                
                if executed and 'accelerometer' in trial_data:
                    try:
                        # Apply DTW alignment if both skeleton and accelerometer are present
                        has_skeleton = 'skeleton' in trial_data and len(trial_data['skeleton']) > 0
                        if has_skeleton and self.use_dtw:
                            trial_data = align_sequence(trial_data, use_dtw=True)
                        else:
                            # Simple alignment for accelerometer-only
                            trial_data = align_sequence(trial_data, use_dtw=False)
                        
                        # Process the data (windowing, etc.)
                        processed_data = self.process(trial_data, label)
                        
                        # Check if we have valid data
                        has_data = any(len(v) > 0 for k, v in processed_data.items() if k != 'labels')
                        
                        if has_data:
                            # Ensure consistent dimensions across all arrays
                            expected_shapes = {}
                            for modality in processed_data:
                                if modality != 'labels' and len(processed_data[modality]) > 0:
                                    if len(processed_data[modality].shape) == 3:
                                        expected_shapes[modality] = processed_data[modality].shape[1:]
                            
                            # Add aligned data to dataset
                            for modality, modality_data in processed_data.items():
                                if modality != 'labels' and len(modality_data) > 0:
                                    # Ensure all arrays have same dimensions
                                    if modality in expected_shapes:
                                        # Safety check for dimension consistency
                                        consistent_data = ensure_3d(modality_data, (1, *expected_shapes[modality]))
                                        if consistent_data is not None and len(consistent_data) > 0:
                                            self.data[modality].append(consistent_data)
                            
                            # Add labels if we have data for at least one modality
                            if any(modality != 'labels' and modality in self.data for modality in processed_data):
                                labels_array = np.array(processed_data['labels'])
                                if 'labels' not in self.data:
                                    self.data['labels'] = []
                                self.data['labels'].append(labels_array)
                                processed_count += 1
                    except Exception as e:
                        logging.error(f"Error in processing trial: {e}")
                        logging.error(traceback.format_exc())
        
        logging.info(f"Processed {processed_count} trials from {len(subjects)} subjects")
        
        # Ensure consistent shapes and concatenate
        for key in list(self.data.keys()):
            if key != 'labels' and len(self.data[key]) > 0:
                try:
                    # Find most common shape
                    shapes = [arr.shape for arr in self.data[key]]
                    shape_counts = Counter(shapes)
                    most_common_shape = shape_counts.most_common(1)[0][0]
                    
                    # Keep only arrays with the most common shape
                    filtered_arrays = []
                    for arr in self.data[key]:
                        if arr.shape == most_common_shape:
                            filtered_arrays.append(arr)
                    
                    if filtered_arrays:
                        self.data[key] = np.concatenate(filtered_arrays, axis=0)
                        if self.verbose:
                            logging.info(f"{key} shape: {self.data[key].shape}")
                    else:
                        logging.warning(f"No consistent shapes for {key}, dropping")
                        del self.data[key]
                except Exception as e:
                    logging.error(f"Error concatenating {key}: {e}")
                    del self.data[key]
        
        # Handle labels separately
        if 'labels' in self.data and len(self.data['labels']) > 0:
            try:
                self.data['labels'] = np.concatenate(self.data['labels'])
                if self.verbose:
                    logging.info(f"labels shape: {self.data['labels'].shape}")
            except Exception as e:
                logging.error(f"Error concatenating labels: {e}")
    
    def normalization(self) -> Dict[str, np.ndarray]:
        for key, value in self.data.items():
            if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                try:
                    num_samples, length = value.shape[:2]
                    scaler = StandardScaler()
                    norm_data = scaler.fit_transform(value.reshape(num_samples*length, -1))
                    self.data[key] = norm_data.reshape(num_samples, length, -1)
                except Exception as e:
                    logging.error(f"Error normalizing {key}: {e}")
        return self.data

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
    
    builder_kwargs = {
        'verbose': arg.dataset_args.get('verbose', False),
        'use_dtw': arg.dataset_args.get('use_dtw', True)
    }
    
    builder = DatasetBuilder(
        sm_dataset, 
        arg.dataset_args['mode'], 
        arg.dataset_args['max_length'],
        arg.dataset_args['task'],
        **builder_kwargs
    )
    return builder

def split_by_subjects_tf(builder: DatasetBuilder, subjects: List[int], fuse: bool) -> Dict[str, np.ndarray]:
    builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    return norm_data

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 64, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
            
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
            self.skl_data = np.zeros((self.num_samples, self.acc_seq, 32, 3), dtype=np.float32)
        
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        self._prepare_data()
        self.indices = np.arange(self.num_samples)
    
    def _prepare_data(self):
        try:
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            if self.use_smv:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            logging.error(traceback.format_exc())
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            batch_indices = self.indices[start_idx:end_idx]
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            batch_data = {}
            batch_acc = tf.gather(self.acc_data, tf_indices)
            
            if self.use_smv:
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, tf_indices)
                else:
                    mean = tf.reduce_mean(batch_acc, axis=1, keepdims=True)
                    zero_mean = batch_acc - mean
                    sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                    batch_smv = tf.sqrt(sum_squared)
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            batch_labels = tf.gather(self.labels, tf_indices)
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logging.error(f"Error in batch generation {idx}: {e}")
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.acc_seq, 32, 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = np.arange(batch_size)
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
