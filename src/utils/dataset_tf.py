#!/usr/bin/env python
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from utils.processor_tf import butterworth_filter, pad_sequence_tf, align_sequence_dtw, selective_sliding_window, sliding_window

logger = logging.getLogger(__name__)

# utils/processor_tf.py (updated csvloader function)
def csvloader(file_path, **kwargs):
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        if activity_data.shape[0] < 10:
            logger.warning(f"File too small: {file_path} has only {activity_data.shape[0]} samples")
            return None
        logger.debug(f"Successfully loaded CSV: {file_path}, shape: {activity_data.shape}")
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        return None
def matloader(file_path, **kwargs):
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        assert key in ['d_iner', 'd_skel'], f'Unsupported key {key}'
        data = loadmat(file_path)[key]
        logger.debug(f"Successfully loaded MAT file: {file_path}, shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading MAT file {file_path}: {e}")
        return None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}


# feeder/make_dataset_tf.py
import tensorflow as tf
import numpy as np

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_smv = use_smv
        self.acc_data = dataset.get('accelerometer')
        self.skl_data = dataset.get('skeleton')
        self.labels = dataset.get('labels')
        self._validate_data()
        self.indices = np.arange(self.num_samples)
    
    def _validate_data(self):
        if self.acc_data is None or len(self.acc_data) == 0:
            self.acc_data = np.zeros((1, 64, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = len(self.acc_data)
        
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                if self.skl_features == 96:  # 32 joints * 3 coords
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, 32, 3)
                elif self.skl_features % 3 == 0:
                    joints = self.skl_features // 3
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) != 4:
                self.skl_data = np.zeros((self.num_samples, 64, 32, 3), dtype=np.float32)
        else:
            self.skl_data = np.zeros((self.num_samples, 64, 32, 3), dtype=np.float32)
        
        if self.labels is None:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
            
        # Input validation - ensure no empty arrays
        if len(self.acc_data) == 0:
            raise ValueError("Accelerometer data is empty")
        if len(self.skl_data) == 0:
            raise ValueError("Skeleton data is empty")
        if len(self.labels) == 0:
            raise ValueError("Labels are empty")
            
        # Convert to tensors with validation
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
    
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared + 1e-8)  # Add epsilon for numerical stability
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Ensure we have at least one sample
        if len(batch_indices) == 0:
            batch_indices = [0]
        
        batch_data = {}
        batch_acc = tf.gather(self.acc_data, batch_indices)
        
        # Validate batch data
        if tf.reduce_any(tf.math.is_nan(batch_acc)):
            batch_acc = tf.where(tf.math.is_nan(batch_acc), tf.zeros_like(batch_acc), batch_acc)
        
        if self.use_smv:
            batch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
        else:
            batch_data['accelerometer'] = batch_acc
        
        batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
        
        # Validate skeleton data
        if tf.reduce_any(tf.math.is_nan(batch_data['skeleton'])):
            batch_data['skeleton'] = tf.where(tf.math.is_nan(batch_data['skeleton']), 
                                            tf.zeros_like(batch_data['skeleton']), 
                                            batch_data['skeleton'])
        
        batch_labels = tf.gather(self.labels, batch_indices)
        return batch_data, batch_labels, batch_indices
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name):
        self.name = name
        self.files = []
    def add_file(self, subject_id, action_id, sequence_number, file_path):
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path

class SmartFallMM:
    def __init__(self, root_dir):
        self.root_dir = root_dir
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
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name: continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                logger.info(f"Loading files from: {modality_dir}")
                if not os.path.exists(modality_dir):
                    logger.warning(f"Directory not found: {modality_dir}")
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
                                logger.warning(f"Error parsing filename {file}: {e}")
    def match_trials(self):
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for file in modality.files:
                    key = (file.subject_id, file.action_id, file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = file.file_path
        required_modalities = list(self.age_groups['young'].keys())
        for key, files_dict in trial_dict.items():
            if all(mod in files_dict for mod in required_modalities):
                subject_id, action_id, sequence_number = key
                trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    trial.add_file(modality_name, file_path)
                self.matched_trials.append(trial)
        logger.info(f"Matched {len(self.matched_trials)} trials across modalities")
    def pipeline(self, age_group, modalities, sensors):
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

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.min_windows = kwargs.get('min_windows', 1)
    def load_file(self, file_path):
        try:
            file_type = file_path.split('.')[-1]
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            if data is None:
                logger.warning(f"Failed to load file or invalid data: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    def process(self, data, label, trial_info):
        try:
            if self.mode == 'avg_pool':
                processed = {}
                for key, value in data.items():
                    processed[key] = pad_sequence_tf(value, self.max_length)
                processed['labels'] = np.array([label])
                return processed
            elif self.mode == 'selective_window':
                result = selective_sliding_window(data, self.max_length, label)
                if 'labels' not in result or len(result.get('labels', [])) < self.min_windows:
                    logger.warning(
                        f"Insufficient windows ({len(result.get('labels', []))}) for trial "
                        f"S{trial_info['subject']:02d}_A{trial_info['action']:02d}_T{trial_info['sequence']:02d}, "
                        f"skipping"
                    )
                    return None
                return result
            else:
                result = sliding_window(data, self.max_length, 32, label)
                if 'labels' not in result or len(result.get('labels', [])) < self.min_windows:
                    logger.warning(
                        f"Insufficient windows ({len(result.get('labels', []))}) for trial "
                        f"S{trial_info['subject']:02d}_A{trial_info['action']:02d}_T{trial_info['sequence']:02d}, "
                        f"skipping"
                    )
                    return None
                return result
        except Exception as e:
            logger.error(f"Error processing data for trial {trial_info}: {e}")
            return None
    def make_dataset(self, subjects, fuse):
        self.data = defaultdict(list)
        successful_trials = 0
        failed_trials = 0
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                trial_info = {
                    'subject': trial.subject_id,
                    'action': trial.action_id,
                    'sequence': trial.sequence_number
                }
                logger.debug(f"Processing trial: S{trial_info['subject']:02d}_A{trial_info['action']:02d}_T{trial_info['sequence']:02d}")
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                trial_data = {}
                for modality, file_path in trial.files.items():
                    data = self.load_file(file_path)
                    if data is None:
                        logger.warning(f"Failed to load {modality} data from {file_path}")
                        continue
                    if len(data) < self.max_length // 2:
                        logger.warning(f"Data too short ({len(data)} samples) in {file_path}")
                        continue
                    trial_data[modality] = data
                if not trial_data:
                    logger.warning(f"No valid data for trial {trial_info}, skipping")
                    failed_trials += 1
                    continue
                try:
                    if len(trial_data) > 1:
                        trial_data = align_sequence_dtw(trial_data)
                    processed_data = self.process(trial_data, label, trial_info)
                    if processed_data and len(processed_data.get('labels', [])) > 0:
                        for key, value in processed_data.items():
                            self.data[key].append(value)
                        successful_trials += 1
                    else:
                        failed_trials += 1
                except Exception as e:
                    logger.error(f"Error processing trial {trial_info}: {e}")
                    failed_trials += 1
        logger.info(f"Successfully processed {successful_trials} trials, failed {failed_trials} trials")
        for key in self.data:
            try:
                self.data[key] = np.concatenate(self.data[key], axis=0)
                logger.info(f"Final data shape for {key}: {self.data[key].shape}")
            except Exception as e:
                logger.error(f"Error concatenating {key} data: {e}")
                self.data[key] = np.array([])
    def normalization(self):
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    num_samples, length = value.shape[:2]
                    norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                    self.data[key] = norm_data.reshape(value.shape)
                    logger.info(f"Normalized {key} data")
                except Exception as e:
                    logger.error(f"Error normalizing {key}: {e}")
        return self.data

def prepare_smartfallmm_tf(arg):
    root_paths = [
        os.path.join(os.getcwd(), 'data/smartfallmm'),
        os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm'),
        '/mmfs1/home/sww35/data/smartfallmm'
    ]
    data_dir = None
    for path in root_paths:
        if os.path.exists(path):
            data_dir = path
            logger.info(f"Found SmartFallMM data directory at: {data_dir}")
            break
    if data_dir is None:
        data_dir = root_paths[0]
        logger.warning(f"SmartFallMM data directory not found, using default: {data_dir}")
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    builder_kwargs = {'min_windows': arg.dataset_args.get('min_windows', 1)}
    builder = DatasetBuilder(
        sm_dataset,
        arg.dataset_args['mode'],
        arg.dataset_args['max_length'],
        arg.dataset_args['task'],
        **builder_kwargs
    )
    return builder

def split_by_subjects_tf(builder, subjects, fuse):
    builder.make_dataset(subjects, fuse)
    return builder.normalization()
