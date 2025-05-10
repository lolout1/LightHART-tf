#!/usr/bin/env python
import os
import logging
from typing import List, Dict
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from utils.processor_tf import (butterworth_filter, pad_sequence_tf, align_sequence_dtw, selective_windowing, sliding_window)

logger = logging.getLogger('dataset-tf')

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    try:
        import pandas as pd
        is_skeleton = 'skeleton' in file_path.lower()
        is_accelerometer = 'accelerometer' in file_path.lower() or 'acc' in file_path.lower()
        is_gyroscope = 'gyroscope' in file_path.lower() or 'gyro' in file_path.lower()
        logger.info(f"Loading CSV file: {file_path} (skeleton={is_skeleton}, acc={is_accelerometer}, gyro={is_gyroscope})")
        try:
            df = pd.read_csv(file_path, index_col=False, header=0)
            df = df.dropna().bfill()
            if is_skeleton:
                expected_cols = 96
                if df.shape[1] >= expected_cols:
                    cols_to_use = list(range(df.shape[1] - expected_cols, df.shape[1]))
                    activity_data = df.iloc[2:, cols_to_use].to_numpy(dtype=np.float32)
                else:
                    activity_data = df.iloc[2:, :].to_numpy(dtype=np.float32)
                if activity_data.shape[1] == 96:
                    frames = activity_data.shape[0]
                    activity_data = activity_data.reshape(frames, 32, 3)
            else:
                expected_cols = 3
                if df.shape[1] >= expected_cols:
                    cols_to_use = list(range(df.shape[1] - expected_cols, df.shape[1]))
                    activity_data = df.iloc[2:, cols_to_use].to_numpy(dtype=np.float32)
                else:
                    activity_data = df.iloc[2:, :].to_numpy(dtype=np.float32)
                    if activity_data.shape[1] < 3:
                        padded_data = np.zeros((activity_data.shape[0], 3), dtype=np.float32)
                        padded_data[:, :activity_data.shape[1]] = activity_data
                        activity_data = padded_data
        except Exception as e:
            logger.warning(f"Standard CSV parsing failed: {e}. Trying alternative...")
            try:
                df = pd.read_csv(file_path, header=None)
                if is_skeleton:
                    if df.shape[1] >= 96:
                        activity_data = df.iloc[2:, -96:].to_numpy(dtype=np.float32)
                        frames = activity_data.shape[0]
                        activity_data = activity_data.reshape(frames, 32, 3)
                    else:
                        activity_data = df.iloc[2:, :].to_numpy(dtype=np.float32)
                else:
                    if df.shape[1] >= 3:
                        activity_data = df.iloc[2:, -3:].to_numpy(dtype=np.float32)
                    else:
                        activity_data = df.iloc[2:, :].to_numpy(dtype=np.float32)
                        if activity_data.shape[1] < 3:
                            padded_data = np.zeros((activity_data.shape[0], 3), dtype=np.float32)
                            padded_data[:, :activity_data.shape[1]] = activity_data
                            activity_data = padded_data
            except Exception as e2:
                logger.error(f"All CSV parsing attempts failed: {e2}")
                if is_skeleton:
                    return np.zeros((10, 32, 3), dtype=np.float32)
                else:
                    return np.zeros((10, 3), dtype=np.float32)
        if (is_accelerometer or is_gyroscope) and len(activity_data) > 30:
            activity_data = butterworth_filter(activity_data, cutoff=7.5, fs=25)
        logger.info(f"Successfully loaded {file_path}: shape={activity_data.shape}")
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        if 'skeleton' in file_path.lower():
            return np.zeros((10, 32, 3), dtype=np.float32)
        else:
            return np.zeros((10, 3), dtype=np.float32)

def matloader(file_path: str, **kwargs) -> np.ndarray:
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            raise ValueError(f'Unsupported key {key} for matlab file')
        logger.info(f"Loading MAT file {file_path} with key {key}")
        data = loadmat(file_path)[key]
        if key == 'd_iner' and data.shape[0] > 30:
            data = butterworth_filter(data, cutoff=7.5, fs=25)
        logger.info(f"Loaded MAT file {file_path}: shape={data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading MAT file {file_path}: {e}")
        return np.zeros((10, 3), dtype=np.float32)

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

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
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
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
                    if not sensor_name: continue
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
                                logger.warning(f"Error parsing file {file}: {e}")
                total_files += files_loaded
                logger.info(f"Loaded {files_loaded} files for {modality_name} in {age_group}")
        logger.info(f"Total files loaded: {total_files}")
    def match_trials(self) -> None:
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path
        required_modalities = list(self.selected_sensors.keys())
        accelerometer_only = len(required_modalities) == 1 and required_modalities[0] == 'accelerometer'
        skeleton_only = len(required_modalities) == 1 and required_modalities[0] == 'skeleton'
        logger.info(f"Required modalities: {required_modalities}")
        logger.info(f"Accelerometer only: {accelerometer_only}, Skeleton only: {skeleton_only}")
        complete_trials = []
        partial_trials = 0
        for key, files_dict in trial_dict.items():
            if accelerometer_only:
                has_required = 'accelerometer' in files_dict
            elif skeleton_only:
                has_required = 'skeleton' in files_dict
            else:
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
    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window', 'selective_window'], f'Unsupported processing method {mode}'
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
        self.data = defaultdict(list)
        matching_trials = [t for t in self.dataset.matched_trials if t.subject_id in subjects]
        logger.info(f"Found {len(matching_trials)} trials for {len(subjects)} subjects")
        if not matching_trials:
            logger.warning("No matching trials found, creating dummy dataset")
            self.data = {'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32), 'skeleton': np.zeros((1, self.max_length, 32, 3), dtype=np.float32), 'labels': np.zeros(1, dtype=np.int32)}
            return
        processed_count = 0
        for i, trial in enumerate(matching_trials):
            if self.verbose and (i % 10 == 0 or i == len(matching_trials) - 1):
                logger.info(f"Processing trial {i+1}/{len(matching_trials)}")
            try:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                trial_data = {}
                for modality, file_path in trial.files.items():
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys and modality.lower() in keys:
                        key = keys[modality.lower()]
                    data = self.load_file(file_path, key=key)
                    if data is not None and len(data) > 10:
                        trial_data[modality] = data
                if not trial_data:
                    logger.warning(f"No valid data for trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
                    continue
                if 'accelerometer' in trial_data and 'skeleton' in trial_data and self.use_dtw:
                    aligned_data = align_sequence_dtw(trial_data, joint_id=9, use_dtw=self.use_dtw)
                else:
                    aligned_data = trial_data
                if self.mode == 'avg_pool':
                    result = {}
                    for key, value in aligned_data.items():
                        result[key] = pad_sequence_tf(value, self.max_length)
                    result['labels'] = np.array([label])
                elif self.mode == 'selective_window':
                    result = selective_windowing(aligned_data, self.max_length, label)
                else:
                    result = sliding_window(aligned_data, self.max_length, 32, label)
                if result and len(result.get('labels', [])) > 0:
                    for key, value in result.items():
                        if len(value) > 0:
                            self.data[key].append(value)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        logger.info(f"Successfully processed {processed_count} trials")
        for key in list(self.data.keys()):
            if len(self.data[key]) > 0:
                try:
                    if key == 'skeleton':
                        consistent_arrays = []
                        for arr in self.data[key]:
                            if len(arr.shape) == 3 and arr.shape[2] == 32:
                                reshaped = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
                                consistent_arrays.append(reshaped)
                            elif len(arr.shape) == 4 and arr.shape[2] == 32 and arr.shape[3] == 3:
                                consistent_arrays.append(arr)
                        if consistent_arrays:
                            self.data[key] = np.concatenate(consistent_arrays, axis=0)
                            logger.info(f"Standardized {key} shape: {self.data[key].shape}")
                        else:
                            logger.warning(f"No consistent {key} arrays found")
                            del self.data[key]
                    else:
                        self.data[key] = np.concatenate(self.data[key], axis=0)
                    logger.info(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {e}")
                    logger.error(f"Inconsistent shapes for {key}: {[arr.shape for arr in self.data[key]]}")
                    del self.data[key]
        if 'labels' not in self.data or len(self.data['labels']) == 0:
            logger.warning("No valid data processed, creating dummy dataset")
            self.data = {'accelerometer': np.zeros((1, self.max_length, 3), dtype=np.float32), 'labels': np.zeros(1, dtype=np.int32)}
            if 'skeleton' in self.kwargs.get('required_modalities', ['accelerometer']):
                self.data['skeleton'] = np.zeros((1, self.max_length, 32, 3), dtype=np.float32)
    def normalization(self) -> Dict[str, np.ndarray]:
        try:
            for key, value in self.data.items():
                if key != 'labels' and isinstance(value, np.ndarray) and len(value) > 0:
                    if len(value.shape) >= 2:
                        num_samples = value.shape[0]
                        orig_shape = value.shape
                        if key == 'skeleton' and len(value.shape) == 4:
                            reshaped = value.reshape(num_samples * value.shape[1], -1)
                        else:
                            reshaped = value.reshape(num_samples * value.shape[1], -1)
                        try:
                            scaler = StandardScaler()
                            norm_data = scaler.fit_transform(reshaped)
                            self.data[key] = norm_data.reshape(orig_shape)
                            logger.info(f"Normalized {key}: shape={self.data[key].shape}")
                        except Exception as e:
                            logger.warning(f"Normalization failed for {key}: {e}. Using original data.")
            return self.data
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return self.data

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        self._validate_and_prepare_data()
        self.indices = np.arange(self.num_samples)
        logger.info(f"UTD_MM_TF initialized with {self.num_samples} samples. SMV: {use_smv}")
    def _validate_and_prepare_data(self):
        if self.acc_data is None or len(self.acc_data) == 0:
            logger.warning("Missing accelerometer data. Creating dummy data.")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = len(self.acc_data)
            if not isinstance(self.acc_data, np.ndarray):
                self.acc_data = np.array(self.acc_data, dtype=np.float32)
            if self.acc_data.dtype != np.float32:
                self.acc_data = self.acc_data.astype(np.float32)
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                if self.skl_data.shape[2] % 3 == 0:
                    num_joints = self.skl_data.shape[2] // 3
                    self.skl_data = self.skl_data.reshape(self.skl_data.shape[0], self.skl_data.shape[1], num_joints, 3)
            if not isinstance(self.skl_data, np.ndarray):
                self.skl_data = np.array(self.skl_data, dtype=np.float32)
            if self.skl_data.dtype != np.float32:
                self.skl_data = self.skl_data.astype(np.float32)
        else:
            logger.warning("Missing skeleton data. Creating dummy data.")
            self.skl_data = np.zeros((self.num_samples, 128, 32, 3), dtype=np.float32)
        if self.labels is None or len(self.labels) == 0:
            logger.warning("Missing labels. Creating dummy labels.")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        else:
            if not isinstance(self.labels, np.ndarray):
                self.labels = np.array(self.labels)
            if self.labels.dtype != np.int32:
                self.labels = self.labels.astype(np.int32)
            if len(self.labels.shape) > 1:
                self.labels = self.labels.flatten()
        min_samples = min(len(self.acc_data), len(self.skl_data), len(self.labels))
        if min_samples < self.num_samples:
            logger.warning(f"Data size mismatch. Truncating to {min_samples} samples.")
            self.acc_data = self.acc_data[:min_samples]
            self.skl_data = self.skl_data[:min_samples]
            self.labels = self.labels[:min_samples]
            self.num_samples = min_samples
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        if self.use_smv:
            try:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logger.info(f"Pre-calculated SMV with shape: {self.smv.shape}")
            except Exception as e:
                logger.error(f"Error calculating SMV: {e}. Will calculate on-the-fly.")
                self.smv = None
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    def __getitem__(self, idx):
        try:
            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.num_samples)
            current_batch_size = batch_end - batch_start
            batch_indices = self.indices[batch_start:batch_end]
            batch_indices_tensor = tf.convert_to_tensor(batch_indices, dtype=tf.int32)
            batch_data = {}
            batch_acc = tf.gather(self.acc_data, batch_indices_tensor)
            if self.use_smv:
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, batch_indices_tensor)
                else:
                    mean = tf.reduce_mean(batch_acc, axis=1, keepdims=True)
                    zero_mean = batch_acc - mean
                    sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                    batch_smv = tf.sqrt(sum_squared)
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices_tensor)
            batch_labels = tf.gather(self.labels, batch_indices_tensor)
            return batch_data, batch_labels, batch_indices
        except Exception as e:
            logger.error(f"Error creating batch {idx}: {e}")
            import traceback
            traceback.print_exc()
            current_batch_size = min(self.batch_size, self.num_samples - batch_start)
            if current_batch_size <= 0:
                current_batch_size = self.batch_size
            acc_channels = 4 if self.use_smv else 3
            dummy_acc = tf.zeros((current_batch_size, self.acc_data.shape[1], acc_channels), dtype=tf.float32)
            dummy_skl = tf.zeros((current_batch_size, self.skl_data.shape[1], self.skl_data.shape[2], self.skl_data.shape[3]), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(current_batch_size, dtype=tf.int32)
            dummy_indices = tf.range(current_batch_size, dtype=tf.int32)
            return dummy_data, dummy_labels, dummy_indices
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    def get_data_shapes(self):
        shapes = {'accelerometer': tuple(self.acc_data.shape), 'skeleton': tuple(self.skl_data.shape), 'labels': tuple(self.labels.shape)}
        if hasattr(self, 'smv') and self.smv is not None:
            shapes['smv'] = tuple(self.smv.shape)
        return shapes
    def get_sample(self, idx):
        if idx < 0 or idx >= self.num_samples:
            logger.error(f"Index {idx} out of range [0, {self.num_samples-1}]")
            return None
        sample = {'accelerometer': self.acc_data[idx].numpy(), 'skeleton': self.skl_data[idx].numpy(), 'label': int(self.labels[idx].numpy())}
        if hasattr(self, 'smv') and self.smv is not None:
            sample['smv'] = self.smv[idx].numpy()
        return sample

def prepare_smartfallmm_tf(arg):
    possible_paths = [os.path.join(os.getcwd(), 'data/smartfallmm'), os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/smartfallmm'), '/mmfs1/home/sww35/data/smartfallmm']
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
            logger.info(f"Found SmartFall data directory at: {data_dir}")
            break
    if data_dir is None:
        logger.warning(f"SmartFall data directory not found in any of: {possible_paths}")
        data_dir = possible_paths[0]
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Created new data directory at: {data_dir}")
    age_group = arg.dataset_args.get('age_group', ['young'])
    modalities = arg.dataset_args.get('modalities', ['accelerometer'])
    sensors = arg.dataset_args.get('sensors', ['watch'])
    logger.info(f"Initializing dataset with: age_group={age_group}, modalities={modalities}, sensors={sensors}")
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(age_group=age_group, modalities=modalities, sensors=sensors)
    builder_kwargs = {'verbose': arg.dataset_args.get('verbose', True), 'use_dtw': arg.dataset_args.get('use_dtw', True), 'required_modalities': modalities}
    builder = DatasetBuilder(sm_dataset, arg.dataset_args.get('mode', 'selective_window'), arg.dataset_args.get('max_length', 128), arg.dataset_args.get('task', 'fd'), **builder_kwargs)
    return builder

def split_by_subjects_tf(builder, subjects, fuse):
    try:
        builder.make_dataset(subjects, fuse)
        norm_data = builder.normalization()
        modalities_to_check = ['accelerometer', 'skeleton']
        has_valid_data = False
        for key in modalities_to_check:
            if key in norm_data and norm_data[key] is not None and len(norm_data[key]) > 0:
                has_valid_data = True
                if np.isnan(norm_data[key]).any():
                    logger.warning(f"NaN values detected in {key}, replacing with zeros")
                    norm_data[key] = np.nan_to_num(norm_data[key])
        if 'skeleton' in norm_data:
            if len(norm_data['skeleton'].shape) != 4:
                logger.warning(f"Unexpected skeleton shape: {norm_data['skeleton'].shape}")
                if len(norm_data['skeleton'].shape) == 3:
                    if norm_data['skeleton'].shape[2] == 32:
                        norm_data['skeleton'] = norm_data['skeleton'].reshape(norm_data['skeleton'].shape[0], norm_data['skeleton'].shape[1], 32, 1)
                    elif norm_data['skeleton'].shape[2] == 96:
                        norm_data['skeleton'] = norm_data['skeleton'].reshape(norm_data['skeleton'].shape[0], norm_data['skeleton'].shape[1], 32, 3)
        if 'labels' not in norm_data or len(norm_data['labels']) == 0:
            if 'accelerometer' in norm_data and len(norm_data['accelerometer']) > 0:
                logger.warning("Creating dummy labels based on accelerometer data length")
                norm_data['labels'] = np.zeros(len(norm_data['accelerometer']), dtype=np.int32)
            else:
                norm_data['labels'] = np.array([0], dtype=np.int32)
        if 'skeleton' not in norm_data and 'accelerometer' in norm_data and 'skeleton' in builder.kwargs.get('required_modalities', []):
            logger.warning("No skeleton data, creating dummy")
            acc_shape = norm_data['accelerometer'].shape
            norm_data['skeleton'] = np.zeros((acc_shape[0], acc_shape[1], 32, 3), dtype=np.float32)
        if not has_valid_data:
            logger.warning("No valid data, creating dummy dataset")
            norm_data = {'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32), 'labels': np.zeros(1, dtype=np.int32)}
            if 'skeleton' in builder.kwargs.get('required_modalities', []):
                norm_data['skeleton'] = np.zeros((1, builder.max_length, 32, 3), dtype=np.float32)
        return norm_data
    except Exception as e:
        logger.error(f"Error in split_by_subjects_tf: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'accelerometer': np.zeros((1, builder.max_length, 3), dtype=np.float32), 'labels': np.zeros(1, dtype=np.int32)}
