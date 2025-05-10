# src/utils/dataset_tf.py
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from utils.processor_tf import butterworth_filter, pad_sequence_tf, align_sequence_dtw, selective_sliding_window, sliding_window

logger = logging.getLogger(__name__)

def csvloader(file_path, **kwargs):
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        file_data = pd.read_csv(file_path, index_col=False, header=0)
        if file_data.empty or len(file_data) < 3:
            logger.warning(f"Empty/insufficient data: {file_path}")
            return None
        cols = 96 if 'skeleton' in file_path else 3
        if len(file_data.columns) < cols:
            logger.warning(f"Insufficient columns: {file_path}")
            return None
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        activity_data = activity_data[~np.isnan(activity_data).any(axis=1)]
        if activity_data.shape[0] < 10:
            logger.warning(f"Too few samples: {file_path}")
            return None
        logger.debug(f"Loaded {file_path}: {activity_data.shape}")
        return activity_data
    except Exception as e:
        logger.error(f"CSV error {file_path}: {e}")
        return None

def matloader(file_path, **kwargs):
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            logger.error(f'Invalid key {key}')
            return None
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logger.error(f"MAT error {file_path}: {e}")
        return None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False, window_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_smv = use_smv
        self.window_size = window_size
        self.acc_data = dataset.get('accelerometer')
        self.skl_data = dataset.get('skeleton')
        self.labels = dataset.get('labels')
        self._validate_data()
        self.indices = np.arange(self.num_samples)
    
    def _validate_data(self):
        if self.acc_data is None or len(self.acc_data) == 0:
            raise ValueError("No accelerometer data")
        self.num_samples = len(self.acc_data)
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                if self.skl_features == 96:
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, 32, 3)
                elif self.skl_features % 3 == 0:
                    joints = self.skl_features // 3
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
        if self.labels is None or len(self.labels) == 0:
            raise ValueError("No labels")
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32) if self.skl_data is not None else tf.zeros((self.num_samples, self.window_size, 32, 3), dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
    
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared + 1e-8)
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        if len(batch_indices) == 0:
            batch_indices = [0]
        batch_data = {}
        batch_acc = tf.gather(self.acc_data, batch_indices)
        batch_acc = tf.where(tf.math.is_nan(batch_acc), tf.zeros_like(batch_acc), batch_acc)
        if self.use_smv:
            batch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
        else:
            batch_data['accelerometer'] = batch_acc
        batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
        batch_data['skeleton'] = tf.where(tf.math.is_nan(batch_data['skeleton']), tf.zeros_like(batch_data['skeleton']), batch_data['skeleton'])
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
                modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                if modality_name != "skeleton":
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name:
                        continue
                    modality_dir = os.path.join(modality_dir, sensor_name)
                logger.info(f"Loading: {modality_dir}")
                if not os.path.exists(modality_dir):
                    logger.warning(f"Not found: {modality_dir}")
                    continue
                try:
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
                                    logger.warning(f"Parse error {file}: {e}")
                except Exception as e:
                    logger.error(f"Dir error {modality_dir}: {e}")
    
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
        matched, unmatched = 0, 0
        for key, files_dict in trial_dict.items():
            if all(mod in files_dict for mod in required_modalities):
                subject_id, action_id, sequence_number = key
                trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    trial.add_file(modality_name, file_path)
                self.matched_trials.append(trial)
                matched += 1
            else:
                unmatched += 1
        logger.info(f"Matched: {matched}, Unmatched: {unmatched}")
    
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
        self.skipped_trials = 0
        self.successful_trials = 0
    
    def load_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                return None
            file_type = file_path.split('.')[-1]
            if file_type not in LOADER_MAP:
                return None
            return LOADER_MAP[file_type](file_path, **self.kwargs)
        except Exception as e:
            logger.error(f"Load error {file_path}: {e}")
            return None
    
    def process(self, data, label, trial_info):
        try:
            if not data:
                return None
            if self.mode == 'avg_pool':
                processed = {}
                for key, value in data.items():
                    if value is None or len(value) == 0:
                        return None
                    processed[key] = pad_sequence_tf(value, self.max_length)
                processed['labels'] = np.array([label])
                return processed
            elif self.mode == 'selective_window':
                result = selective_sliding_window(data, self.max_length, label)
                if not result or 'labels' not in result or len(result.get('labels', [])) == 0:
                    logger.debug(f"No windows: S{trial_info['subject']:02d}_A{trial_info['action']:02d}")
                    return None
                return result
            else:
                result = sliding_window(data, self.max_length, 32, label)
                if not result or 'labels' not in result or len(result.get('labels', [])) == 0:
                    logger.debug(f"No windows: S{trial_info['subject']:02d}_A{trial_info['action']:02d}")
                    return None
                return result
        except Exception as e:
            logger.error(f"Process error {trial_info}: {e}")
            return None
    
    def make_dataset(self, subjects, fuse):
        self.data = defaultdict(list)
        self.successful_trials = 0
        self.skipped_trials = 0
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                trial_info = {'subject': trial.subject_id, 'action': trial.action_id, 'sequence': trial.sequence_number}
                try:
                    logger.debug(f"Processing: S{trial_info['subject']:02d}_A{trial_info['action']:02d}_T{trial_info['sequence']:02d}")
                    label = int(trial.action_id > 9) if self.task == 'fd' else (int(trial.subject_id < 29 or trial.subject_id > 46) if self.task == 'age' else trial.action_id - 1)
                    trial_data = {}
                    for modality, file_path in trial.files.items():
                        data = self.load_file(file_path)
                        if data is None:
                            logger.debug(f"Failed load: {modality} from {file_path}")
                            continue
                        if len(data) < self.max_length // 2:
                            logger.debug(f"Too short: {len(data)} samples in {file_path}")
                            continue
                        if modality == 'accelerometer':
                            try:
                                data = butterworth_filter(data, cutoff=7.5, fs=25)
                            except Exception as e:
                                logger.debug(f"Filter error: {e}")
                        trial_data[modality] = data
                    if not trial_data:
                        logger.debug(f"No data: {trial_info}")
                        self.skipped_trials += 1
                        continue
                    if len(trial_data) > 1:
                        try:
                            trial_data = align_sequence_dtw(trial_data)
                        except Exception as e:
                            logger.debug(f"DTW error: {e}")
                    processed_data = self.process(trial_data, label, trial_info)
                    if processed_data and len(processed_data.get('labels', [])) > 0:
                        for key, value in processed_data.items():
                            self.data[key].append(value)
                        self.successful_trials += 1
                    else:
                        self.skipped_trials += 1
                except Exception as e:
                    logger.error(f"Trial error {trial_info}: {e}")
                    self.skipped_trials += 1
        logger.info(f"Dataset: {self.successful_trials} success, {self.skipped_trials} skip")
        for key in self.data:
            try:
                if len(self.data[key]) > 0:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                    logger.info(f"{key} shape: {self.data[key].shape}")
                else:
                    self.data[key] = np.array([])
            except Exception as e:
                logger.error(f"Concat error {key}: {e}")
                self.data[key] = np.array([])
    
    def normalization(self, acc_mean=None, acc_std=None, skl_mean=None, skl_std=None, compute_stats_only=False):
        """Normalize data using mean and std statistics.

        Args:
            acc_mean: Optional precomputed mean for accelerometer data
            acc_std: Optional precomputed std for accelerometer data
            skl_mean: Optional precomputed mean for skeleton data
            skl_std: Optional precomputed std for skeleton data
            compute_stats_only: If True, only compute and return normalization stats

        Returns:
            dict: Normalized data or statistics dictionary
        """
        normalized_data = {}
        stats = {}

        # If we only need to compute stats, collect them without normalizing
        if compute_stats_only:
            for key, value in self.data.items():
                if key == 'labels' or len(value) == 0:
                    continue

                try:
                    if key == 'accelerometer' or key.startswith('acc'):
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples*length, -1)
                        scaler = StandardScaler().fit(reshaped_data)
                        stats['acc_mean'] = scaler.mean_
                        stats['acc_std'] = scaler.scale_
                        logger.info(f"Computed accelerometer stats: mean={scaler.mean_.mean():.4f}, std={scaler.scale_.mean():.4f}")

                    elif key == 'skeleton' or key.startswith('skl'):
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples*length, -1)
                        scaler = StandardScaler().fit(reshaped_data)
                        stats['skl_mean'] = scaler.mean_
                        stats['skl_std'] = scaler.scale_
                        logger.info(f"Computed skeleton stats: mean={scaler.mean_.mean():.4f}, std={scaler.scale_.mean():.4f}")

                except Exception as e:
                    logger.error(f"Stats computation error for {key}: {e}")

            return stats

        # Apply normalization with provided or computed statistics
        for key, value in self.data.items():
            if key == 'labels' or len(value) == 0:
                normalized_data[key] = value
                continue

            try:
                num_samples, length = value.shape[:2]
                reshaped_data = value.reshape(num_samples*length, -1)

                # Use provided stats or compute new ones
                if key == 'accelerometer' or key.startswith('acc'):
                    if acc_mean is not None and acc_std is not None:
                        # Handle dimension mismatch if needed
                        if len(acc_mean) != reshaped_data.shape[1]:
                            logger.warning(f"Dimension mismatch in acc stats, using computed values")
                            norm_data = StandardScaler().fit_transform(reshaped_data)
                        else:
                            # Apply pre-computed normalization
                            norm_data = (reshaped_data - acc_mean) / acc_std
                            logger.info(f"Applied pre-computed normalization for {key}")
                    else:
                        # Compute new normalization
                        norm_data = StandardScaler().fit_transform(reshaped_data)

                elif key == 'skeleton' or key.startswith('skl'):
                    if skl_mean is not None and skl_std is not None:
                        # Handle dimension mismatch if needed
                        if len(skl_mean) != reshaped_data.shape[1]:
                            logger.warning(f"Dimension mismatch in skl stats, using computed values")
                            norm_data = StandardScaler().fit_transform(reshaped_data)
                        else:
                            # Apply pre-computed normalization
                            norm_data = (reshaped_data - skl_mean) / skl_std
                            logger.info(f"Applied pre-computed normalization for {key}")
                    else:
                        # Compute new normalization
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                else:
                    # For other modalities, always compute new normalization
                    norm_data = StandardScaler().fit_transform(reshaped_data)

                # Reshape back to original shape
                normalized_data[key] = norm_data.reshape(value.shape)
                logger.info(f"Normalized {key}")

            except Exception as e:
                logger.error(f"Normalization error for {key}: {e}")
                normalized_data[key] = value

        return normalized_data

def prepare_smartfallmm_tf(arg):
    root_paths = [os.path.join(os.getcwd(), 'data/smartfallmm'), os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm'), '/mmfs1/home/sww35/data/smartfallmm']
    data_dir = next((path for path in root_paths if os.path.exists(path)), root_paths[0])
    logger.info(f"Data dir: {data_dir}")
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(arg.dataset_args['age_group'], arg.dataset_args['modalities'], arg.dataset_args['sensors'])
    builder = DatasetBuilder(sm_dataset, arg.dataset_args['mode'], arg.dataset_args['max_length'], arg.dataset_args['task'], **{'min_windows': arg.dataset_args.get('min_windows', 1)})
    return builder

def split_by_subjects_tf(builder, subjects, fuse, compute_stats_only=False, acc_mean=None, acc_std=None, skl_mean=None, skl_std=None):
    """Split dataset by subjects with consistent normalization across splits.

    Args:
        builder: DatasetBuilder instance
        subjects: List of subject IDs to include
        fuse: Whether to fuse modalities
        compute_stats_only: If True, only compute normalization statistics
        acc_mean: Optional precomputed mean for accelerometer
        acc_std: Optional precomputed std for accelerometer
        skl_mean: Optional precomputed mean for skeleton
        skl_std: Optional precomputed std for skeleton

    Returns:
        dict: Normalized data or statistics
    """
    builder.make_dataset(subjects, fuse)
    return builder.normalization(
        acc_mean=acc_mean,
        acc_std=acc_std,
        skl_mean=skl_mean,
        skl_std=skl_std,
        compute_stats_only=compute_stats_only
    )
