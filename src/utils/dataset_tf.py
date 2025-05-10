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
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
        num_col = file_data.shape[1]
        if 'skeleton' in file_path:
            cols = 96
        else:
            cols = 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        activity_data = activity_data[~np.isnan(activity_data).any(axis=1)]
        if activity_data.shape[0] < 10:
            logger.warning(f"Too few samples in {file_path}: {activity_data.shape}")
            return None
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return None

def matloader(file_path, **kwargs):
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            raise ValueError(f'Unsupported key {key} for matlab file')
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {e}")
        return None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def select_subwindow_pandas(unimodal_data):
    n = len(unimodal_data)
    magnitude = np.linalg.norm(unimodal_data, axis=1)
    df = pd.DataFrame({"values": magnitude})
    df["variance"] = df["values"].rolling(window=125).var()
    max_idx = df["variance"].idxmax()
    final_start = max(0, max_idx - 100)
    final_end = min(n, max_idx + 100)
    return unimodal_data[final_start:final_end, :]

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
            self.acc_data = np.zeros((1, self.window_size, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = len(self.acc_data)
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                if self.skl_features == 96:
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, 32, 3)
                elif self.skl_features % 3 == 0:
                    joints = self.skl_features // 3
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
        else:
            self.skl_data = np.zeros((self.num_samples, self.window_size, 32, 3), dtype=np.float32)
        if self.labels is None:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        return smv
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        batch_data = {}
        batch_acc = tf.gather(self.acc_data, batch_indices)
        if self.use_smv:
            batch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
        else:
            batch_data['accelerometer'] = batch_acc
        batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
        batch_labels = tf.gather(self.labels, batch_indices)
        return batch_data, batch_labels, batch_indices
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = False
    def load_file(self, file_path):
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        return data
    def _import_loader(self, file_path):
        file_type = file_path.split('.')[-1]
        if file_type not in LOADER_MAP:
            raise ValueError(f'Unsupported file type {file_type}')
        return LOADER_MAP[file_type]
    def process(self, data, label):
        try:
            if self.mode == 'avg_pool':
                processed = {}
                for key, value in data.items():
                    processed[key] = pad_sequence_tf(value, self.max_length)
                processed['labels'] = np.array([label])
                return processed
            elif self.mode == 'selective_window':
                return selective_sliding_window(data, self.max_length, label)
            else:
                return sliding_window(data, self.max_length, 32, label)
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    def make_dataset(self, subjects, fuse):
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
                trial_data = {}
                for modality, file_path in trial.files.items():
                    try:
                        unimodal_data = self.load_file(file_path)
                        if unimodal_data is None:
                            continue
                        if modality == 'accelerometer':
                            unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                            if unimodal_data.shape[0] > 250:
                                unimodal_data = select_subwindow_pandas(unimodal_data)
                        trial_data[modality] = unimodal_data
                    except Exception as e:
                        logger.error(f"Error loading {modality} from {file_path}: {e}")
                        continue
                if not trial_data:
                    continue
                if len(trial_data) > 1:
                    trial_data = align_sequence_dtw(trial_data)
                processed_data = self.process(trial_data, label)
                if processed_data and len(processed_data.get('labels', [])) > 0:
                    for key, value in processed_data.items():
                        self.data[key].append(value)
        for key in self.data:
            if len(self.data[key]) > 0:
                self.data[key] = np.concatenate(self.data[key], axis=0)
            else:
                self.data[key] = np.array([])
    def normalization(self, acc_mean=None, acc_std=None, skl_mean=None, skl_std=None, compute_stats_only=False):
        if compute_stats_only:
            stats = {}
            for key, value in self.data.items():
                if key == 'labels' or len(value) == 0:
                    continue
                num_samples, length = value.shape[:2]
                norm_data = value.reshape(num_samples * length, -1)
                scaler = StandardScaler()
                scaler.fit(norm_data)
                if key == 'accelerometer':
                    stats['acc_mean'] = scaler.mean_
                    stats['acc_std'] = scaler.scale_
                elif key == 'skeleton':
                    stats['skl_mean'] = scaler.mean_
                    stats['skl_std'] = scaler.scale_
            return stats
        normalized_data = {}
        for key, value in self.data.items():
            if key == 'labels':
                normalized_data[key] = value
                continue
            if len(value) == 0:
                normalized_data[key] = value
                continue
            num_samples, length = value.shape[:2]
            norm_data = value.reshape(num_samples * length, -1)
            if key == 'accelerometer' and acc_mean is not None and acc_std is not None:
                norm_data = (norm_data - acc_mean) / acc_std
            elif key == 'skeleton' and skl_mean is not None and skl_std is not None:
                norm_data = (norm_data - skl_mean) / skl_std
            else:
                scaler = StandardScaler()
                norm_data = scaler.fit_transform(norm_data)
            normalized_data[key] = norm_data.reshape(value.shape)
        return normalized_data

def prepare_smartfallmm_tf(arg):
    from utils.dataset_sf import SmartFallMM
    root_paths = [os.path.join(os.getcwd(), 'data/smartfallmm'), os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm'), '/path/to/data/smartfallmm']
    data_dir = None
    for path in root_paths:
        if os.path.exists(path):
            data_dir = path
            break
    if data_dir is None:
        raise ValueError(f"Data directory not found. Tried: {root_paths}")
    logger.info(f"Using data directory: {data_dir}")
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(age_group=arg.dataset_args['age_group'], modalities=arg.dataset_args['modalities'], sensors=arg.dataset_args['sensors'])
    builder = DatasetBuilder(sm_dataset, arg.dataset_args['mode'], arg.dataset_args['max_length'], arg.dataset_args['task'])
    return builder

def split_by_subjects_tf(builder, subjects, fuse=False, compute_stats_only=False, acc_mean=None, acc_std=None, skl_mean=None, skl_std=None):
    builder.make_dataset(subjects, fuse)
    return builder.normalization(acc_mean=acc_mean, acc_std=acc_std, skl_mean=skl_mean, skl_std=skl_std, compute_stats_only=compute_stats_only)
