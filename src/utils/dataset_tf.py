import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

class SmartFallMM_TF:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
        self.args = None
    
    def set_args(self, args):
        self.args = args
    
    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        if modality_name not in self.age_groups[age_group]:
            self.age_groups[age_group][modality_name] = []
    
    def select_sensor(self, modality_name, sensor_name=None):
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self):
        total_files = 0
        loaded_files = 0
        
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities:
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name: continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                
                if not os.path.exists(modality_dir):
                    logging.warning(f"Directory not found: {modality_dir}")
                    continue
                
                file_count = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        total_files += 1
                        if file.endswith('.csv'):
                            try:
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                self._add_to_matched_trials(modality_name, subject_id, action_id, sequence_number, file_path)
                                file_count += 1
                                loaded_files += 1
                            except:
                                pass
                
                logging.info(f"Loaded {file_count} files from {modality_dir}")
        
        logging.info(f"Found {len(self.matched_trials)} matched trials from {loaded_files}/{total_files} files")
    
    def _add_to_matched_trials(self, modality_name, subject_id, action_id, sequence_number, file_path):
        matched_trial = None
        for trial in self.matched_trials:
            if (trial["subject_id"] == subject_id and trial["action_id"]==action_id and trial["sequence_number"] == sequence_number):
                matched_trial = trial
                break
        
        if not matched_trial:
            matched_trial = {
                "subject_id": subject_id,
                "action_id": action_id,
                "sequence_number": sequence_number,
                "files": {}
            }
            self.matched_trials.append(matched_trial)
        
        matched_trial["files"][modality_name] = file_path
    
    def pipe_line(self, age_group, modalities, sensors):
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        self.load_files()

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.acc_data = dataset.get('accelerometer', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            self.acc_data = np.zeros((1, 64, 3), dtype=np.float32)
            self.num_samples = 1
            self.acc_seq = 64
            self.channels = 3
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
            
        self.skl_data = dataset.get('skeleton', None)
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
        
        self.labels = dataset.get('labels', None)
        if self.labels is None or len(self.labels) == 0:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples:
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                last_label = self.labels[-1]
                self.labels = np.concatenate([
                    self.labels, 
                    np.full(self.num_samples - len(self.labels), last_label, dtype=self.labels.dtype)
                ])
        
        self._prepare_data()
    
    def _prepare_data(self):
        try:
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
            zero_mean = self.acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            self.smv = tf.sqrt(sum_squared)
            
            if self.skl_data is not None and len(self.skl_data) > 0:
                self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        except Exception as e:
            self.smv = tf.zeros((self.num_samples, self.acc_seq, 1), dtype=tf.float32)
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            if start_idx >= self.num_samples:
                start_idx = 0
                end_idx = min(self.batch_size, self.num_samples)
            
            indices = tf.range(start_idx, end_idx)
            batch_acc = tf.gather(self.acc_data, indices)
            batch_smv = tf.gather(self.smv, indices)
            
            data = {}
            data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            
            if hasattr(self, 'skl_data') and self.skl_data is not None and len(self.skl_data) > 0:
                if len(self.skl_data.shape) == 4:
                    batch_skl = tf.gather(self.skl_data, indices)
                    data['skeleton'] = batch_skl
                else:
                    data['skeleton'] = tf.gather(self.skl_data, indices)
            
            batch_labels = tf.gather(self.labels, indices)
            return data, batch_labels, indices.numpy()
            
        except Exception as e:
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            return dummy_data, dummy_labels, np.arange(batch_size)

def _process_trial(args):
    from utils.loaders import csvloader_tf, butterworth_filter_tf, align_sequence_tf 
    
    trial, subjects, builder_args, verbose, use_dtw = args
    
    if trial["subject_id"] not in subjects:
        return None
    
    try:
        if builder_args and 'task' in builder_args:
            task = builder_args['task']
            if task == 'fd':
                label = int(trial["action_id"] > 9)
            elif task == 'age':
                label = int(trial["subject_id"] < 29 or trial["subject_id"] > 46)
            else:
                label = trial["action_id"] - 1
        else:
            label = trial["action_id"] - 1
        
        trial_data = defaultdict(np.ndarray)
        required_modalities = builder_args.get('modalities', []) if builder_args else []
        has_accelerometer = False
        
        for modality in required_modalities:
            if modality in trial["files"]:
                file_path = trial["files"][modality]
                try:
                    data = csvloader_tf(file_path, verbose=False)
                    
                    if data is not None and len(data) > 10:
                        if modality == 'accelerometer':
                            data = butterworth_filter_tf(data, cutoff=7.5, fs=25, order=4)
                            has_accelerometer = True
                        trial_data[modality] = data
                except Exception as e:
                    if verbose:
                        logging.warning(f"Error loading {modality} from {file_path}: {e}")
        
        if 'fd' in builder_args.get('task', '') and not has_accelerometer:
            return None
            
        if len(trial_data) > 1 and use_dtw:
            aligned_data = align_sequence_tf(trial_data, verbose=False, use_dtw=use_dtw)
            if aligned_data and ('accelerometer' in aligned_data and len(aligned_data['accelerometer']) > 10):
                trial_data = aligned_data
        
        if builder_args and 'mode' in builder_args:
            mode = builder_args['mode']
            max_length = builder_args.get('max_length', 64)
            
            if mode == 'sliding_window':
                stride = 32
                processed_data = sliding_window_tf(
                    trial_data, 
                    max_length, 
                    label, 
                    stride=stride, 
                    verbose=False,
                    min_windows=1
                )
            else:
                processed_data = {}
                for k, v in trial_data.items():
                    if k != 'labels' and len(v) > 0:
                        if len(v) > max_length:
                            processed_data[k] = v[:max_length]
                        else:
                            padded = np.zeros((max_length,) + v.shape[1:], dtype=v.dtype)
                            padded[:len(v)] = v
                            processed_data[k] = padded[np.newaxis, ...]
                
                if processed_data:
                    processed_data['labels'] = np.array([label], dtype=np.int64)
        else:
            processed_data = sliding_window_tf(trial_data, 64, label, stride=32, verbose=False)
        
        if processed_data:
            if ('accelerometer' in processed_data and len(processed_data['accelerometer']) > 0) or \
               ('skeleton' in processed_data and len(processed_data['skeleton']) > 0):
                
                sample_count = 0
                if 'accelerometer' in processed_data:
                    sample_count = len(processed_data['accelerometer'])
                elif 'skeleton' in processed_data:
                    sample_count = len(processed_data['skeleton'])
                
                if sample_count > 0:
                    processed_data['labels'] = np.full(sample_count, label, dtype=np.int64)
                    return processed_data
        
        return None
    except Exception as e:
        if verbose:
            logging.error(f"Error processing trial {trial['subject_id']}-{trial['action_id']}: {e}")
        return None

def sliding_window_tf(data, window_size, label=None, stride=32, verbose=False, min_windows=1):
    try:
        if not data:
            return None
            
        has_data = False
        for key, value in data.items():
            if key != 'labels' and value is not None and len(value) > 0:
                has_data = True
                break
                
        if not has_data:
            return None
        
        windowed_data = {}
        window_counts = {}
        
        for key, value in data.items():
            if key != 'labels' and value is not None and len(value) > 0:
                if len(value) < 10:
                    continue
                    
                windows = []
                for i in range(0, max(1, len(value) - window_size + 1), stride):
                    window = value[i:i+window_size]
                    
                    if len(window) == window_size:
                        windows.append(window)
                    elif len(window) > window_size/2:
                        padded = np.zeros((window_size,) + window.shape[1:], dtype=window.dtype)
                        padded[:len(window)] = window
                        windows.append(padded)
                
                if len(windows) >= min_windows:
                    try:
                        windowed_data[key] = np.stack(windows)
                        window_counts[key] = len(windows)
                    except Exception as e:
                        if verbose:
                            logging.error(f"Error stacking windows for {key}: {e}")
                        continue
        
        if not windowed_data:
            return None
        
        if len(window_counts) > 1:
            min_count = min(window_counts.values())
            for key in windowed_data:
                windowed_data[key] = windowed_data[key][:min_count]
        
        if label is not None:
            if len(windowed_data) > 0:
                count = len(next(iter(windowed_data.values())))
                windowed_data['labels'] = np.full(count, label, dtype=np.int64)
        
        return windowed_data
    except Exception as e:
        if verbose:
            logging.error(f"Windowing error: {e}")
        return None

def prepare_smartfallmm_tf(arg):
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    logging.info(f"Loading data from {data_dir}")
    
    sm_dataset = SmartFallMM_TF(root_dir=data_dir)
    sm_dataset.set_args(arg)
    
    verbose = arg.dataset_args.get('verbose', True)
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    return sm_dataset 

def split_by_subjects_tf(builder, subjects, fuse=False, use_dtw=True, verbose=False):
    import time
    from multiprocessing import Pool, cpu_count
    from utils.loaders import normalize_data_tf
    
    start_time = time.time()
    logging.info(f"Processing data for subjects {subjects} using parallel processing")
    
    num_cores = min(cpu_count(), 8)
    logging.info(f"Using {num_cores} CPU cores for preprocessing")
    
    process_args = [
        (trial, subjects, builder.args.dataset_args if hasattr(builder, 'args') else None, verbose, use_dtw) 
        for trial in builder.matched_trials
    ]
    
    raw_data = defaultdict(list)
    processed_trials = 0
    failed_trials = 0
    labels = []
    
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(_process_trial, process_args),
            total=len(process_args),
            desc=f"Processing {len(process_args)} trials"
        ))
    
    for result in results:
        if result is not None:
            if 'labels' in result and len(result['labels']) > 0:
                trial_labels = result['labels']
                labels.extend(trial_labels)
                
                for key, value in result.items():
                    if key != 'labels' and len(value) > 0:
                        raw_data[key].append(value)
                        
                processed_trials += 1
            else:
                failed_trials += 1
        else:
            failed_trials += 1
    
    logging.info(f"Successfully processed {processed_trials} trials, failed {failed_trials} trials")
    
    data = {}
    
    for key in raw_data:
        if len(raw_data[key]) > 0:
            try:
                logging.info(f"Concatenating {len(raw_data[key])} {key} arrays...")
                data[key] = np.concatenate(raw_data[key], axis=0)
                logging.info(f"Final shape of {key}: {data[key].shape}")
            except Exception as e:
                logging.error(f"Error concatenating {key} data: {e}")
    
    if labels:
        data['labels'] = np.array(labels, dtype=np.int64)
        
        if 'accelerometer' in data:
            acc_len = len(data['accelerometer'])
            if len(data['labels']) != acc_len:
                logging.warning(f"Fixing label length: {len(data['labels'])} to match accelerometer: {acc_len}")
                if len(data['labels']) > acc_len:
                    data['labels'] = data['labels'][:acc_len]
                else:
                    padding = np.full(acc_len - len(data['labels']), data['labels'][-1], dtype=np.int64)
                    data['labels'] = np.concatenate([data['labels'], padding])
        
        unique_labels, counts = np.unique(data['labels'], return_counts=True)
        logging.info(f"Final labels shape: {data['labels'].shape}, distribution: {list(zip(unique_labels, counts))}")
    
    logging.info("Normalizing data...")
    try:
        if data:
            data = normalize_data_tf(data, verbose=verbose)
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
    
    logging.info(f"Total preprocessing time: {time.time()-start_time:.2f}s")
    return data
