import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from .loaders import csvloader_tf, butterworth_filter_tf, align_sequence_tf, sliding_window_tf, normalize_data_tf

class SmartFallMM_TF:
    """TensorFlow version of SmartFallMM dataset."""
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
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities:
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name:
                        continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                
                if os.path.exists(modality_dir):
                    print(f"Loading files from {modality_dir}")
                    for root, _, files in os.walk(modality_dir):
                        for file in files:
                            if file.endswith('.csv'):
                                try:
                                    subject_id = int(file[1:3])
                                    action_id = int(file[4:6])
                                    sequence_number = int(file[7:9])
                                    file_path = os.path.join(root, file)
                                    self._add_to_matched_trials(modality_name, subject_id, action_id, sequence_number, file_path)
                                except Exception as e:
                                    print(f"Error processing file {file}: {e}")
                else:
                    print(f"Warning: Directory not found: {modality_dir}")
    
    def _add_to_matched_trials(self, modality_name, subject_id, action_id, sequence_number, file_path):
        matched_trial = None
        for trial in self.matched_trials:
            if (trial["subject_id"] == subject_id and trial["action_id"] == action_id and trial["sequence_number"] == sequence_number):
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
        print(f"Setting up pipeline for age groups: {age_group}, modalities: {modalities}, sensors: {sensors}")
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        self.load_files()
        print(f"Found {len(self.matched_trials)} matched trials")

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow data loader for multimodal data."""
    def __init__(self, dataset, batch_size):
        self.acc_data = dataset.get('accelerometer', None)
        self.gyro_data = dataset.get('gyroscope', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        self.subjects = dataset.get('subjects', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            print("Warning: No accelerometer data found, using empty array")
            self.acc_data = np.zeros((0, 128, 3))
            self.num_samples = 0
            self.acc_seq = 128
            self.channels = 3
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
        
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                self.skl_data = np.reshape(self.skl_data, (self.skl_seq, self.skl_length, joints, 3))
            elif len(self.skl_data.shape) == 4:
                self.skl_seq, self.skl_length, joints, dims = self.skl_data.shape
        
        if self.labels is None or len(self.labels) == 0:
            print("Warning: No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples and self.num_samples > 0:
            print(f"Warning: Labels length {len(self.labels)} doesn't match data samples {self.num_samples}")
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                self.labels = np.pad(self.labels, (0, self.num_samples - len(self.labels)), 'constant')
        
        self.batch_size = batch_size
        self.crop_size = 64
    
    def __len__(self):
        return max(1, int(np.ceil(self.num_samples / self.batch_size)))
    
    def __getitem__(self, idx):
        if self.num_samples == 0:
            empty_data = {'accelerometer': tf.zeros((0, self.acc_seq, self.channels+1))}
            if self.skl_data is not None:
                empty_data['skeleton'] = tf.zeros((0, self.skl_length, self.skl_data.shape[2], 3))
            return empty_data, tf.zeros(0, dtype=tf.int32), np.array([])
        
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_data = {}
        
        try:
            batch_acc = self.acc_data[batch_indices]
            watch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([watch_smv, batch_acc], axis=-1)
            
            if self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.convert_to_tensor(
                    self.skl_data[batch_indices], dtype=tf.float32)
            
            batch_labels = tf.convert_to_tensor(self.labels[batch_indices], dtype=tf.int32)
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            print(f"Error in data batch generation: {e}")
            empty_data = {'accelerometer': tf.zeros((1, self.acc_seq, self.channels+1))}
            if self.skl_data is not None:
                empty_data['skeleton'] = tf.zeros((1, self.skl_length, self.skl_data.shape[2], 3))
            return empty_data, tf.zeros(1, dtype=tf.int32), np.array([0])
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector."""
        if len(sample.shape) < 3:
            return tf.zeros((*sample.shape[:-1], 1), dtype=tf.float32)
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)

def prepare_smartfallmm_tf(arg):
    """Create and initialize SmartFallMM dataset."""
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    print(f"Preparing SmartFallMM dataset from {data_dir}")
    sm_dataset = SmartFallMM_TF(root_dir=data_dir)
    sm_dataset.set_args(arg)
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    return sm_dataset

def split_by_subjects_tf(builder, subjects, fuse=False):
    """Split dataset by subjects for training/validation/testing."""
    print(f"Splitting data for subjects: {subjects}")
    data = defaultdict(list)
    processed_trials = 0
    
    for trial in builder.matched_trials:
        if trial["subject_id"] in subjects:
            try:
                if builder.args and 'task' in builder.args.dataset_args:
                    task = builder.args.dataset_args['task']
                    if task == 'fd':
                        label = int(trial["action_id"] > 9)
                    elif task == 'age':
                        label = int(trial["subject_id"] < 29 or trial["subject_id"] > 46)
                    else:
                        label = trial["action_id"] - 1
                else:
                    label = trial["action_id"] - 1
                
                trial_data = defaultdict(np.ndarray)
                for modality, file_path in trial["files"].items():
                    try:
                        unimodal_data = csvloader_tf(file_path)
                        if unimodal_data is not None and len(unimodal_data) > 0:
                            if modality == 'accelerometer':
                                unimodal_data = butterworth_filter_tf(unimodal_data)
                            trial_data[modality] = unimodal_data
                    except Exception as e:
                        print(f"Error processing {modality} file {file_path}: {e}")
                
                if not trial_data:
                    continue
                
                if len(trial_data) > 0:
                    trial_data = align_sequence_tf(trial_data)
                    
                    if builder.args and 'mode' in builder.args.dataset_args:
                        mode = builder.args.dataset_args['mode']
                        max_length = builder.args.dataset_args.get('max_length', 128)
                        
                        if mode == 'sliding_window':
                            processed_data = sliding_window_tf(trial_data, max_length, label)
                        else:
                            processed_data = {k: v[:max_length] if len(v) > max_length else v for k, v in trial_data.items()}
                            processed_data['labels'] = np.array([label])
                    else:
                        processed_data = sliding_window_tf(trial_data, 128, label)
                    
                    if processed_data and any(len(v) > 0 for k, v in processed_data.items() if k != 'labels'):
                        for key, value in processed_data.items():
                            if key != 'labels' and len(value) > 0:
                                data[key].append(value)
                        
                        if 'labels' in processed_data and len(processed_data['labels']) > 0:
                            if 'labels' not in data:
                                data['labels'] = []
                            data['labels'].extend(processed_data['labels'])
                        
                        processed_trials += 1
            except Exception as e:
                print(f"Error processing trial for subject {trial['subject_id']}: {e}")
    
    if processed_trials == 0:
        print(f"Warning: No trials were successfully processed for subjects {subjects}")
        return {}
    
    print(f"Successfully processed {processed_trials} trials")
    
    for key in list(data.keys()):
        if key != 'labels':
            if len(data[key]) > 0:
                try:
                    data[key] = np.concatenate(data[key], axis=0)
                    print(f"Shape of {key} data: {data[key].shape}")
                except Exception as e:
                    print(f"Error concatenating {key} data: {e}")
                    del data[key]
            else:
                del data[key]
        else:
            data[key] = np.array(data[key])
            print(f"Shape of labels: {data[key].shape}")
    
    try:
        if data:
            data = normalize_data_tf(data)
    except Exception as e:
        print(f"Error normalizing data: {e}")
    
    return data
