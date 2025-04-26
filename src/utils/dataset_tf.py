import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .loaders import csvloader_tf, butterworth_filter_tf, align_sequence_tf, sliding_window_tf, normalize_data_tf

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
        matched_count = 0
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities:
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if not sensor_name:
                        continue
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                
                if not os.path.exists(modality_dir):
                    print(f"Directory not found: {modality_dir}")
                    continue
                    
                file_count = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                self._add_to_matched_trials(modality_name, subject_id, action_id, sequence_number, file_path)
                                file_count += 1
                            except Exception as e:
                                # Skip files with naming errors
                                pass
                
                print(f"Loaded {file_count} files from {modality_dir}")
        
        print(f"Found {len(self.matched_trials)} matched trials")
    
    def _add_to_matched_trials(self, modality_name, subject_id, action_id, sequence_number, file_path):
        matched_trial = None
        for trial in self.matched_trials:
            if (trial["subject_id"] == subject_id and 
                trial["action_id"] == action_id and 
                trial["sequence_number"] == sequence_number):
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
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            self.acc_data = np.zeros((1, 128, 3))
            self.num_samples = 1
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
            
        if self.labels is None or len(self.labels) == 0:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples and self.num_samples > 0:
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                self.labels = np.pad(self.labels, (0, self.num_samples - len(self.labels)), 'constant')
        
        self.batch_size = batch_size
        
        # Pre-compute signal magnitude vector to speed up batch retrieval
        self._prepare_data()
    
    def _prepare_data(self):
        """Pre-process and prepare data for faster batch retrieval."""
        # Convert to TensorFlow tensors for faster operations
        try:
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.smv_cache = self.cal_smv(self.acc_data)
            
            if self.skl_data is not None and len(self.skl_data) > 0:
                self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
                
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        except Exception as e:
            print(f"Error preparing data: {e}")
    
    def __len__(self):
        return max(1, int(np.ceil(self.num_samples / self.batch_size)))
    
    def __getitem__(self, idx):
        """Retrieve a batch with optimized performance."""
        try:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.num_samples)
            
            batch_indices = tf.range(start_idx, end_idx)
            
            # Gather data for the batch
            batch_acc = tf.gather(self.acc_data, batch_indices)
            batch_smv = tf.gather(self.smv_cache, batch_indices)
            batch_data = {'accelerometer': tf.concat([batch_smv, batch_acc], axis=-1)}
            
            if hasattr(self, 'skl_data') and self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
            
            batch_labels = tf.gather(self.labels, batch_indices)
            
            return batch_data, batch_labels, batch_indices.numpy()
            
        except Exception as e:
            # Return dummy data on error
            dummy_data = {'accelerometer': tf.zeros((1, self.acc_seq, self.channels+1))}
            return dummy_data, tf.zeros(1, dtype=tf.int32), np.array([0])
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector efficiently."""
        try:
            mean = tf.reduce_mean(sample, axis=1, keepdims=True)
            zero_mean = sample - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            return tf.sqrt(sum_squared)
        except Exception as e:
            # Return zero tensor on error
            return tf.zeros((*sample.shape[:-1], 1), dtype=tf.float32)

def prepare_smartfallmm_tf(arg):
    """Prepare SmartFallMM dataset."""
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    print(f"Loading data from {data_dir}")
    
    sm_dataset = SmartFallMM_TF(root_dir=data_dir)
    sm_dataset.set_args(arg)
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    return sm_dataset

def _process_trial(args):
    """Process a single trial (for parallel processing)."""
    trial, subjects, builder_args, verbose = args
    
    if trial["subject_id"] not in subjects:
        return None
        
    try:
        # Determine label based on task
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
        
        # Load data for each modality
        trial_data = defaultdict(np.ndarray)
        for modality, file_path in trial["files"].items():
            try:
                unimodal_data = csvloader_tf(file_path, verbose=verbose)
                if unimodal_data is not None and len(unimodal_data) > 0:
                    if modality == 'accelerometer':
                        unimodal_data = butterworth_filter_tf(unimodal_data, verbose=verbose)
                    trial_data[modality] = unimodal_data
            except Exception as e:
                if verbose:
                    print(f"Error loading {modality} file {file_path}: {e}")
        
        # Skip if modalities are missing
        if not trial_data or len(trial_data) < len(trial["files"]):
            return None
        
        if len(trial_data) > 0:
            # Apply DTW alignment
            trial_data = align_sequence_tf(trial_data, verbose=verbose)
            
            # Apply windowing based on mode
            if builder_args and 'mode' in builder_args:
                mode = builder_args['mode']
                max_length = builder_args.get('max_length', 128)
                
                if mode == 'sliding_window':
                    processed_data = sliding_window_tf(trial_data, max_length, label, verbose=verbose)
                else:
                    processed_data = {k: v[:max_length] if len(v) > max_length else v for k, v in trial_data.items()}
                    processed_data['labels'] = np.array([label])
            else:
                processed_data = sliding_window_tf(trial_data, 128, label, verbose=verbose)
            
            # Check if processed data is valid
            if processed_data and any(len(v) > 0 for k, v in processed_data.items() if k != 'labels'):
                return processed_data
        
        return None
    except Exception as e:
        if verbose:
            print(f"Error processing trial for subject {trial['subject_id']}: {e}")
        return None

def split_by_subjects_tf(builder, subjects, fuse=False):
    """Split dataset by subjects using parallel processing with enhanced error handling."""
    import time
    from multiprocessing import Pool, cpu_count
    
    print(f"Processing data for subjects {subjects} using parallel processing")
    start_time = time.time()
    
    # Determine number of CPU cores to use (limit to 48)
    num_cores = min(cpu_count(), 48)
    print(f"Using {num_cores} CPU cores for preprocessing")
    
    # Prepare arguments for parallel processing
    verbose = False  # Set to True for detailed debugging
    process_args = [(trial, subjects, builder.args.dataset_args if builder.args else None, verbose) 
                   for trial in builder.matched_trials]
    
    data = defaultdict(list)
    processed_trials = 0
    
    # Use multiprocessing to process trials in parallel
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(_process_trial, process_args),
            total=len(process_args),
            desc=f"Processing {len(process_args)} trials"
        ))
    
    # Collect results
    for result in results:
        if result is not None:
            for key, value in result.items():
                if key != 'labels' and len(value) > 0:
                    data[key].append(value)
            
            if 'labels' in result and len(result['labels']) > 0:
                if 'labels' not in data:
                    data['labels'] = []
                data['labels'].extend(result['labels'])
            
            processed_trials += 1
    
    print(f"Successfully processed {processed_trials} trials in {time.time()-start_time:.2f}s")
    
    # Concatenate data
    for key in list(data.keys()):
        if key != 'labels':
            if len(data[key]) > 0:
                try:
                    print(f"Concatenating {len(data[key])} {key} arrays...")
                    data[key] = np.concatenate(data[key], axis=0)
                    print(f"Final shape of {key}: {data[key].shape}")
                except Exception as e:
                    print(f"Error concatenating {key} data: {e}")
                    del data[key]
            else:
                del data[key]
        else:
            data[key] = np.array(data[key])
            print(f"Final labels shape: {data[key].shape}, class distribution: {np.unique(data[key], return_counts=True)}")
    
    # Normalize the data
    print("Normalizing data...")
    norm_start_time = time.time()
    try:
        if data:
            data = normalize_data_tf(data, verbose=verbose)
    except Exception as e:
        print(f"Error normalizing data: {e}")
    
    print(f"Normalization completed in {time.time()-norm_start_time:.2f}s")
    print(f"Total preprocessing time: {time.time()-start_time:.2f}s")
    
    return data
