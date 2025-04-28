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
                            except Exception as e:
                                pass
                
                logging.info(f"Loaded {file_count} files from {modality_dir}")
        
        logging.info(f"Found {len(self.matched_trials)} matched trials from {loaded_files}/{total_files} files")
    
    def _add_to_matched_trials(self, modality_name, subject_id, action_id, sequence_number, file_path):
        # Find or create a matched trial
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
            logging.warning("Empty accelerometer data in dataset")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
        
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                try:
                    joints = self.skl_data.shape[2] // 3
                    self.skl_data = np.reshape(
                        self.skl_data,
                        (self.skl_data.shape[0], self.skl_data.shape[1], joints, 3)
                    )
                except Exception as e:
                    logging.error(f"Error reshaping skeleton data: {e}")
        
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels in dataset, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int64)
        elif len(self.labels) != self.num_samples:
            logging.warning(f"Labels length {len(self.labels)} doesn't match data samples {self.num_samples}")
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                self.labels = np.pad(
                    self.labels, 
                    (0, self.num_samples - len(self.labels)), 
                    'constant'
                )
        
        self.batch_size = batch_size
        self.smv_cache = None
        self._prepare_data()
    
    def _prepare_data(self):
        try:
            # Convert to tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate signal magnitude vector (SMV)
            mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
            zero_mean = self.acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            self.smv_cache = tf.sqrt(sum_squared)
            
            if self.skl_data is not None and len(self.skl_data) > 0:
                self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
    
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
            
            # Get batch data
            batch_acc = tf.gather(self.acc_data, indices)
            batch_smv = tf.gather(self.smv_cache, indices)
            
            # Combine accelerometer and SMV
            batch_data = {'accelerometer': tf.concat([batch_smv, batch_acc], axis=-1)}
            
            # Add skeleton data if available
            if hasattr(self, 'skl_data') and self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.gather(self.skl_data, indices)
            
            # Get batch labels
            batch_labels = tf.gather(self.labels, indices)
            
            return batch_data, batch_labels, indices.numpy()
            
        except Exception as e:
            logging.error(f"Error in batch generation {idx}: {e}")
            # Return dummy data on error
            dummy_data = {'accelerometer': tf.zeros((1, 128, 4), dtype=tf.float32)}
            return dummy_data, tf.zeros(1, dtype=tf.int32), np.array([0])

def split_by_subjects_tf(builder, subjects, fuse=False, use_dtw=True, verbose=False):
    import time
    from multiprocessing import Pool, cpu_count
    from utils.loaders import normalize_data_tf, _process_trial
    
    start_time = time.time()
    logging.info(f"Processing data for subjects {subjects} using parallel processing")
    
    # Limit CPU cores for stability
    num_cores = min(cpu_count(), 8)
    logging.info(f"Using {num_cores} CPU cores for preprocessing")
    
    # Prepare arguments for parallel processing
    process_args = [
        (trial, subjects, builder.args.dataset_args if hasattr(builder, 'args') else None, verbose, use_dtw) 
        for trial in builder.matched_trials
    ]
    
    data = defaultdict(list)
    processed_trials = 0
    failed_trials = 0
    
    # Process trials in parallel
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(_process_trial, process_args),
            total=len(process_args),
            desc=f"Processing {len(process_args)} trials"
        ))
    
    # Collect and merge results
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
        else:
            failed_trials += 1
    
    logging.info(f"Successfully processed {processed_trials} trials, failed {failed_trials} trials")
    
    # Concatenate data
    for key in list(data.keys()):
        if key != 'labels':
            if len(data[key]) > 0:
                try:
                    logging.info(f"Concatenating {len(data[key])} {key} arrays...")
                    data[key] = np.concatenate(data[key], axis=0)
                    logging.info(f"Final shape of {key}: {data[key].shape}")
                except Exception as e:
                    logging.error(f"Error concatenating {key} data: {e}")
                    del data[key]
            else:
                del data[key]
        else:
            data[key] = np.array(data[key], dtype=np.int64)  # Ensure int64
            unique_labels, counts = np.unique(data[key], return_counts=True)
            logging.info(f"Final labels shape: {data[key].shape}, distribution: {list(zip(unique_labels, counts))}")
    
    # Normalize the data
    logging.info("Normalizing data...")
    try:
        if data:
            data = normalize_data_tf(data, verbose=verbose)
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
    
    logging.info(f"Total preprocessing time: {time.time()-start_time:.2f}s")
    return data

def prepare_smartfallmm_tf(arg):
    data_dir = os.path.join(os.getcwd(), 'data/smartfallmm')
    logging.info(f"Loading data from {data_dir}")
    
    sm_dataset = SmartFallMM_TF(root_dir=data_dir)
    sm_dataset.set_args(arg)
    
    verbose = arg.dataset_args.get('verbose', True)
    
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    return sm_dataset 
