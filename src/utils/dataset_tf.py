import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class SmartFallDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
    
    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            logging.warning(f"Invalid age group: {age_group}")
            return
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
                
                if not os.path.exists(modality_dir):
                    logging.warning(f"Directory not found: {modality_dir}")
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
                                self._add_file(modality_name, subject_id, action_id, sequence_number, file_path)
                                file_count += 1
                            except Exception as e:
                                pass
                
                logging.info(f"Loaded {file_count} files from {modality_dir}")
    
    def _add_file(self, modality_name, subject_id, action_id, sequence_number, file_path):
        # Find or create a matched trial
        found = False
        for trial in self.matched_trials:
            if (trial["subject_id"] == subject_id and 
                trial["action_id"] == action_id and 
                trial["sequence_number"] == sequence_number):
                trial["files"][modality_name] = file_path
                found = True
                break
        
        if not found:
            self.matched_trials.append({
                "subject_id": subject_id,
                "action_id": action_id,
                "sequence_number": sequence_number,
                "files": {modality_name: file_path}
            })
    
    def filter_matched_trials(self, required_modalities):
        """Filter trials to ensure they have all required modalities"""
        total_trials = len(self.matched_trials)
        filtered_trials = []
        
        for trial in self.matched_trials:
            if all(modality in trial["files"] for modality in required_modalities):
                filtered_trials.append(trial)
        
        self.matched_trials = filtered_trials
        logging.info(f"Found {len(self.matched_trials)} complete matched trials out of {total_trials} total trials")
    
    def setup(self, age_groups, modalities, sensors):
        """Complete setup process"""
        # Add all modalities and sensors
        for age in age_groups:
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        # Load all files
        self.load_files()
        
        # Filter to ensure complete trials
        self.filter_matched_trials(modalities)

class DatasetTF(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and validate data"""
        self.num_samples = 0
        
        # Check necessary components
        if 'accelerometer' not in self.data or len(self.data['accelerometer']) == 0:
            self.data['accelerometer'] = np.zeros((1, 64, 3), dtype=np.float32)
        
        # Set sample count from accelerometer
        self.num_samples = len(self.data['accelerometer'])
        
        # Process labels
        if 'labels' not in self.data or len(self.data['labels']) == 0:
            self.data['labels'] = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.data['labels']) != self.num_samples:
            # Ensure labels match sample count
            if len(self.data['labels']) > self.num_samples:
                self.data['labels'] = self.data['labels'][:self.num_samples]
            else:
                self.data['labels'] = np.pad(
                    self.data['labels'],
                    (0, self.num_samples - len(self.data['labels'])),
                    'constant'
                )
        
        # Process skeleton data if present
        if 'skeleton' in self.data and len(self.data['skeleton']) > 0:
            if len(self.data['skeleton']) != self.num_samples:
                logging.warning(f"Skeleton samples {len(self.data['skeleton'])} doesn't match accelerometer samples {self.num_samples}")
                # Match samples for consistency
                min_samples = min(len(self.data['skeleton']), self.num_samples)
                self.data['accelerometer'] = self.data['accelerometer'][:min_samples]
                self.data['skeleton'] = self.data['skeleton'][:min_samples]
                self.data['labels'] = self.data['labels'][:min_samples]
                self.num_samples = min_samples
        
        # Convert to tensors and add SMV
        self.data['accelerometer'] = tf.convert_to_tensor(self.data['accelerometer'], dtype=tf.float32)
        self.data['labels'] = tf.convert_to_tensor(self.data['labels'], dtype=tf.int32)
        
        # Calculate signal magnitude vector
        acc_data = self.data['accelerometer']
        mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
        zero_mean = acc_data - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        
        # Create accelerometer with SMV
        self.data['accelerometer_with_smv'] = tf.concat([smv, acc_data], axis=-1)
        
        if 'skeleton' in self.data and len(self.data['skeleton']) > 0:
            self.data['skeleton'] = tf.convert_to_tensor(self.data['skeleton'], dtype=tf.float32)
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        """Get batch at index"""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # Handle empty dataset edge case
        if start_idx >= self.num_samples:
            start_idx = 0
            end_idx = min(self.batch_size, self.num_samples)
        
        # Get indices for this batch
        indices = tf.range(start_idx, end_idx)
        
        # Create batch data
        batch_data = {'accelerometer': tf.gather(self.data['accelerometer_with_smv'], indices)}
        
        # Add skeleton if available
        if 'skeleton' in self.data and len(self.data['skeleton']) > 0:
            batch_data['skeleton'] = tf.gather(self.data['skeleton'], indices)
        
        # Get batch labels and indices
        batch_labels = tf.gather(self.data['labels'], indices)
        
        return batch_data, batch_labels, indices.numpy()

def prepare_dataset(root_dir, age_groups, modalities, sensors):
    """Prepare the dataset from raw files"""
    dataset = SmartFallDataset(root_dir)
    dataset.setup(age_groups, modalities, sensors)
    return dataset

def process_dataset(dataset, subjects, window_size=64, stride=None, max_workers=8):
    """Process the dataset in parallel"""
    from utils.loaders import process_trial, normalize_data
    
    logging.info(f"Processing data for subjects {subjects} using {max_workers} workers")
    
    results = defaultdict(list)
    processed_count = 0
    failed_count = 0
    
    # Use concurrent processing for speed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing jobs
        future_to_trial = {
            executor.submit(process_trial, trial, subjects, window_size, stride): trial
            for trial in dataset.matched_trials
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc=f"Processing {len(future_to_trial)} trials"):
            result = future.result()
            if result:
                processed_count += 1
                for key, value in result.items():
                    if key == 'labels':
                        results[key].extend(value)
                    else:
                        results[key].append(value)
            else:
                failed_count += 1
    
    logging.info(f"Successfully processed {processed_count} trials, failed {failed_count} trials")
    
    # Concatenate data
    processed_data = {}
    for key, values in results.items():
        if key != 'labels':
            if values:
                try:
                    logging.info(f"Concatenating {len(values)} {key} arrays...")
                    processed_data[key] = np.concatenate(values, axis=0)
                    logging.info(f"Final shape of {key}: {processed_data[key].shape}")
                except Exception as e:
                    logging.error(f"Error concatenating {key} data: {e}")
            else:
                logging.warning(f"No data for {key}")
        else:
            processed_data[key] = np.array(values, dtype=np.int32)
            unique, counts = np.unique(processed_data[key], return_counts=True)
            logging.info(f"Final labels shape: {processed_data[key].shape}, distribution: {list(zip(unique, counts))}")
    
    # Normalize the data
    if processed_data:
        logging.info("Normalizing data...")
        processed_data = normalize_data(processed_data)
    
    return processed_data
