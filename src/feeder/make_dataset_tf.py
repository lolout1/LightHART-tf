import tensorflow as tf
import numpy as np

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        # Initialize from dataset with robust error handling
        self.acc_data = dataset.get('accelerometer', None)
        self.gyro_data = dataset.get('gyroscope', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        self.subjects = dataset.get('subjects', None)
        
        # Handle missing data
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
            
        # Process skeleton data if available
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                # Default reshape for 3D data
                joints = self.skl_features // 3
                self.skl_data = np.reshape(self.skl_data, (self.skl_seq, self.skl_length, joints, 3))
            elif len(self.skl_data.shape) == 4:
                self.skl_seq, self.skl_length, joints, dims = self.skl_data.shape
        
        # Ensure labels match data samples
        if self.labels is None or len(self.labels) == 0:
            print("Warning: No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples and self.num_samples > 0:
            print(f"Warning: Labels length {len(self.labels)} doesn't match data samples {self.num_samples}")
            # Truncate or pad labels to match data
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                self.labels = np.pad(self.labels, (0, self.num_samples - len(self.labels)), 'constant')
        
        self.batch_size = batch_size
        self.crop_size = 64
    
    def __len__(self):
        return max(1, int(np.ceil(self.num_samples / self.batch_size)))
    
    def __getitem__(self, idx):
        # Empty batch case
        if self.num_samples == 0:
            empty_data = {'accelerometer': tf.zeros((0, self.acc_seq, self.channels+1))}
            if self.skl_data is not None:
                empty_data['skeleton'] = tf.zeros((0, self.skl_length, self.skl_data.shape[2], 3))
            return empty_data, tf.zeros(0, dtype=tf.int32), np.array([])
        
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_indices = np.arange(start_idx, end_idx)
        
        # Get data for batch
        batch_data = {}
        
        try:
            # Process accelerometer data
            batch_acc = self.acc_data[batch_indices]
            
            # Add signal magnitude vector
            watch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([watch_smv, batch_acc], axis=-1)
            
            # Process skeleton data if available
            if self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.convert_to_tensor(
                    self.skl_data[batch_indices], dtype=tf.float32)
            
            # Get labels
            batch_labels = tf.convert_to_tensor(self.labels[batch_indices], dtype=tf.int32)
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            print(f"Error in data batch generation: {e}")
            # Return empty batch on error
            empty_data = {'accelerometer': tf.zeros((1, self.acc_seq, self.channels+1))}
            if self.skl_data is not None:
                empty_data['skeleton'] = tf.zeros((1, self.skl_length, self.skl_data.shape[2], 3))
            return empty_data, tf.zeros(1, dtype=tf.int32), np.array([0])
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector"""
        if len(sample.shape) < 3:
            # Handle unexpected shape
            return tf.zeros((*sample.shape[:-1], 1), dtype=tf.float32)
            
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
    def calculate_weight(self, sample):
        """Calculate magnitude of accelerometer data"""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        return tf.sqrt(tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True))
    
    def calculate_pitch(self, data):
        """Calculate pitch from accelerometer data"""
        ax = data[:, :, 0]
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.math.atan2(-ax, tf.sqrt(ay**2 + az**2)), axis=-1)
    
    def calculate_roll(self, data):
        """Calculate roll from accelerometer data"""
        ax = data[:, :, 0]
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.math.atan2(ay, az), axis=-1)
