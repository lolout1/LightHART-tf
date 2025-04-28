import tensorflow as tf
import numpy as np
import logging

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.acc_data = dataset.get('accelerometer', None)
        self.gyro_data = dataset.get('gyroscope', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 64, 3), dtype=np.float32)
            self.num_samples = 1
            self.acc_seq = 64
            self.channels = 3
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
            
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) == 4:
                self.skl_seq, self.skl_length, joints, dims = self.skl_data.shape
        
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples:
            logging.warning(f"Labels length {len(self.labels)} doesn't match data samples {self.num_samples}")
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
            logging.error(f"Error preparing data: {e}")
            self.smv = tf.zeros((self.num_samples, self.acc_seq, 1), dtype=tf.float32)
    
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
    def calculate_weight(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        return tf.sqrt(tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True))
    
    def calculate_pitch(self, data):
        ax = data[:, :, 0]
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.math.atan2(-ax, tf.sqrt(ay**2 + az**2)), axis=-1)
    
    def calculate_roll(self, data):
        ax = data[:, :, 0]
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.math.atan2(ay, az), axis=-1)
    
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
            logging.error(f"Error in batch generation {idx}: {e}")
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            return dummy_data, dummy_labels, np.arange(batch_size)
