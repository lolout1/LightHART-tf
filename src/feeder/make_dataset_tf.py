import tensorflow as tf
import numpy as np
import logging
import traceback

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        self.acc_data = dataset.get('accelerometer', None)
        self.gyro_data = dataset.get('gyroscope', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
            
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                # Handle flattened skeleton data
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    # Reshape to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) == 4:
                self.skl_seq, self.skl_length, self.skl_joints, self.skl_dims = self.skl_data.shape
        else:
            # Create dummy skeleton data if needed
            logging.warning("No skeleton data in dataset, creating dummy data")
            self.skl_data = np.zeros((self.num_samples, self.acc_seq, 32, 3), dtype=np.float32)
        
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Prepare data
        self._prepare_data()
        self.indices = np.arange(self.num_samples)
    
    def _prepare_data(self):
        try:
            # Convert to TensorFlow tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate SMV if requested
            if self.use_smv:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logging.info(f"SMV calculated with shape: {self.smv.shape}")
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            logging.error(traceback.format_exc())
    
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
    def __len__(self):
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            batch_indices = self.indices[start_idx:end_idx]
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            batch_data = {}
            
            # Get accelerometer data
            batch_acc = tf.gather(self.acc_data, tf_indices)
            
            # Add SMV if requested
            if self.use_smv:
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, tf_indices)
                else:
                    batch_smv = self.cal_smv(batch_acc)
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            # Add skeleton data
            batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            
            # Get labels
            batch_labels = tf.gather(self.labels, tf_indices)
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logging.error(f"Error in batch generation {idx}: {e}")
            logging.error(traceback.format_exc())
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.acc_seq, 32, 3), dtype=tf.float32)
            
            dummy_data = {
                'accelerometer': dummy_acc,
                'skeleton': dummy_skl
            }
            
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = np.arange(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
