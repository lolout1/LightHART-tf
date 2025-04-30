import tensorflow as tf
import numpy as np
import logging
import traceback

class UTD_MM_TF(tf.keras.utils.Sequence):
    '''TensorFlow data feeder compatible with PyTorch UTD_MM'''
    def __init__(self, dataset, batch_size, use_smv=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv

        # Extract data
        self.acc_data = dataset.get('accelerometer', None)
        self.gyro_data = dataset.get('gyroscope', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)

        # Validate data
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
                # Handle flattened skeleton data
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                joints = self.skl_features // 3
                if joints * 3 == self.skl_features:
                    # Reshape to [batch, frames, joints, 3]
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
            elif len(self.skl_data.shape) == 4:
                self.skl_seq, self.skl_length, self.skl_joints, self.skl_dims = self.skl_data.shape
        
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples:
            logging.warning(f"Labels length {len(self.labels)} doesn't match data samples {self.num_samples}")
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                # Pad with last label
                last_label = self.labels[-1]
                self.labels = np.concatenate([
                    self.labels, 
                    np.full(self.num_samples - len(self.labels), last_label, dtype=self.labels.dtype)
                ])
        
        # Initialize data for TensorFlow
        self._prepare_data()
        
        # Store indices for more reliable batch generation
        self.indices = np.arange(self.num_samples)
    
    def _prepare_data(self):
        """Prepare data for TensorFlow - compatible with PyTorch implementation"""
        try:
            # Convert to TensorFlow tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate signal magnitude vector (SMV) if requested
            if self.use_smv:
                # Calculate SMV precisely as in PyTorch implementation
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logging.info(f"SMV calculated with shape: {self.smv.shape}")
            
            # Convert skeleton data if available
            if self.skl_data is not None and len(self.skl_data) > 0:
                self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
                
            # Log shapes for debugging
            logging.info(f"Prepared accelerometer data shape: {self.acc_data.shape}")
            if self.use_smv and hasattr(self, 'smv'):
                logging.info(f"Prepared SMV data shape: {self.smv.shape}")
            if hasattr(self, 'skl_data') and self.skl_data is not None:
                logging.info(f"Prepared skeleton data shape: {self.skl_data.shape}")
                
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            logging.error(traceback.format_exc())
            
            # Create fallback tensors
            if not hasattr(self, 'acc_seq'):
                self.acc_seq = self.acc_data.shape[1] if hasattr(self, 'acc_data') and len(self.acc_data.shape) > 1 else 64
            
            if self.use_smv and not hasattr(self, 'smv'):
                self.smv = tf.zeros((self.num_samples, self.acc_seq, 1), dtype=tf.float32)
    
    def cal_smv(self, sample):
        """Calculate SMV for a single sample - matches PyTorch implementation"""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
    def __len__(self):
        """Number of batches - compatible with PyTorch implementation"""
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        """Get a batch - compatible with PyTorch implementation"""
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            # Ensure valid range
            if start_idx >= self.num_samples:
                start_idx = 0
                end_idx = min(self.batch_size, self.num_samples)
            
            # Use stored indices for batch generation
            batch_indices = self.indices[start_idx:end_idx]
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            # Create batch data
            batch_data = {}
            
            # Add accelerometer data (with or without SMV)
            batch_acc = tf.gather(self.acc_data, tf_indices)
            
            if self.use_smv:
                # Either use precalculated SMV or calculate on the fly
                if hasattr(self, 'smv') and self.smv is not None:
                    batch_smv = tf.gather(self.smv, tf_indices)
                else:
                    batch_smv = self.cal_smv(batch_acc)
                
                # Concatenate SMV with accelerometer data
                batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
            else:
                batch_data['accelerometer'] = batch_acc
            
            # Add skeleton if available
            if hasattr(self, 'skl_data') and self.skl_data is not None and len(self.skl_data) > 0:
                batch_data['skeleton'] = tf.gather(self.skl_data, tf_indices)
            
            # Get labels
            batch_labels = tf.gather(self.labels, tf_indices)
            
            # Log first batch for debugging
            if idx == 0:
                for key, value in batch_data.items():
                    logging.info(f"First batch {key} shape: {value.shape}")
                logging.info(f"First batch labels shape: {batch_labels.shape}")
            
            return batch_data, batch_labels, batch_indices
            
        except Exception as e:
            logging.error(f"Error in batch generation {idx}: {e}")
            logging.error(traceback.format_exc())
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            channels = 4 if self.use_smv else 3
            
            dummy_acc = tf.zeros((batch_size, self.acc_seq, channels), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = np.arange(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        """Called at the end of each epoch - allows reshuffling for training"""
        # Optionally shuffle indices for next epoch
        # np.random.shuffle(self.indices)
        pass
