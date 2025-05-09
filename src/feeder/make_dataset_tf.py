# feeder/make_dataset_tf.py

import tensorflow as tf
import numpy as np
import logging
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.io import loadmat

class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow implementation of the UTD-MM dataset loader"""
    
    def __init__(self, dataset, batch_size, use_smv=False):
        """
        Initialize the dataset loader
        
        Args:
            dataset: Dictionary containing modality data and labels
            batch_size: Batch size for training
            use_smv: Whether to use Signal Magnitude Vector calculation
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_smv = use_smv
        
        # Extract data from dataset dictionary
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        
        # Handle missing data with appropriate logging
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset")
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]
        
        # Process skeleton data
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
        
        # Handle missing labels
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Prepare data for TensorFlow
        self._prepare_data()
        self.indices = np.arange(self.num_samples)
        
        logging.info(f"Initialized UTD_MM_TF with {self.num_samples} samples")
    
    def _prepare_data(self):
        """Prepare data for TensorFlow processing"""
        try:
            # Convert to TensorFlow tensors
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            
            # Calculate SMV if requested (Signal Magnitude Vector)
            if self.use_smv:
                mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
                zero_mean = self.acc_data - mean
                sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
                self.smv = tf.sqrt(sum_squared)
                logging.info(f"SMV calculated with shape: {self.smv.shape}")
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector (SMV) for a batch of samples"""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    
    def __len__(self):
        """Return the number of batches"""
        return max(1, (self.num_samples + self.batch_size - 1) // self.batch_size)
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        try:
            # Get indices for this batch
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            tf_indices = tf.convert_to_tensor(batch_indices)
            
            # Gather data for this batch
            batch_data = {}
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
            
            # Return dummy data in case of error
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.acc_seq, 4 if self.use_smv else 3), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.acc_seq, 32, 3), dtype=tf.float32)
            dummy_data = {'accelerometer': dummy_acc, 'skeleton': dummy_skl}
            dummy_labels = tf.zeros(batch_size, dtype=tf.int32)
            dummy_indices = tf.range(batch_size)
            
            return dummy_data, dummy_labels, dummy_indices
    
    def on_epoch_end(self):
        """Called at the end of each epoch to shuffle data"""
        np.random.shuffle(self.indices)
