import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

def csvloader(file_path, **kwargs):
    '''Loads csv data'''
    file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
    
    if 'skeleton' in file_path: 
        cols = 96
    else: 
        cols = 3
        
    activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path, **kwargs):
    '''Loads MatLab files'''
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv': csvloader, 
    'mat': matloader
}

def avg_pool_tf(sequence, window_size=5, stride=1, max_length=512):
    '''Executes average pooling to smoothen out the data'''
    # Convert to tensorflow
    sequence_tensor = tf.convert_to_tensor(sequence, dtype=tf.float32)
    
    # Reshape for 1D convolution
    orig_shape = tf.shape(sequence_tensor)
    sequence_reshaped = tf.reshape(sequence_tensor, [1, orig_shape[0], -1])
    sequence_transposed = tf.transpose(sequence_reshaped, [0, 2, 1])
    
    # Calculate stride
    if max_length < orig_shape[0]:
        stride = (orig_shape[0] // max_length) + 1
    else:
        stride = 1
    
    # Apply average pooling
    pooled = tf.keras.layers.AveragePooling1D(
        pool_size=window_size,
        strides=stride,
        padding='valid'
    )(sequence_transposed)
    
    # Reshape back
    pooled_transposed = tf.transpose(pooled, [0, 2, 1])
    pooled_reshaped = tf.reshape(pooled_transposed, [-1, orig_shape[1], orig_shape[2]])
    
    return pooled_reshaped.numpy()

def pad_sequence_tf(sequence, max_sequence_length, input_shape):
    '''Pools and pads the sequence to uniform length'''
    # Create output shape
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    # Pool sequence
    pooled_sequence = avg_pool_tf(sequence, max_length=max_sequence_length)
    
    # Create zero tensor
    new_sequence = np.zeros(shape, dtype=sequence.dtype)
    
    # Copy pooled sequence
    seq_len = min(pooled_sequence.shape[0], max_sequence_length)
    new_sequence[:seq_len] = pooled_sequence[:seq_len]
    
    return new_sequence

def sliding_window_tf(data, clearing_time_index, max_time, sub_window_size, stride_size):
    '''Sliding Window implementation'''
    assert clearing_time_index >= sub_window_size - 1, "Clearing value needs to be greater or equal to (window size - 1)"
    
    start = clearing_time_index - sub_window_size + 1
    
    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1
    
    # Create indices for sub-windows
    indices = []
    for i in range(0, max_time, stride_size):
        window_indices = np.arange(start + i, start + i + sub_window_size)
        indices.append(window_indices)
    
    # Stack indices
    indices = np.stack(indices)
    
    # Extract windows
    windows = tf.gather(data, indices)
    
    return windows.numpy()

def selective_sliding_window_tf(data, window_size, height, distance):
    '''Selective sliding window implementation'''
    # Calculate signal magnitude
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    sqrt_sum = tf.sqrt(tf.reduce_sum(tf.square(data_tensor), axis=1))
    
    # Find peaks
    peaks, _ = find_peaks(sqrt_sum.numpy(), height=height, distance=distance)
    
    # Create windows around peaks
    windows = []
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(data), start + window_size)
        
        # Skip if window is too small
        if end - start < window_size:
            continue
            
        window = data[start:end]
        windows.append(window)
    
    # Stack windows if any exist
    if windows:
        return np.stack(windows)
    else:
        return np.array([])

class ProcessorTF:
    '''Data Processor for TensorFlow'''
    def __init__(self, file_path, mode, max_length, label, **kwargs):
        assert mode in ['sliding_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.label = label
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs
    
    def set_input_shape(self, sequence):
        '''Sets the input shape based on sequence'''
        self.input_shape = sequence.shape
    
    def _import_loader(self, file_path):
        '''Imports the correct loader based on file type'''
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def load_file(self, file_path):
        '''Loads file data'''
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        self.set_input_shape(data)
        return data
    
    def process(self, data):
        '''Processes data based on mode'''
        if self.mode == 'avg_pool':
            data = pad_sequence_tf(
                sequence=data, 
                max_sequence_length=self.max_length,
                input_shape=self.input_shape
            )
        else:
            if self.label == 1:
                # Fall data
                data = selective_sliding_window_tf(
                    data, 
                    window_size=self.max_length, 
                    height=1.4, 
                    distance=50
                )
            else:
                # Non-fall data
                data = selective_sliding_window_tf(
                    data, 
                    window_size=self.max_length, 
                    height=1.2, 
                    distance=100
                )
        
        return data
