#!/usr/bin/env python
import numpy as np
import logging
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm
import tensorflow as tf

logger = logging.getLogger(__name__)

def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    try:
        if len(data) < 10: 
            logger.warning(f"Data too short for filtering: {len(data)} samples")
            return data
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        logger.error(f"Error in butterworth filter: {e}")
        return data

def avg_pool_tf(sequence, max_length=128, window_size=5):
    try:
        shape = sequence.shape
        if shape[0] <= max_length: 
            return sequence
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0,2,1)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        stride = max(1, ((sequence.shape[2]-1)//max_length)+1)
        sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1,0)
        sequence = sequence.reshape(-1, *shape[1:])
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool_tf: {e}")
        return sequence[:max_length] if len(sequence) > max_length else sequence

def pad_sequence_tf(sequence, max_length):
    try:
        shape = list(sequence.shape)
        if shape[0] == max_length: 
            return sequence
        if shape[0] > max_length:
            pooled_sequence = avg_pool_tf(sequence, max_length)
        else:
            pooled_sequence = sequence
        shape[0] = max_length
        new_sequence = np.zeros(shape, sequence.dtype)
        new_sequence[:len(pooled_sequence)] = pooled_sequence
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_tf: {e}")
        return sequence

def align_sequence_dtw(data, joint_id=9):
    try:
        if 'skeleton' not in data or 'accelerometer' not in data:
            logger.warning("Missing skeleton or accelerometer data for DTW alignment")
            return data
        skeleton_joint_data = data['skeleton'][:, (joint_id-1)*3:joint_id*3]
        inertial_data = data['accelerometer']
        if len(skeleton_joint_data) < 10 or len(inertial_data) < 10:
            logger.warning(f"Data too short for DTW alignment: skeleton={len(skeleton_joint_data)}, inertial={len(inertial_data)}")
            min_len = min(len(skeleton_joint_data), len(inertial_data))
            for key in data:
                data[key] = data[key][:min_len]
            return data
        dynamic_keys = [k for k in data.keys() if k not in ['skeleton', 'labels']]
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)
        distance, path = fastdtw(inertial_frob_norm[:, np.newaxis], skeleton_frob_norm[:, np.newaxis], dist=euclidean)
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        if len(inertial_ids) < 10 or len(skeleton_ids) < 10:
            logger.warning(f"DTW alignment produced too few indices: inertial={len(inertial_ids)}, skeleton={len(skeleton_ids)}")
            min_len = min(len(skeleton_joint_data), len(inertial_data))
            for key in data:
                data[key] = data[key][:min_len]
            return data
        data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_ids))
        for key in dynamic_keys:
            data[key] = filter_data_by_ids(data[key], list(inertial_ids))
        logger.debug(f"DTW alignment completed: {len(skeleton_ids)} skeleton frames, {len(inertial_ids)} inertial frames")
        return data
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        min_len = min(len(data['skeleton']), len(data['accelerometer']))
        for key in data:
            if key != 'labels':
                data[key] = data[key][:min_len]
        return data

def filter_repeated_ids(path):
    seen_first, seen_second = set(), set()
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    return seen_first, seen_second

def filter_data_by_ids(data, ids):
    return data[ids, :]

def selective_sliding_window(data, window_size, label):
    try:
        if 'accelerometer' not in data or len(data['accelerometer']) < window_size:
            logger.warning(f"Insufficient accelerometer data for windowing: {len(data.get('accelerometer', []))} < {window_size}")
            return {'labels': np.array([])}
        if label == 1:
            height, distance = 1.4, 50
        else:
            height, distance = 1.2, 100
        sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        logger.debug(f"Found {len(peaks)} peaks for label {label}")
        if len(peaks) == 0:
            logger.warning(f"No peaks found for label {label}, using center position")
            peaks = [len(sqrt_sum) // 2]
        windowed_data = {}
        for key, value in data.items():
            if key == 'labels': 
                continue
            windows = []
            for peak in peaks:
                start = max(0, peak - window_size//2)
                end = min(len(value), start + window_size)
                if end - start < window_size * 0.8:
                    logger.warning(f"Window too small at peak {peak}: {end-start} < {window_size * 0.8}")
                    continue
                window = np.zeros((window_size,) + value.shape[1:], dtype=value.dtype)
                window[:end-start] = value[start:end]
                windows.append(window)
            if windows:
                windowed_data[key] = np.stack(windows)
                logger.debug(f"Created {len(windows)} windows for {key}")
            else:
                logger.warning(f"No valid windows created for {key}")
        if not windowed_data:
            logger.warning("No valid windows created for any modality")
            return {'labels': np.array([])}
        windowed_data['labels'] = np.repeat(label, len(windows))
        return windowed_data
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {e}")
        return {'labels': np.array([])}

def sliding_window(data, max_length, stride=10, label=0):
    try:
        windowed_data = {}
        reference_key = next((k for k in data.keys() if k != 'labels'), None)
        if reference_key is None:
            logger.error("No valid data keys found")
            return {'labels': np.array([])}
        seq_length = len(data[reference_key])
        if seq_length < max_length:
            logger.warning(f"Sequence too short for sliding window: {seq_length} < {max_length}")
            windows_start = [0]
        else:
            windows_start = list(range(0, seq_length - max_length + 1, stride))
        for key, value in data.items():
            if key == 'labels': 
                continue
            windows = []
            for start in windows_start:
                end = start + max_length
                if end <= len(value):
                    windows.append(value[start:end])
            if windows:
                windowed_data[key] = np.stack(windows)
                logger.debug(f"Created {len(windows)} sliding windows for {key}")
            else:
                logger.warning(f"No valid windows created for {key}")
        if not windowed_data:
            logger.warning("No valid windows created for any modality")
            return {'labels': np.array([])}
        windowed_data['labels'] = np.repeat(label, len(windows))
        return windowed_data
    except Exception as e:
        logger.error(f"Error in sliding_window: {e}")
        return {'labels': np.array([])}
