#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import logging
from scipy.signal import butter, filtfilt, find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

logger = logging.getLogger('processor-tf')

def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    try:
        if len(data) < 10: return data
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        logger.warning(f"Butterworth filter error: {e}")
        return data

def avg_pool_tf(sequence, max_length=128, window_size=5):
    try:
        shape = sequence.shape
        if shape[0] <= max_length: return sequence
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        stride = max(1, ((sequence.shape[2] - 1) // max_length) + 1)
        sequence = tf.nn.avg_pool1d(sequence, ksize=window_size, strides=stride, padding='VALID')
        sequence = sequence.numpy().squeeze(0).transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool_tf: {e}")
        if len(sequence) > max_length: return sequence[:max_length]
        return sequence

def pad_sequence_tf(sequence, max_length):
    try:
        shape = list(sequence.shape)
        if shape[0] == max_length: return sequence
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
        shape = list(sequence.shape)
        shape[0] = max_length
        padded = np.zeros(shape, dtype=sequence.dtype)
        copy_length = min(len(sequence), max_length)
        padded[:copy_length] = sequence[:copy_length]
        return padded

def filter_data_by_ids(data, ids):
    if not isinstance(ids, list): ids = list(ids)
    if len(ids) == 0: return data[:0]
    return data[ids]

def filter_repeated_ids(path):
    seen_first, seen_second = set(), set()
    for first, second in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    return seen_first, seen_second

def align_sequence_dtw(data, joint_id=9, use_dtw=True):
    try:
        has_skeleton = 'skeleton' in data and data['skeleton'] is not None and len(data['skeleton']) > 0
        has_accelerometer = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
        
        if (not has_skeleton or not has_accelerometer) or not use_dtw:
            logger.info(f"DTW skipped (use_dtw={use_dtw}, has_skeleton={has_skeleton}, has_acc={has_accelerometer})")
            return data
        
        logger.info(f"Performing DTW alignment between skeleton and accelerometer data")
        
        if len(data['skeleton'].shape) == 4:
            skeleton_joint_data = data['skeleton'][:, joint_id-1, :]
        elif len(data['skeleton'].shape) == 3 and data['skeleton'].shape[1] >= joint_id * 3:
            start_idx = (joint_id - 1) * 3
            end_idx = joint_id * 3
            skeleton_joint_data = data['skeleton'][:, start_idx:end_idx]
        else:
            logger.warning(f"Skeleton shape not compatible: {data['skeleton'].shape}")
            return data
        
        inertial_data = data['accelerometer']
        dynamic_keys = [k for k in data.keys() if k != 'skeleton' and k != 'labels']
        
        if len(dynamic_keys) > 1:
            min_length = min(len(data[key]) for key in dynamic_keys)
            for key in dynamic_keys:
                data[key] = data[key][:min_length]
        
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)
        
        if len(skeleton_frob_norm) == 0 or len(inertial_frob_norm) == 0:
            logger.warning(f"Empty norm vectors for DTW, skipping")
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        try:
            distance, path = fastdtw(
                inertial_frob_norm.reshape(-1, 1), 
                skeleton_frob_norm.reshape(-1, 1),
                dist=euclidean,
                radius=10
            )
            logger.info(f"DTW completed with distance: {distance:.2f}")
        except Exception as e:
            logger.error(f"DTW calculation error: {e}")
            min_length = min(len(inertial_data), len(data['skeleton']))
            for key in dynamic_keys:
                data[key] = data[key][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
            return data
        
        inertial_ids, skeleton_ids = filter_repeated_ids(path)
        inertial_ids = sorted(list(inertial_ids))
        skeleton_ids = sorted(list(skeleton_ids))
        
        inertial_ids = [idx for idx in inertial_ids if idx < len(inertial_data)]
        skeleton_ids = [idx for idx in skeleton_ids if idx < len(data['skeleton'])]
        
        min_required_length = 128
        if len(inertial_ids) < min_required_length or len(skeleton_ids) < min_required_length:
            logger.warning(f"DTW sequences too short: {len(inertial_ids)}/{len(skeleton_ids)} < {min_required_length}")
            min_length = min(len(inertial_data), len(data['skeleton']))
            for key in dynamic_keys:
                data[key] = data[key][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
        else:
            logger.info(f"Applying DTW alignment: {len(inertial_ids)} accelerometer frames mapped to {len(skeleton_ids)} skeleton frames")
            data['skeleton'] = filter_data_by_ids(data['skeleton'], skeleton_ids)
            for key in dynamic_keys:
                data[key] = filter_data_by_ids(data[key], inertial_ids)
        
        return data
    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        if 'accelerometer' in data and 'skeleton' in data:
            min_length = min(len(data['accelerometer']), len(data['skeleton']))
            data['accelerometer'] = data['accelerometer'][:min_length]
            data['skeleton'] = data['skeleton'][:min_length]
        return data

def create_dummy_windows(data, label, window_size):
    result = {'labels': np.array([label])}
    for key in data:
        if key != 'labels':
            if key == 'accelerometer':
                result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
            elif key == 'skeleton':
                result[key] = np.zeros((1, window_size, 32, 3), dtype=np.float32)
            elif key == 'gyroscope':
                result[key] = np.zeros((1, window_size, 3), dtype=np.float32)
    return result

def selective_windowing(data, window_size, label):
    try:
        required_modalities = [k for k in data.keys() if k != 'labels' and len(data[k]) > 0]
        logger.info(f"Available modalities for windowing: {required_modalities}")
        
        if not required_modalities:
            logger.warning("No valid modalities for windowing")
            return create_dummy_windows(data, label, window_size)
        
        peak_modality = 'accelerometer' if 'accelerometer' in required_modalities else required_modalities[0]
        peak_data = data[peak_modality]
        
        if len(peak_data) < window_size:
            logger.warning(f"Insufficient {peak_modality} data: {len(peak_data)} < {window_size}")
            return create_dummy_windows(data, label, window_size)
        
        sqrt_sum = np.sqrt(np.sum(peak_data**2, axis=1))
        
        if label == 1:  # Fall
            height, distance = 1.4, 50
        else:  # Non-fall
            height, distance = 1.2, 100
        
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        if len(peaks) == 0:
            peaks = [len(sqrt_sum) // 2]
            logger.info(f"No peaks found for label {label}, using center point")
        
        # Verify windows across ALL modalities first
        valid_windows_indices = []
        for peak in peaks:
            valid_for_all_modalities = True
            for key in required_modalities:
                start = max(0, peak - window_size // 2)
                end = min(len(data[key]), start + window_size)
                if end - start < window_size * 0.9:
                    valid_for_all_modalities = False
                    break
            if valid_for_all_modalities:
                valid_windows_indices.append(peak)
        
        if not valid_windows_indices:
            logger.warning(f"No valid windows across all modalities for label {label}")
            return create_dummy_windows(data, label, window_size)
        
        windows_dict = {}
        for key in required_modalities:
            windows = []
            for peak in valid_windows_indices:
                start = max(0, peak - window_size // 2)
                end = min(len(data[key]), start + window_size)
                window = np.zeros((window_size,) + data[key].shape[1:], dtype=data[key].dtype)
                copy_length = end - start
                window[:copy_length] = data[key][start:end]
                windows.append(window)
            if windows:
                windows_dict[key] = np.stack(windows)
                logger.info(f"Created {len(windows)} windows for {key}, shape: {windows_dict[key].shape}")
        
        if not windows_dict:
            logger.warning(f"Failed to create windows for any modality")
            return create_dummy_windows(data, label, window_size)
        
        windows_dict['labels'] = np.full(len(valid_windows_indices), label, dtype=np.int32)
        logger.info(f"Created {len(windows_dict['labels'])} synchronized windows for label {label}")
        return windows_dict
    except Exception as e:
        logger.error(f"Error in selective_windowing: {e}")
        return create_dummy_windows(data, label, window_size)

def sliding_window(data, max_length, stride=10, label=0):
    try:
        available_modalities = [k for k in data.keys() if k != 'labels' and len(data[k]) > 0]
        logger.info(f"Sliding window for modalities: {available_modalities}")
        
        if not available_modalities:
            logger.warning("No valid modalities for sliding window")
            return {'labels': np.array([label])}
        
        first_modality = available_modalities[0]
        seq_length = len(data[first_modality])
        
        if seq_length < max_length:
            logger.warning(f"Insufficient {first_modality} data: {seq_length} < {max_length}")
            return create_dummy_windows(data, label, max_length)
        
        window_starts = range(0, seq_length - max_length + 1, stride)
        windows_dict = {}
        
        for key in available_modalities:
            windows = []
            for start in window_starts:
                end = start + max_length
                window = data[key][start:end]
                if len(window) == max_length:
                    windows.append(window)
            
            if windows:
                try:
                    windows_dict[key] = np.stack(windows)
                    logger.info(f"Created {len(windows)} sliding windows for {key}, shape: {windows_dict[key].shape}")
                except Exception as e:
                    logger.error(f"Error stacking windows for {key}: {e}")
        
        if windows_dict:
            first_key = next(iter(windows_dict))
            windows_dict['labels'] = np.full(len(windows_dict[first_key]), label, dtype=np.int32)
            return windows_dict
        else:
            return create_dummy_windows(data, label, max_length)
    except Exception as e:
        logger.error(f"Error in sliding_window: {e}")
        return create_dummy_windows(data, label, max_length)
