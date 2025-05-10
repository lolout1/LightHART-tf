# src/utils/processor_tf.py
import numpy as np
import logging
from collections import defaultdict
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

def align_sequence_dtw(data, joint_id=9, radius=1, dtw_max_dist=None):
    """Align accelerometer and skeleton data using DTW.

    Args:
        data: Dictionary containing 'skeleton' and 'accelerometer' data
        joint_id: Joint to use for alignment (default=9 which is left wrist joint)
        radius: Radius for fastdtw search (larger = more accurate but slower)
        dtw_max_dist: Maximum allowed DTW distance (alignment quality threshold)

    Returns:
        dict: Data with aligned modalities
    """
    try:
        # Validate input data
        if 'skeleton' not in data or 'accelerometer' not in data:
            logger.warning("Missing skeleton or accelerometer data for DTW alignment")
            return data

        # Check valid joint ID (max of 32 joints in common formats)
        if joint_id <= 0 or joint_id > 32 or (joint_id-1)*3 >= data['skeleton'].shape[1]:
            logger.warning(f"Invalid joint_id {joint_id} for skeleton of shape {data['skeleton'].shape}")
            # Fall back to joint 9 if possible
            if (9-1)*3 < data['skeleton'].shape[1]:
                joint_id = 9
                logger.warning(f"Falling back to joint_id 9 (left wrist)")
            else:
                # If joint 9 isn't available, use the first joint
                joint_id = 1
                logger.warning(f"Falling back to joint_id 1 (first available joint)")

        # Log original sequence lengths
        skeleton_len = len(data['skeleton'])
        accel_len = len(data['accelerometer'])
        logger.info(f"Before alignment - Skeleton: {skeleton_len}, Accelerometer: {accel_len} frames")

        # Extract data for alignment
        skeleton_joint_data = data['skeleton'][:, (joint_id-1)*3:joint_id*3]
        inertial_data = data['accelerometer']

        # Check sufficient data length
        min_length_threshold = 10  # Minimum length required for reliable DTW
        if len(skeleton_joint_data) < min_length_threshold or len(inertial_data) < min_length_threshold:
            logger.warning(f"Data too short for DTW alignment: Skeleton={len(skeleton_joint_data)}, Accel={len(inertial_data)}")
            # Simple truncation
            min_len = min(len(skeleton_joint_data), len(inertial_data))
            for key in data:
                if key != 'labels':
                    data[key] = data[key][:min_len]
            return data

        # Identify which keys need alignment (all except skeleton and labels)
        dynamic_keys = [k for k in data.keys() if k not in ['skeleton', 'labels']]

        # Compute Frobenius norms for both signals
        skeleton_frob_norm = norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = norm(inertial_data, axis=1)

        # Apply smoothing to reduce noise impact
        from scipy.signal import savgol_filter
        try:
            # Apply light smoothing (window length 5, polynomial order 2)
            skeleton_frob_norm = savgol_filter(skeleton_frob_norm, 5, 2)
            inertial_frob_norm = savgol_filter(inertial_frob_norm, 5, 2)
        except Exception as smooth_err:
            logger.warning(f"Smoothing failed, using raw signals: {smooth_err}")

        # Run FastDTW algorithm
        distance, path = fastdtw(
            inertial_frob_norm[:, np.newaxis],
            skeleton_frob_norm[:, np.newaxis],
            dist=euclidean,
            radius=radius  # Search radius - higher is more accurate
        )

        # Check alignment quality
        alignment_quality = distance / (len(inertial_frob_norm) + len(skeleton_frob_norm))
        logger.info(f"DTW alignment quality: {alignment_quality:.4f} (lower is better)")

        # If DTW distance exceeds threshold, reject alignment
        if dtw_max_dist is not None and distance > dtw_max_dist:
            logger.warning(f"DTW alignment rejected - distance {distance:.2f} > threshold {dtw_max_dist:.2f}")
            # Simple truncation
            min_len = min(len(skeleton_joint_data), len(inertial_data))
            for key in data:
                if key != 'labels':
                    data[key] = data[key][:min_len]
            return data

        # Filter repeated indices to ensure one-to-one mapping
        inertial_ids, skeleton_ids = filter_repeated_ids(path)

        # Check we have enough indices after filtering
        if len(inertial_ids) < min_length_threshold or len(skeleton_ids) < min_length_threshold:
            logger.warning(f"DTW alignment produced too few indices: Inertial={len(inertial_ids)}, Skeleton={len(skeleton_ids)}")
            # Simple truncation
            min_len = min(len(skeleton_joint_data), len(inertial_data))
            for key in data:
                if key != 'labels':
                    data[key] = data[key][:min_len]
            return data

        # Apply alignment by selecting indices
        data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_ids))
        for key in dynamic_keys:
            data[key] = filter_data_by_ids(data[key], list(inertial_ids))

        # Log alignment results
        logger.info(f"After alignment - Skeleton: {len(data['skeleton'])}, Accelerometer: {len(data['accelerometer'])} frames")
        return data

    except Exception as e:
        logger.error(f"Error in DTW alignment: {e}")
        # Simple truncation fallback
        try:
            min_len = min(len(data['skeleton']), len(data['accelerometer']))
            for key in data:
                if key != 'labels':
                    data[key] = data[key][:min_len]
            logger.warning(f"Fallback to simple truncation: length={min_len}")
        except Exception as fallback_err:
            logger.error(f"Fallback failed: {fallback_err}")
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
            logger.warning(f"Insufficient data for windowing")
            return {'labels': np.array([])}
        if label == 1:
            height, distance = 1.4, 50
        else:
            height, distance = 1.2, 100
        sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        if len(peaks) == 0:
            logger.warning(f"No peaks found for label {label}")
            return {'labels': np.array([])}
        windowed_data = defaultdict(list)
        for modality, modality_data in data.items():
            if modality == 'labels':
                continue
            for peak in peaks:
                start = max(0, peak - window_size // 2)
                end = min(len(modality_data), start + window_size)
                if end - start < window_size:
                    if end == len(modality_data):
                        start = max(0, end - window_size)
                        end = start + window_size
                if end - start == window_size and end <= len(modality_data):
                    window = modality_data[start:end]
                    windowed_data[modality].append(window)
                else:
                    logger.debug(f"Dropping window at peak {peak} - size {end-start} != {window_size}")
        if not windowed_data or len(windowed_data[next(iter(windowed_data))]) == 0:
            logger.warning(f"No valid windows created for label {label}")
            return {'labels': np.array([])}
        for key in windowed_data:
            windowed_data[key] = np.array(windowed_data[key])
        num_windows = len(windowed_data[next(iter(windowed_data))])
        windowed_data['labels'] = np.repeat(label, num_windows)
        return dict(windowed_data)
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
            return {'labels': np.array([])}
        windows_start = list(range(0, seq_length - max_length + 1, stride))
        for key, value in data.items():
            if key == 'labels':
                continue
            windows = []
            for start in windows_start:
                end = start + max_length
                if end <= len(value):
                    window = value[start:end]
                    if len(window) == max_length:
                        windows.append(window)
            if windows:
                windowed_data[key] = np.stack(windows)
            else:
                logger.warning(f"No valid windows created for {key}")
        if not windowed_data:
            logger.warning("No valid windows created for any modality")
            return {'labels': np.array([])}
        num_windows = len(windowed_data[next(iter(windowed_data))])
        windowed_data['labels'] = np.repeat(label, num_windows)
        return windowed_data
    except Exception as e:
        logger.error(f"Error in sliding_window: {e}")
        return {'labels': np.array([])}
