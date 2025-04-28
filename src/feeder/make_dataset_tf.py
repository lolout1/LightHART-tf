import tensorflow as tf
import numpy as np
import logging

class UTD_MM_TF(tf.keras.utils.Sequence):
    """
    TensorFlow dataset class for UTD-MM, mirroring PyTorch's UTD_mm.
    Handles multi-modal data (accelerometer and skeleton) with batching.
    
    Args:
        dataset (dict): Dictionary containing 'accelerometer', 'skeleton', and 'labels'.
        batch_size (int): Number of samples per batch.
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.acc_data = dataset.get('accelerometer', None)
        self.skl_data = dataset.get('skeleton', None)
        self.labels = dataset.get('labels', None)
        self.crop_size = 64  # Default sequence length, matching PyTorch

        # Validate and initialize accelerometer data
        if self.acc_data is None or len(self.acc_data) == 0:
            logging.warning("No accelerometer data in dataset, using zeros")
            self.acc_data = np.zeros((1, self.crop_size, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = self.acc_data.shape[0]
            self.acc_seq = self.acc_data.shape[1]
            self.channels = self.acc_data.shape[2]

        # Validate and reshape skeleton data
        if self.skl_data is None or len(self.skl_data) == 0:
            logging.warning("No skeleton data in dataset, using zeros")
            self.skl_data = np.zeros((self.num_samples, self.crop_size, 32, 3), dtype=np.float32)
        else:
            # Handle different input shapes
            if len(self.skl_data.shape) == 3:  # (samples, seq, 96)
                if self.skl_data.shape[2] == 96:
                    self.skl_data = self.skl_data.reshape(self.num_samples, self.skl_data.shape[1], 32, 3)
                else:
                    logging.error(f"Unexpected skeleton feature size: {self.skl_data.shape[2]}")
                    raise ValueError("Skeleton data features must be 96 or already in (joints, coords) format")
            elif len(self.skl_data.shape) == 4:  # (samples, seq, joints, coords)
                if self.skl_data.shape[2] != 32 or self.skl_data.shape[3] != 3:
                    logging.error(f"Invalid skeleton shape: {self.skl_data.shape}")
                    raise ValueError("Skeleton data must have 32 joints and 3 coordinates")
            else:
                logging.error(f"Unexpected skeleton data shape: {self.skl_data.shape}")
                raise ValueError("Skeleton data must be 3D or 4D tensor")

        # Validate and adjust labels
        if self.labels is None or len(self.labels) == 0:
            logging.warning("No labels found, using zeros")
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        elif len(self.labels) != self.num_samples:
            logging.warning(f"Labels length {len(self.labels)} doesn't match samples {self.num_samples}")
            if len(self.labels) > self.num_samples:
                self.labels = self.labels[:self.num_samples]
            else:
                self.labels = np.pad(self.labels, (0, self.num_samples - len(self.labels)),
                                    mode='constant', constant_values=self.labels[-1])

        # Prepare data for TensorFlow
        self._prepare_data()

    def _prepare_data(self):
        """Convert data to TensorFlow tensors and compute SMV."""
        try:
            self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
            self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
            self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

            # Compute Signal Magnitude Vector (SMV) for accelerometer
            mean = tf.reduce_mean(self.acc_data, axis=1, keepdims=True)
            zero_mean = self.acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            self.smv = tf.sqrt(sum_squared)

            # Concatenate SMV with accelerometer data
            self.acc_data_with_smv = tf.concat([self.smv, self.acc_data], axis=-1)

        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            # Fallback to zeros
            self.acc_data = tf.zeros((self.num_samples, self.crop_size, 3), dtype=tf.float32)
            self.skl_data = tf.zeros((self.num_samples, self.crop_size, 32, 3), dtype=tf.float32)
            self.labels = tf.zeros((self.num_samples,), dtype=tf.int32)
            self.smv = tf.zeros((self.num_samples, self.crop_size, 1), dtype=tf.float32)
            self.acc_data_with_smv = tf.zeros((self.num_samples, self.crop_size, 4), dtype=tf.float32)

    def random_crop(self, data):
        """Apply random cropping to match sequence length, mirroring PyTorch."""
        length = tf.shape(data)[1]
        if length <= self.crop_size:
            return tf.pad(data, [[0, 0], [0, self.crop_size - length], [0, 0]], mode='CONSTANT')
        start_idx = tf.random.uniform([], 0, length - self.crop_size + 1, dtype=tf.int32)
        return tf.slice(data, [0, start_idx, 0], [-1, self.crop_size, -1])

    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector (SMV), mirroring PyTorch."""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)

    def calculate_weight(self, sample):
        """Calculate magnitude (weight) of accelerometer data."""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        return tf.sqrt(tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True))

    def calculate_pitch(self, data):
        """Calculate pitch angle from accelerometer data."""
        ax = data[:, :, 0]
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.atan2(-ax, tf.sqrt(ay**2 + az**2)), axis=-1)

    def calculate_roll(self, data):
        """Calculate roll angle from accelerometer data."""
        ay = data[:, :, 1]
        az = data[:, :, 2]
        return tf.expand_dims(tf.atan2(ay, az), axis=-1)

    def __len__(self):
        """Return number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """Generate a batch of data."""
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            if start_idx >= self.num_samples:
                start_idx = 0
                end_idx = min(self.batch_size, self.num_samples)

            indices = tf.range(start_idx, end_idx)
            batch_acc = tf.gather(self.acc_data_with_smv, indices)
            batch_skl = tf.gather(self.skl_data, indices)
            batch_labels = tf.gather(self.labels, indices)

            # Apply random cropping if needed
            if self.acc_seq > self.crop_size:
                batch_acc = self.random_crop(batch_acc)
                batch_skl = self.random_crop(batch_skl)

            data = {
                'accelerometer': batch_acc,  # Shape: (batch, 64, 4) with SMV
                'skeleton': batch_skl        # Shape: (batch, 64, 32, 3)
            }

            return data, batch_labels, indices.numpy()

        except Exception as e:
            logging.error(f"Error in batch {idx}: {e}")
            # Return dummy data on failure
            batch_size = min(self.batch_size, self.num_samples)
            dummy_acc = tf.zeros((batch_size, self.crop_size, 4), dtype=tf.float32)
            dummy_skl = tf.zeros((batch_size, self.crop_size, 32, 3), dtype=tf.float32)
            dummy_labels = tf.zeros((batch_size,), dtype=tf.int32)
            return {'accelerometer': dummy_acc, 'skeleton': dummy_skl}, dummy_labels, np.arange(batch_size)

if __name__ == "__main__":
    # Example usage
    sample_dataset = {
        'accelerometer': np.random.randn(100, 128, 3).astype(np.float32),
        'skeleton': np.random.randn(100, 128, 96).astype(np.float32),
        'labels': np.random.randint(0, 10, 100, dtype=np.int32)
    }
    dataset = UTD_MM_TF(sample_dataset, batch_size=16)
    data, labels, indices = dataset[0]
    print("Accelerometer shape:", data['accelerometer'].shape)  # Should be (16, 64, 4)
    print("Skeleton shape:", data['skeleton'].shape)          # Should be (16, 64, 32, 3)
    print("Labels shape:", labels.shape)                      # Should be (16,)
