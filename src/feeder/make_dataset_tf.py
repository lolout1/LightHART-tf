import tensorflow as tf
import numpy as np

class UTD_MM_TF(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, use_smv=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_smv = use_smv
        self.acc_data = dataset.get('accelerometer')
        self.skl_data = dataset.get('skeleton')
        self.labels = dataset.get('labels')
        self._validate_data()
        self.indices = np.arange(self.num_samples)
    def _validate_data(self):
        if self.acc_data is None or len(self.acc_data) == 0:
            self.acc_data = np.zeros((1, 128, 3), dtype=np.float32)
            self.num_samples = 1
        else:
            self.num_samples = len(self.acc_data)
        if self.skl_data is not None and len(self.skl_data) > 0:
            self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
            if self.skl_features % 3 == 0:
                joints = self.skl_features // 3
                self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
        else:
            self.skl_data = np.zeros((self.num_samples, 128, 32, 3), dtype=np.float32)
        if self.labels is None:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
    def cal_smv(self, sample):
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        return tf.sqrt(sum_squared)
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        batch_data = {}
        batch_acc = tf.gather(self.acc_data, batch_indices)
        if self.use_smv:
            batch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
        else:
            batch_data['accelerometer'] = batch_acc
        batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
        batch_labels = tf.gather(self.labels, batch_indices)
        return batch_data, batch_labels, batch_indices
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
