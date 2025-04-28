import tensorflow as tf
from tensorflow.keras import layers
import logging

class MMTransformerTF(tf.keras.Model):
    def __init__(self, num_classes=1, num_joints=32, embed_dim=128, num_heads=4, num_layers=2, window_size=128, **kwargs):
        """Initialize MMTransformerTF for skeleton and/or accelerometer data."""
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_coords = 3
        self.logger = logging.getLogger(__name__)

        # Skeleton processing
        self.skeleton_conv = tf.keras.Sequential([
            layers.Conv2D(filters=embed_dim, kernel_size=(1, 9), padding='same', name='skeleton_conv1'),
            layers.BatchNormalization(name='skeleton_bn1'),
            layers.ReLU(name='skeleton_relu1'),
            layers.Conv2D(filters=embed_dim // 2, kernel_size=(1, 9), padding='same', name='skeleton_conv2'),
            layers.BatchNormalization(name='skeleton_bn2'),
            layers.ReLU(name='skeleton_relu2')
        ])

        self.skeleton_encoder = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=3, padding='same', name='skeleton_encoder_conv'),
            layers.BatchNormalization(name='skeleton_encoder_bn'),
            layers.ReLU(name='skeleton_encoder_relu')
        ])

        # Accelerometer processing
        self.acc_conv = tf.keras.Sequential([
            layers.Conv1D(filters=embed_dim, kernel_size=3, padding='same', name='acc_conv'),
            layers.BatchNormalization(name='acc_bn'),
            layers.ReLU(name='acc_relu')
        ])

        # Transformer blocks
        self.transformer_blocks = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name=f'mha_{i}')
            for i in range(num_layers)
        ]
        self.transformer_norms = [layers.LayerNormalization(epsilon=1e-6, name=f'norm_{i}') for i in range(num_layers)]

        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6, name='final_norm'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

    def add_smv(self, acc_data):
        """Add Signal Magnitude Vector to accelerometer data."""
        try:
            mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
            zero_mean = acc_data - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            smv = tf.sqrt(sum_squared)
            return tf.concat([smv, acc_data], axis=-1)
        except Exception as e:
            self.logger.error(f"Error adding SMV: {str(e)}")
            raise

    def call(self, inputs, training=False):
        """Forward pass."""
        try:
            if not isinstance(inputs, dict) or not inputs:
                raise ValueError("Inputs must be a dict with 'skeleton' or 'accelerometer' keys")
            if 'skeleton' in inputs and 'accelerometer' in inputs:
                skl_data = inputs['skeleton']
                acc_data = inputs['accelerometer']
                self.logger.debug(f"Dual modalities: skeleton={skl_data.shape}, accel={acc_data.shape}")
                acc_data = self.add_smv(acc_data)
                x_skl = self.skeleton_conv(skl_data)
                x_skl = tf.reduce_mean(x_skl, axis=-1)
                x_skl = self.skeleton_encoder(x_skl)
                x_acc = self.acc_conv(acc_data)
                x = (x_skl + x_acc) / 2
            elif 'skeleton' in inputs:
                skl_data = inputs['skeleton']
                self.logger.debug(f"Skeleton only: {skl_data.shape}")
                x_skl = self.skeleton_conv(skl_data)
                x_skl = tf.reduce_mean(x_skl, axis=-1)
                x_skl = self.skeleton_encoder(x_skl)
                x = x_skl
            elif 'accelerometer' in inputs:
                acc_data = inputs['accelerometer']
                self.logger.debug(f"Accel only: {acc_data.shape}")
                acc_data = self.add_smv(acc_data)
                x_acc = self.acc_conv(acc_data)
                x = x_acc
            else:
                raise ValueError("Input must contain 'skeleton' or 'accelerometer'")
            for attn, norm in zip(self.transformer_blocks, self.transformer_norms):
                attn_output = attn(x, x)
                x = norm(x + attn_output)
            features = tf.reduce_mean(x, axis=1)
            logits = self.class_head(features)
            return logits if training else (logits, features)
        except Exception as e:
            self.logger.error(f"Forward pass error: {str(e)}")
            raise
