import tensorflow as tf
from tensorflow.keras import layers

class MMTransformerTF(tf.keras.Model):
    def __init__(self, num_classes=1, num_joints=32, embed_dim=32, num_heads=4, num_layers=2, window_size=64, **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_coords = 3

        # Skeleton processing (no input_shape)
        self.skeleton_conv = tf.keras.Sequential([
            layers.Conv2D(filters=embed_dim, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=embed_dim // 2, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.skeleton_encoder = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Accelerometer processing (no input_shape)
        self.acc_conv = tf.keras.Sequential([
            layers.Conv1D(filters=embed_dim, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Transformer blocks
        self.transformer_blocks = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
            for _ in range(num_layers)
        ]
        self.transformer_norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]

        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ])

    def add_smv(self, acc_data):
        """Add Signal Magnitude Vector to accelerometer data."""
        mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
        zero_mean = acc_data - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        return tf.concat([smv, acc_data], axis=-1)

    def call(self, inputs, training=False):
        acc_data = inputs['accelerometer']  # Shape: (batch, window_size, 3)
        skl_data = inputs['skeleton']       # Shape: (batch, window_size, 32, 3)

        # Dynamically add SMV to accelerometer data
        acc_data = self.add_smv(acc_data)   # Shape: (batch, window_size, 4)

        # Process skeleton data
        x_skl = self.skeleton_conv(skl_data)
        x_skl = tf.reduce_mean(x_skl, axis=-1)
        x_skl = self.skeleton_encoder(x_skl)

        # Process accelerometer data
        x_acc = self.acc_conv(acc_data)

        # Combine modalities
        x = (x_skl + x_acc) / 2

        # Transformer blocks
        for attn, norm in zip(self.transformer_blocks, self.transformer_norms):
            attn_output = attn(x, x)
            x = norm(x + attn_output)

        # Classification
        x = tf.reduce_mean(x, axis=1)
        logits = self.class_head(x)
        return logits, x
