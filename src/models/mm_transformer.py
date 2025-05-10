import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MMTransformer(Model):
    def __init__(
        self,
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        num_patch=2,
        acc_coords=3,
        spatial_embed=32,
        sdepth=4,
        adepth=4,
        tdepth=2,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
        num_classes=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configuration
        self.num_patch = num_patch
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.joint_coords = in_chans
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.tdepth = tdepth
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Match PyTorch initialization exactly
        temp_embed = spatial_embed
        
        # Spatial convolution layers (matches PyTorch)
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(in_chans, (1, 9), 1, padding='same', name='spatial_conv1'),
            layers.BatchNormalization(name='spatial_bn1'),
            layers.ReLU(),
            layers.Conv2D(1, (1, 9), 1, padding='same', name='spatial_conv2'),
            layers.BatchNormalization(name='spatial_bn2'),
            layers.ReLU()
        ])
        
        # Spatial encoder (matches PyTorch)
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(temp_embed, 3, 1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        # Temporal positional embedding
        self.temporal_pos_embed = self.add_weight(
            name='temporal_pos_embed',
            shape=(1, 1, spatial_embed),
            initializer='zeros',
            trainable=True
        )
        
        # Joint relation block
        self.joint_block = TransformerBlock(
            dim=temp_embed,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate
        )
        
        # Temporal transformer blocks
        self.temporal_blocks = [
            TransformerBlock(
                dim=temp_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate * i / max(tdepth - 1, 1)
            ) for i in range(tdepth)
        ]
        
        # Norms
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ])
        
    def call(self, inputs, training=False):
        # Handle input types
        if isinstance(inputs, dict):
            acc_data = inputs.get('accelerometer')
            skl_data = inputs.get('skeleton')
        else:
            # Assume input is (acc_data, skl_data)
            acc_data, skl_data = inputs
            
        # Get shapes
        batch_size = tf.shape(skl_data)[0]
        frames = tf.shape(skl_data)[1]
        joints = tf.shape(skl_data)[2]
        channels = tf.shape(skl_data)[3]
        
        # Spatial processing (matches PyTorch)
        x = tf.transpose(skl_data, [0, 3, 1, 2])  # B, C, F, J
        x = self.spatial_conv(x)
        x = tf.transpose(x, [0, 2, 3, 1])  # B, F, J, C
        x = tf.reshape(x, [batch_size, frames, -1])  # B, F, J*C
        
        # Spatial encoder
        x = self.spatial_encoder(x)
        
        # Joint relation block
        x = self.joint_block(x, training=training)
        
        # Add temporal positional embedding
        x = x + self.temporal_pos_embed
        
        # Temporal transformer blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
            
        # Normalize
        x = self.temporal_norm(x)
        features = x
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Classification
        logits = self.class_head(x)
        
        return logits, features


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop,
            use_bias=qkv_bias
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else layers.Lambda(lambda x: x)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
        
    def call(self, x, training=False):
        # Multi-head attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, training=training)
        x = self.drop_path(x, training=training)
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + residual
        
        return x


class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_features)
        self.act = layers.ReLU()
        self.fc2 = layers.Dense(in_features)
        self.drop = layers.Dropout(drop)
        
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        
    def call(self, x, training=False):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * binary_tensor
