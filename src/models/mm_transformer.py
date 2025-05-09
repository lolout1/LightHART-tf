# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers
import logging
import numpy as np

# Register the model class for serialization
@tf.keras.saving.register_keras_serializable(package="models")
class MMTransformer(tf.keras.Model):
    def __init__(self, mocap_frames=128, acc_frames=128, num_joints=32, in_chans=3, num_patch=4, acc_coords=3, 
                 spatial_embed=16, sdepth=4, adepth=4, tdepth=2, num_heads=2, mlp_ratio=2, qkv_bias=True, 
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.sdepth = sdepth
        self.adepth = adepth
        self.tdepth = tdepth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        
        # Build model components
        self._build_layers()
        
    def _build_layers(self):
        """Create all model layers"""
        # Tokens and positional embeddings
        self.temp_token = self.add_weight(
            name="temp_token", 
            shape=(1, 1, self.spatial_embed),
            initializer="zeros", 
            trainable=True
        )
        
        self.temporal_pos_embed = self.add_weight(
            name="temporal_pos_embed",
            shape=(1, 1, self.spatial_embed),
            initializer="zeros",
            trainable=True
        )
        
        # Spatial convolutional layers for skeleton processing
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(filters=self.in_chans, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=1, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_conv")
        
        # Spatial encoder for skeleton features
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(filters=self.spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="spatial_encoder")
        
        # Feature transformation layer
        self.transform = layers.Dense(self.spatial_embed, activation='relu', name="transform")
        
        # Temporal transformer blocks
        self.temporal_blocks = []
        for i in range(self.tdepth):
            block = TransformerBlock(
                dim=self.spatial_embed,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate * i / max(1, self.tdepth-1),
                name=f"temporal_block_{i}"
            )
            self.temporal_blocks.append(block)
        
        # Joint relation block
        self.joint_block = TransformerBlock(
            dim=self.spatial_embed,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            name="joint_block"
        )
        
        # Normalization layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.num_classes)
        ], name="class_head")
    
    def call(self, inputs, training=False):
        """Forward pass handling different input types"""
        # Handle different input types gracefully
        if isinstance(inputs, dict):
            acc_data = inputs.get('accelerometer')
            skl_data = inputs.get('skeleton')
            
            if skl_data is None:
                # Create dummy skeleton data
                batch_size = tf.shape(acc_data)[0]
                skl_data = tf.zeros((batch_size, self.mocap_frames, self.num_joints, self.in_chans), dtype=tf.float32)
            
            if acc_data is None:
                # Create dummy accelerometer data
                batch_size = tf.shape(skl_data)[0]
                acc_data = tf.zeros((batch_size, self.acc_frames, self.acc_coords), dtype=tf.float32)
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            acc_data, skl_data = inputs
        else:
            raise ValueError("Input must be dictionary with modality keys or tuple (acc_data, skl_data)")
        
        # Process skeleton data
        batch_size = tf.shape(skl_data)[0]
        
        # Reshape to [batch, channels, frames, joints]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x, training=training)
        
        # Reshape to [batch, frames, features]
        x = tf.reshape(x, [batch_size, self.mocap_frames, -1])
        
        # Apply spatial encoding
        x = self.spatial_encoder(x, training=training)
        
        # Feature transformation
        x = self.transform(x)
        
        # Process through joint block
        x = self.joint_block(x, training=training)
        
        # Add class token
        class_token = tf.repeat(self.temp_token, repeats=batch_size, axis=0)
        x = tf.concat([x, class_token], axis=1)
        
        # Add positional embedding
        x = x + self.temporal_pos_embed
        
        # Process through transformer blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Apply normalization
        x = self.temporal_norm(x)
        
        # Extract features from sequence (for distillation)
        seq_len = tf.shape(x)[1] - 1
        sequence = x[:, :seq_len, :]
        distill_features = tf.reduce_mean(sequence, axis=1)
        
        # Global average pooling and classification
        final_features = tf.reduce_mean(sequence, axis=1)
        logits = self.class_head(final_features, training=training)
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        return logits, distill_features
    
    def get_config(self):
        """Return serializable config for model saving/loading"""
        config = super().get_config()
        config.update({
            'mocap_frames': self.mocap_frames,
            'acc_frames': self.acc_frames,
            'num_joints': self.num_joints,
            'in_chans': self.in_chans,
            'acc_coords': self.acc_coords,
            'spatial_embed': self.spatial_embed,
            'sdepth': self.sdepth,
            'adepth': self.adepth,
            'tdepth': self.tdepth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'num_classes': self.num_classes,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="models")
class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_rate=0.0, 
                 attn_drop_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi-head attention
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop_rate
        )
        
        # MLP block
        self.mlp_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(self.mlp_dim, activation='gelu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])
    
    def call(self, x, training=False):
        # Normalization before attention
        x_norm = self.norm1(x)
        
        # Multi-head attention
        attn_output = self.attn(x_norm, x_norm, x_norm, training=training)
        
        # Apply dropout path if training
        if training and self.drop_path_rate > 0:
            attn_output = tf.nn.dropout(attn_output, rate=self.drop_path_rate)
        
        # First residual connection
        x = x + attn_output
        
        # Normalization before MLP
        x_norm = self.norm2(x)
        
        # MLP block
        mlp_output = self.mlp(x_norm, training=training)
        
        # Apply dropout path if training
        if training and self.drop_path_rate > 0:
            mlp_output = tf.nn.dropout(mlp_output, rate=self.drop_path_rate)
        
        # Second residual connection
        x = x + mlp_output
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate
        })
        return config
