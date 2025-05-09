#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mm_transformer.py - Teacher model for TensorFlow LightHART
Exact match to PyTorch implementation with TF compatibility
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np
import logging

logger = logging.getLogger('mm_transformer')

# Version compatibility for serialization
def get_serializable_decorator():
    """Return appropriate serialization decorator based on TF version"""
    # Check for newer API (TF 2.5+)
    if hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'register_keras_serializable'):
        return tf.keras.saving.register_keras_serializable
    # Check for older API
    elif hasattr(tf.keras, 'utils') and hasattr(tf.keras.utils, 'register_keras_serializable'):
        return tf.keras.utils.register_keras_serializable
    # Fallback to no-op decorator
    else:
        def dummy_decorator(*args, **kwargs):
            def wrapper(cls):
                return cls
            return wrapper
        logger.warning("TensorFlow version doesn't support keras serialization decorators")
        return dummy_decorator

# Get appropriate serialization decorator
register_keras_serializable = get_serializable_decorator()

class MMTransformer(Model):
    """
    Multi-modal transformer for accelerometer and skeleton data.
    Takes skeleton data and accelerometer data as input and outputs binary classification.
    
    Architecture matches the PyTorch implementation in models/experimental_cvtransformer.py
    """
    def __init__(
        self, 
        mocap_frames=128,
        acc_frames=128, 
        num_joints=32, 
        in_chans=3, 
        num_patch=4, 
        acc_coords=3,
        spatial_embed=16, 
        sdepth=4, 
        adepth=4, 
        tdepth=2, 
        num_heads=2, 
        mlp_ratio=2, 
        qkv_bias=True,
        drop_rate=0.2, 
        attn_drop_rate=0.2, 
        drop_path_rate=0.2, 
        num_classes=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store configuration
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
        self.num_patch = num_patch
        self.skl_patch_size = mocap_frames // num_patch
        
        logger.info(f"Initializing MMTransformer with: frames={mocap_frames}, joints={num_joints}, "
                    f"embed_dim={spatial_embed}, heads={num_heads}")
        
        # Build all model layers
        self._build_layers()
    
    def _build_layers(self):
        """Create all layers for the model"""
        # === Tokens and position embeddings ===
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
        
        # === Spatial convolutional layers for skeleton processing ===
        # Matches PyTorch implementation exactly
        self.spatial_conv = Sequential([
            layers.Conv2D(filters=self.in_chans, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=1, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_conv")
        
        # === Spatial encoder for skeleton features ===
        self.spatial_encoder = Sequential([
            layers.Conv1D(filters=self.spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="spatial_encoder")
        
        # === Feature transformation layer ===
        self.transform = layers.Dense(self.spatial_embed, activation='relu', name="transform")
        
        # === Dropout for regularization ===
        self.pos_drop = layers.Dropout(self.drop_rate)
        
        # === Joint relation block ===
        self.joint_block = TransformerBlock(
            dim=self.spatial_embed,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            name="joint_relation_block"
        )
        
        # === Temporal transformer blocks ===
        self.temporal_blocks = []
        for i in range(self.tdepth):
            drop_path = self.drop_path_rate * i / max(1, self.tdepth-1)
            block = TransformerBlock(
                dim=self.spatial_embed,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=drop_path,
                name=f"temporal_block_{i}"
            )
            self.temporal_blocks.append(block)
        
        # === Normalization layers ===
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        self.spatial_norm = layers.LayerNormalization(epsilon=1e-6, name="spatial_norm")
        
        # === Classification head ===
        self.class_head = Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.num_classes)
        ], name="class_head")
    
    def call(self, inputs, training=False):
        """Forward pass handling different input types"""
        # Handle different input formats
        if isinstance(inputs, dict):
            # Dictionary input with named modalities
            acc_data = inputs.get('accelerometer')
            skl_data = inputs.get('skeleton')
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            # Tuple input (acc_data, skl_data)
            acc_data, skl_data = inputs
        else:
            raise ValueError("Input must be dictionary with modality keys or tuple (acc_data, skl_data)")
        
        # Ensure data exists
        if acc_data is None:
            raise ValueError("Accelerometer data is required")
        if skl_data is None:
            raise ValueError("Skeleton data is required")
        
        # Get batch size
        batch_size = tf.shape(skl_data)[0]
        
        # === Process skeleton data ===
        # Reshape to [batch, channels, frames, joints] - matching PyTorch implementation
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x, training=training)
        
        # Reshape for further processing
        x = tf.reshape(x, [batch_size, -1, self.mocap_frames])
        x = tf.transpose(x, [0, 2, 1])  # [batch, frames, features]
        
        # Apply spatial encoding
        x = self.spatial_encoder(x, training=training)
        
        # Apply feature transformation
        x = self.transform(x)
        
        # === Joint relation processing ===
        x = self.joint_block(x, training=training)
        
        # Save features before temporal processing
        features_before_temporal = x
        
        # === Add class token and positional embedding ===
        # Replicate the class token for each item in the batch
        class_tokens = tf.repeat(self.temp_token, repeats=batch_size, axis=0)
        
        # Concatenate with the processed features
        x = tf.concat([x, class_tokens], axis=1)
        
        # Add positional embedding
        x = x + self.temporal_pos_embed
        
        # Apply dropout
        x = self.pos_drop(x, training=training)
        
        # === Process through transformer blocks ===
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Apply normalization
        x = self.temporal_norm(x)
        
        # Extract features from sequence (for distillation)
        seq_len = tf.shape(x)[1] - 1  # exclude class token
        sequence = x[:, :seq_len, :]
        
        # Global average pooling from feature sequence
        pooled_features = tf.reduce_mean(sequence, axis=1)
        
        # Classification
        logits = self.class_head(pooled_features, training=training)
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        # Return both logits and features for distillation
        return logits, features_before_temporal
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'mocap_frames': self.mocap_frames,
            'acc_frames': self.acc_frames,
            'num_joints': self.num_joints,
            'in_chans': self.in_chans,
            'num_patch': self.num_patch,
            'acc_coords': self.acc_coords,
            'spatial_embed': self.spatial_embed,
            'sdepth': self.sdepth,
            'adepth': self.adepth,
            'tdepth': self.tdepth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': True,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate,
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration"""
        return cls(**config)

class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention and MLP
    Matches the PyTorch implementation exactly
    """
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4.0, 
        drop_rate=0.0,
        attn_drop_rate=0.0, 
        drop_path_rate=0.0, 
        **kwargs
    ):
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
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])
    
    def call(self, x, training=False):
        # Normalize inputs (pre-norm architecture)
        attn_input = self.norm1(x)
        
        # Multi-head self-attention
        attn_output = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            training=training
        )
        
        # Apply dropout path during training
        if training and self.drop_path_rate > 0:
            attn_output = tf.nn.dropout(
                attn_output,
                rate=self.drop_path_rate
            )
        
        # First residual connection
        out1 = x + attn_output
        
        # Second pre-norm
        ff_input = self.norm2(out1)
        
        # MLP block
        ff_output = self.mlp(ff_input, training=training)
        
        # Apply dropout path during training
        if training and self.drop_path_rate > 0:
            ff_output = tf.nn.dropout(
                ff_output,
                rate=self.drop_path_rate
            )
        
        # Second residual connection
        output = out1 + ff_output
        
        return output
    
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

# For testing the model directly
if __name__ == "__main__":
    # Create test data
    batch_size = 2
    frames = 128
    joints = 32
    coords = 3
    
    # Create model
    model = MMTransformer(
        mocap_frames=frames,
        acc_frames=frames,
        num_joints=joints,
        in_chans=coords,
        num_patch=4,
        acc_coords=coords,
        spatial_embed=16,
        sdepth=2,
        adepth=2,
        tdepth=2,
        num_heads=2,
        num_classes=1
    )
    
    # Create dummy inputs
    acc_data = tf.random.normal((batch_size, frames, coords))
    skl_data = tf.random.normal((batch_size, frames, joints, coords))
    
    # Forward pass
    inputs = {
        'accelerometer': acc_data,
        'skeleton': skl_data
    }
    
    logits, features = model(inputs, training=True)
    
    print(f"Model successfully created and ran")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
