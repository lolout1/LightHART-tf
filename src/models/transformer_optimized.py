#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger('transformer-tf')

class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, activation='relu', norm_first=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, inputs, training=False):
        if self.norm_first:
            attn_output = self.attention_block(self.norm1(inputs), training=training)
            x = inputs + attn_output
            outputs = x + self.mlp_block(self.norm2(x), training=training)
        else:
            attn_output = self.attention_block(inputs, training=training)
            x = self.norm1(inputs + attn_output)
            outputs = self.norm2(x + self.mlp_block(x, training=training))
        
        return outputs
    
    def attention_block(self, inputs, training=False):
        attention_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        return self.dropout1(attention_output, training=training)
    
    def mlp_block(self, inputs, training=False):
        return self.mlp(inputs, training=training)

class TransModel(tf.keras.Model):
    def __init__(
        self,
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=0.5,
        activation='relu',
        norm_first=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(
                filters=embed_dim,
                kernel_size=8,
                padding='same',
                use_bias=True,
                name="input_proj_conv"
            ),
            layers.BatchNormalization(name="input_proj_bn")
        ], name="input_projection")
        
        self.transformer_blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=embed_dim*2,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
        
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        self.output_layer = layers.Dense(num_classes, name="output_layer")
        
        logger.info(f"TransModel initialized: frames={acc_frames}, embed_dim={embed_dim}, layers={num_layers}, heads={num_heads}")
    
    def call(self, inputs, training=False, **kwargs):
        if isinstance(inputs, dict):
            acc_data = inputs.get('accelerometer')
            if acc_data is None:
                raise ValueError("Accelerometer data is required")
        else:
            acc_data = inputs
        
        # Process for both original data and SMV-augmented
        if isinstance(acc_data, tf.Tensor) and len(acc_data.shape) == 3:
            x = acc_data
        else:
            raise ValueError(f"Expected 3D input [batch, frames, features], got shape {tf.shape(acc_data)}")
        
        # Project input - convert to channels-first for Conv1D
        x = tf.transpose(x, [0, 2, 1])  # [batch, channels, frames]
        x = self.input_proj(x, training=training)
        x = tf.transpose(x, [0, 2, 1])  # [batch, frames, channels]
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Final normalization - save features for distillation
        features = self.temporal_norm(x)
        
        # Global pooling and output
        pooled = tf.reduce_mean(features, axis=1)
        logits = self.output_layer(pooled)
        
        # Ensure correct output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        # Return both logits and features for distillation
        return logits, features
