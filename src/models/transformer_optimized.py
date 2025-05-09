#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transformer_optimized.py - TensorFlow implementation of the TransModel student
Matches PyTorch implementation exactly with proper feature extraction
"""

import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger('transformer-tf')

class TransModel(tf.keras.Model):
    """
    TensorFlow implementation of TransModel student
    Exactly matches the PyTorch implementation structure and function
    """
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
        # Configuration
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # Input projection - matches PyTorch implementation exactly
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
        
        # Transformer layers
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
        
        # Output layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        self.output_layer = layers.Dense(num_classes, name="output_layer")
        
        logger.info(f"TransModel initialized: frames={acc_frames}, embed_dim={embed_dim}, "
                    f"layers={num_layers}, heads={num_heads}")
    
    def call(self, inputs, training=False, **kwargs):
        """Forward pass handling different input types"""
        # Handle different input formats
        if isinstance(inputs, dict):
            # Dictionary input with named modalities
            x = inputs.get('accelerometer')
            if x is None:
                raise ValueError("Accelerometer data is required")
        else:
            # Direct tensor input
            x = inputs
        
        # Ensure 3D input [batch, frames, features]
        if len(tf.shape(x)) != 3:
            raise ValueError(f"Expected 3D input [batch, frames, features], got shape {tf.shape(x)}")
        
        # Reorganize if SMV is present (4 channels instead of 3)
        input_channels = tf.shape(x)[-1]
        
        # Project input - [batch, frames, channels] -> [batch, frames, embed_dim]
        x = self.input_proj(x, training=training)
        
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
    
    def export_to_tflite(self, save_path, input_shape=None, quantize=False):
        """Export model to TFLite format"""
        if input_shape is None:
            input_shape = [1, self.acc_frames, self.acc_coords]
        
        try:
            # Create a concrete function that processes raw accelerometer data
            @tf.function(input_signature=[
                tf.TensorSpec(shape=input_shape, dtype=tf.float32)
            ])
            def serving_fn(accelerometer):
                # Create input dictionary
                inputs = {'accelerometer': accelerometer}
                # Get model output (only logits)
                logits, _ = self(inputs, training=False)
                # For binary classification, ensure output is of shape [1, 1]
                if self.num_classes == 1:
                    logits = tf.reshape(logits, [1, 1])
                return {'output': logits}
            
            # Convert to SavedModel
            temp_dir = save_path + "_temp"
            tf.saved_model.save(self, temp_dir, signatures=serving_fn)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            
            # Apply optimization if requested
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save to file
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info(f"Model exported to TFLite: {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export to TFLite: {e}")
            return False


class TransformerBlock(layers.Layer):
    """Transformer encoder block implementation"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, 
                 activation='relu', norm_first=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout layers
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        # MLP block - exactly like PyTorch imp
