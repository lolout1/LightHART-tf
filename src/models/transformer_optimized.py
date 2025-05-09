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
        # Handle different input types
        x = inputs
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        
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
    
    def export_to_tflite(self, save_path, input_length=128, quantize=False):
        """Export model to TFLite format"""
        try:
            # Create a concrete function that processes raw accelerometer data
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, input_length, self.acc_coords], dtype=tf.float32)
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


class TransformerBlock(tf.keras.layers.Layer):
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
        
        # MLP block - exactly like PyTorch implementation
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, x, training=False):
        if self.norm_first:
            # Pre-norm architecture (like PyTorch implementation)
            # First multi-head attention block
            attn_input = self.norm1(x)
            attn_output = self.mha(
                query=attn_input, 
                key=attn_input, 
                value=attn_input,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = x + attn_output  # First residual connection
            
            # Second feed-forward block
            ff_input = self.norm2(out1)
            ff_output = self.mlp(ff_input, training=training)
            return out1 + ff_output  # Second residual connection
        else:
            # Post-norm architecture (original Transformer)
            # First multi-head attention block
            attn_output = self.mha(
                query=x, 
                key=x, 
                value=x,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.norm1(x + attn_output)  # First residual + norm
            
            # Second feed-forward block
            ff_output = self.mlp(out1, training=training)
            return self.norm2(out1 + ff_output)  # Second residual + norm
