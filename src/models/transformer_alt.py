import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np
import traceback
import shutil

class TransModel(tf.keras.Model):
    def __init__(
        self,
        acc_frames=64,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=0.5,
        activation='relu',
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
        
        # Define layers
        self.conv_layer = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=(8, 1),
            padding='same',
            name="conv_projection"
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        
        # Create attention layers
        self.attention_layers = []
        self.ffn_layers = []
        self.layer_norms1 = []
        self.layer_norms2 = []
        
        for i in range(self.num_layers):
            # Create attention layer
            attn_layer = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name=f"mha_{i}"
            )
            self.attention_layers.append(attn_layer)
            
            # Create FFN
            ffn = tf.keras.Sequential([
                layers.Dense(self.embed_dim * 2, activation=self.activation, name=f"ffn_dense1_{i}"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.embed_dim, name=f"ffn_dense2_{i}"),
                layers.Dropout(self.dropout_rate)
            ], name=f"ffn_{i}")
            self.ffn_layers.append(ffn)
            
            # Create layer norms
            ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")
            ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")
            self.layer_norms1.append(ln1)
            self.layer_norms2.append(ln2)
        
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        self.output_dense = layers.Dense(self.num_classes, name="output_dense")
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, acc_frames, acc_coords), dtype=tf.float32)
        self(dummy_input, training=False)
        
        logging.info(f"TransModel initialized: frames={acc_frames}, coords={acc_coords}, embed_dim={embed_dim}")

    def call(self, inputs, training=False):
        # Handle either dict or tensor input
        x = inputs
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
            
        # Apply convolutional projection
        x = tf.expand_dims(x, axis=2)  # Add channel dimension
        x = self.conv_layer(x)
        x = tf.squeeze(x, axis=2)      # Remove channel dimension
        
        # Apply initial normalization
        x = self.layer_norm(x)
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            # Self-attention
            attn_output = self.attention_layers[i](
                query=x,
                key=x,
                value=x,
                training=training
            )
            x = self.layer_norms1[i](x + attn_output)
            
            # Feed-forward network
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.layer_norms2[i](x + ffn_output)
        
        # Final processing
        x = self.final_norm(x)
        x = self.global_pool(x)
        logits = self.output_dense(x)
        
        # Reshape to ensure consistent output shape
        if self.num_classes == 1:
            return tf.reshape(logits, [-1, 1])
        else:
            return logits

    # Simple method returning trainable parameters for compatibility
    def get_weights_for_export(self):
        return self.trainable_variables
