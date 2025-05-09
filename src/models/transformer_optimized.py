#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
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
        
        # MLP layers
        self.mlp = Sequential([
            layers.Dense(mlp_dim, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, inputs, training=False):
        # Pre-norm architecture
        if self.norm_first:
            # First attention block
            norm_inputs = self.norm1(inputs)
            attn_output = self.mha(
                query=norm_inputs, 
                key=norm_inputs, 
                value=norm_inputs,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            inputs_plus_attn = inputs + attn_output  # First residual connection
            
            # Second MLP block
            norm_inputs_plus_attn = self.norm2(inputs_plus_attn)
            mlp_output = self.mlp(norm_inputs_plus_attn, training=training)
            output = inputs_plus_attn + mlp_output  # Second residual connection
        else:
            # Post-norm architecture
            attn_output = self.mha(
                query=inputs, 
                key=inputs, 
                value=inputs,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            inputs_plus_attn = self.norm1(inputs + attn_output)
            
            mlp_output = self.mlp(inputs_plus_attn, training=training)
            output = self.norm2(inputs_plus_attn + mlp_output)
        
        return output


class TransModel(Model):
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
        
        # Create all layers directly
        self.feature_proj = layers.Dense(embed_dim, name="feature_projection")
        self.input_dropout = layers.Dropout(dropout)
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Create transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
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
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        self.output_layer = layers.Dense(num_classes, name="output_layer")
        
        logger.info(f"TransModel initialized with: embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
    
    def call(self, inputs, training=False, **kwargs):
        # Handle different input formats
        if isinstance(inputs, dict):
            x = inputs.get('accelerometer')
            if x is None:
                raise ValueError("Accelerometer data is required")
        else:
            x = inputs
        
        # Project input features to embedding dimension
        x = self.feature_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x, training=training)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Extract features for distillation
        features = self.temporal_norm(x)
        
        # Global pooling and classification
        x = self.global_pool(features)
        logits = self.output_layer(x)
        
        # Make sure output shape is correct for binary classification
        if self.num_classes == 1:
            batch_size = tf.shape(inputs['accelerometer'])[0] if isinstance(inputs, dict) else tf.shape(inputs)[0]
            logits = tf.reshape(logits, [batch_size, 1])
        
        return logits, features
