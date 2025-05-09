# models/transformer_optimized.py
import tensorflow as tf
from tensorflow.keras import layers
import logging

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
        
        # Input projection - consistent with PyTorch implementation
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(
                filters=embed_dim,
                kernel_size=8,
                padding='same',
                use_bias=True,
                name="conv_projection"
            ),
            layers.BatchNormalization(name="batch_norm")
        ], name="input_projection")
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=embed_dim * 2,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_idx=i
            )
            self.transformer_blocks.append(block)
        
        # Output layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        self.output_dense = layers.Dense(num_classes, name="output_dense")
        
        logging.info(f"TransModel initialized: frames={acc_frames}, embed_dim={embed_dim}")
    
    def call(self, inputs, training=False, **kwargs):
        # Handle different input types
        x = inputs
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        
        # Project input features [batch, frames, channels] -> [batch, frames, embed_dim]
        x = self.input_proj(x, training=training)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Final normalization
        features = self.temporal_norm(x)
        
        # Global pooling and classification
        pooled = tf.reduce_mean(features, axis=1)
        logits = self.output_dense(pooled)
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        # Return logits and features for distillation
        return logits, features


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block implementation for TensorFlow"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, activation='relu', norm_first=True, layer_idx=0):
        super().__init__(name=f"transformer_block_{layer_idx}")
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
            dropout=dropout,
            name=f"mha_{layer_idx}"
        )
        
        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{layer_idx}")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{layer_idx}")
        
        # Dropout layers
        self.dropout1 = layers.Dropout(dropout, name=f"dropout1_{layer_idx}")
        self.dropout2 = layers.Dropout(dropout, name=f"dropout2_{layer_idx}")
        
        # MLP block
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation, name=f"mlp_dense1_{layer_idx}"),
            layers.Dropout(dropout),
            layers.Dense(dim, name=f"mlp_dense2_{layer_idx}"),
            layers.Dropout(dropout)
        ], name=f"mlp_{layer_idx}")
    
    def call(self, inputs, training=False):
        # Shapes are maintained as [batch, seq_len, features] throughout
        
        if self.norm_first:
            # Pre-norm implementation (like PyTorch)
            # First multi-head attention block
            x = self.norm1(inputs)
            attn_output = self.mha(
                query=x, 
                key=x, 
                value=x,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + attn_output  # First residual connection
            
            # Second feed-forward block
            x = self.norm2(out1)
            mlp_output = self.mlp(x, training=training)
            return out1 + mlp_output  # Second residual connection
        else:
            # Post-norm implementation
            # First multi-head attention block
            attn_output = self.mha(
                query=inputs, 
                key=inputs, 
                value=inputs,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.norm1(inputs + attn_output)  # First residual + norm
            
            # Second feed-forward block
            mlp_output = self.mlp(out1, training=training)
            return self.norm2(out1 + mlp_output)  # Second residual + norm
