#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import logging

logger = logging.getLogger('transformer-tf')

class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention and MLP feedforward.
    Matches PyTorch implementation for compatibility.
    """
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
        
        # MLP block
        self.mlp = Sequential([
            layers.Dense(mlp_dim, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, inputs, training=False):
        # Pre-norm architecture (as per PyTorch implementation)
        if self.norm_first:
            # First attention block with pre-norm
            norm_inputs = self.norm1(inputs)
            attn_output = self.mha(
                query=norm_inputs,
                key=norm_inputs,
                value=norm_inputs,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + attn_output  # Residual connection
            
            # Second MLP block with pre-norm
            norm_out1 = self.norm2(out1)
            mlp_output = self.mlp(norm_out1, training=training)
            output = out1 + mlp_output  # Residual connection
        else:
            # Post-norm architecture
            attn_output = self.mha(
                query=inputs,
                key=inputs,
                value=inputs,
                training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.norm1(inputs + attn_output)
            
            mlp_output = self.mlp(out1, training=training)
            output = self.norm2(out1 + mlp_output)
        
        return output


class TransModel(Model):
    """
    Transformer model for accelerometer data processing.
    Matches PyTorch LightHART implementation for fall detection.
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
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # Input projection layer - matches PyTorch implementation
        self.input_proj = Sequential([
            layers.Conv1D(filters=embed_dim, kernel_size=8, strides=1, padding='same'),
            layers.BatchNormalization()
        ], name="input_projection")
        
        # Input dropout for regularization
        self.input_dropout = layers.Dropout(dropout)
        
        # Create transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=embed_dim * 2,  # Standard transformer ratio
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
        
        # Output normalization
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        
        # Classification head
        self.output_layer = layers.Dense(num_classes, name="output_layer")
        
        logger.info(f"TransModel initialized with: embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
    
    def call(self, inputs, training=False, **kwargs):
        """
        Forward pass handling different input types.
        
        Args:
            inputs: Either dict with 'accelerometer' key or tensor directly
            training: Whether in training mode
        
        Returns:
            tuple: (logits, features) for distillation compatibility
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            x = inputs.get('accelerometer')
            if x is None:
                raise ValueError("Accelerometer data is required")
        else:
            x = inputs
        
        # Get batch size for consistent reshaping
        batch_size = tf.shape(x)[0]
        
        # Check input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch, time, features), got shape {x.shape}")
        
        # Input projection: [batch, time, features] -> [batch, time, embed_dim]
        # Conv1D expects [batch, time, channels] format
        x = self.input_proj(x, training=training)
        
        # Apply input dropout
        x = self.input_dropout(x, training=training)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Normalize features
        features = self.temporal_norm(x)
        
        # Global average pooling for classification
        # [batch, time, embed_dim] -> [batch, embed_dim]
        pooled_features = tf.reduce_mean(features, axis=1)
        
        # Classification output
        logits = self.output_layer(pooled_features)
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [batch_size, 1])
        
        # Return both logits and features for distillation
        return logits, features
    
    def build(self, input_shape):
        """Explicitly build the model"""
        # Handle dict input
        if isinstance(input_shape, dict):
            input_shape = input_shape.get('accelerometer', (None, 128, 3))
        
        # Build with proper shape
        super().build(input_shape)
        
        # Force build all layers
        dummy_input = tf.zeros((1,) + input_shape[1:])
        _ = self.call(dummy_input, training=False)
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'acc_frames': self.acc_frames,
            'num_classes': self.num_classes,
            'num_heads': self.num_heads,
            'acc_coords': self.acc_coords,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation,
            'norm_first': self.norm_first
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration"""
        return cls(**config)


# For testing the model directly
if __name__ == "__main__":
    # Create test data
    batch_size = 4
    time_steps = 128
    features = 3
    
    # Create model
    model = TransModel(
        acc_frames=time_steps,
        num_classes=1,
        num_heads=4,
        acc_coords=features,
        embed_dim=32,
        num_layers=2,
        dropout=0.5,
        activation='relu',
        norm_first=True
    )
    
    # Create dummy inputs
    dummy_acc = tf.random.normal((batch_size, time_steps, features))
    
    # Test with tensor input
    logits, features = model(dummy_acc, training=True)
    print(f"Tensor input - Logits shape: {logits.shape}, Features shape: {features.shape}")
    
    # Test with dict input
    inputs = {'accelerometer': dummy_acc}
    logits, features = model(inputs, training=True)
    print(f"Dict input - Logits shape: {logits.shape}, Features shape: {features.shape}")
    
    # Test model summary
    model.build((None, time_steps, features))
    model.summary()
