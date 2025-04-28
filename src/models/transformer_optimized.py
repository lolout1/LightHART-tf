# models/transformer_optimized.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TransModel(Model):
    """
    Transformer model for accelerometer data, optimized for TensorFlow.
    Equivalent to the PyTorch version in Models/transformer.py
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
        **kwargs  # Accept extra kwargs but don't use them
    ):
        super().__init__()
        
        # Store parameters
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Initial projection - convert accelerometer data to embedding
        # Input shape will be (batch, time, features) with features = 3 or 4 with SMV
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=8, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=embed_dim,
                num_heads=num_heads,
                dff=embed_dim*2,
                dropout_rate=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ]
        
        # Normalization and output layers
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = layers.Dense(num_classes)
        
    def call(self, inputs, training=False):
        # Handle different input formats
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        elif isinstance(inputs, tuple) and len(inputs) >= 1:
            x = inputs[0]  # First element is accelerometer data
        else:
            raise ValueError(f"Unsupported input format: {type(inputs)}. "
                            f"Expected dict with 'accelerometer' key or tuple with acc_data as first element.")
        
        # Initial projection - reshape if needed
        if len(x.shape) == 3:  # [batch, frames, features]
            # Convert from [batch, frames, features] to [batch, features, frames]
            x = tf.transpose(x, [0, 2, 1])
            x = self.input_proj(x)
            # Convert back to [frames, batch, features] for transformer
            x = tf.transpose(x, [2, 0, 1])
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [batch, frames, features]")
        
        # Apply transformer layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        # Convert back to [batch, frames, features]
        x = tf.transpose(x, [1, 0, 2])
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Extract features before output
        features = x
        
        # Global average pooling across time dimension
        x = tf.reduce_mean(x, axis=1)
        
        # Final output
        logits = self.output_layer(x)
        
        return logits, features


class TransformerEncoderLayer(layers.Layer):
    """
    Single transformer encoder layer
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, activation='relu'):
        super().__init__()
        
        self.mha = layers.MultiHeadAttention(
            key_dim=d_model // num_heads,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation=activation),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        # x shape: [seq_len, batch, d_model]
        
        # Self-attention
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


# Test code for debugging
if __name__ == "__main__":
    # Create sample data
    batch_size = 4
    acc_frames = 128
    acc_coords = 3
    
    # Test with dictionary input
    inputs_dict = {
        'accelerometer': tf.random.normal([batch_size, acc_frames, acc_coords+1])  # +1 for SMV
    }
    
    # Test with tuple input
    inputs_tuple = (tf.random.normal([batch_size, acc_frames, acc_coords+1]),)
    
    # Create model
    model = TransModel(
        acc_frames=acc_frames,
        num_classes=1,
        acc_coords=acc_coords
    )
    
    # Test with dictionary input
    logits_dict, features_dict = model(inputs_dict)
    print(f"Dict input - logits shape: {logits_dict.shape}, features shape: {features_dict.shape}")
    
    # Test with tuple input
    logits_tuple, features_tuple = model(inputs_tuple)
    print(f"Tuple input - logits shape: {logits_tuple.shape}, features shape: {features_tuple.shape}")
