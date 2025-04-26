import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim//num_heads,
            value_dim=embed_dim//num_heads
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Pre-norm transformer architecture
        x = self.layernorm1(inputs)
        
        # Multi-head attention with residual connection
        attn_output = self.att(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        # Feed-forward network with residual connection
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output, None  # Return None for attention weights to save memory

class StudentTransformerTF(Model):
    """
    Efficient Transformer model for accelerometer-based human activity recognition.
    Optimized for compatibility with TensorFlow's GPU operations.
    """
    def __init__(
            self,
            acc_frames=128,
            num_classes=1,
            num_heads=2,
            acc_coords=4,
            num_layers=2,
            embed_dim=32,
            ff_dim=64,
            dropout=0.2,
            **kwargs
        ):
        super(StudentTransformerTF, self).__init__(**kwargs)
        
        # Model parameters
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # Explicitly set data_format to channels_last for better compatibility
        self.input_projection = tf.keras.Sequential([
            layers.Conv1D(
                embed_dim, 
                kernel_size=7, 
                strides=1, 
                padding='same',
                data_format='channels_last',
                kernel_initializer='glorot_uniform'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        
        # Create positional encoding
        self.pos_encoding = self._positional_encoding(acc_frames, embed_dim)
        
        # Create transformer encoder blocks
        self.encoder_blocks = []
        for _ in range(num_layers):
            self.encoder_blocks.append(
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
            )
        
        # Output layers
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.pooling = layers.GlobalAveragePooling1D()
        self.final_dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        
        # Initialize with dummy input to build the model
        self._build_model()
    
    def _positional_encoding(self, max_len, d_model):
        """Create standard transformer positional encoding."""
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(positions * div_term)
        pos_encoding[:, 1::2] = np.cos(positions * div_term)
        
        # Add batch dimension and convert to TF tensor
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def _build_model(self):
        """Initialize model weights by running a forward pass with dummy data."""
        dummy_input = tf.zeros((1, self.acc_frames, self.acc_coords))
        self({"accelerometer": dummy_input}, training=False)
    
    @tf.function
    def call(self, inputs, training=False):
        """Forward pass with memory optimization and XLA compatibility."""
        # Extract input data
        if isinstance(inputs, dict):
            x = inputs['accelerometer']
        else:
            x = inputs
        
        # Apply input projection
        x = self.input_projection(x, training=training)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            x, _ = encoder_block(x, training=training)
        
        # Apply normalization and global pooling
        x = self.layernorm(x)
        features = self.pooling(x)
        
        # Apply dropout during training
        if training:
            features = self.final_dropout(features, training=True)
        
        # Final classification
        logits = self.classifier(features)
        
        return logits, features
