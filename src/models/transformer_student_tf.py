# src/models/transformer_student_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block with multi-head attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Normalization before attention (pre-norm transformer)
        x = self.layernorm1(inputs)
        
        # Multi-head attention
        attn_output, attn_weights = self.att(
            query=x, key=x, value=x, 
            return_attention_scores=True, 
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        # Feed-forward network
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output, attn_weights

class StudentTransformerTF(Model):
    """
    Student Transformer model for human activity recognition.
    A lighter version designed to be trained with knowledge distillation.
    """
    
    def __init__(
            self,
            acc_frames=128,
            num_classes=2,
            num_heads=2,
            acc_coords=3,
            num_layers=2,
            embed_dim=32,
            ff_dim=64,
            dropout=0.2,
            **kwargs
        ):
        super(StudentTransformerTF, self).__init__(**kwargs)
        
        # Input shape definition
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # Input projection: Map the raw accelerometer data to embedding space
        self.input_projection = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        
        # Positional encoding
        self.pos_encoding = self.positional_encoding(acc_frames, embed_dim)
        
        # Transformer encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        
        # Output layers
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.pooling = layers.GlobalAveragePooling1D()
        self.final_dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(num_classes)

    def positional_encoding(self, max_len, d_model):
        """Create standard transformer positional encoding."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pe[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs, training=False):
        """Forward pass of the student model."""
        # Handle different input formats
        if isinstance(inputs, dict):
            if 'accelerometer' in inputs:
                x = inputs['accelerometer']
            elif 'acc' in inputs:
                x = inputs['acc']
            else:
                raise ValueError("Accelerometer data is required in the input")
        else:
            x = inputs
        
        # Ensure input has the right shape
        if len(x.shape) < 3:
            x = tf.expand_dims(x, axis=-1)
        
        # Apply input projection
        x = self.input_projection(x, training=training)
        
        # Add positional encoding
        x += self.pos_encoding[:, :x.shape[1], :]
        
        # Store attention weights for visualization/distillation
        attention_weights = []
        
        # Apply transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            x, weights = encoder_block(x, training=training)
            attention_weights.append(weights)
        
        # Apply layer normalization
        x = self.layernorm(x)
        
        # Extract features via global pooling
        features = self.pooling(x)
        features = self.final_dropout(features, training=training)
        
        # Final classification
        logits = self.classifier(features)
        
        return logits, features
