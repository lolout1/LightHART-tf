import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim//num_heads
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        # Save parameters for serialization
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        })
        return config
    
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
        return out1 + ffn_output

class StudentTransformerTF(Model):
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
        
        # Save parameters for serialization
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Input projection
        self.input_projection = tf.keras.Sequential([
            layers.Conv1D(
                embed_dim, 
                kernel_size=7, 
                strides=1, 
                padding='same',
                data_format='channels_last'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        
        # Initialize encoder blocks
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
        self.classifier = layers.Dense(num_classes)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "acc_frames": self.acc_frames,
            "num_classes": self.num_classes,
            "num_heads": self.num_heads,
            "acc_coords": self.acc_coords,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        })
        return config
    
    def build(self, input_shape):
        # Create positional encoding
        positions = np.arange(self.acc_frames)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pos_encoding = np.zeros((self.acc_frames, self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(positions * div_term)
        pos_encoding[:, 1::2] = np.cos(positions * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """Forward pass"""
        # Extract input data
        if isinstance(inputs, dict):
            x = inputs['accelerometer']
        else:
            x = inputs
        
        # Apply input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer encoder blocks
        features = x
        for encoder_block in self.encoder_blocks:
            features = encoder_block(features, training=training)
        
        # Apply normalization and global pooling
        features = self.layernorm(features)
        features_pooled = self.pooling(features)
        
        # Apply dropout during training
        if training:
            features_pooled = self.final_dropout(features_pooled)
        
        # Final classification
        logits = self.classifier(features_pooled)
        
        return logits, features
