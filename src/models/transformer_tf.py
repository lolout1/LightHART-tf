import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
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
        x = self.layernorm1(inputs)
        attn_output, attn_weights = self.att(query=x, key=x, value=x, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output, attn_weights

class StudentTransformerTF(Model):
    def __init__(self, acc_frames=128, num_classes=1, num_heads=2, acc_coords=4, num_layers=2, embed_dim=32, ff_dim=64, dropout=0.2, **kwargs):
        super(StudentTransformerTF, self).__init__(**kwargs)
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        self.input_projection = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        
        self.pos_encoding = self._positional_encoding(acc_frames, embed_dim)
        
        self.encoder_blocks = [
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ]
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.pooling = layers.GlobalAveragePooling1D()
        self.final_dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(num_classes)
        
        # Initialize with dummy input
        dummy_input = {'accelerometer': tf.zeros((1, acc_frames, acc_coords))}
        self(dummy_input)

    def _positional_encoding(self, max_len, d_model):
        import numpy as np
        pos = np.arange(max_len)[:, np.newaxis]
        angles = np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * np.exp(angles))
        pe[:, 1::2] = np.cos(pos * np.exp(angles))
        return tf.cast(pe[np.newaxis, ...], dtype=tf.float32)
    
    def build(self, input_shape):
        if isinstance(input_shape, dict):
            self.built = True
        else:
            super().build(input_shape)
    
    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            x = inputs['accelerometer']
        else:
            x = inputs
        
        # Apply input projection
        x = self.input_projection(x, training=training)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        
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
