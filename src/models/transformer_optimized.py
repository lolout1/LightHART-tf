import tensorflow as tf
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger('transformer-tf')

class TransModel(tf.keras.Model):
    def __init__(self, acc_frames=64, num_classes=1, num_heads=4, acc_coords=3, 
                 embed_dim=32, num_layers=2, dropout=0.5, activation='relu', 
                 norm_first=True, **kwargs):
        super().__init__()
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # Input projection using standard Conv1D
        self.conv_proj = layers.Conv1D(embed_dim, kernel_size=8, padding='same', use_bias=False, name='conv_proj')
        self.bn_proj = layers.BatchNormalization(name='bn_proj')
        
        # Encoder layers
        self.encoder_blocks = []
        for i in range(num_layers):
            self.encoder_blocks.append(TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                name=f'transformer_block_{i}'
            ))
        
        # Output layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name='temporal_norm')
        self.output_dense = layers.Dense(num_classes, name='output_dense')
        
        logger.info(f"TransModel initialized with: embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
    
    def call(self, inputs, training=False, **kwargs):
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        else:
            x = inputs
        
        # Input projection
        x = self.conv_proj(x)
        x = self.bn_proj(x, training=training)
        
        # Transformer encoding
        for encoder in self.encoder_blocks:
            x = encoder(x, training=training)
        
        # Output processing
        features = x
        x = self.temporal_norm(x)
        x = tf.reduce_mean(x, axis=1)  # Global average pooling
        logits = self.output_dense(x)
        
        return logits, features
    
    def get_config(self):
        return {
            'acc_frames': self.acc_frames,
            'num_classes': self.num_classes,
            'num_heads': self.num_heads,
            'acc_coords': self.acc_coords,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation,
            'norm_first': self.norm_first
        }

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout, activation, norm_first, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 2, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=False):
        if self.norm_first:
            # Pre-norm architecture
            attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), training=training)
            x = x + self.dropout1(attn_out, training=training)
            
            ffn_out = self.ffn(self.norm2(x), training=training)
            x = x + ffn_out
        else:
            # Post-norm architecture
            attn_out = self.attn(x, x, x, training=training)
            x = self.norm1(x + self.dropout1(attn_out, training=training))
            
            ffn_out = self.ffn(x, training=training)
            x = self.norm2(x + ffn_out)
        
        return x
