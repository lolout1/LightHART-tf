import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block implementation matching PyTorch"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.5):
        super(TransformerEncoderBlock, self).__init__()
        
        # Save parameters for serialization
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        # Initialize attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim//num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    
    def get_config(self):
        """Configuration for serialization"""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config
    
    def call(self, inputs, training=False):
        """Forward pass with residual connections"""
        # Layer normalization and attention
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, x, training=training)
        
        # First residual connection
        out1 = inputs + attn_output
        
        # Layer normalization and feed-forward
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x, training=training)
        
        # Second residual connection
        return out1 + ffn_output


class StudentTransformerTF(Model):
    """Student transformer model matched to PyTorch implementation"""
    def __init__(
            self,
            acc_frames=128,
            num_classes=1,
            num_heads=2,
            acc_coords=4,
            num_layers=2,
            embed_dim=32,
            ff_dim=64,
            dropout=0.5,
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
        self.dropout_rate = dropout
        
        # Input projection layers (separately defined for clarity)
        self.conv_layer = layers.Conv1D(
            filters=embed_dim, 
            kernel_size=8, 
            strides=1, 
            padding='same',
            data_format='channels_last',
            name="input_conv"
        )
        
        self.batch_norm = layers.BatchNormalization(name="input_bn")
        
        # Initialize encoder blocks
        self.encoder_blocks = []
        for i in range(num_layers):
            self.encoder_blocks.append(
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=embed_dim*2,  # Match original (embed_dim*2)
                    dropout=dropout
                )
            )
        
        # Output layers
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")
        self.global_pooling = layers.GlobalAveragePooling1D(name="global_pool")
        self.classifier = layers.Dense(num_classes, name="classifier")
    
    def get_config(self):
        """Configuration for serialization"""
        config = super().get_config()
        config.update({
            "acc_frames": self.acc_frames,
            "num_classes": self.num_classes,
            "num_heads": self.num_heads,
            "acc_coords": self.acc_coords,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config
    
    def call(self, inputs, training=False):
        """Forward pass handling both dict and tensor inputs"""
        # Handle different input types
        if isinstance(inputs, dict) and "accelerometer" in inputs:
            x = inputs["accelerometer"]
        else:
            x = inputs
        
        # Apply input projection
        x = self.conv_layer(x)
        x = self.batch_norm(x, training=training)
        
        # Pass through encoder blocks
        features = x
        for encoder_block in self.encoder_blocks:
            features = encoder_block(features, training=training)
        
        # Apply normalization
        features = self.layernorm(features)
        
        # Global pooling
        x = self.global_pooling(features)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits, features
