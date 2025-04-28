import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, activation='relu', norm_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dim_feedforward, activation=activation),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        if self.norm_first:
            # Self-attention with pre-normalization
            attn_input = self.layernorm1(inputs)
            attn_output = self.att(attn_input, attn_input)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + attn_output

            # Feed-forward network with pre-normalization
            ffn_input = self.layernorm2(out1)
            ffn_output = self.ffn(ffn_input)
            ffn_output = self.dropout2(ffn_output, training=training)
            return out1 + ffn_output
        else:
            # Original post-normalization order
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

class TransModel(Model):
    def __init__(self, num_classes=1, num_heads=4, embed_dim=32, num_layers=2, 
                 dim_feedforward=64, dropout=0.5, activation='relu', norm_first=True, **kwargs):
        super(TransModel, self).__init__()
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(filters=embed_dim, kernel_size=8, strides=1, padding='same', input_shape=(128, 4)),
            layers.BatchNormalization()
        ])
        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            ) for _ in range(num_layers)
        ]
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        else:
            x = inputs
            
        x = self.input_proj(x)
        
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        x = self.temporal_norm(x)
        features = x
        
        x = self.global_avg_pool(x)
        output = self.output_layer(x)
        
        return output, features

# For backward compatibility with previous API
StudentTransformerTF = TransModel
