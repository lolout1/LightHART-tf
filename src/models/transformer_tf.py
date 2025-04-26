import tensorflow as tf
from tensorflow.keras import layers

class TransModelTF(tf.keras.Model):
    def __init__(self,
                mocap_frames=128,
                num_joints=32,
                acc_frames=128,
                num_classes=1, 
                num_heads=2, 
                acc_coords=3, 
                av=False,
                num_layers=2, 
                norm_first=True, 
                embed_dim=32, 
                activation='relu',
                **kwargs):
        super(TransModelTF, self).__init__()

        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(embed_dim, kernel_size=8, strides=1, padding='same'),
            layers.BatchNormalization()
        ])

        # Create encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            # Create multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=0.5
            )
            
            # Create feedforward network
            ffn = tf.keras.Sequential([
                layers.Dense(embed_dim * 2, activation=activation),
                layers.Dropout(0.5),
                layers.Dense(embed_dim)
            ])
            
            # Add to list
            self.encoder_layers.append({
                'attention': attention,
                'attention_norm': layers.LayerNormalization(epsilon=1e-6),
                'ffn': ffn,
                'ffn_norm': layers.LayerNormalization(epsilon=1e-6)
            })
        
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # Extract inputs
        acc_data = inputs['accelerometer']
        skl_data = inputs.get('skeleton', None)
        
        # Process accelerometer data
        x = tf.transpose(acc_data, perm=[0, 2, 1])  # [batch, channels, length]
        x = self.input_proj(x, training=training)
        x = tf.transpose(x, perm=[0, 2, 1])  # [batch, length, channels]
        
        # Transformer encoder
        attention_weights = []
        for layer in self.encoder_layers:
            # Apply layer normalization first (if norm_first=True)
            attn_input = layer['attention_norm'](x)
            
            # Apply attention
            attn_output, attention_scores = layer['attention'](
                query=attn_input, 
                key=attn_input, 
                value=attn_input,
                training=training,
                return_attention_scores=True
            )
            
            # Add attention output to input (residual)
            x = x + attn_output
            attention_weights.append(attention_scores)
            
            # Apply FFN with residual connection
            ffn_input = layer['ffn_norm'](x)
            ffn_output = layer['ffn'](ffn_input, training=training)
            x = x + ffn_output
        
        # Final layer normalization
        x = self.temporal_norm(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits, x  # Return logits and features
