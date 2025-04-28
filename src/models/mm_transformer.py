# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MMTransformer(Model):
    def __init__(
        self,
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        num_patch=2,
        acc_coords=4,  # Match the config
        spatial_embed=96,  # Match the config
        sdepth=2,
        adepth=2,
        tdepth=2,
        num_heads=2,
        mlp_ratio=2,
        num_classes=1,
        **kwargs
    ):
        super().__init__()
        
        # Store parameters
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.num_patch = num_patch
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.num_classes = num_classes
        
        # Learnable tokens and embeddings
        self.temp_token = self.add_weight(
            shape=(1, 1, spatial_embed),
            initializer="random_normal",
            trainable=True,
            name="temp_token"
        )
        
        self.temporal_pos_embed = self.add_weight(
            shape=(1, spatial_embed),  # Changed dimension to fix concat issue
            initializer="random_normal",
            trainable=True,
            name="temporal_pos_embed"
        )
        
        # Spatial encoder
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv1D(spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv1D(spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ], name='spatial_encoder')
        
        # Spatial convolution for skeleton data
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(in_chans, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(1, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ], name='spatial_conv')
        
        # Joint transformer block
        self.joint_blocks = [
            TransformerBlock(
                embed_dim=spatial_embed,
                num_heads=num_heads,
                ff_dim=spatial_embed * mlp_ratio,
                dropout=0.1
            ) for _ in range(1)
        ]
        
        # Temporal transformer blocks
        self.temporal_blocks = [
            TransformerBlock(
                embed_dim=spatial_embed,
                num_heads=num_heads,
                ff_dim=spatial_embed * mlp_ratio,
                dropout=0.1
            ) for _ in range(tdepth)
        ]
        
        # Normalization layers
        self.spatial_norm = layers.LayerNormalization(epsilon=1e-6)
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Transform layer
        self.transform = tf.keras.Sequential([
            layers.Dense(spatial_embed),  # Match with spatial_embed
            layers.Activation('relu')
        ], name='transform')
        
        # Output classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ], name='class_head')
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        # Handle different input formats, with special handling for inference mode
        if isinstance(inputs, dict):
            # For training: require both modalities
            if 'accelerometer' in inputs and 'skeleton' in inputs:
                # Ignore accelerometer as requested, only use skeleton
                skl_data = inputs['skeleton']
            # For inference: allow only accelerometer input
            elif 'accelerometer' in inputs and training is False:
                # This is inference mode with only accelerometer
                # Create a dummy tensor with correct skeleton dimensions
                acc_data = inputs['accelerometer']
                batch_size = tf.shape(acc_data)[0]
                seq_len = tf.shape(acc_data)[1]
                # Create a dummy skeleton tensor filled with zeros
                skl_data = tf.zeros([batch_size, seq_len, self.num_joints, 3], dtype=tf.float32)
            else:
                raise ValueError("Expected 'skeleton' key in inputs dict for training, or 'accelerometer' for inference")
        elif isinstance(inputs, tuple) and len(inputs) >= 2:
            # If tuple format, second element is skeleton
            _, skl_data = inputs[0], inputs[1]
        else:
            raise ValueError(f"Unsupported input format: {type(inputs)}")
        
        # Get batch size
        batch_size = tf.shape(skl_data)[0]
        
        # Process skeleton data
        # [batch, frames, joints, coords] -> [batch, coords, frames, joints]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x)
        
        # Get shape after convolution for dynamic reshape
        shape_after_conv = tf.shape(x)
        frames = shape_after_conv[2]
        joints = shape_after_conv[3]
        
        # Reshape using dynamic dimensions
        # [batch, channels, frames, joints] -> [batch, frames, joints]
        x = tf.reshape(x, [batch_size, frames*3, joints])
        
        # Apply spatial encoder
        x = self.spatial_encoder(x)
        
        # Apply transform
        x = self.transform(x)
        
        # Extract features from joint block
        features = None
        
        # Apply joint transformer blocks
        for i, block in enumerate(self.joint_blocks):
            x = block(x, training=training)
            
            if i == 0:
                # Extract features for knowledge distillation
                features = tf.identity(x)
                features = self.spatial_norm(features)
                # Global pooling for feature vector
                features = tf.reduce_mean(features, axis=1)
        
        # Add class token for temporal attention
        class_tokens = tf.tile(self.temp_token, [batch_size, 1, 1])
        
        # Make sure x and class_tokens have compatible dimensions for concatenation
        # x shape should be [batch, frames, embed_dim]
        # class_tokens shape should be [batch, 1, embed_dim]
        x_embed_dim = tf.shape(x)[-1]
        class_token_embed_dim = tf.shape(class_tokens)[-1]
        
        # Ensure they have the same embed dimension
        if x_embed_dim != class_token_embed_dim:
            # Project if needed
            x = tf.keras.layers.Dense(class_token_embed_dim)(x)
        
        # Concatenate on sequence dimension (axis=1)
        x = tf.concat([x, class_tokens], axis=1)
        
        # Add positional embedding (broadcast to all positions)
        # Reshape temporal_pos_embed to [1, 1, embed_dim] for broadcasting
        pos_embed = tf.reshape(self.temporal_pos_embed, [1, 1, -1])
        x = x + pos_embed
        
        # Apply temporal transformer blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Apply normalization
        x = self.temporal_norm(x)
        
        # Global average pooling (excluding class token)
        x = tf.reduce_mean(x[:, :-1, :], axis=1)
        
        # Apply classification head
        logits = self.class_head(x)
        
        return logits, features


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Calculate key dimension to avoid shape issues
        key_dim = max(1, embed_dim // num_heads)
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        # Projection layers to ensure shape compatibility
        self.proj = layers.Dense(embed_dim)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        # Record input shape for residual connection compatibility
        input_shape = tf.shape(inputs)
        
        # Self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        
        # Ensure output shape matches input shape
        attn_output = self.proj(attn_output)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Ensure compatible shapes for addition
        if tf.shape(attn_output)[2] != input_shape[2]:
            attn_output = tf.keras.layers.Dense(input_shape[2])(attn_output)
            
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Ensure compatible shapes for addition
        if tf.shape(ffn_output)[2] != tf.shape(out1)[2]:
            ffn_output = tf.keras.layers.Dense(tf.shape(out1)[2])(ffn_output)
            
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
