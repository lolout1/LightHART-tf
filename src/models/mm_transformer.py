# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Block(tf.keras.layers.Layer):
    """Transformer encoder block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., blocktype=None):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim // num_heads, 
            dropout=attn_drop
        )
        self.drop_path_rate = drop_path
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])
        
    def call(self, x, training=False):
        shortcut = x
        attn_output = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = shortcut + attn_output
        
        shortcut = x
        x = shortcut + self.mlp(self.norm2(x), training=training)
        
        return x

class MMTransformer(tf.keras.Model):
    def __init__(self, 
                 mocap_frames=128, 
                 acc_frames=128, 
                 num_joints=32, 
                 in_chans=3, 
                 num_patch=2,
                 acc_coords=3, 
                 spatial_embed=32, 
                 sdepth=2, 
                 adepth=2, 
                 tdepth=2, 
                 num_heads=2, 
                 mlp_ratio=2, 
                 qkv_bias=True, 
                 qk_scale=None, 
                 op_type='all', 
                 embed_type='lin', 
                 drop_rate=0.2, 
                 attn_drop_rate=0.2, 
                 drop_path_rate=0.2, 
                 num_classes=1,
                 **kwargs):
        super().__init__()
        
        # Save configuration parameters
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.num_patch = num_patch
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.op_type = op_type
        self.embed_type = embed_type
        
        # Embeddings
        temp_embed = spatial_embed
        acc_embed = temp_embed
        
        # Token and position embeddings
        self.temp_token = self.add_weight(
            name="temp_token",
            shape=(1, 1, spatial_embed),
            initializer=tf.zeros_initializer()
        )
        
        self.temporal_pos_embed = self.add_weight(
            name="temporal_pos_embed",
            shape=(1, 1, spatial_embed),
            initializer=tf.zeros_initializer()
        )
        
        # Spatial transformer components
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(in_chans, kernel_size=(1, 9), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, kernel_size=(1, 9), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(temp_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        # Transform layer
        self.transform = tf.keras.Sequential([
            layers.Dense(spatial_embed),
            layers.ReLU()
        ])
        
        # Joint relation blocks
        self.joint_block = [
            Block(
                dim=spatial_embed, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=drop_path_rate
            )
        ]
        
        # Temporal blocks
        tdpr = [x for x in np.linspace(0.0, drop_path_rate, tdepth)]
        self.temporal_blocks = [
            Block(
                dim=temp_embed, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=tdpr[i]
            )
            for i in range(tdepth)
        ]
        
        # Norm layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ])
    
    def build(self, input_shape):
        # Ensure the model is built properly
        super().build(input_shape)
    
    def temporal_forward(self, x, training=False):
        """Process through temporal transformer blocks"""
        # Get batch size
        b = tf.shape(x)[0]
        
        # Reshape class token to match feature dimensions
        # Expected input shape: [batch, time, embed_dim]
        class_token = tf.tile(self.temp_token, [b, 1, 1])
        
        # Concatenate along the time dimension
        x = tf.concat([x, class_token], axis=1)
        
        # Add position embedding
        x = x + self.temporal_pos_embed
        
        # Process through transformer blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Apply normalization
        x = self.temporal_norm(x)
        
        # Output based on type
        if self.op_type == 'cls':
            # Use class token
            return x[:, -1, :]
        else:
            # Use mean pooling over sequence dimension
            return tf.reduce_mean(x[:, :-1, :], axis=1)
    
    def call(self, inputs, training=False):
        """Forward pass through the model"""
        # Parse inputs
        if isinstance(inputs, dict):
            acc_data = inputs['accelerometer']
            skl_data = inputs['skeleton']
        elif isinstance(inputs, tuple):
            acc_data, skl_data = inputs
        else:
            raise ValueError("Inputs must be a dictionary or tuple")
        
        # Get batch size
        b = tf.shape(acc_data)[0]
        
        # Process skeleton data through spatial encoder
        # Reshape: [batch, time, joints, coords] -> [batch, coords, time, joints]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x)
        
        # Reshape: [batch, 1, time, joints] -> [batch, time, joints]
        x = tf.squeeze(x, axis=1)
        
        # Apply spatial encoder: [batch, time, joints] -> [batch, time, embed_dim]
        x = self.spatial_encoder(x)
        
        # Transform to embed_dim
        x = self.transform(x)
        
        # Process through joint attention blocks
        feature = None
        for i, block in enumerate(self.joint_block):
            x = block(x, training=training)
            if i == 0:
                # Save features for distillation
                feature = x
        
        # Temporal processing
        x = self.temporal_forward(x, training=training)
        
        # Output logits
        logits = self.class_head(x)
        
        return logits, feature
