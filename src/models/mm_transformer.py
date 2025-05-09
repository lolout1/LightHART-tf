# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers
import logging

class MMTransformer(tf.keras.Model):
    def __init__(
        self, 
        mocap_frames=128, 
        acc_frames=128, 
        num_joints=32, 
        in_chans=3, 
        num_patch=4, 
        acc_coords=3, 
        spatial_embed=16, 
        sdepth=4, 
        adepth=4, 
        tdepth=2, 
        num_heads=2, 
        mlp_ratio=2, 
        qkv_bias=True, 
        qk_scale=None, 
        op_type='pool', 
        embed_type='lin', 
        drop_rate=0.2, 
        attn_drop_rate=0.2, 
        drop_path_rate=0.2, 
        num_classes=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.num_patch = num_patch
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.sdepth = sdepth
        self.adepth = adepth
        self.tdepth = tdepth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.op_type = op_type
        self.embed_type = embed_type
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        
        # Learnable tokens and weights
        self.temp_token = self.add_weight(
            shape=(1, 1, spatial_embed),
            initializer="zeros",
            trainable=True,
            name="temp_token"
        )
        
        self.Temporal_pos_embed = self.add_weight(
            shape=(1, 1, spatial_embed),
            initializer="zeros",
            trainable=True,
            name="temporal_pos_embed"
        )
        
        # Spatial encoder (processes skeleton data)
        self.Spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(spatial_embed, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(spatial_embed, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(spatial_embed, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_encoder")
        
        # Transform layer
        self.transform = layers.Dense(spatial_embed, activation='relu')
        
        # Spatial convolutional layers
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(in_chans, (1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, (1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_conv")
        
        # Transformer blocks
        self.Temporal_blocks = []
        for i in range(tdepth):
            self.Temporal_blocks.append(
                TransformerBlock(
                    dim=spatial_embed,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate * i / max(1, tdepth-1)
                )
            )
        
        # Joint relation block
        self.joint_block = TransformerBlock(
            dim=spatial_embed,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Layer normalization
        self.Temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ])
    
    def Temp_forward_features(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # Ensure x is properly shaped [batch, seq_len, features]
        if len(tf.shape(x)) == 3 and tf.shape(x)[2] == self.spatial_embed:
            # This is [batch, seq_len, features] - correct shape
            pass
        elif len(tf.shape(x)) == 3 and tf.shape(x)[1] == self.spatial_embed:
            # This is [batch, features, seq_len] - needs transpose
            x = tf.transpose(x, [0, 2, 1])
        
        # Create class token for each batch item
        class_token = tf.repeat(self.temp_token, repeats=batch_size, axis=0)
        
        # Concatenate class token to sequence
        x = tf.concat([x, class_token], axis=1)
        
        # Create positional embedding that matches sequence length
        pos_embed = tf.repeat(self.Temporal_pos_embed, repeats=tf.shape(x)[1], axis=1)
        pos_embed = pos_embed[:, :tf.shape(x)[1], :]
        
        # Add positional embedding
        x = x + pos_embed
        
        # Process through transformer blocks
        for blk in self.Temporal_blocks:
            x = blk(x, training=training)
        
        # Apply normalization
        x = self.Temporal_norm(x)
        
        # Extract features based on mode
        if self.op_type == 'cls':
            # Extract class token
            return x[:, -1, :]
        else:
            # Take all tokens except class token for pooling
            seq_len = tf.shape(x)[1] - 1
            tokens = x[:, :seq_len, :]
            # Global average pooling
            return tf.reduce_mean(tokens, axis=1)
    
    def call(self, inputs, training=False):
        # Handle both dictionary and tuple inputs
        if isinstance(inputs, dict) and 'accelerometer' in inputs and 'skeleton' in inputs:
            acc_data = inputs['accelerometer']
            skl_data = inputs['skeleton']
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            acc_data, skl_data = inputs
        else:
            raise ValueError("Input must be a dictionary with 'accelerometer' and 'skeleton' keys or a tuple of (acc_data, skl_data)")
        
        # Get batch size
        batch_size = tf.shape(skl_data)[0]
        
        # Process skeleton data only (like PyTorch implementation)
        # Convert to [batch, channels, frames, joints]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x)
        
        # Reshape to [batch, frames, features]
        x = tf.reshape(x, [batch_size, tf.shape(skl_data)[1], -1])
        
        # Spatial encoding
        x = self.Spatial_encoder(x)
        
        # Feature transformation
        x = self.transform(x)
        
        # Process through joint block
        x = self.joint_block(x, training=training)
        
        # Save features for knowledge distillation
        feature = x
        feature = tf.transpose(feature, [0, 2, 1])
        feature = tf.reduce_mean(feature, axis=2, keepdims=True)
        feature = tf.reshape(feature, [batch_size, -1])
        
        # Temporal features processing
        cls_features = self.Temp_forward_features(x, training=training)
        
        # Classification
        logits = self.class_head(cls_features)
        
        # Format output for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        return logits, feature


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4, 
        drop_rate=0.0, 
        attn_drop_rate=0.0, 
        drop_path_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop_rate
        )
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='relu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])
        
        self.drop_path_rate = drop_path_rate
    
    def call(self, x, training=False):
        # Normalize then attention
        x_norm = self.norm1(x)
        attn_output = self.mha(x_norm, x_norm, x_norm, training=training)
        
        # Apply drop path if training
        if training and self.drop_path_rate > 0:
            attn_output = tf.nn.dropout(attn_output, self.drop_path_rate)
        
        # First residual
        x = x + attn_output
        
        # Normalize then MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm, training=training)
        
        # Apply drop path if training
        if training and self.drop_path_rate > 0:
            mlp_output = tf.nn.dropout(mlp_output, self.drop_path_rate)
        
        # Second residual
        x = x + mlp_output
        
        return x
