# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np

class MMTransformer(tf.keras.Model):
    """
    Multi-Modal Transformer model for sensor fusion of accelerometer and skeleton data
    Converted from PyTorch implementation to TensorFlow
    """
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
        
        # Save configuration
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
        
        # Derived parameters
        temp_embed = spatial_embed
        acc_embed = temp_embed
        self.skl_patch_size = mocap_frames // num_patch
        self.acc_patch_size = acc_frames // num_patch
        self.skl_patch = self.skl_patch_size * (num_joints-8-8)//2
        self.temp_frames = mocap_frames
        self.skl_encoder_size = temp_embed
        
        # Learnable tokens and embeddings
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
        
        self.Acc_pos_embed = self.add_weight(
            shape=(1, 1, acc_embed),
            initializer="zeros",
            trainable=True,
            name="acc_pos_embed"
        )
        
        self.acc_token = self.add_weight(
            shape=(1, 1, acc_embed),
            initializer="zeros",
            trainable=True,
            name="acc_token"
        )
        
        # Embedding layers
        if self.embed_type == 'lin':
            self.Spatial_patch_to_embedding = layers.Dense(
                spatial_embed,
                name="spatial_patch_embedding"
            )
            
            self.Acc_coords_to_embedding = layers.Dense(
                acc_embed,
                name="acc_coords_embedding"
            )
        else:
            self.Spatial_patch_to_embedding = layers.Conv1D(
                filters=spatial_embed,
                kernel_size=1,
                strides=1,
                name="spatial_patch_conv"
            )
            
            self.Acc_coords_to_embedding = layers.Conv1D(
                filters=acc_embed,
                kernel_size=1,
                strides=1,
                name="acc_coords_conv"
            )
        
        # Prepare drop path rates
        sdpr = tf.linspace(0.0, drop_path_rate, sdepth).numpy()
        adpr = tf.linspace(0.0, drop_path_rate, adepth).numpy()
        tdpr = tf.linspace(0.0, drop_path_rate, tdepth).numpy()
        
        # Spatial encoder
        self.Spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(self.skl_encoder_size, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(self.skl_encoder_size, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(temp_embed, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_encoder")
        
        # Temporal transformer blocks
        self.Temporal_blocks = [
            TransformerBlock(
                dim=temp_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=tdpr[i],
                name=f"temporal_block_{i}"
            )
            for i in range(self.tdepth)
        ]
        
        # Joint relation block
        self.joint_block = TransformerBlock(
            dim=32,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=tdpr[0],
            name="joint_block"
        )
        
        # Normalization layers
        self.Spatial_norm = layers.LayerNormalization(epsilon=1e-6, name="spatial_norm")
        self.Acc_norm = layers.LayerNormalization(epsilon=1e-6, name="acc_norm")
        self.Temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        
        # Dropout
        self.pos_drop = layers.Dropout(drop_rate)
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ], name="class_head")
        
        # Spatial convolutional layers
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(in_chans, (1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, (1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_conv")
        
        # Transform layer
        self.transform = tf.keras.Sequential([
            layers.Dense(32),
            layers.ReLU()
        ], name="transform")
        
        # Initialize with dummy input
        dummy_acc = tf.zeros((1, acc_frames, acc_coords))
        dummy_skl = tf.zeros((1, mocap_frames, num_joints, in_chans))
        self._initialize_model(dummy_acc, dummy_skl)
        
        logging.info(f"MMTransformer initialized: frames={mocap_frames}, embed_dim={spatial_embed}, classes={num_classes}")
    
    def _initialize_model(self, acc_data, skl_data):
        """Initialize model weights with dummy inputs"""
        inputs = {'accelerometer': acc_data, 'skeleton': skl_data}
        _ = self(inputs, training=False)
    
    def Temp_forward_features(self, x, training=False):
        """Temporal features forward pass"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Replicate class token for each batch item
        class_token = tf.tile(self.temp_token, [batch_size, 1, 1])
        
        # Concatenate with input sequence
        x = tf.concat([x, class_token], axis=1)
        
        # Add positional embedding
        x = x + self.Temporal_pos_embed
        
        # Pass through transformer blocks
        for blk in self.Temporal_blocks:
            x = blk(x, training=training)
        
        # Apply normalization
        x = self.Temporal_norm(x)
        
        # Handle output based on mode
        if self.op_type == 'cls':
            # Extract class token
            cls_token = x[:, -1, :]
            return cls_token
        else:
            # Pool sequence tokens
            x = x[:, :seq_len, :]
            x = tf.transpose(x, [0, 2, 1])  # b f St -> b St f
            x = tf.reduce_mean(x, axis=2)  # Global average pooling
            return x
    
    def call(self, inputs, training=False):
        """Forward pass of the model"""
        # Handle different input types
        if isinstance(inputs, dict) and 'accelerometer' in inputs and 'skeleton' in inputs:
            acc_data = inputs['accelerometer']
            skl_data = inputs['skeleton']
        else:
            # If inputs is not a dict with the expected keys, check for other formats
            if isinstance(inputs, tuple) and len(inputs) == 2:
                acc_data, skl_data = inputs
            else:
                raise ValueError("Input must be a dictionary with 'accelerometer' and 'skeleton' keys or a tuple of (acc_data, skl_data)")
        
        batch_size = tf.shape(acc_data)[0]
        
        # Add channel dimension to accelerometer data if needed
        if len(acc_data.shape) == 3 and acc_data.shape[-1] == self.acc_coords:
            acc_data = tf.expand_dims(acc_data, axis=2)  # [batch, frames, channels] -> [batch, frames, 1, channels]
        
        # Spatial processing for skeleton data
        x = tf.transpose(skl_data, [0, 3, 1, 2])  # b f j c -> b c f j
        x = self.spatial_conv(x)
        x = tf.reshape(x, [batch_size, tf.shape(skl_data)[1], -1])  # b c f j -> b f (j c)
        x = self.Spatial_encoder(x)
        x = self.transform(x)
        
        # Process through joint block
        feature = None
        x = self.joint_block(x, training=training)
        
        # Extract features for knowledge distillation
        feature = x[:, :tf.shape(skl_data)[1], :]
        feature = tf.transpose(feature, [0, 2, 1])  # b j f -> b f j
        feature = self.Temporal_norm(feature)
        feature = tf.reduce_mean(feature, axis=2, keepdims=True)  # Global average pooling
        feature = tf.reshape(feature, [batch_size, -1])  # Flatten
        
        # Temporal features
        x = tf.transpose(x, [0, 2, 1])  # b f c -> b c f
        x = self.Temp_forward_features(x, training=training)
        
        # Classification
        logits = self.class_head(x)
        
        # Format output for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        return logits, feature
    
    # Support for TFLite export
    def export_to_tflite(self, save_path, input_shape=(1, 128, 3), quantize=False):
        """Export model to TFLite format"""
        from utils.tflite_converter import convert_to_tflite
        return convert_to_tflite(
            model=self,
            save_path=save_path,
            input_shape=input_shape,
            quantize=quantize
        )

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and MLP"""
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4, 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0.0, 
        attn_drop_rate=0.0, 
        drop_path_rate=0.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate
        )
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop_rate
        )
        
        # Stochastic depth (drop path)
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else layers.Lambda(lambda x: x)
    
    def call(self, x, training=False):
        # Multi-head attention with residual connection
        attn_output = self.attn(self.norm1(x), training=training)
        x = x + self.drop_path(attn_output, training=training)
        
        # MLP with residual connection
        mlp_output = self.mlp(self.norm2(x), training=training)
        x = x + self.drop_path(mlp_output, training=training)
        
        return x

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention implementation"""
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0.0, 
        proj_drop=0.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)
        
        # QKV projection
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = layers.Dropout(attn_drop)
        
        # Output projection
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        
        # Split Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        # Apply attention to V
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, seq_len, self.dim])
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        
        return x

class MLP(tf.keras.layers.Layer):
    """MLP module"""
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=layers.ReLU, 
        drop=0.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = layers.Dense(hidden_features)
        self.act = act_layer()
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)
    
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class DropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
    
    def call(self, x, training=False):
        if self.drop_prob == 0. or not training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        batch_size = tf.shape(x)[0]
        random_tensor = keep_prob + tf.random.uniform([batch_size, 1, 1], dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(x, keep_prob) * binary_tensor
        return output
