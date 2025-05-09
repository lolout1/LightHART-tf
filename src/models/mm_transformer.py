# models/mm_transformer.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np

class MMTransformer(tf.keras.Model):
    def __init__(self, mocap_frames=128, acc_frames=128, num_joints=32, in_chans=3, num_patch=4, acc_coords=3, 
                 spatial_embed=16, sdepth=4, adepth=4, tdepth=2, num_heads=2, mlp_ratio=2, qkv_bias=True, 
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords
        self.spatial_embed = spatial_embed
        self.sdepth = sdepth
        self.adepth = adepth
        self.tdepth = tdepth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        
        # Build model components
        self._build_layers()
        
        # Initialize with dummy inputs to create variables
        self._initialize_variables()
        
    def _build_layers(self):
        """Create all model layers"""
        # Tokens and positional embeddings
        self.temp_token = self.add_weight(
            name="temp_token", 
            shape=(1, 1, self.spatial_embed),
            initializer="zeros", 
            trainable=True
        )
        
        self.temporal_pos_embed = self.add_weight(
            name="temporal_pos_embed",
            shape=(1, 1, self.spatial_embed),
            initializer="zeros",
            trainable=True
        )
        
        # Spatial convolutional layers for skeleton processing
        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(filters=self.in_chans, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=1, kernel_size=(1, 9), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_conv")
        
        # Spatial encoder for skeleton features
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv1D(filters=self.spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(filters=self.spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(filters=self.spatial_embed, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="spatial_encoder")
        
        # Feature transformation layer
        self.transform = layers.Dense(self.spatial_embed, activation='relu', name="transform")
        
        # Temporal transformer blocks
        self.temporal_blocks = []
        for i in range(self.tdepth):
            self.temporal_blocks.append(
                TransformerBlock(
                    dim=self.spatial_embed,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=self.drop_path_rate * i / max(1, self.tdepth-1),
                    name=f"temporal_block_{i}"
                )
            )
        
        # Joint relation block
        self.joint_block = TransformerBlock(
            dim=self.spatial_embed,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            name="joint_block"
        )
        
        # Normalization layers
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")
        
        # Classification head
        self.class_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.num_classes)
        ], name="class_head")
    
    def _initialize_variables(self):
        """Initialize all variables with dummy inputs"""
        try:
            # Create dummy inputs
            dummy_acc = tf.zeros((2, self.acc_frames, self.acc_coords), dtype=tf.float32)
            dummy_skl = tf.zeros((2, self.mocap_frames, self.num_joints, self.in_chans), dtype=tf.float32)
            
            # Forward pass to build variables
            _ = self({"accelerometer": dummy_acc, "skeleton": dummy_skl}, training=False)
            logging.info("MMTransformer variables initialized successfully")
        except Exception as e:
            logging.warning(f"Variable initialization failed (this is expected during initialization): {e}")
    
    def build(self, input_shape):
        """Explicitly build the model based on input shapes"""
        # Handle dictionary input
        if isinstance(input_shape, dict):
            acc_shape = input_shape.get('accelerometer')
            skl_shape = input_shape.get('skeleton')
            
            if acc_shape is not None and skl_shape is not None:
                # Both modalities available
                pass
            elif acc_shape is not None:
                # Only accelerometer available - create dummy skeleton
                skl_shape = (acc_shape[0], self.mocap_frames, self.num_joints, self.in_chans)
                logging.warning("Building with only accelerometer - using dummy skeleton shape")
            elif skl_shape is not None:
                # Only skeleton available - create dummy accelerometer
                acc_shape = (skl_shape[0], self.acc_frames, self.acc_coords)
                logging.warning("Building with only skeleton - using dummy accelerometer shape")
            else:
                raise ValueError("Neither accelerometer nor skeleton shapes provided")
        else:
            raise ValueError("Input shape must be a dictionary with modality keys")
        
        # Let each layer build itself with correct shapes
        self.built = True
    
    def process_skeleton(self, skl_data, training=False):
        """Process skeleton data through spatial network"""
        batch_size = tf.shape(skl_data)[0]
        
        # Reshape to [batch, channels, frames, joints]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        
        # Apply spatial convolution
        x = self.spatial_conv(x, training=training)
        
        # Reshape to [batch, frames, features]
        x = tf.reshape(x, [batch_size, self.mocap_frames, -1])
        
        # Apply spatial encoding
        x = self.spatial_encoder(x, training=training)
        
        # Feature transformation
        x = self.transform(x)
        
        # Process through joint block
        x = self.joint_block(x, training=training)
        
        return x
    
    def temporal_forward(self, x, training=False):
        """Process features through temporal network"""
        batch_size = tf.shape(x)[0]
        
        # Add class token
        class_token = tf.repeat(self.temp_token, repeats=batch_size, axis=0)
        x = tf.concat([x, class_token], axis=1)
        
        # Add positional embedding
        x = x + self.temporal_pos_embed
        
        # Process through transformer blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Apply normalization
        x = self.temporal_norm(x)
        
        # Extract features from sequence
        seq_len = tf.shape(x)[1] - 1
        sequence = x[:, :seq_len, :]
        
        # Global average pooling for features
        pooled = tf.reduce_mean(sequence, axis=1)
        
        return pooled
    
    def call(self, inputs, training=False):
        """Forward pass handling different input types"""
        # Handle different input types gracefully
        if isinstance(inputs, dict):
            acc_data = inputs.get('accelerometer')
            skl_data = inputs.get('skeleton')
            
            if skl_data is None:
                logging.warning("No skeleton data provided, outputs may be unreliable for teacher model")
                # Create dummy skeleton data
                batch_size = tf.shape(acc_data)[0]
                skl_data = tf.zeros((batch_size, self.mocap_frames, self.num_joints, self.in_chans), dtype=tf.float32)
            
            if acc_data is None:
                logging.warning("No accelerometer data provided, using only skeleton")
                # Create dummy accelerometer data
                batch_size = tf.shape(skl_data)[0]
                acc_data = tf.zeros((batch_size, self.acc_frames, self.acc_coords), dtype=tf.float32)
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            acc_data, skl_data = inputs
        else:
            raise ValueError("Input must be dictionary with modality keys or tuple (acc_data, skl_data)")
        
        # Process skeleton data
        features = self.process_skeleton(skl_data, training=training)
        
        # Extract and save features for knowledge distillation
        distill_features = tf.transpose(features, [0, 2, 1])
        batch_size = tf.shape(distill_features)[0]
        distill_features = tf.reduce_mean(distill_features, axis=2, keepdims=True)
        distill_features = tf.reshape(distill_features, [batch_size, -1])
        
        # Process through temporal network
        temporal_features = self.temporal_forward(features, training=training)
        
        # Final classification
        logits = self.class_head(temporal_features, training=training)
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        return logits, distill_features


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_rate=0.0, 
                 attn_drop_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = int(dim * mlp_ratio)
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi-head attention
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop_rate
        )
        
        # MLP block
        self.mlp = tf.keras.Sequential([
            layers.Dense(self.mlp_dim, activation='gelu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])
        
        self.drop_path_rate = drop_path_rate
    
    def build(self, input_shape):
        """Build layer based on input shape"""
        self.built = True
    
    def call(self, x, training=False):
        # Normalization before attention
        x_norm = self.norm1(x)
        
        # Multi-head attention
        attn_output = self.attn(x_norm, x_norm, x_norm, training=training)
        
        # Apply dropout path if training
        if training and self.drop_path_rate > 0:
            attn_output = tf.nn.dropout(attn_output, rate=self.drop_path_rate)
        
        # First residual connection
        x = x + attn_output
        
        # Normalization before MLP
        x_norm = self.norm2(x)
        
        # MLP block
        mlp_output = self.mlp(x_norm, training=training)
        
        # Apply dropout path if training
        if training and self.drop_path_rate > 0:
            mlp_output = tf.nn.dropout(mlp_output, rate=self.drop_path_rate)
        
        # Second residual connection
        x = x + mlp_output
        
        return x
