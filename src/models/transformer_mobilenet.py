import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np

class TransModel(tf.keras.Model):
    def __init__(
        self,
        acc_frames=64,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=0.5,
        activation='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        # Save configuration
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.head_dim = embed_dim // num_heads
        
        # Input processing layers (TFLite-friendly)
        self.conv_projection = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=(8, 1),
            padding='same',
            use_bias=True,  # Enable bias for better folding
            name="conv_projection"
        )
        self.input_norm = layers.LayerNormalization(epsilon=1e-6, name="input_norm")
        
        # Define transformer blocks
        self.transformer_blocks = []
        for i in range(self.num_layers):
            block = TFLiteTransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.embed_dim * 2,
                dropout=self.dropout_rate,
                activation=self.activation,
                layer_idx=i
            )
            self.transformer_blocks.append(block)
        
        # Output layers
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        self.global_avg_pool = layers.GlobalAveragePooling1D(name="global_pool")
        self.classifier = layers.Dense(self.num_classes, name="output_dense")
        
        # Initialize with dummy input for variable creation
        dummy_input = tf.zeros((1, acc_frames, acc_coords), dtype=tf.float32)
        self(dummy_input, training=False)
        
        logging.info(f"TFLite-optimized TransModel initialized: frames={acc_frames}, embed_dim={embed_dim}")

    def call(self, inputs, training=False):
        # Handle different input types
        x = inputs
        if isinstance(inputs, dict) and 'accelerometer' in inputs:
            x = inputs['accelerometer']
        
        # Input shape: [batch, frames, channels]
        batch_size = tf.shape(x)[0]
        
        # TFLite-friendly reshaping
        x = tf.expand_dims(x, axis=2)                      # [batch, frames, 1, channels]
        x = self.conv_projection(x)                         # [batch, frames, 1, embed_dim]
        x = tf.squeeze(x, axis=2)                           # [batch, frames, embed_dim]
        x = self.input_norm(x)                              # [batch, frames, embed_dim]
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)                 # [batch, frames, embed_dim]
        
        # Output processing
        x = self.final_norm(x)                              # [batch, frames, embed_dim]
        x = self.global_avg_pool(x)                         # [batch, embed_dim]
        x = self.classifier(x)                              # [batch, num_classes]
        
        # Ensure consistent output shape for binary classification
        if self.num_classes == 1:
            return tf.reshape(x, [-1, 1])
        return x


class TFLiteTransformerBlock(tf.keras.layers.Layer):
    """TFLite-optimized transformer block with manually implemented attention"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, activation='relu', layer_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.head_dim = dim // num_heads
        self.idx = layer_idx
        
        # Attention components
        self.q_dense = layers.Dense(dim, name=f"attn_q_{layer_idx}")
        self.k_dense = layers.Dense(dim, name=f"attn_k_{layer_idx}")
        self.v_dense = layers.Dense(dim, name=f"attn_v_{layer_idx}")
        self.attn_output = layers.Dense(dim, name=f"attn_output_{layer_idx}")
        
        # Layer normalization and dropout
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{layer_idx}")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{layer_idx}")
        self.dropout1 = layers.Dropout(dropout, name=f"dropout1_{layer_idx}")
        self.dropout2 = layers.Dropout(dropout, name=f"dropout2_{layer_idx}")
        
        # MLP layers
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation, name=f"mlp_dense1_{layer_idx}"),
            layers.Dropout(dropout),
            layers.Dense(dim, name=f"mlp_dense2_{layer_idx}"),
            layers.Dropout(dropout)
        ], name=f"mlp_{layer_idx}")

    def call(self, inputs, training=False):
        # Normalize first
        x_norm = self.norm1(inputs)  # [batch, seq_len, dim]
        
        # Compute Q, K, V
        q = self.q_dense(x_norm)  # [batch, seq_len, dim]
        k = self.k_dense(x_norm)  # [batch, seq_len, dim]
        v = self.v_dense(x_norm)  # [batch, seq_len, dim]
        
        # Reshape for multi-head attention
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scale query
        q = q * (float(self.head_dim) ** -0.5)
        
        # Calculate attention scores
        attention_scores = tf.matmul(q, k, transpose_b=True)  # [batch, num_heads, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention weights to values
        context = tf.matmul(attention_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Transpose and reshape to original dimensions
        context = tf.transpose(context, [0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
        context = tf.reshape(context, [batch_size, seq_len, self.dim])  # [batch, seq_len, dim]
        
        # Apply output projection
        attention_output = self.attn_output(context)
        attention_output = self.dropout1(attention_output, training=training)
        
        # First residual connection
        output1 = attention_output + inputs
        
        # Second normalization and MLP
        output2 = self.norm2(output1)
        mlp_output = self.mlp(output2, training=training)
        
        # Second residual connection
        return mlp_output + output1


class MobileNetBlock(tf.keras.layers.Layer):
    """Optional MobileNet-style block for more efficient feature extraction"""
    def __init__(self, filters, strides=1, expansion_factor=4, name="mobile_block"):
        super().__init__(name=name)
        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor
        
        # Expansion
        self.expand = layers.Conv2D(
            filters=filters * expansion_factor,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=f"{name}_expand"
        )
        self.expand_bn = layers.BatchNormalization(name=f"{name}_expand_bn")
        self.expand_act = layers.ReLU(6., name=f"{name}_expand_relu")
        
        # Depthwise
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding='same',
            use_bias=False,
            name=f"{name}_depthwise"
        )
        self.depthwise_bn = layers.BatchNormalization(name=f"{name}_depthwise_bn")
        self.depthwise_act = layers.ReLU(6., name=f"{name}_depthwise_relu")
        
        # Projection
        self.project = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=f"{name}_project"
        )
        self.project_bn = layers.BatchNormalization(name=f"{name}_project_bn")
        
        # Skip connection if input and output have same shape
        self.use_residual = strides == 1
        
    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.expand_bn(x, training=training)
        x = self.expand_act(x)
        
        x = self.depthwise(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_act(x)
        
        x = self.project(x)
        x = self.project_bn(x, training=training)
        
        if self.use_residual and inputs.shape[-1] == self.filters:
            return x + inputs
        return x
