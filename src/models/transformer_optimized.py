# src/models/transformer_optimized.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransModel(tf.keras.Model):
    """TensorFlow implementation of LightHART TransModel with TFLite export support.
    Closely matches PyTorch implementation but optimized for TFLite conversion."""
    
    def __init__(
        self,
        acc_frames=128,
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
        
        # Store configuration parameters
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        
        # Input projection - equivalent to PyTorch Conv1d
        self.input_proj = tf.keras.Sequential([
            layers.Conv1D(
                filters=embed_dim,
                kernel_size=8,
                strides=1,
                padding='same',
                name="input_projection"
            ),
            layers.BatchNormalization(name="input_batch_norm")
        ], name="input_block")
        
        # Create encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                self._build_encoder_layer(i)
            )
        
        # Final normalization (matches PyTorch LayerNorm)
        self.temporal_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="temporal_norm"
        )
        
        # Output layer
        self.output_layer = layers.Dense(
            num_classes,
            name="output_projection"
        )
    
    def _build_encoder_layer(self, idx):
        """Build a single transformer encoder layer matching PyTorch implementation"""
        # Create attention layer
        attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name=f"mha_{idx}"
        )
        
        # Create feed-forward network (matches PyTorch MLP)
        ffn = tf.keras.Sequential([
            layers.Dense(
                units=self.embed_dim * 2,
                activation=self.activation,
                name=f"ffn_dense1_{idx}"
            ),
            layers.Dropout(self.dropout_rate, name=f"ffn_dropout1_{idx}"),
            layers.Dense(
                units=self.embed_dim,
                name=f"ffn_dense2_{idx}"
            ),
            layers.Dropout(self.dropout_rate, name=f"ffn_dropout2_{idx}")
        ], name=f"ffn_{idx}")
        
        # Create normalization layers
        ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{idx}")
        ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{idx}")
        
        # Define layer as a custom Layer subclass
        class EncoderBlock(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, x, training=False):
                # Self-attention with residual connection
                attn_output = attention(x, x, x, training=training)
                x = x + attn_output
                x = ln1(x)
                
                # Feed forward with residual connection
                ffn_output = ffn(x, training=training)
                x = x + ffn_output
                x = ln2(x)
                
                return x
            
            def get_config(self):
                return super().get_config()
        
        return EncoderBlock(name=f"encoder_block_{idx}")
    
    def preprocess_acc_data(self, acc_data):
        """Add Signal Magnitude Vector to accelerometer data"""
        # Calculate signal magnitude vector (SMV)
        mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
        zero_mean = acc_data - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        
        # Concatenate SMV with original accelerometer data
        return tf.concat([smv, acc_data], axis=-1)
    
    def call(self, inputs, training=False):
        """Forward pass through the model.
        Handles both direct tensor inputs and dictionary inputs with multiple modalities.
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            if 'accelerometer' in inputs:
                acc_data = inputs['accelerometer']
                
                # Check if SMV needs to be added (expecting 3 channels)
                if acc_data.shape[-1] == self.acc_coords:
                    acc_data = self.preprocess_acc_data(acc_data)
                
                x = acc_data
            else:
                raise ValueError("Input must contain 'accelerometer' data")
        else:
            # Direct tensor input - assume it's accelerometer data
            # Check if SMV needs to be added (expecting 3 channels)
            if inputs.shape[-1] == self.acc_coords:
                x = self.preprocess_acc_data(inputs)
            else:
                x = inputs
        
        # Input projection
        x = self.input_proj(x, training=training)
        
        # Process through encoder layers
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        
        # Final normalization
        x = self.temporal_norm(x)
        features = x  # Save features for knowledge distillation
        
        # Global pooling (matching PyTorch's mean reduction)
        x = tf.reduce_mean(x, axis=1)
        
        # Output projection
        logits = self.output_layer(x)
        
        return logits, features
    
    def get_config(self):
        """Get model configuration for serialization"""
        config = super().get_config()
        config.update({
            "acc_frames": self.acc_frames,
            "num_classes": self.num_classes,
            "num_heads": self.num_heads, 
            "acc_coords": self.acc_coords,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "activation": self.activation,
        })
        return config
    
    def create_tflite_signatures(self):
        """Create signatures for TFLite conversion with accelerometer-only input"""
        # Create concrete function for accelerometer-only input
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, self.acc_frames, self.acc_coords], 
                         dtype=tf.float32, name='accelerometer')
        ])
        def serving_function(accelerometer):
            # Preprocess data by adding SMV
            acc_with_smv = self.preprocess_acc_data(accelerometer)
            return self(acc_with_smv, training=False)
        
        # Create signature for batch size 1 (for TFLite)
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, self.acc_frames, self.acc_coords], 
                         dtype=tf.float32, name='accelerometer')
        ])
        def tflite_function(accelerometer):
            # Preprocess data by adding SMV
            acc_with_smv = self.preprocess_acc_data(accelerometer)
            return self(acc_with_smv, training=False)
        
        self.serving_function = serving_function
        self.tflite_function = tflite_function
        
        return {
            'serving_default': serving_function,
            'tflite': tflite_function
        }
