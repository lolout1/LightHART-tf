# src/models/transformer_optimized.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import logging
import numpy as np
import shutil

class TransModel(tf.keras.Model):
    """TensorFlow implementation of fall detection transformer model.
    Optimized for TFLite conversion with raw accelerometer data (x,y,z)."""
    
    def __init__(
        self,
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=3,  # Exactly 3 for raw accelerometer (x,y,z)
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
        self.acc_coords = 3  # Force to 3 for raw accelerometer data
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        
        # Build model components with explicit naming
        self._build_components()
        
        # Initialize with dummy input to create variables
        dummy_input = tf.zeros((1, self.acc_frames, self.acc_coords), dtype=tf.float32)
        self(dummy_input, training=False)
        
        logging.info(f"TransModel initialized with acc_frames={self.acc_frames}, "
                    f"acc_coords={self.acc_coords}, embed_dim={self.embed_dim}")
    
    def _build_components(self):
        """Build all model components with explicit tracking"""
        # Input projection - single Conv1D layer
        self.conv_layer = layers.Conv1D(
            filters=self.embed_dim,
            kernel_size=8,
            strides=1,
            padding='same',
            name="conv_projection"
        )
        self.batch_norm = layers.BatchNormalization(name="batch_norm")
        
        # Encoder blocks - using standard layers for better tracking
        self.attention_layers = []
        self.ffn_layers = []
        self.layer_norms1 = []
        self.layer_norms2 = []
        
        for i in range(self.num_layers):
            # Multi-head attention
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.embed_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    name=f"mha_{i}"
                )
            )
            
            # Feed-forward network
            self.ffn_layers.append(tf.keras.Sequential([
                layers.Dense(
                    units=self.embed_dim * 2,
                    activation=self.activation,
                    name=f"ffn_dense1_{i}"
                ),
                layers.Dropout(self.dropout_rate),
                layers.Dense(
                    units=self.embed_dim,
                    name=f"ffn_dense2_{i}"
                ),
                layers.Dropout(self.dropout_rate)
            ], name=f"ffn_{i}"))
            
            # Layer normalization
            self.layer_norms1.append(
                layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")
            )
            self.layer_norms2.append(
                layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")
            )
        
        # Final normalization
        self.final_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="final_norm"
        )
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        
        # Output layer - ensuring we maintain dimensions
        self.output_dense = layers.Dense(
            self.num_classes,
            name="output_dense"
        )
    
    def call(self, inputs, training=False):
        """Forward pass through the model.
        Handles both dictionary and tensor inputs.
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            if 'accelerometer' in inputs:
                x = inputs['accelerometer']
            else:
                logging.warning("Input dictionary missing 'accelerometer' key")
                # Get batch size from any available key
                for key in inputs:
                    batch_size = tf.shape(inputs[key])[0]
                    break
                # Return safe output
                if training:
                    return tf.zeros([batch_size, 1]), tf.zeros([batch_size, self.acc_frames, self.embed_dim])
                else:
                    return tf.zeros([batch_size, 1])
        else:
            x = inputs
        
        # Input projection
        x = self.conv_layer(x)
        x = self.batch_norm(x, training=training)
        
        # Process through encoder layers
        for i in range(self.num_layers):
            # Self-attention with residual connection
            attn_output = self.attention_layers[i](x, x, x, training=training)
            x = x + attn_output
            x = self.layer_norms1[i](x)
            
            # Feed forward with residual connection
            ffn_output = self.ffn_layers[i](x, training=training)
            x = x + ffn_output
            x = self.layer_norms2[i](x)
        
        # Final normalization
        x = self.final_norm(x)
        features = x  # Save features for knowledge distillation
        
        # Global pooling
        x = self.global_pool(x)
        
        # Output projection with proper shape
        logits = self.output_dense(x)
        
        # For binary classification, ensure shape is [batch_size, 1]
        if self.num_classes == 1:
            logits = tf.reshape(logits, [-1, 1])
        
        # Return outputs based on training mode
        if training:
            return logits, features
        else:
            return logits
    
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
    
    def export_to_tflite(self, save_path):
        """Export model to TFLite format with robust approach compatible with TF 2.x"""
        try:
            logging.info("Exporting TransModel to TFLite...")
            
            # Add .tflite extension if not present
            if not save_path.endswith('.tflite'):
                save_path += '.tflite'
            
            # Create a standalone model for TFLite export
            class ExportModel(tf.keras.Model):
                def __init__(self, parent_model):
                    super().__init__()
                    self.parent = parent_model
                
                @tf.function
                def call(self, inputs):
                    # Directly pass to parent model with dictionary wrapping
                    acc_dict = {'accelerometer': inputs}
                    outputs = self.parent(acc_dict, training=False)
                    # Ensure output shape is [batch_size, 1]
                    return tf.reshape(outputs, [-1, 1])
            
            # Create export model
            export_model = ExportModel(self)
            
            # Generate sample data matching the expected input shape
            sample_input = tf.zeros([1, self.acc_frames, 3], dtype=tf.float32)
            
            # Run once to ensure variables are created
            _ = export_model(sample_input)
            
            # Setup temporary export directory
            temp_dir = os.path.join(os.path.dirname(save_path), "temp_export")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save model to temporary directory
            tf.saved_model.save(
                export_model,
                temp_dir,
                signatures=export_model.call.get_concrete_function(
                    tf.TensorSpec([None, self.acc_frames, 3], tf.float32)
                )
            )
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
                
            logging.info(f"TFLite model saved to {save_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            return True
            
        except Exception as e:
            logging.error(f"TFLite export failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            temp_dir = os.path.join(os.path.dirname(save_path), "temp_export")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
            return False
