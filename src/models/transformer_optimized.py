import tensorflow as tf

class TransModel(tf.keras.Model):
    """TensorFlow implementation of LightHART TransModel with TFLite export support."""
    
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
        
        # Store configuration
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        
        # Input projection
        self.input_proj_conv = tf.keras.layers.Conv1D(
            filters=embed_dim, 
            kernel_size=8, 
            strides=1, 
            padding='same',
            name="input_projection"
        )
        self.input_proj_bn = tf.keras.layers.BatchNormalization(name="batch_norm_projection")
        
        # Transformer encoder layers - using individual components instead of Sequential
        self.encoder_norm1_layers = []
        self.encoder_attn_layers = []
        self.encoder_norm2_layers = []
        self.encoder_ffn1_layers = []
        self.encoder_dropout1_layers = []
        self.encoder_ffn2_layers = []
        self.encoder_dropout2_layers = []
        
        for i in range(num_layers):
            # First normalization layer
            self.encoder_norm1_layers.append(
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{i}")
            )
            
            # Multi-head attention layer
            self.encoder_attn_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout,
                    name=f"mha_{i}"
                )
            )
            
            # Second normalization layer
            self.encoder_norm2_layers.append(
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{i}")
            )
            
            # FFN first layer
            self.encoder_ffn1_layers.append(
                tf.keras.layers.Dense(
                    units=embed_dim * 2,
                    activation=activation,
                    name=f"ffn1_{i}"
                )
            )
            
            # First dropout layer
            self.encoder_dropout1_layers.append(
                tf.keras.layers.Dropout(dropout, name=f"dropout1_{i}")
            )
            
            # FFN second layer
            self.encoder_ffn2_layers.append(
                tf.keras.layers.Dense(
                    units=embed_dim,
                    name=f"ffn2_{i}"
                )
            )
            
            # Second dropout layer
            self.encoder_dropout2_layers.append(
                tf.keras.layers.Dropout(dropout, name=f"dropout2_{i}")
            )
        
        # Final normalization layer
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, 
            name="final_norm"
        )
        
        # Global pooling layer
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")
        
        # Output projection
        self.output_proj = tf.keras.layers.Dense(
            num_classes, 
            name="output_projection"
        )
        
        # Build the model with a sample input
        self.build_sample_model()
    
    def build_sample_model(self):
        """Initialize the model by running a sample input through it"""
        sample_input = tf.zeros((1, self.acc_frames, self.acc_coords))
        self(sample_input, training=False)
    
    def add_smv(self, acc_data):
        """Add Signal Magnitude Vector to accelerometer data"""
        # Calculate mean
        mean = tf.reduce_mean(acc_data, axis=1, keepdims=True)
        
        # Center data
        zero_mean = acc_data - mean
        
        # Calculate sum of squares
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        
        # Calculate signal magnitude vector
        smv = tf.sqrt(sum_squared)
        
        # Concatenate SMV with original data
        return tf.concat([smv, acc_data], axis=-1)
    
    def call(self, inputs, training=False):
        """Forward pass through the model"""
        # Handle dictionary input
        if isinstance(inputs, dict):
            if 'accelerometer' in inputs:
                acc_data = inputs['accelerometer']
                
                # Check if SMV needs to be added
                if acc_data.shape[-1] == self.acc_coords:
                    acc_data = self.add_smv(acc_data)
                
                x = acc_data
            else:
                raise ValueError("Input dict must contain 'accelerometer' key")
        else:
            # Handle direct tensor input
            if inputs.shape[-1] == self.acc_coords:
                x = self.add_smv(inputs)
            else:
                x = inputs
        
        # Apply input projection
        x = self.input_proj_conv(x)
        x = self.input_proj_bn(x, training=training)
        
        # Apply transformer encoder layers
        for i in range(self.num_layers):
            # Layer norm 1
            norm1_output = self.encoder_norm1_layers[i](x)
            
            # Self-attention
            attn_output = self.encoder_attn_layers[i](
                query=norm1_output,
                value=norm1_output,
                training=training
            )
            
            # Residual connection
            x1 = x + attn_output
            
            # Layer norm 2
            norm2_output = self.encoder_norm2_layers[i](x1)
            
            # FFN
            ffn1_output = self.encoder_ffn1_layers[i](norm2_output)
            ffn1_dropout = self.encoder_dropout1_layers[i](ffn1_output, training=training)
            ffn2_output = self.encoder_ffn2_layers[i](ffn1_dropout)
            ffn2_dropout = self.encoder_dropout2_layers[i](ffn2_output, training=training)
            
            # Residual connection
            x = x1 + ffn2_dropout
        
        # Apply final normalization
        features = self.final_norm(x)
        
        # Apply global pooling
        pooled = self.global_pool(features)
        
        # Apply output projection
        logits = self.output_proj(pooled)
        
        # Return logits and features
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
    
    def save_tflite(self, filepath):
        """Export model to TFLite format with accelerometer-only input"""
        try:
            # Create concrete function for TFLite conversion
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, self.acc_frames, self.acc_coords], dtype=tf.float32)
            ])
            def serving_fn(input_tensor):
                # Add SMV to input
                processed = self.add_smv(input_tensor)
                # Forward pass
                outputs, _ = self(processed, training=False)
                return outputs
            
            # Create a saved model with the concrete function
            saved_model_dir = f"{filepath}_saved_model"
            tf.saved_model.save(
                obj=self,
                export_dir=saved_model_dir,
                signatures={'serving_default': serving_fn}
            )
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            # Set optimization flags
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set supported ops
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            # Convert to TFLite format
            tflite_model = converter.convert()
            
            # Save to file
            with open(filepath, 'wb') as f:
                f.write(tflite_model)
                
            # Clean up saved model directory if needed
            import shutil
            try:
                shutil.rmtree(saved_model_dir)
            except:
                pass
            
            return True
        except Exception as e:
            print(f"Error exporting TFLite model: {e}")
            import traceback
            traceback.print_exc()
            return False
