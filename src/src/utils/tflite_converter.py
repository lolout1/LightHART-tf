import os
import tensorflow as tf
import logging
import numpy as np

def convert_to_tflite(model, save_path, input_shape=(1, 128, 3), quantize=False):
    """Convert model to TFLite format with accelerometer-only input.
    
    This function creates a TFLite model that only takes accelerometer data as input,
    even if the original model was trained with both skeleton and accelerometer data.
    
    Args:
        model: TensorFlow model to convert
        save_path: Path to save the TFLite model
        input_shape: Input shape for accelerometer data (batch, frames, channels)
        quantize: Whether to apply quantization
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Add .tflite extension if not present
        if not save_path.endswith('.tflite'):
            save_path += '.tflite'
        
        # Create concrete function for accelerometer-only input
        @tf.function(input_signature=[
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='accelerometer')
        ])
        def serving_function(accelerometer):
            # Calculate signal magnitude vector (SMV)
            mean = tf.reduce_mean(accelerometer, axis=1, keepdims=True)
            zero_mean = accelerometer - mean
            sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
            smv = tf.sqrt(sum_squared)
            
            # Concatenate SMV with original data
            acc_with_smv = tf.concat([smv, accelerometer], axis=-1)
            
            # Create a dictionary for the model if it expects one
            inputs = {'accelerometer': acc_with_smv}
            
            # Forward pass (handle both tuple and single output)
            outputs = model(inputs, training=False)
            
            # Return only logits if model returns (logits, features)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0]
            return outputs
        
        # Create SavedModel with the concrete function
        saved_model_dir = os.path.join(os.path.dirname(save_path), "temp_saved_model")
        if os.path.exists(saved_model_dir):
            import shutil
            shutil.rmtree(saved_model_dir)
            
        tf.saved_model.save(
            model, 
            saved_model_dir,
            signatures={
                'serving_default': serving_function
            }
        )
        
        # Create converter from saved model
        converter = tf.lite.TFLiteConverter.from_saved_model(
            saved_model_dir,
            signature_keys=['serving_default']
        )
        
        # Set optimization options
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure TFLite options
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Include TF ops for better compatibility
        ]
        
        # Enable custom ops if needed
        converter.allow_custom_ops = True
        
        # Convert to TFLite format
        logging.info("Converting model to TFLite...")
        tflite_model = converter.convert()
        
        # Save the converted model
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        logging.info(f"TFLite model saved to {save_path}")
        
        # Test the model with sample input
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.info(f"Input details: {input_details}")
        logging.info(f"Output details: {output_details}")
        
        # Create sample input
        sample_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Test inference
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logging.info(f"TFLite test successful - output shape: {output.shape}")
        
        # Clean up temporary SavedModel
        import shutil
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
            
        return True
    except Exception as e:
        logging.error(f"TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
