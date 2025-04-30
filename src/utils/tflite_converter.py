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
        
        # Use the model's built-in export method if available
        if hasattr(model, 'export_to_tflite'):
            logging.info("Using model's built-in TFLite export method")
            return model.export_to_tflite(save_path, input_shape=input_shape, quantize=quantize)
        
        logging.info(f"Starting TFLite conversion with input shape {input_shape}")
        
        # Create concrete function for accelerometer-only input
        class WrapperModel(tf.keras.Model):
            def __init__(self, parent_model):
                super().__init__()
                self.parent = parent_model
            
            @tf.function
            def call(self, inputs):
                # Wrap accelerometer data in dictionary for parent model
                inputs_dict = {'accelerometer': inputs}
                # Get output from parent model
                return self.parent(inputs_dict, training=False)
        
        # Create and initialize wrapper model
        wrapper_model = WrapperModel(model)
        dummy_input = tf.zeros(input_shape, dtype=tf.float32)
        _ = wrapper_model(dummy_input)  # Initialize variables
        
        # Create concrete function for TF 2.x compatibility
        concrete_func = wrapper_model.call.get_concrete_function(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='accelerometer_input')
        )
        
        # Create temporary saved model 
        temp_saved_model_dir = os.path.join(os.path.dirname(save_path), "temp_savedmodel")
        if os.path.exists(temp_saved_model_dir):
            import shutil
            shutil.rmtree(temp_saved_model_dir)
        
        # Save model with concrete function
        tf.saved_model.save(
            wrapper_model,
            temp_saved_model_dir,
            signatures=concrete_func
        )
        
        # Create converter from saved model
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model_dir)
        
        # Set optimization options
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure TFLite conversion options for TF 2.19 compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Include TF ops for better compatibility
        ]
        
        # Convert the model
        logging.info("Converting model to TFLite...")
        tflite_model = converter.convert()
        
        # Save the converted model
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        logging.info(f"TFLite model saved to {save_path}")
        
        # Validate the TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.info(f"TFLite input details: {input_details}")
        logging.info(f"TFLite output details: {output_details}")
        
        # Create sample input
        sample_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Test inference
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logging.info(f"TFLite test successful - output shape: {output.shape}")
        
        # Clean up temporary SavedModel
        import shutil
        if os.path.exists(temp_saved_model_dir):
            shutil.rmtree(temp_saved_model_dir)
            
        return True
    except Exception as e:
        logging.error(f"TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary model if exists
        try:
            import shutil
            temp_saved_model_dir = os.path.join(os.path.dirname(save_path), "temp_savedmodel")
            if os.path.exists(temp_saved_model_dir):
                shutil.rmtree(temp_saved_model_dir)
        except:
            pass
            
        return False
