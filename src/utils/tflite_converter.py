# utils/tflite_converter.py
import os
import logging
import tensorflow as tf
import numpy as np
import traceback

def convert_to_tflite(model, save_path, input_shape=(1, 128, 3), quantize=False):
    """Convert model to TFLite format with raw accelerometer input."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Add .tflite extension if not present
        if not save_path.endswith('.tflite'):
            save_path += '.tflite'
        
        # Check if model has built-in export method
        if hasattr(model, 'export_to_tflite'):
            logging.info("Using model's built-in TFLite export method")
            return model.export_to_tflite(save_path)
        
        logging.info("Using standard TFLite conversion approach")
        
        # Create a wrapper model for TFLite export
        class TFLiteModel(tf.keras.Model):
            def __init__(self, orig_model):
                super().__init__()
                self.orig_model = orig_model
            
            def call(self, inputs):
                # Wrap in dictionary for original model
                inputs_dict = {'accelerometer': inputs}
                # Get output from original model
                return self.orig_model(inputs_dict, training=False)
        
        # Create wrapper model
        tflite_model = TFLiteModel(model)
        
        # Initialize with sample input
        sample_input = tf.zeros(input_shape, dtype=tf.float32)
        tflite_model(sample_input)
        
        # Create concrete function for TFLite conversion
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, input_shape[1], input_shape[2]], dtype=tf.float32)
        ])
        def serving_function(inputs):
            return tflite_model(inputs)
        
        # Convert using concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [serving_function.get_concrete_function()]
        )
        
        # Set optimization options
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure supported operations
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Convert to TFLite format
        logging.info("Converting model to TFLite...")
        tflite_buffer = converter.convert()
        
        # Save the converted model
        with open(save_path, 'wb') as f:
            f.write(tflite_buffer)
        
        logging.info(f"TFLite model saved to {save_path}")
        
        # Test the TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_buffer)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.info(f"Input details: {input_details}")
        logging.info(f"Output details: {output_details}")
        
        # Create sample input
        test_input = np.zeros(input_shape, dtype=np.float32)
        
        # Test inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logging.info(f"TFLite test successful - output shape: {output.shape}")
        
        return True
    except Exception as e:
        logging.error(f"TFLite conversion failed: {e}")
        traceback.print_exc()
        return False
