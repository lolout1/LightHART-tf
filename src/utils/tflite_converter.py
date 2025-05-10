import os
import tensorflow as tf
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

def convert_to_tflite(model, save_path, input_shape=(1, 64, 3), quantize=False, optimize_for_mobile=True):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.endswith('.tflite'):
            save_path += '.tflite'
        
        logger.info(f"Starting TFLite conversion with input shape {input_shape}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Simple wrapper for accelerometer-only input
        class AccelerometerOnlyModel(tf.keras.Model):
            def __init__(self, parent_model):
                super().__init__()
                self.parent = parent_model
            
            @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
            def call(self, inputs):
                # Direct input without SMV calculation
                inputs_dict = {'accelerometer': inputs}
                outputs = self.parent(inputs_dict, training=False)
                if isinstance(outputs, tuple):
                    return outputs[0]
                return outputs
        
        wrapper_model = AccelerometerOnlyModel(model)
        
        # Get concrete function
        concrete_func = wrapper_model.call.get_concrete_function(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32)
        )
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Set optimization flags
        if optimize_for_mobile:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        
        if quantize:
            converter.representative_dataset = lambda: representative_dataset_gen(input_shape)
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        logger.info("Converting model to TFLite...")
        start_time = time.time()
        tflite_model = converter.convert()
        conversion_time = time.time() - start_time
        logger.info(f"Conversion completed in {conversion_time:.2f} seconds")
        
        # Save model
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"TFLite model saved to {save_path} (Size: {model_size:.2f} MB)")
        
        # Verify model
        verify_tflite_model(save_path, input_shape)
        return True
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def representative_dataset_gen(input_shape, num_samples=100):
    for _ in range(num_samples):
        data = np.random.randn(*input_shape).astype(np.float32)
        yield [data]

def verify_tflite_model(model_path, input_shape):
    try:
        logger.info("Verifying TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        output = interpreter.get_tensor(output_details[0]['index'])
        logger.info(f"Test inference successful - Output shape: {output.shape}")
        logger.info(f"Inference time: {inference_time*1000:.2f} ms")
        return True
        
    except Exception as e:
        logger.error(f"TFLite verification failed: {e}")
        return False
