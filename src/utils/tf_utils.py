# src/utils/tf_utils.py
"""TensorFlow utility functions for LightHART-TF."""
import tensorflow as tf
import os
import logging

def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            return gpus
        except RuntimeError as e:
            logging.warning(f"Error configuring GPU: {e}")
    logging.info("No GPU found, using CPU")
    return []

def enable_mixed_precision(enabled=True):
    """Enable mixed precision training if requested."""
    if enabled:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info(f"Mixed precision enabled with policy: {policy}")

def build_model_with_inputs(model, input_shapes):
    """Build model with dummy inputs to ensure it's properly initialized."""
    try:
        # Create dummy batch
        inputs = {}
        for name, shape in input_shapes.items():
            inputs[name] = tf.zeros((2,) + shape, dtype=tf.float32)
        
        # Forward pass to build the model
        _ = model(inputs, training=False)
        
        # Verify the model was built successfully
        if hasattr(model, 'built') and model.built:
            logging.info("Model successfully built")
        else:
            logging.warning("Model may not be fully built")
        
        return True
    except Exception as e:
        logging.error(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_model_weights(model, filepath):
    """Save model weights with proper file extension."""
    try:
        # Ensure the filepath has the correct extension
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights
        model.save_weights(filepath)
        logging.info(f"Model weights saved to {filepath}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving model weights: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_tflite(model, filepath):
    """Export model to TFLite format."""
    try:
        # Ensure the filepath has the correct extension
        if not filepath.endswith('.tflite'):
            filepath = filepath + '.tflite'
        
        # Create converter from Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
            
        logging.info(f"TFLite model exported to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error exporting TFLite model: {e}")
        return False
