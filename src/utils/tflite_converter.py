#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFLite Converter for LightHART-TF

Handles converting TensorFlow models to TFLite format with proper preprocessing
for accelerometer-only input, supporting inference on mobile and edge devices.
"""
import os
import logging
import shutil
import traceback
from typing import Tuple, Optional, Union
import tensorflow as tf
import numpy as np

logger = logging.getLogger('lighthart-tf')

def convert_to_tflite(model, save_path, input_shape=(1, 64, 3), quantize=False):
    """Convert model to TFLite format with accelerometer-only input.
    
    Args:
        model: TensorFlow model to convert
        save_path: Path to save the TFLite model
        input_shape: Input shape for accelerometer data (batch, frames, channels)
        quantize: Whether to apply quantization
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
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
def convert_to_tflite_quantized(
    model: tf.keras.Model, 
    save_path: str,
    input_shape: Tuple[int, int, int] = (1, 128, 3)
) -> bool:
    """Shortcut function to create a quantized TFLite model.
    
    Args:
        model: TensorFlow model to convert
        save_path: Path to save the TFLite model
        input_shape: Input shape for accelerometer data
        
    Returns:
        bool: Success status of the conversion
    """
    return convert_to_tflite(
        model=model,
        save_path=save_path,
        input_shape=input_shape,
        quantize=True,
        optimize=True
    )

def load_tflite_model(model_path: str) -> Optional[tf.lite.Interpreter]:
    """Load a TFLite model for inference.
    
    Args:
        model_path: Path to the TFLite model
        
    Returns:
        tf.lite.Interpreter or None: Loaded interpreter or None if loading failed
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        logger.error(f"Error loading TFLite model: {e}")
        return None

def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    accelerometer_data: np.ndarray
) -> Union[np.ndarray, None]:
    """Run inference with a TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        accelerometer_data: Raw accelerometer data with shape (frames, channels)
        
    Returns:
        np.ndarray or None: Inference result or None if inference failed
    """
    try:
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ensure input has batch dimension
        if len(accelerometer_data.shape) == 2:
            # Add batch dimension
            accelerometer_data = np.expand_dims(accelerometer_data, axis=0)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], accelerometer_data.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return output
    except Exception as e:
        logger.error(f"Error during TFLite inference: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="TFLite Converter Tool")
    parser.add_argument("--model", type=str, required=True, help="Path to TensorFlow model")
    parser.add_argument("--output", type=str, required=True, help="Output TFLite model path")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--frames", type=int, default=128, help="Number of frames")
    parser.add_argument("--channels", type=int, default=3, help="Number of accelerometer channels")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.model)
        logger.info(f"Model loaded from {args.model}")
        
        # Convert to TFLite
        success = convert_to_tflite(
            model=model,
            save_path=args.output,
            input_shape=(1, args.frames, args.channels),
            quantize=args.quantize
        )
        
        if success:
            logger.info("Conversion successful")
            
            # Test the model
            interpreter = load_tflite_model(args.output)
            if interpreter:
                dummy_input = np.random.randn(1, args.frames, args.channels).astype(np.float32)
                result = run_tflite_inference(interpreter, dummy_input)
                
                if result is not None:
                    logger.info(f"Test inference successful, output shape: {result.shape}")
        else:
            logger.error("Conversion failed")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
