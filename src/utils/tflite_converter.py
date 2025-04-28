#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import tensorflow as tf
import numpy as np

def convert_to_tflite_model(model, save_path, input_shape=(1, 64, 3)):
    """Create a TFLite-compatible model and convert it"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract dimensions
        batch_size, sequence_length, num_features = input_shape
        
        # Create standalone model for TFLite
        inputs = tf.keras.Input(shape=(sequence_length, num_features), name='input')
        
        # Calculate SMV
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        zero_mean = inputs - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        processed = tf.concat([smv, inputs], axis=-1)
        
        # Create model inputs dict
        model_inputs = {'accelerometer': processed}
        
        # Forward pass
        outputs = model(model_inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Create new model
        tflite_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
        
        # Set optimization and options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        
        # Convert model
        tflite_buffer = converter.convert()
        
        # Save model
        with open(save_path, 'wb') as f:
            f.write(tflite_buffer)
        
        return True
        
    except Exception as e:
        logging.error(f"TFLite conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tflite_model(tflite_path, test_input=None):
    """Test a TFLite model with sample input"""
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input if not provided
        if test_input is None:
            input_shape = input_details[0]['shape']
            test_input = np.random.random(input_shape).astype(np.float32)
        elif len(test_input.shape) < len(input_details[0]['shape']):
            # Add batch dimension if needed
            test_input = np.expand_dims(test_input, axis=0)
        
        # Make sure input is float32
        test_input = test_input.astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return output
    except Exception as e:
        logging.error(f"Error testing TFLite model: {e}")
        return None
