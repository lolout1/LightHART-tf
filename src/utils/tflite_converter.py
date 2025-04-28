import os
import logging
import tensorflow as tf
import numpy as np
import time

def convert_to_tflite(model, save_path, input_shape=(1, 128, 3), quantize=False, 
                     use_lite_runtime=False, conversion_timeout=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not save_path.endswith('.tflite'):
        save_path += '.tflite'
    
    logging.info(f"Starting TFLite conversion for {save_path}")
    start_time = time.time()
    
    try:
        inputs = tf.keras.Input(shape=input_shape[1:], name='accelerometer_input')
        
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        zero_mean = inputs - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        processed = tf.concat([smv, inputs], axis=-1)
        
        model_inputs = {'accelerometer': processed}
        outputs = model(model_inputs, training=False)
        
        if isinstance(outputs, tuple) and len(outputs) > 0:
            outputs = outputs[0]
        
        tflite_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        saved_model_dir = os.path.join(os.path.dirname(save_path), f"temp_saved_model_{int(time.time())}")
        tflite_model.save(saved_model_dir)
        logging.info(f"Temporary SavedModel created at {saved_model_dir}")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        
        if use_lite_runtime:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        
        logging.info("Running TFLite conversion...")
        tflite_buffer = converter.convert()
        
        with open(save_path, 'wb') as f:
            f.write(tflite_buffer)
        
        logging.info(f"TFLite model saved to {save_path} (in {time.time() - start_time:.2f}s)")
        
        if os.path.exists(saved_model_dir):
            import shutil
            shutil.rmtree(saved_model_dir)
            logging.info(f"Temporary SavedModel removed")
            
        return True
    except Exception as e:
        logging.error(f"TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tflite_model(tflite_path, input_shape=(1, 128, 3)):
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.info(f"TFLite model input: {input_details}")
        logging.info(f"TFLite model output: {output_details}")
        
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logging.info(f"TFLite test successful - output shape: {output.shape}")
        return True
    except Exception as e:
        logging.error(f"Error testing TFLite model: {e}")
        return False
