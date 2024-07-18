import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# model = transformer()

# #load weight
# weight_path = f'exp/smartfallmm/models/transformer
# model.load_weights(weight_path)
def lite_converter(file_path : str) -> None : 
  '''
    Funciton to converter to tf lite
  '''
  model = tf.keras.models.load_model(file_path, compile = True)

  #convert to tflite
  converter = tf.lite.TFLiteConverter.from_keras_model(model)

  #adding special ops
  # converter.target_spec.supported_ops = [
  #   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  #   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops. <-- Add this line
  # ]
  # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]


  #converting to tflite 
  tflite_model = converter.convert()

  #writing the tflite model to a file 
  with open(f'{file_path}.tflite', 'wb') as f:
      f.write(tflite_model)

  # X_test, y_test = process_data(TEST, WINDOW, STRIDE)
  X_test = tf.random.normal([32, 128, 3])
  y_test = tf.random.uniform(shape = [32], minval = 0, maxval = 0)
  #using interpreter to test
  interpreter = tf.lite.Interpreter(model_path=f"{file_path}.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  data = X_test[1, :, :]
  data = data[np.newaxis, :]

  # Set input tensor to the interpreter
  interpreter.set_tensor(input_details[0]['index'], data)
  # Run inference
  interpreter.invoke()

  # Get the output tensor and post-process the results (example)
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print("Inference result:", output_data)


if __name__ =='__main__':
   lite_converter(file_path='exp/smartfallmm/test/models/transformer')