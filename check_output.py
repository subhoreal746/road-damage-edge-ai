import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input, or load a sample image as input
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(input_details[0]['dtype'])

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output shape:", output_data.shape)
print("Output sample (first 10 elements):", output_data[0][:10])

