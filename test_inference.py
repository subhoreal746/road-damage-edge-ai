import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a sample input (random or one of your images resized to model input size)
input_shape = input_details[0]['shape']
# Example with random input:
input_data = np.random.rand(*input_shape).astype(input_details[0]['dtype'])

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output shape:", output_data.shape)
print("Output sample:", output_data)

