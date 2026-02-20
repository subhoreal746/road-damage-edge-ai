import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(100):
        # YOLO input size (adjust if your model differs)
        data = np.random.rand(1, 640, 640, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open("best_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved as best_int8.tflite")

