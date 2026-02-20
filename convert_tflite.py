import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Path to your SavedModel directory
saved_model_dir = "tf_model"

# Path to folder with calibration images (must contain sample images)
calibration_images_folder = "/Users/subhojitbaidya/INT8"

# Number of calibration samples to use (can be less than total images if you want)
NUM_CALIBRATION_IMAGES = 140

def representative_data_gen():
    images = os.listdir(calibration_images_folder)
    count = 0
    for image_name in images:
        if count >= NUM_CALIBRATION_IMAGES:
            break
        image_path = os.path.join(calibration_images_folder, image_name)

        # Load image with PIL
        img = Image.open(image_path).convert('RGB')

        # Resize to model input size: 640x640
        img = img.resize((640, 640))

        # Convert to numpy array and normalize (0-1 float32)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Add batch dimension: (1, 640, 640, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Yield the image as a list (TFLite expects a list of inputs)
        yield [img_array]

        count += 1


def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Set optimization flag for full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for calibration
    converter.representative_dataset = representative_data_gen

    # Ensure input and output types are int8 for full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Convert the model
    tflite_quant_model = converter.convert()

    # Save the quantized TFLite model
    with open("best_int8_640.tflite", "wb") as f:
        f.write(tflite_quant_model)
    print("INT8 quantized TFLite model saved as best_int8_640.tflite")



if __name__ == "__main__":
    convert_to_tflite()

