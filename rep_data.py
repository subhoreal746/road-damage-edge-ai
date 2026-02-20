import cv2
import numpy as np
import glob

def representative_dataset():
    # Adjust this path to where your images are stored
    image_paths = glob.glob("/Users/subhojitbaidya/INT8/*.jpg")[:200]  # 140 images available

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))  # Your model input size
        img = img.astype(np.float32) / 255.0  # Normalize as per your training (usually /255)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        yield [img]

