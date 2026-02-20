import os
import cv2
import numpy as np
import tensorflow as tf

# Define your class names (adjust according to your model)
class_names = ["class0", "class1", "class2", "class3", "class4", "class5", "class6", "class7"]

# Threshold to filter weak detections
CONFIDENCE_THRESHOLD = 0.5

def load_image(image_path, input_shape):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    return img, input_data

def postprocess_output(output_data, conf_threshold=CONFIDENCE_THRESHOLD):
    # Your model outputs shape (1, 8, 8400) or similar. 
    # Adjust this parsing logic depending on your model output format
    
    detections = []  # will hold boxes, confidences, class_ids

    # Example dummy parsing: 
    # This must be replaced by actual parsing logic based on your model output
    # For example, each detection may contain [x, y, w, h, confidence, class_id, ...]
    for det in output_data[0]:
        # If your output format is different, adjust unpacking accordingly
        if len(det) < 6:
            continue
        x, y, w, h, conf, class_id = det[:6]
        if conf >= conf_threshold:
            detections.append((int(x), int(y), int(w), int(h), conf, int(class_id)))

    if not detections:
        return [], [], []

    boxes, confidences, class_ids = zip(*[(d[0:4], d[4], d[5]) for d in detections])
    return boxes, confidences, class_ids

def draw_boxes(image, boxes, confidences, class_ids, class_names, conf_threshold=CONFIDENCE_THRESHOLD):
    for i, box in enumerate(boxes):
        if confidences[i] >= conf_threshold:
            x, y, w, h = box
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)  # green box

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Draw label text above the box
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    return image

def run_inference_on_folder(model_path="best_int8.tflite",
                            input_folder="Users/subhojitbaidya/INT8",
                            output_folder="output_results"):

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(input_folder, filename)
        original_img, input_data = load_image(image_path, input_details[0]['shape'])

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, confidences, class_ids = postprocess_output(output_data)

        if boxes:
            # Draw bounding boxes on original image (BGR)
            output_img = draw_boxes(original_img, boxes, confidences, class_ids, class_names)
        else:
            output_img = original_img

        # Save output image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, output_img)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    run_inference_on_folder()

