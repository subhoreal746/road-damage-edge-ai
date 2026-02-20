import os
import cv2
import numpy as np
import tensorflow as tf

# Paths - update these as needed
model_path = "best_int8.tflite"
input_folder = "/Users/subhojitbaidya/INT8"
output_folder = "output_results"

# Create output folder if doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = img_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, img

def postprocess_output(output_data, conf_threshold=0.5):
    # output_data shape: (1, 8, 8400)
    detections = output_data[0].T  # (8400, 8)
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        x, y, w, h, conf = detection[:5]
        class_probs = detection[5:]
        class_id = np.argmax(class_probs)
        confidence = conf * class_probs[class_id]

        if confidence > conf_threshold:
            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

    return boxes, confidences, class_ids

def draw_boxes(image, boxes, confidences, class_ids):
    h_img, w_img = image.shape[:2]
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x, y, w, h = box
        # Convert from normalized center x,y,w,h to box coordinates
        left = int((x - w/2) * w_img)
        top = int((y - h/2) * h_img)
        right = int((x + w/2) * w_img)
        bottom = int((y + h/2) * h_img)

        # Clamp coordinates
        left = max(0, left)
        top = max(0, top)
        right = min(w_img - 1, right)
        bottom = min(h_img - 1, bottom)

        color = (0, 255, 0)  # Green box
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label = f"ID:{cls_id} {conf:.2f}"
        cv2.putText(image, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def run_inference_on_folder():
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(input_folder, filename)
        input_data, original_img = preprocess_image(image_path)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, confidences, class_ids = postprocess_output(output_data)

        img_with_boxes = draw_boxes(original_img, boxes, confidences, class_ids)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img_with_boxes)
        print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    run_inference_on_folder()

