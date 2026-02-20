import cv2
import numpy as np
import os
import time

# --- CONFIG ---
VIDEO_PATH = "Road_Classes.mp4" 
MODEL_PATH = "best_full_integer_quant.tflite"
CONF_THRESHOLD = 0.20 # Based on your 0.36 HIT, this is safe

def main():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    o_scale, o_zero = output_details[0]['quantization']

    cap = cv2.VideoCapture(VIDEO_PATH)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret: break
        t_start = time.time()
        
        # Pre-process
        resized = cv2.resize(frame, (320, 320))
        input_data = (resized.astype(np.float32) / 255.0 * 255 - 128).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data, axis=0))
        interpreter.invoke()
        
        # Post-process
        output = (interpreter.get_tensor(output_details[0]['index']).astype(np.float32) - o_zero) * o_scale
        output = np.squeeze(output)
        if output.shape[0] < output.shape[1]: output = output.T

        # Get Best Detection
        conf_scores = output[:, 4:]
        max_idx = np.argmax(conf_scores)
        row_idx, col_idx = divmod(max_idx, conf_scores.shape[1])
        best_conf = conf_scores[row_idx, col_idx]

        # --- DRAWING LOGIC ---
        if best_conf > CONF_THRESHOLD:
            row = output[row_idx]
            cx, cy, w, h = row[:4]

            # AUTO-DETECTION OF COORDINATE TYPE
            # If values are very small (0-1), they are normalized. 
            # If they are large (0-320), they are pixel-based.
            if cx < 1.0 and cy < 1.0:
                # Normalized -> Multiply by full resolution
                x1 = int((cx - w/2) * w_orig)
                y1 = int((cy - h/2) * h_orig)
                x2 = int((cx + w/2) * w_orig)
                y2 = int((cy + h/2) * h_orig)
            else:
                # Pixel-based (0-320) -> Scale to resolution
                x1 = int((cx - w/2) * (w_orig / 320))
                y1 = int((cy - h/2) * (h_orig / 320))
                x2 = int((cx + w/2) * (w_orig / 320))
                y2 = int((cy + h/2) * (h_orig / 320))

            # Draw Neon Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"ANOMALY {best_conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Overlay
        fps = 1.0 / (time.time() - t_start)
        cv2.rectangle(frame, (0,0), (300, 80), (0,0,0), -1)
        cv2.putText(frame, f"CONF: {best_conf:.2f}", (10, 30), 1, 1.5, (255,255,255), 2)
        cv2.putText(frame, f"FPS: {round(fps,1)}", (10, 65), 1, 1.5, (255,255,255), 2)

        cv2.imshow("ARM Bharat Final Validation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
