# 🛣️ AI-Powered Real-Time Road Anomaly Detection
**Project for ARM Bharat | Optimized for Edge Intelligence**

This repository contains an end-to-end pipeline for detecting road damage (potholes and cracks) using a custom-trained YOLOv8 model optimized for ARM-based hardware (Mac Silicon & Raspberry Pi).



## 🚀 Key Achievements
* **Phase 4 (Validation):** Achieved **95+ FPS** on Mac M-Series hardware using TFLite optimization.
* **Phase 5 (Deployment):** Successful real-time deployment on **Raspberry Pi** with a stable **5.6 FPS**, suitable for mobile road monitoring.
* **Quantization:** Converted FP32 weights to **INT8 Full Integer Quantization**, reducing model size by **75%** (to 3.1MB) for maximum ARM NEON efficiency.

## 📂 Project Structure
* `deployment/`: Contains `mac_validation.py` for testing and the production scripts.
* `model_optimization/`: The home of our `best_int8.tflite` ARM-optimized brain.
* `data/`: Sample road footage for verification.
* `hardware_config/`: Scripts for locking CPU frequency to prevent thermal throttling on the Pi.

## 🛠️ Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/subhoreal746/road-damage-edge-ai.git](https://github.com/subhoreal746/road-damage-edge-ai.git)
   cd road-damage-edge-ai
