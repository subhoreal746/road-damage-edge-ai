from onnxruntime.quantization import quantize_dynamic, QuantType

fp32_model = "runs/detect/train2/weights/best.onnx"
int8_model = "runs/detect/train2/weights/best_int8.onnx"

quantize_dynamic(
    model_input=fp32_model,
    model_output=int8_model,
    weight_type=QuantType.QInt8
)

print("✅ INT8 quantized model saved as best_int8.onnx")

