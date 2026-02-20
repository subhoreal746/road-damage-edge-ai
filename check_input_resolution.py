import tensorflow as tf

saved_model_dir = "tf_model"  # relative path inside rdd_yolo folder
loaded = tf.saved_model.load(saved_model_dir)
infer = loaded.signatures['serving_default']

print("Input Signature:")
print(infer.structured_input_signature)

print("\nDetailed Input Tensors:")
for input_key, tensor_spec in infer.structured_input_signature[1].items():
    print(f"Input tensor name: {input_key}")
    print(f"Shape: {tensor_spec.shape}")
    print(f"Dtype: {tensor_spec.dtype}")

