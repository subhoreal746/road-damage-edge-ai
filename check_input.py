import tensorflow as tf

saved_model_dir = "tf_model"
loaded = tf.saved_model.load(saved_model_dir)
infer = loaded.signatures['serving_default']
print(infer.structured_input_signature)

