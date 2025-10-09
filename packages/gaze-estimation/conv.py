# convert_to_onnx.py
import tensorflow as tf
import tf2onnx
import numpy as np
import os

# import your model builder from the original file where CNNResNetSWAttention is defined
# assume original file is named gaze_tf.py (adjust if different)
from a import CNNResNetSWAttention, BASE_PATH, model_path

def convert_keras_to_onnx(onnx_path="Models/rn_sw_attention.onnx", opset=13):
    # Build the Keras model architecture and load weights
    model = CNNResNetSWAttention.create_resnet18_sw__attention(
        input_shape=(60, 36, 1),
        bn=False,
        first_dense_units=512,
        fc_layer_units=[2048, 1024],
        debug=False,
    )
    model.load_weights(model_path)
    model.trainable = False  # inference mode

    # Optional: run once to build shapes
    dummy_r = np.zeros((1, 60, 36, 1), dtype=np.float32)
    dummy_l = np.zeros((1, 60, 36, 1), dtype=np.float32)
    model.predict([dummy_r, dummy_l])

    # Create input signature matching the Keras inputs
    input_signature = [
        tf.TensorSpec((None, 60, 36, 1), tf.float32, name="eye_right_input"),
        tf.TensorSpec((None, 60, 36, 1), tf.float32, name="eye_left_input"),
    ]

    # Convert and save
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=onnx_path,
    )
    print(f"Saved ONNX model to: {onnx_path}")

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    convert_keras_to_onnx(onnx_path=os.path.join("Models", "rn_sw_attention.onnx"), opset=13)
