import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Multiply,
    Concatenate,
    Conv2D,
    ReLU,
    Add,
    AveragePooling2D,
    Flatten,
)


def process_face_grid(face_grid_2d, face_grid_size):
    face_grid_2d = face_grid_2d.T
    face_grid_2d = face_grid_2d * face_grid_size
    face_grid_2d = face_grid_2d.astype(int)

    x_max, y_max = np.amax(face_grid_2d, axis=1)
    x_min, y_min = np.amin(face_grid_2d, axis=1)

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def relu_batch_normalization() -> list:
    return [ReLU(), BatchNormalization()]


def residual_block(downsample: bool, filters: int, kernel_size: int = 3) -> dict:
    block = {
        "conv_layers": [],
        "downsample_layer": None,
        "add_layer": None,
        "output_layers": [],
    }

    block["conv_layers"].append(
        Conv2D(
            kernel_size=kernel_size,
            strides=(1 if not downsample else 2),
            filters=filters,
            padding="same",
        )
    )
    block["conv_layers"] = block["conv_layers"] + relu_batch_normalization()
    block["conv_layers"].append(
        Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")
    )

    if downsample:
        block["downsample_layer"] = Conv2D(
            kernel_size=1, strides=2, filters=filters, padding="same"
        )

    block["add_layer"] = Add()
    block["output_layers"] = relu_batch_normalization()

    return block


def create_residual_blocks(num_blocks_list: list, num_filters: int) -> list:
    residual_blocks = []
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            residual_blocks.append(
                residual_block(downsample=(j == 0 and i != 0), filters=num_filters)
            )
        num_filters *= 2

    return residual_blocks


class ResNet18Model(Model):
    def __init__(self, num_filters: int = 64, num_blocks_list: list = None):
        super(ResNet18Model, self).__init__()
        if num_blocks_list is None:
            num_blocks_list = [2, 2, 2, 2]

        self.bn_1 = BatchNormalization()
        self.conv_1 = Conv2D(
            kernel_size=3, strides=1, filters=num_filters, padding="same"
        )
        self.relu_bn_1 = relu_batch_normalization()

        self.residual_blocks = create_residual_blocks(num_blocks_list, num_filters)

        self.pooling = AveragePooling2D(4)
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.bn_1(inputs)
        x = self.conv_1(x)

        for layer in self.relu_bn_1:
            x = layer(x)

        for block in self.residual_blocks:
            block_input = x

            if block["downsample_layer"] is not None:
                block_input = block["downsample_layer"](block_input)

            for layer in block["conv_layers"]:
                x = layer(x)

            if block["add_layer"] is not None:
                x = block["add_layer"]([x, block_input])

            for layer in block["output_layers"]:
                x = layer(x)

        x = self.pooling(x)
        x = self.flatten(x)

        return x


class CNNResNetSWAttention:
    @staticmethod
    def create_resnet18_sw__attention(
        input_shape: Tuple = (60, 36, 1),
        bn: bool = False,
        first_dense_units: int = 256,
        fc_layer_units: list = None,
        debug: bool = True,
    ) -> Model:
        fc_layer_units = fc_layer_units if fc_layer_units is not None else [256, 512]
        if debug:
            print(
                "!Note: Using model ResNet18Model with shared weights between the two eyes"
            )

        cnn_feature_extractor = ResNet18Model()
        cnn_attention_extractor = ResNet18Model()

        eye_right_input = Input(input_shape)
        eye_right_out = cnn_feature_extractor(eye_right_input)
        eye_right_out = Dense(units=first_dense_units, activation="relu")(eye_right_out)
        eye_right_out = BatchNormalization()(eye_right_out)

        eye_right_attention = cnn_attention_extractor(eye_right_input)
        eye_right_attention = Dense(units=first_dense_units, activation="sigmoid")(
            eye_right_attention
        )

        eye_right_out = Multiply()([eye_right_out, eye_right_attention])

        eye_left_input = Input(input_shape)
        eye_left_out = cnn_feature_extractor(eye_left_input)
        eye_left_out = Dense(units=first_dense_units, activation="relu")(eye_left_out)
        eye_left_out = BatchNormalization()(eye_left_out)

        eye_left_attention = cnn_attention_extractor(eye_left_input)
        eye_left_attention = Dense(units=first_dense_units, activation="sigmoid")(
            eye_left_attention
        )

        eye_left_out = Multiply()([eye_left_out, eye_left_attention])

        concat_layer = Concatenate(axis=1)
        output = concat_layer([eye_right_out, eye_left_out])

        for layer_units in fc_layer_units:
            output = Dense(units=layer_units, activation="relu")(output)
            if bn:
                output = BatchNormalization()(output)

        output = Dense(units=2, activation="sigmoid")(output)

        return Model(inputs=[eye_right_input, eye_left_input], outputs=output)


model_path = "Models/rn_sw_attention__tf_model.h5"

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


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
    convert_keras_to_onnx(
        onnx_path=os.path.join("Models", "rn_sw_attention.onnx"), opset=13
    )
