import os
import sys
import time
from typing import Tuple

import cv2
import dlib
import numpy as np
import scipy.io as sio
import tensorflow as tf
from imutils import face_utils
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

CAMERA_MATRIX = np.array(
    [
        [1.49454593e03, 0.0, 9.55794289e02],
        [0.0, 1.49048883e03, 5.18040731e02],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CAMERA_DISTORTION = np.array(
    [[0.08599595, -0.37972518, -0.0059906, -0.00468435, 0.45227431]],
    dtype=np.float64,
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
        self.conv_1 = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")
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


class MixedNormalizer:
    def __init__(
        self,
        eye_roi_size=(60, 36),
        eyes_together=False,
        face_grid_size=25,
        equalization="Hist",
        base_path=".",
    ):
        model_file = os.path.join(
            base_path, "Models", "OpencvDNN", "res10_300x300_ssd_iter_140000.caffemodel"
        )
        model_config_file = os.path.join(
            base_path, "Models", "OpencvDNN", "deploy.prototxt.txt"
        )

        self.detector = cv2.dnn.readNetFromCaffe(model_config_file, model_file)
        self.predictor = dlib.shape_predictor(
            os.path.join(
                base_path, "ShapePredictors", "shape_predictor_68_face_landmarks.dat"
            )
        )

        self.confidence_threshold = 0.5

        self.camera_calibration = None
        self.camera_matrix = None
        self.camera_distortion = None

        self.focal_length_norm = 960
        self.distance_norm = 600
        self.eye_roi_size = eye_roi_size
        self.eyes_together = eyes_together

        self.generic_3d_face_coordinates = np.array(
            [
                [
                    -45.0967681126441,
                    -21.3128582097374,
                    21.3128582097374,
                    45.0967681126441,
                    -26.2995769055718,
                    26.2995769055718,
                ],
                [
                    -0.483773045049757,
                    0.483773045049757,
                    0.483773045049757,
                    -0.483773045049757,
                    68.5950352778326,
                    68.5950352778326,
                ],
                [
                    2.39702984214363,
                    -2.39702984214363,
                    -2.39702984214363,
                    2.39702984214363,
                    -9.86076131526265 * (10 ** -32),
                    -9.86076131526265 * (10 ** -32),
                ],
            ]
        )
        self.generic_3d_face_coordinates_T = self.generic_3d_face_coordinates.T

        self.face_grid_size = face_grid_size
        self.equalization = equalization
        self.clahe = cv2.createCLAHE() if self.equalization == "Clahe" else None

    def load_calibration_parameters(self, camera_file):
        self.camera_calibration = sio.loadmat(camera_file)
        self.camera_matrix = self.camera_calibration["cameraMatrix"]
        self.camera_distortion = self.camera_calibration["distCoeffs"]

    def set_calibration_parameters(self, camera_matrix, camera_distortion):
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

    def undistort_image(self, img):
        return cv2.undistort(img, self.camera_matrix, self.camera_distortion)

    def estimate_head_pose(self, face_2d):
        _, rot_vec, trans_vec = cv2.solvePnP(
            self.generic_3d_face_coordinates_T,
            face_2d,
            self.camera_matrix,
            self.camera_distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        _, rot_vec, trans_vec = cv2.solvePnP(
            self.generic_3d_face_coordinates_T,
            face_2d,
            self.camera_matrix,
            self.camera_distortion,
            rot_vec,
            trans_vec,
            True,
        )

        return rot_vec, trans_vec

    def retrieve_eyes(self, rot_vec, trans_vec):
        head_translation = trans_vec.reshape((3, 1))
        head_rotation = cv2.Rodrigues(rot_vec)[0]

        face_landmarks_3d = np.dot(head_rotation, self.generic_3d_face_coordinates) + head_translation

        right_eye = 0.5 * (face_landmarks_3d[:, 0] + face_landmarks_3d[:, 1]).reshape((3, 1))
        left_eye = 0.5 * (face_landmarks_3d[:, 2] + face_landmarks_3d[:, 3]).reshape((3, 1))

        return [right_eye, left_eye], head_rotation

    def retrieve_eyes_together(self, rot_vec, trans_vec):
        head_translation = trans_vec.reshape((3, 1))
        head_rotation = cv2.Rodrigues(rot_vec)[0]
        face_landmarks_3d = np.dot(head_rotation, self.generic_3d_face_coordinates) + head_translation
        eyes_center = 0.5 * (face_landmarks_3d[:, 0] + face_landmarks_3d[:, 3]).reshape((3, 1))
        return eyes_center, head_rotation

    def normalize_eye(self, eye, head_rotation, img_gray):
        distance = np.linalg.norm(eye)
        z_scale = self.distance_norm / distance

        camera_norm = np.array(
            [
                [self.focal_length_norm, 0, self.eye_roi_size[0] / 2],
                [0, self.focal_length_norm, self.eye_roi_size[1] / 2],
                [0, 0, 1.0],
            ]
        )

        scaling_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, z_scale],
            ]
        )

        forward = (eye / distance).reshape(3)
        down = np.cross(forward, head_rotation[:, 0])
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)

        rotation_matrix = np.c_[right, down, forward].T

        transformation_matrix = np.dot(
            np.dot(camera_norm, scaling_matrix),
            np.dot(rotation_matrix, np.linalg.inv(self.camera_matrix)),
        )

        img_warped = cv2.warpPerspective(img_gray, transformation_matrix, self.eye_roi_size)

        if self.equalization == "Hist":
            img_warped = cv2.equalizeHist(img_warped)
        elif self.equalization == "Clahe" and self.clahe is not None:
            img_warped = self.clahe.apply(img_warped)

        return img_warped

    def detect_face(self, img):
        img_height, img_width = img.shape[:2]
        img_blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
        )

        self.detector.setInput(img_blob)

        faces = self.detector.forward()
        for idx in range(faces.shape[2]):
            confidence = faces[0, 0, idx, 2]

            if confidence > self.confidence_threshold:
                x1 = faces[0, 0, idx, 3]
                y1 = faces[0, 0, idx, 4]
                x2 = faces[0, 0, idx, 5]
                y2 = faces[0, 0, idx, 6]

                bounding_box = dlib.rectangle(
                    int(x1 * img_width),
                    int(y1 * img_height),
                    int(x2 * img_width),
                    int(y2 * img_height),
                )

                face_grid_2d = np.array(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x1, y2],
                        [x2, y2],
                    ],
                    dtype=np.float64,
                )

                return bounding_box, face_grid_2d

        raise Exception("[MixedNormalizer] - No face could be detected!")

    def normalize_image(self, file_name):
        img = cv2.imread(file_name)
        img = self.undistort_image(img)
        bounding_box, face_grid_2d = self.detect_face(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_grid = process_face_grid(face_grid_2d, self.face_grid_size)
        landmarks = face_utils.shape_to_np(self.predictor(img, bounding_box))
        face_2d = np.array(
            [
                landmarks[36],
                landmarks[39],
                landmarks[42],
                landmarks[45],
                landmarks[48],
                landmarks[54],
            ],
            dtype=np.float64,
        )

        rot_vec, trans_vec = self.estimate_head_pose(face_2d)

        if self.eyes_together:
            eyes_center, head_rotation = self.retrieve_eyes_together(rot_vec, trans_vec)
            processed_eyes = self.normalize_eye(eyes_center, head_rotation, img)
            eye_screen_distance = np.linalg.norm(eyes_center)

            return (
                processed_eyes,
                face_grid,
                eye_screen_distance,
            )

        eyes, head_rotation = self.retrieve_eyes(rot_vec, trans_vec)
        processed_eyes = []
        eye_screen_distance = np.mean(
            [np.linalg.norm(eyes[0]), np.linalg.norm(eyes[1])]
        )

        for eye in eyes:
            processed_eyes.append(self.normalize_eye(eye, head_rotation, img))

        return processed_eyes, face_grid, eye_screen_distance


model_path = "Models/rn_sw_attention__tf_model.h5"

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


def detect_screen_resolution():
    root = None
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        return width, height
    except Exception as exc:
        raise RuntimeError("Unable to detect primary screen resolution") from exc
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def load_rage_net_model():
    bn = False
    first_dense_units = 512
    fc_layer_units = [2048, 1024]
    lr = [5e-6, 2.5e-4]

    cnn_model = CNNResNetSWAttention.create_resnet18_sw__attention(
        input_shape=(60, 36, 1),
        bn=bn,
        first_dense_units=first_dense_units,
        fc_layer_units=fc_layer_units,
        debug=False,
    )

    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    cnn_model.load_weights(model_path)
    return cnn_model


def create_normalizer():
    normalizer = MixedNormalizer(base_path=BASE_PATH)
    normalizer.set_calibration_parameters(CAMERA_MATRIX, CAMERA_DISTORTION)
    return normalizer


def normalize_frame(normalizer, frame_bgr):
    undistorted = normalizer.undistort_image(frame_bgr)
    bounding_box, _ = normalizer.detect_face(undistorted)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    landmarks = face_utils.shape_to_np(normalizer.predictor(gray, bounding_box))
    face_2d = np.array(
        [
            landmarks[36],
            landmarks[39],
            landmarks[42],
            landmarks[45],
            landmarks[48],
            landmarks[54],
        ],
        dtype=np.float64,
    )
    rot_vec, trans_vec = normalizer.estimate_head_pose(face_2d)
    eyes, head_rotation = normalizer.retrieve_eyes(rot_vec, trans_vec)
    processed_eyes = [
        normalizer.normalize_eye(eye, head_rotation, gray) for eye in eyes
    ]
    return processed_eyes


def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (36, 60), interpolation=cv2.INTER_LINEAR)
    eye = eye.astype(np.float32) / 255.0
    eye = (eye - 128.0) / 128.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye


def main():
    cnn_model = load_rage_net_model()
    normalizer = create_normalizer()
    try:
        screen_width, screen_height = detect_screen_resolution()
    except RuntimeError as exc:
        print(f"Screen resolution detection failed: {exc}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        sys.exit()

    frame_rate = 30
    prev = 0
    gaze_display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    while True:
        time_elapsed = time.time() - prev
        if time_elapsed > 1.0 / frame_rate:
            prev = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            gaze_point = None
            try:
                right_eye_img, left_eye_img = normalize_frame(normalizer, frame)
                right_eye_proc = preprocess_eye(right_eye_img)
                left_eye_proc = preprocess_eye(left_eye_img)

                pred = cnn_model.predict([right_eye_proc, left_eye_proc], verbose=0)

                gaze_x = pred[0][0] * screen_width
                gaze_y = pred[0][1] * screen_height
                gaze_point = (
                    int(np.clip(gaze_x, 0, screen_width - 1)),
                    int(np.clip(gaze_y, 0, screen_height - 1)),
                )

                print(f"Gaze point: ({gaze_x:.2f}, {gaze_y:.2f})")
            except Exception as exc:
                print(f"Tracking lost: {exc}")

            cv2.imshow("Webcam", frame)
            gaze_display[:] = 0
            if gaze_point is not None:
                cv2.circle(gaze_display, gaze_point, 15, (0, 0, 255), -1)
            cv2.imshow("Gaze Point", gaze_display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
