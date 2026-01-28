'''
Example:

import cv2

from rtmlib import Body, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

body = Body(to_openpose=openpose_skeleton,
                      backend=backend,
                      device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''
import numpy as np
import os

class Body:
    MODE = {
        'performance': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',  # noqa
            'det_input_size': (416, 416),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    RTMO_MODE = {
        'performance': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        },
        'lightweight': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        },
        'balanced': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        }
    }

    def __init__(self,
                 pose_input_size: tuple = (288, 384),
                 weight: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
            
        from .. import RTMO

        if weight in self.RTMO_MODE:
            pose = self.RTMO_MODE[weight]['pose']
            pose_input_size = self.RTMO_MODE[weight]['pose_input_size']
        else:
            if not (os.path.exists(weight)):
                raise ValueError("Weight must be one of the predefined options in RTMO_MODE or a valid .zip file path.")
            pose = weight

        pose_input_size = self.RTMO_MODE[weight]['pose_input_size']
        self.pose_model = RTMO(pose,
                                model_input_size=pose_input_size,
                                to_openpose=to_openpose,
                                backend=backend,
                                device=device)

    def __call__(self, image: np.ndarray):
        bboxes, keypoints, scores = self.pose_model(image)

        return bboxes, keypoints, scores
