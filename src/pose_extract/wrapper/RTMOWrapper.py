import time

import cv2
import imageio
import numpy as np

# from rtmlib import Body, draw_skeleton
from vendor.rtmlib.rtmlib import RTMO
from config.config import ConfigManager

class RTMOWrapper:
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

    def __init__(self, config:ConfigManager, bond_filter_ratio=0.01):
        self._session_active = True
        # config = ConfigManager()
        weight_path = config.get("weight_path", "./models/rtmo-l/end2end.onnx")
        input_size = config.get("input_size")
        backend = 'onnxruntime'
        openpose_skeleton=False
        device='cuda'

        self.bond_filter_ratio = bond_filter_ratio

        self.pose_model = RTMO(
            onnx_model=weight_path,
            to_openpose=openpose_skeleton,  # True for openpose-style, False for mmpose-style
            backend=backend,
            device=device)

    def __call__(self, image: np.ndarray):
        final_bboxes, keypoints, final_boxes_scores, keypoints_scores = self.pose_model(image)

        # 邊界篩選
        filtered_bonding_boxes = []
        filtered_keypoints = []
        filtered_boxes_scores = []
        filtered_keypoints_scores = []
        height, width = image.shape[:2]
        for i, bbox in enumerate(final_bboxes):
            x1, y1, x2, y2 = bbox
            if x1 > width * self.bond_filter_ratio and \
            y1 > height * self.bond_filter_ratio and \
            x2 < width * (1 - self.bond_filter_ratio) and \
            y2 < height * (1 - self.bond_filter_ratio):
                filtered_bonding_boxes.append(bbox)
                filtered_keypoints.append(keypoints[i])
                filtered_boxes_scores.append(final_boxes_scores[i])
                filtered_keypoints_scores.append(keypoints_scores[i])

        if len(filtered_bonding_boxes) > 0:
            filtered_bonding_boxes = np.array(filtered_bonding_boxes)
            filtered_keypoints = np.array(filtered_keypoints)
            filtered_boxes_scores = np.array(filtered_boxes_scores)
            filtered_keypoints_scores = np.array(filtered_keypoints_scores)
        else:
            # Create empty arrays with the correct dimensions
            filtered_bonding_boxes = np.expand_dims(np.zeros_like(final_bboxes[0]), axis=0)
            filtered_keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
            filtered_boxes_scores = np.expand_dims(np.zeros_like(final_boxes_scores[0]), axis=0)
            filtered_keypoints_scores = np.expand_dims(np.zeros_like(keypoints_scores[0]), axis=0)

        return filtered_bonding_boxes, filtered_keypoints, filtered_boxes_scores, filtered_keypoints_scores

    def batch_inference(self, images: list):
        """Batch inference for multiple images.
        
        Args:
            images: List of images (np.ndarray), each in shape (H, W, 3).
            
        Returns:
            list: List of tuples (bboxes, keypoints, box_scores, keypoint_scores) for each image,
                  with edge filtering applied.
        """
        # Get batch results from underlying model
        batch_results = self.pose_model.batch_call(images)
        
        # Apply edge filtering to each result
        filtered_results = []
        for i, (bboxes, keypoints, box_scores, kpt_scores) in enumerate(batch_results):
            height, width = images[i].shape[:2]
            
            filtered_bboxes = []
            filtered_kpts = []
            filtered_box_scores = []
            filtered_kpt_scores = []
            
            for j, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                if x1 > width * self.bond_filter_ratio and \
                   y1 > height * self.bond_filter_ratio and \
                   x2 < width * (1 - self.bond_filter_ratio) and \
                   y2 < height * (1 - self.bond_filter_ratio):
                    filtered_bboxes.append(bbox)
                    filtered_kpts.append(keypoints[j])
                    filtered_box_scores.append(box_scores[j])
                    filtered_kpt_scores.append(kpt_scores[j])
            
            if len(filtered_bboxes) > 0:
                filtered_bboxes = np.array(filtered_bboxes)
                filtered_kpts = np.array(filtered_kpts)
                filtered_box_scores = np.array(filtered_box_scores)
                filtered_kpt_scores = np.array(filtered_kpt_scores)
            else:
                # Create empty arrays with correct dimensions
                filtered_bboxes = np.expand_dims(np.zeros_like(bboxes[0]), axis=0)
                filtered_kpts = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
                filtered_box_scores = np.expand_dims(np.zeros_like(box_scores[0]), axis=0)
                filtered_kpt_scores = np.expand_dims(np.zeros_like(kpt_scores[0]), axis=0)
            
            filtered_results.append((filtered_bboxes, filtered_kpts, filtered_box_scores, filtered_kpt_scores))
        
        return filtered_results
    
    def close(self):
        """顯式釋放 ONNX 會話與 CUDA 資源"""
        if hasattr(self, 'pose_model') and hasattr(self.pose_model, 'session'):
            del self.pose_model.session
            if 'cuda' in getattr(self.pose_model, 'device', ''):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    # Python 正在關閉或 torch 不可用，忽略錯誤
                    pass
        self._session_active = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Python 正在關閉時忽略清理錯誤
            pass