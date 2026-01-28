from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose
from ..object_detection.post_processings import multiclass_nms


class RTMO(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = None,
                 std: tuple = None,
                 nms_thr: float = 0.45,
                 score_thr: float = 0.7,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray, nms_thr: float = None, score_thr: float = None):
        nms_thr = nms_thr if nms_thr is not None else self.nms_thr
        score_thr = score_thr if score_thr is not None else self.score_thr

        image, ratio = self.preprocess(image)
        outputs = self.inference(image)

        final_bboxes, keypoints, final_boxes_scores, keypoints_scores = self.postprocess(outputs, ratio, nms_thr, score_thr)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return final_bboxes, keypoints, final_boxes_scores, keypoints_scores

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
        nms_thr: float = None,
        score_thr: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMO model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        det_outputs, pose_outputs = outputs

        # onnx contains nms module (?)
        final_boxes, final_boxes_scores = (det_outputs[0, :, :4], det_outputs[0, :, 4])   # boxes scores
        final_boxes /= ratio
        keypoints, keypoints_scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio

        # apply nms
        dets, keep = multiclass_nms(final_boxes, 
                    final_boxes_scores[:, np.newaxis],
                    nms_thr=nms_thr,
                    score_thr=score_thr)
        if keep is not None:
            keypoints = keypoints[keep]
            keypoints_scores = keypoints_scores[keep]
            final_boxes = dets[:, :4]  # bboxes axis
            final_boxes_scores = final_boxes_scores[keep]
        else:
            keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
            keypoints_scores = np.expand_dims(np.zeros_like(keypoints_scores[0]), axis=0)   # keypoint scores
            final_boxes = np.expand_dims(np.zeros_like(final_boxes[0]), axis=0)
            final_boxes_scores = np.expand_dims(np.zeros_like(final_boxes_scores[0]), axis=0)     
            # final_bboxes = np.empty((0, 4), dtype=final_boxes.dtype)
            # final_boxes_scores = np.empty((0,), dtype=final_boxes_scores.dtype)


        return final_boxes, keypoints, final_boxes_scores, keypoints_scores

    def batch_preprocess(self, images: list):
        """Preprocess multiple images for batch inference.

        Args:
            images (list): List of input images, each in shape (H, W, 3).

        Returns:
            tuple:
            - batch_imgs (np.ndarray): Preprocessed images in shape (N, H, W, 3).
            - ratios (list): List of ratios for each image.
        """
        batch_imgs = []
        ratios = []
        
        for img in images:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114

            ratio = min(self.model_input_size[0] / img.shape[0],
                        self.model_input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

            # normalize image
            if self.mean is not None:
                mean = np.array(self.mean)
                std = np.array(self.std)
                padded_img = (padded_img - mean) / std

            batch_imgs.append(padded_img)
            ratios.append(ratio)
        
        return np.array(batch_imgs), ratios

    def batch_call(self, images: list, nms_thr: float = None, score_thr: float = None):
        """Process multiple images in batch.

        Args:
            images (list): List of input images.
            nms_thr (float): NMS threshold.
            score_thr (float): Score threshold.

        Returns:
            list: List of tuples (bboxes, keypoints, box_scores, keypoint_scores) for each image.
        """
        nms_thr = nms_thr if nms_thr is not None else self.nms_thr
        score_thr = score_thr if score_thr is not None else self.score_thr

        # Batch preprocess
        batch_imgs, ratios = self.batch_preprocess(images)
        
        # Batch inference
        outputs = self.batch_inference(batch_imgs)
        
        # outputs[0]: det_outputs shape (N, num_dets, 5)
        # outputs[1]: pose_outputs shape (N, num_dets, num_kpts, 3)
        det_outputs, pose_outputs = outputs
        
        # Postprocess each image
        results = []
        batch_size = len(images)
        
        for i in range(batch_size):
            ratio = ratios[i]
            
            # Extract per-image outputs
            det_out = det_outputs[i:i+1]  # (1, num_dets, 5)
            pose_out = pose_outputs[i:i+1]  # (1, num_dets, num_kpts, 3)
            
            final_boxes, keypoints, final_boxes_scores, keypoints_scores = self.postprocess(
                [det_out, pose_out], ratio, nms_thr, score_thr
            )
            
            if self.to_openpose:
                keypoints, keypoints_scores = convert_coco_to_openpose(keypoints, keypoints_scores)
            
            results.append((final_boxes, keypoints, final_boxes_scores, keypoints_scores))
        
        return results
