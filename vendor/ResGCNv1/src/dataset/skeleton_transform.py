"""
Skeleton Transform Pipeline

Shared data transformation for training and inference.
Ensures consistent preprocessing between coco.py Feeder and processor.predict().
"""

import numpy as np
from typing import Optional, List

from .data_utils import multi_input


class SkeletonTransform:
    """Skeleton data transformation pipeline
    
    Transforms raw skeleton data (C, T, V, M) to model input format (4, 6, T, V, M).
    Ensures identical processing for training, evaluation, and inference.
    
    Processing steps:
    1. Mask eye/ear joints (indices 1,2,3,4 in COCO format)
    2. Apply multi_input to generate 4-branch, 6-channel format
       (joint, velocity, bone, acceleration)
    3. Optional augmentation (training only)
    """
    
    # COCO joint indices to mask (eyes and ears)
    MASK_JOINTS = [1, 2, 3, 4]  # left_eye, right_eye, left_ear, right_ear
    
    def __init__(
        self,
        connect_joint: List[int],
        augmentor: Optional[object] = None
    ):
        """
        Args:
            connect_joint: List of parent joint indices for bone computation
            augmentor: Optional SkeletonAugmentor instance (training only)
        """
        self.connect_joint = connect_joint
        self.augmentor = augmentor
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Transform skeleton data
        
        Args:
            data: Raw skeleton data with shape (C, T, V, M)
                  C=3 (x, y, score), T=frames, V=joints, M=persons
        
        Returns:
            Transformed data with shape (4, 6, T, V, M)
        """
        # 1. Mask eye/ear joints
        data = data.copy()
        data[:, :, self.MASK_JOINTS, :] = 0
        
        # 2. Apply augmentation (if available, typically training only)
        if self.augmentor is not None:
            data = self.augmentor(data)
        
        # 3. Apply multi_input to generate 3-branch format
        data = multi_input(data, self.connect_joint)
        
        return data.astype(np.float32)
    
    @classmethod
    def from_graph(cls, dataset: str, augmentor: Optional[object] = None) -> 'SkeletonTransform':
        """Create transform from dataset name
        
        Args:
            dataset: Dataset name (e.g., 'coco')
            augmentor: Optional augmentor for training
            
        Returns:
            SkeletonTransform instance
        """
        from .graph import Graph
        graph = Graph(dataset)
        return cls(connect_joint=graph.connect_joint, augmentor=augmentor)
