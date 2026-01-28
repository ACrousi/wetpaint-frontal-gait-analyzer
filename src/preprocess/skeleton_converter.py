"""
骨架資料格式轉換器

將 JSON 骨架資料轉換為 ResGCN 模型所需的張量格式
供訓練資料生成和預測推論共用
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Any


class SkeletonDataConverter:
    """骨架資料格式轉換器
    
    將 JSON 骨架資料轉換為 ResGCN 模型所需的張量格式 (C, T, V, M)
    - C: 通道數 (x, y, score) = 3
    - T: 時間幀數 (max_frame)
    - V: 關節數 (num_joint)
    - M: 人數 (num_person)
    """
    
    def __init__(
        self, 
        num_joint: int = 17,      # COCO 17 joints
        max_frame: int = 150,     # 最大幀數
        num_person: int = 1       # 單人
    ):
        self.num_joint = num_joint
        self.max_frame = max_frame
        self.num_person = num_person
    
    def json_to_tensor(self, json_path: str) -> np.ndarray:
        """將 JSON 骨架檔案轉為 ResGCN 格式張量
        
        Args:
            json_path: JSON 檔案路徑
            
        Returns:
            np.ndarray: shape (C, T, V, M) = (3, max_frame, num_joint, num_person)
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Keypoint JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 提取 frames: [T, V, C]
        frames = np.array(json_data['frames'])
        
        return self.frames_to_tensor(frames)
    
    def frames_to_tensor(self, frames: np.ndarray) -> np.ndarray:
        """將 frames 陣列轉為 ResGCN 格式張量
        
        Args:
            frames: shape [T, V, C] 的骨架資料
            
        Returns:
            np.ndarray: shape (C, T, V, M) = (3, max_frame, num_joint, num_person)
        """
        T, V, C = frames.shape
        
        if V != self.num_joint:
            raise ValueError(f"Expected {self.num_joint} joints, got {V}")
        
        # 建立資料張量 (C, T, V, M)
        data = np.zeros(
            (3, self.max_frame, self.num_joint, self.num_person), 
            dtype=np.float32
        )
        
        # 填入 x, y 座標：從 (T, V, 2) 轉為 (2, T, V)
        # 注意：與 coco_generator.read_xyz() 保持一致，只填入 x, y
        # channel 2 (score) 維持 0，與訓練資料格式相同
        actual_frames = min(T, self.max_frame)
        data[:2, :actual_frames, :, 0] = frames[:actual_frames, :, :2].transpose(2, 0, 1)
        
        # 不填入 score（保持 channel 2 = 0）
        # 與 coco_generator.read_xyz() 訓練資料格式一致
        
        return data
    
    def extract_features(
        self, 
        json_path: str, 
        gait_columns: List[str]
    ) -> np.ndarray:
        """從 JSON 提取步態特徵
        
        Args:
            json_path: JSON 檔案路徑
            gait_columns: 要提取的特徵欄位名稱
            
        Returns:
            np.ndarray: 步態特徵向量
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Keypoint JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        features = json_data.get('features', {})
        
        gait_values = []
        for col in gait_columns:
            val = features.get(col, 0.0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                gait_values.append(float(val))
            else:
                gait_values.append(0.0)
        
        return np.array(gait_values, dtype=np.float32)
    
    def batch_convert(self, json_paths: List[str]) -> np.ndarray:
        """批次轉換多個 JSON 檔案
        
        Args:
            json_paths: JSON 檔案路徑列表
            
        Returns:
            np.ndarray: shape (N, C, T, V, M)
        """
        tensors = []
        for path in json_paths:
            tensor = self.json_to_tensor(path)
            tensors.append(tensor)
        
        return np.stack(tensors, axis=0)
