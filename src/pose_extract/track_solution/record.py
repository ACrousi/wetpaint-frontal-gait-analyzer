# record.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Optional # 移除 Union
import numpy as np # 如果要在類型提示中使用 np.ndarray

class TrackState(Enum):
    NEW = auto()
    TRACKED = auto()
    REMOVED = auto()

@dataclass
class TrackRecord:
    track_id: int
    positions: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    mean: Dict[int, List[float]] = field(default_factory=dict)
    keypoints: Dict[int, "Optional[np.ndarray]"] = field(default_factory=dict)
    keypoint_scores: Dict[int, Any] = field(default_factory=dict)

    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    states: TrackState = TrackState.NEW
    scores: Dict[int, float] = field(default_factory=dict)
    is_interpolated: Dict[int, bool] = field(default_factory=dict)

    # 新的欄位結構，用於兩階段分段
    # 階段一: 幀級別的條件判斷結果
    # 鍵是 SegmentType.value (例如 "standing", "moving", "torso_ratio_valid")
    # 值是 Dict[frame_id, bool]
    frame_conditions: Dict[str, Dict[int, bool]] = field(default_factory=dict)

    # 階段二: 基於 frame_conditions 計算出的分段結果
    # 鍵是 SegmentType.value (例如 "standing", "moving", "torso_ratio_valid")
    # 值是 List of (start_frame, end_frame) 元組
    segments: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    # # --- 由 BodyProportionMetric 計算 ---
    # median_sitting_index: Optional[float] = None
    # median_head_ratio: Optional[float] = None # Corresponds to median of body_to_head_ratio

    # # Frame-wise body proportion metrics, as requested
    # sitting_height_index_at_frame: Dict[int, float] = field(default_factory=dict)
    # body_to_head_ratio_at_frame: Dict[int, float] = field(default_factory=dict)
    # sitting_height_raw_at_frame: Dict[int, float] = field(default_factory=dict)
    # total_height_raw_at_frame: Dict[int, Optional[float]] = field(default_factory=dict) # Can be None
    # head_height_raw_at_frame: Dict[int, float] = field(default_factory=dict)
    # # --- 結束 BodyProportionMetric 相關屬性 ---

    # # --- 由 TorsoProportionMetric 計算 ---
    # median_hip_width: Optional[float] = None
    # median_torso_length: Optional[float] = None
    # median_torso_to_hip_ratio: Optional[float] = None

    # hip_width_at_frame: Dict[int, float] = field(default_factory=dict)
    # torso_length_at_frame: Dict[int, float] = field(default_factory=dict)
    # torso_to_hip_ratio_at_frame: Dict[int, float] = field(default_factory=dict)
    # # --- 結束 TorsoProportionMetric 相關屬性 ---

    # is_standing_at_frame: Dict[int, bool] = field(default_factory=dict)

    # 由 KeypointStandardizationStrategy (如果有的話) 計算
    # 或者由 TrackManager.standardize_track_keypoints 方法填充
    keypoints_standardized: Dict[int, "Optional[np.ndarray]"] = field(default_factory=dict)
    
    # 由 KeypointsNormalizationPreprocessor 計算的正規化關鍵點
    keypoints_normalized: Dict[int, "Optional[np.ndarray]"] = field(default_factory=dict)


    def add_detection(
        self,
        frame_id: int,
        position: Tuple[float, float, float, float],
        state: TrackState, # This parameter is present but its direct use to update self.states was commented out in the original.
        score: float,
        mean: List[float],
        keypoints: Optional[np.ndarray] = None,
        keypoint_scores: Optional[Any] = None,
        is_interpolated: bool = False,
    ):
        if self.first_frame is None or frame_id < self.first_frame:
            self.first_frame = frame_id
        if self.last_frame is None or frame_id > self.last_frame:
            self.last_frame = frame_id

        self.positions[frame_id] = position
        if self.states == TrackState.NEW: # Only update from NEW to TRACKED
             self.states = TrackState.TRACKED
        # If the 'state' argument should always update self.states, it would be:
        # self.states = state

        self.scores[frame_id] = score
        self.mean[frame_id] = mean
        self.keypoints[frame_id] = keypoints
        self.keypoint_scores[frame_id] = keypoint_scores
        self.is_interpolated[frame_id] = is_interpolated
