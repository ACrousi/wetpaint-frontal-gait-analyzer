from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from src.pose_extract.track_solution import TrackRecord


logger = logging.getLogger(__name__)


class InvalidSegmentTypeError(ValueError):
    """無效的 SegmentType 值異常"""
    pass


class SegmentType(Enum):
    MOVING = "moving"
    STANDING = "standing"
    TORSO_RATIO_VALID = "torso_ratio_valid"
    COMBINED = "combined"  # 這個類型可以用來表示所有的分段結果
    WALKING = "walking" # 新增走路分段類型
    HIP_ORIENTATION = "hip_orientation" # 新增臀部朝向分段類型
    TIME_RANGE = "time_range" # 新增時間範圍分段類型
    FIXED_LENGTH_CUTTING = "fixed_length_cutting" # 新增固定長度切割分段類型
    # 其他如 STATIONARY, NOT_STANDING, TORSO_RATIO_INVALID
    # 可以通過 label_map 的 False 值來表示

    @classmethod
    def from_string(cls, value: str) -> 'SegmentType':
        """
        從字串解析 SegmentType，若無效則拋出異常。

        Args:
            value: 要解析的字串值

        Returns:
            解析後的 SegmentType 實例

        Raises:
            InvalidSegmentTypeError: 若值無效
        """
        try:
            return cls(value)
        except ValueError:
            raise InvalidSegmentTypeError(f"無效的 segment type '{value}'。支援的值：{[member.value for member in cls]}")

    @property
    def label_map(self) -> Dict[bool, str]:
        if self is SegmentType.MOVING:
            return {True: "Moving", False: "Stop"}
        if self is SegmentType.STANDING:
            return {True: "Standing", False: "Not Standing"}
        if self is SegmentType.TORSO_RATIO_VALID:
            return {True: "Torso Ratio Valid", False: "Torso Ratio Invalid"}
        if self is SegmentType.WALKING:
            return {True: "Walking", False: "Not Walking"}
        if self is SegmentType.HIP_ORIENTATION:
            return {True: "Facing Front", False: "Facing Back"}
        if self is SegmentType.TIME_RANGE:
            return {True: "In Time Range", False: "Out of Time Range"}
        if self is SegmentType.FIXED_LENGTH_CUTTING:
            return {True: "Fixed Length Segment", False: "Not Fixed Length Segment"}
        # 預設回傳，或者可以引發錯誤如果枚舉成員未被處理
        return {True: "True", False: "False"}


# class AnalysisStrategy(ABC):
#     @abstractmethod
#     def analyze(self, track: TrackRecord) -> None:
#         pass


# class MetricAnalysisStrategy(AnalysisStrategy):
#     """單幀 / 單 track 指標：實作時回傳 {frame_id: value} 或累積統計"""


# class TargetIdentificationResult:
#     """目標識別策略的結果"""
#     def __init__(self, kept_track_ids: List[int], removed_track_ids: List[int]):
#         self.kept_track_ids = kept_track_ids
#         self.removed_track_ids = removed_track_ids


# class TargetIdentificationStrategy(ABC):
#     """
#     目標識別策略基類。
#     與 MetricAnalysisStrategy 和 SegmentAnalysisStrategy 不同，
#     此策略操作整個軌跡集合而不是單個軌跡。
#     """
    
#     @abstractmethod
#     def identify(self, repository) -> TargetIdentificationResult:
#         """
#         從軌跡集合中識別目標軌跡。
        
#         Args:
#             repository: 包含所有軌跡的 TrackRepository
            
#         Returns:
#             TargetIdentificationResult: 包含保留和移除的軌跡ID列表
#         """
#         pass


# class SegmentAnalysisStrategy(AnalysisStrategy):
#     """跨幀段落分析：
#     第一階段策略：計算每幀的條件並寫入 track.frame_conditions。
#     第二階段策略：根據 track.frame_conditions 計算分段並寫入 track.segments。
#     """

#     @property
#     @abstractmethod
#     def segment_type(self) -> SegmentType:
#         """定義此策略分析的分段類型。"""
#         pass

#     @property
#     @abstractmethod
#     def min_len(self) -> int:
#         """定義分段的最小長度。"""
#         pass

#     @property
#     def gap_tolerance(self) -> int:
#         """定義在一個 'True' 分段中允許的連續 'False' 幀的數量。預設為 0。"""
#         return 0

#     @property
#     def treat_false_as_separate_segment(self) -> bool:
#         """是否也為 False 條件產生分段。預設為 False。"""
#         return False

#     @abstractmethod
#     def generate_frame_conditions(
#         self, track: TrackRecord, **kwargs: Any
#     ) -> Dict[int, bool]:
#         """
#         Generates frame conditions for a given track for the specific segment_type.

#         Args:
#             track: The track record to analyze.
#             kwargs: Additional arguments for the analysis.

#         Returns:
#             A dictionary mapping frame_id to a boolean condition.
#         """
#         pass

#     def analyze(self, track: TrackRecord, **kwargs: Any) -> None:
#         """
#         執行兩階段分段分析：
#         1. 計算幀條件並存儲到 track.frame_conditions。
#         2. 根據幀條件計算分段並存儲到 track.segments。
#         """
#         # 階段一：計算幀條件
#         frame_conditions = self.generate_frame_conditions(track, **kwargs)
        
#         # 確保 track.frame_conditions 中有對應 segment_type.value 的鍵
#         if self.segment_type.value not in track.frame_conditions:
#             track.frame_conditions[self.segment_type.value] = {}
#         track.frame_conditions[self.segment_type.value].update(frame_conditions)

#         # 階段二：根據幀條件計算分段
#         # 從 track.frame_conditions 獲取此 segment_type 的條件
#         conditions_for_segment_type = track.frame_conditions.get(self.segment_type.value)
#         if conditions_for_segment_type is None:
#             logger.warning(f"Track {track.track_id} 沒有找到 segment type '{self.segment_type.value}' 的幀條件。")
#             return

#         segments_true, segments_false = self._convert_conditions_to_segments(
#             frame_conditions=conditions_for_segment_type,
#             min_len=self.min_len,
#             gap_tolerance=self.gap_tolerance,
#             treat_false_as_separate_segment=self.treat_false_as_separate_segment
#         )

#         # 確保 track.segments 中有對應 segment_type.value 的鍵
#         if self.segment_type.value not in track.segments:
#             track.segments[self.segment_type.value] = []
        
#         # 更新分段結果
#         track.segments[self.segment_type.value] = segments_true
#         logger.debug(f"Track {track.track_id}, SegmentType {self.segment_type.value}: True segments {segments_true}")
#         print(f"Track {track.track_id}, SegmentType {self.segment_type.value}: True segments {segments_true}")
        
#         # 如果需要，也可以處理 False 分段，例如存儲到不同的鍵或以某種方式標記
#         if self.treat_false_as_separate_segment and segments_false:
#             # 這裡可以決定如何處理 False 分段，例如使用不同的鍵
#             false_segment_key = f"{self.segment_type.value}_false"
#             track.segments[false_segment_key] = segments_false
#             logger.debug(f"Track {track.track_id}, SegmentType {self.segment_type.value}: False segments {segments_false}")


#     @staticmethod
#     def _convert_conditions_to_segments(
#         frame_conditions: Dict[int, bool],
#         min_len: int,
#         gap_tolerance: int = 0,
#         treat_false_as_separate_segment: bool = False
#     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
#         segments_true: List[Tuple[int, int]] = []
#         segments_false: List[Tuple[int, int]] = []

#         if not frame_conditions:
#             return segments_true, segments_false

#         sorted_frames = sorted(frame_conditions.keys())
#         if not sorted_frames:
#             return segments_true, segments_false

#         current_segment_start_frame: Optional[int] = None
#         current_segment_is_true: Optional[bool] = None # True if current segment is for True condition, False for False condition
#         # last_true_frame_in_potential_segment 用於處理 gap_tolerance
#         last_true_frame_in_potential_segment: Optional[int] = None
#         gap_count = 0

#         for i, frame_id in enumerate(sorted_frames):
#             condition_at_frame = frame_conditions[frame_id]

#             if current_segment_start_frame is None: # 處理第一個幀
#                 current_segment_start_frame = frame_id
#                 current_segment_is_true = condition_at_frame
#                 if condition_at_frame:
#                     last_true_frame_in_potential_segment = frame_id
#                 gap_count = 0
#                 continue

#             if current_segment_is_true: # 目前正在追蹤一個 True 條件的潛在分段
#                 if condition_at_frame: # 條件仍然為 True
#                     last_true_frame_in_potential_segment = frame_id
#                     gap_count = 0
#                 else: # 條件變為 False
#                     gap_count += 1
#                     if gap_count > gap_tolerance: # 超出容忍的間斷
#                         # 結束 True 分段，結束點是 last_true_frame_in_potential_segment
#                         if last_true_frame_in_potential_segment is not None and \
#                            last_true_frame_in_potential_segment - current_segment_start_frame + 1 >= min_len:
#                             segments_true.append((current_segment_start_frame, last_true_frame_in_potential_segment))
                        
#                         # 開始一個新的 False 分段
#                         current_segment_start_frame = sorted_frames[i - gap_count + 1] # False 分段從第一個 False 幀開始
#                         current_segment_is_true = False
#                         last_true_frame_in_potential_segment = None
#                         gap_count = 0 # 重置 gap_count for the new False segment
#             else: # 目前正在追蹤一個 False 條件的潛在分段
#                 if not condition_at_frame: # 條件仍然為 False
#                     # 不需要 gap_count 或 last_true_frame_in_potential_segment
#                     pass
#                 else: # 條件變為 True
#                     if treat_false_as_separate_segment:
#                         # 結束 False 分段，結束點是前一個幀
#                         end_false_segment_frame = sorted_frames[i-1]
#                         if end_false_segment_frame - current_segment_start_frame + 1 >= min_len:
#                             segments_false.append((current_segment_start_frame, end_false_segment_frame))
                    
#                     # 開始一個新的 True 分段
#                     current_segment_start_frame = frame_id
#                     current_segment_is_true = True
#                     last_true_frame_in_potential_segment = frame_id
#                     gap_count = 0
        
#         # 處理最後一個分段
#         if current_segment_start_frame is not None:
#             final_frame_in_data = sorted_frames[-1]
#             if current_segment_is_true: # 最後是 True 分段
#                 # 如果 gap_count > 0 且 <= gap_tolerance，表示結尾有一些 False 幀，但不超過容忍範圍
#                 # 這種情況下，分段的結束點應該是 last_true_frame_in_potential_segment
#                 effective_end_frame = last_true_frame_in_potential_segment if last_true_frame_in_potential_segment is not None else current_segment_start_frame
#                 if effective_end_frame >= current_segment_start_frame and \
#                    effective_end_frame - current_segment_start_frame + 1 >= min_len:
#                     segments_true.append((current_segment_start_frame, effective_end_frame))

#             elif treat_false_as_separate_segment: # 最後是 False 分段
#                 if final_frame_in_data - current_segment_start_frame + 1 >= min_len:
#                     segments_false.append((current_segment_start_frame, final_frame_in_data))
            
#         return segments_true, segments_false
