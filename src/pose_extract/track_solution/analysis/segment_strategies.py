from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
from scipy.signal import find_peaks
from .base import SegmentType, InvalidSegmentTypeError, logger
from src.pose_extract.track_solution import TrackRecord

# 新的架構
from .analysis_strategies import SegmentStrategy
from .analysis_results import AnalysisResult

class FixedLengthCuttingStrategy(SegmentStrategy):
    """
    根據指定的 segmenttype 進行固定長度切割的策略。

    從頭開始切固定長度的分段，可選擇是否保留剩餘的片段。
    """

    def __init__(
        self,
        target_segment_type: SegmentType | str,
        fixed_length: int,
        min_len: int = 1,
        gap_tolerance: int = 0,
        step_offset: Optional[int] = None,
    ):
        """
        初始化固定長度切割策略

        Args:
            target_segment_type: 要切割的目標 segment type，可以是 SegmentType 或字符串
            fixed_length: 固定切割長度
            min_len: 分段的最小長度 (用於基類)
            gap_tolerance: 分段中允許的短暫間隙
            step_offset: 下一個起始切割位置的偏移量，如果為 None 則正常切割
        """
        if fixed_length <= 0:
            raise ValueError("fixed_length must be greater than 0")
        if step_offset is not None and step_offset <= 0:
            raise ValueError("step_offset must be greater than 0 if provided")

        # 處理 target_segment_type，支持字符串或 SegmentType
        # if isinstance(target_segment_type, str):
        #     try:
        #         self._target_segment_type = SegmentType.from_string(target_segment_type)
        #     except InvalidSegmentTypeError as e:
        #         raise ValueError(f"無效的 segment type '{target_segment_type}': {e}")
        # elif isinstance(target_segment_type, SegmentType):
        #     self._target_segment_type = target_segment_type
        # else:
        #     raise ValueError(f"target_segment_type 必須是 SegmentType 或字符串，得到 {type(target_segment_type)}")

        self._target_segment_type = target_segment_type
        self._fixed_length = fixed_length
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance
        self._step_offset = step_offset

    @property
    def name(self) -> str:
        return SegmentType.FIXED_LENGTH_CUTTING.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance


    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        """
        由於 analyze 被重寫，此方法不再使用，返回空字典。
        """
        return {}

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> Tuple[pd.Series, List[Tuple[int, int]]]:
        """
        自定義的 analyze 實現，直接返回切割後的 segments，而不通過 frame_conditions。
        """
        # 直接獲取目標 segments
        target_segments = dependencies.get_segments(self._target_segment_type)
        if not target_segments:
            logger.warning(
                f"Track {track.track_id}: No segments found for target segment type '{self._target_segment_type}'. "
                f"Cannot perform fixed length cutting."
            )
            # 返回空的條件和空的 segments
            return pd.Series(dtype=bool, name=self.name), []

        segments = []

        if self._step_offset is None:
            # 正常切割：對每個目標 segment 進行固定長度切割，丟棄不滿 fixed_length 的剩餘
            for segment_start, segment_end in target_segments:
                segment_length = segment_end - segment_start + 1
                if segment_length < self._fixed_length:
                    continue
                current_pos = segment_start
                while current_pos <= segment_end:
                    cut_end = min(current_pos + self._fixed_length - 1, segment_end)
                    cut_length = cut_end - current_pos + 1
                    if cut_length >= self._fixed_length:
                        segments.append((current_pos, cut_end))
                    current_pos = cut_end + 1
                    if current_pos > segment_end:
                        break
        else:
            # 使用 step_offset 的自定義切割邏輯
            segments = self._custom_fixed_length_cutting(target_segments)

        # 生成包含 segments 的 conditions Series
        frame_conditions_dict = {}
        if track.first_frame is not None and track.last_frame is not None:
            for frame_id in range(track.first_frame, track.last_frame + 1):
                frame_conditions_dict[frame_id] = False

        for start, end in segments:
            for frame_id in range(start, end + 1):
                frame_conditions_dict[frame_id] = True

        conditions_series = pd.Series(frame_conditions_dict, name=self.name).sort_index()

        return conditions_series, segments

    def _custom_fixed_length_cutting(self, target_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        自定義固定長度切割邏輯，支持 step_offset 和重疊檢查。
        """
        segments = []

        for segment_start, segment_end in target_segments:
            segment_length = segment_end - segment_start + 1

            if segment_length < self._fixed_length:
                continue

            # 使用 step_offset 進行切割
            current_pos = segment_start
            while current_pos <= segment_end:
                cut_end = min(current_pos + self._fixed_length - 1, segment_end)
                cut_length = cut_end - current_pos + 1

                if cut_length >= self._fixed_length:
                    segments.append((current_pos, cut_end))

                # 移動到下一個起始位置，使用 step_offset
                current_pos += self._step_offset

                if current_pos > segment_end:
                    break

        # 檢查最後一個片段的重疊邏輯
        if len(segments) >= 3:
            last_segment = segments[-1]
            second_last_segment = segments[-2]
            third_last_segment = segments[-3]

            last_length = last_segment[1] - last_segment[0] + 1
            if last_length < self._fixed_length:
                # 從最後一偵往前數 fixed_length 個幀
                potential_start = last_segment[1] - self._fixed_length + 1
                potential_end = last_segment[1]

                # 檢查是否與倒數第三個片段有重疊
                if potential_start <= third_last_segment[1] and potential_end >= third_last_segment[0]:
                    # 有重疊，丟棄倒數第二個
                    segments.pop(-2)

        return segments

class MotionSegment(SegmentStrategy):
    """以速度判斷 moving / stationary (現在只輸出 moving)"""
    required_metrics = ["motion_metric"]
    required_segments = []

    def __init__(
        self,
        min_len_moving: int = 10, # 移動分段的最小長度
        gap_tolerance_moving: int = 0, # 移動分段中允許的短暫靜止幀數
    ):
        self._min_len_moving = min_len_moving
        self._gap_tolerance_moving = gap_tolerance_moving

    @property
    def name(self) -> str:
        return SegmentType.MOVING.value

    @property
    def min_len(self) -> int:
        return self._min_len_moving

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance_moving

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        # 關鍵改變：從 dependencies 獲取 motion_metric 的結果
        motion_metric_df = dependencies.get_metric("motion_metric")

        if motion_metric_df is None or 'is_moving' not in motion_metric_df.columns:
            logger.warning(f"Track {track.track_id}: 依賴項 'motion_metric' 未找到，無法生成移動分段。")
            return {}

        # 直接從 DataFrame 轉換為字典，非常乾淨
        return motion_metric_df['is_moving'].to_dict()

class StandingSegment(SegmentStrategy):
    required_metrics = ["standing_metric"]
    required_segments = []

    def __init__(self, min_len: int = 1, gap_tolerance: int = 5):
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance

    @property
    def name(self) -> str:
        return SegmentType.STANDING.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        # 關鍵改變：從 dependencies 獲取 Metric 的結果
        standing_metric_df = dependencies.get_metric("standing_metric")

        if standing_metric_df is None or 'is_standing' not in standing_metric_df.columns:
            logger.warning(f"Track {track.track_id}: 依賴項 'standing_metric' 未找到，無法生成站立分段。")
            return {}

        # 直接從 DataFrame 轉換為字典，非常乾淨
        return standing_metric_df['is_standing'].to_dict()

class TorsoRatioSegment(SegmentStrategy):
    required_metrics = ["torso_proportion"]
    required_segments = []

    def __init__(
        self,
        max_ratio_threshold = 5.0,
        min_len: int = 15,
        gap_tolerance: int = 0,
    ):
        self._max_ratio_threshold = max_ratio_threshold
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance

    @property
    def name(self) -> str:
        return SegmentType.TORSO_RATIO_VALID.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        # 關鍵改變：從 dependencies 獲取 torso_proportion 的結果
        torso_proportion_df = dependencies.get_metric("torso_proportion")

        if torso_proportion_df is None or 'torso_to_hip_ratio' not in torso_proportion_df.columns:
            logger.warning(f"Track {track.track_id}: 依賴項 'torso_proportion' 未找到，無法生成 torso ratio 分段。")
            return {}

        # 直接從 DataFrame 轉換為字典，並應用閾值
        conditions = {}
        for frame_id, ratio in torso_proportion_df['torso_to_hip_ratio'].dropna().items():
            if np.isfinite(ratio):
                conditions[int(frame_id)] = (ratio <= self._max_ratio_threshold)
            else:
                conditions[int(frame_id)] = False

        return conditions

class CombinedAnalysisStrategy(SegmentStrategy):
    def __init__(self,
                 criteria: Dict[SegmentType, Optional[bool]],
                 min_len: int = 1, # 調整預設值，或由用戶指定
                 gap_tolerance: int = 0 # 通常交集後的 gap_tolerance 為 0
                ):
        """
        初始化 CombinedAnalysisStrategy

        Args:
            criteria: Dict[SegmentType, Optional[bool]] 或 Dict[str, Optional[bool]]
                     - True: 該 SegmentType 必須為 True
                     - False: 該 SegmentType 必須為 False
                     - None: 該 SegmentType 可以是 True 或 False（不限制）
            min_len: 分段的最小長度
            gap_tolerance: 分段中允許的短暫間隙

        Example:
            criteria = {
                SegmentType.MOVING: True,      # 必須在移動
                SegmentType.STANDING: False,   # 必須不在站立
                SegmentType.WALKING: None      # 不限制是否在走路
            }
            或
            criteria = {
                "moving": True,      # 必須在移動
                "standing": False,   # 必須不在站立
                "walking": None      # 不限制是否在走路
            }
        """
        if not criteria:
            raise ValueError("criteria cannot be empty")

        # 處理字串鍵，轉換為 SegmentType
        processed_criteria = {}
        for key, value in criteria.items():
            if isinstance(key, str):
                try:
                    segment_type = SegmentType.from_string(key)
                    processed_criteria[segment_type] = value
                except InvalidSegmentTypeError as e:
                    logger.warning(f"無效的 segment type '{key}'，跳過: {e}")
            elif isinstance(key, SegmentType):
                processed_criteria[key] = value
            else:
                logger.warning(f"無效的 criteria 鍵類型 '{type(key)}'，跳過")

        if not processed_criteria:
            raise ValueError("沒有有效的 criteria")

        self.criteria = processed_criteria
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance
        # 動態設定依賴
        self.required_segments = [seg.value for seg in processed_criteria.keys()]

    @property
    def name(self) -> str:
        return SegmentType.COMBINED.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        # 關鍵改變：從 dependencies 獲取其他 Segment 的 conditions 結果

        all_series = []
        for seg_type, required_value in self.criteria.items():
            if required_value is None: continue # 忽略不限制的條件

            conditions_series = dependencies.get_conditions(seg_type.value)
            if conditions_series is None:
                logger.warning(f"Track {track.track_id}: 依賴項 '{seg_type.value}' 的 conditions 未找到。")
                return {}

            if required_value is True:
                all_series.append(conditions_series)
            else: # required_value is False
                all_series.append(~conditions_series)

        if not all_series: return {}

        # 使用 pandas 的交集運算 (&)，非常高效
        combined_df = pd.concat(all_series, axis=1)
        final_series = combined_df.all(axis=1) # 只有所有條件都滿足的幀才為 True

        return final_series.to_dict()

class WalkingDetectionByAnkleAlternationStrategy(SegmentStrategy):
    """
    使用腳踝 Y 座標差值的交替模式（波峰-波谷序列）來偵測走路片段。
    偵測連續的波峰 (左腳高) 和波谷 (右腳高) 作為步態交替的標記。
    """
    required_metrics = ["ankle_alternation"]
    required_segments = []

    def __init__(
        self,
        min_alternating_cycles: int = 7, # 判斷走路所需的最小交替週期數 (e.g., 3 = 波峰-波谷-波峰)
        segment_padding: int = 10,   # 在連續序列前後擴展的幀數
        min_len: int = 15,               # 分段的最小長度
        gap_tolerance: int = 15,         # 分段中允許的短暫非走路幀數
    ):
        if min_alternating_cycles < 1:
            raise ValueError("min_alternating_cycles must be at least 1.")
        if segment_padding < 0:
            raise ValueError("segment_padding cannot be negative.")

        self._min_alternating_cycles = min_alternating_cycles
        self._segment_padding = segment_padding
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance

    @property
    def name(self) -> str:
        return SegmentType.WALKING.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        if track.first_frame is not None and track.last_frame is not None:
            conditions: Dict[int, bool] = {fid: False for fid in range(track.first_frame, track.last_frame + 1)}
        else:
            conditions: Dict[int, bool] = {}

        # 從指標獲取 alternating_sequences
        alternation_metric_df = dependencies.get_metric("ankle_alternation")
        if alternation_metric_df is None or 'alternating_sequences' not in alternation_metric_df.columns:
            logger.warning(f"Track {track.track_id}: 依賴項 'ankle_alternation' 未找到，無法生成走路分段。")
            return conditions

        alternating_sequences = alternation_metric_df['alternating_sequences'].iloc[0]
        logger.debug(f'alternating_sequences: {alternating_sequences}')

        for sequence in alternating_sequences:
            if len(sequence) >= self._min_alternating_cycles:  # e.g., 3 cycles 需要至少5點 (峰-谷-峰-谷-峰)
                segment_start_fid = sequence[0] - self._segment_padding
                segment_end_fid = sequence[-1] + self._segment_padding
                actual_start = max(track.first_frame, segment_start_fid)
                actual_end = min(track.last_frame, segment_end_fid)
                if actual_end - actual_start + 1 >= self._min_len:
                    for fid_to_mark in range(actual_start, actual_end + 1):
                        conditions[fid_to_mark] = True

        return conditions

    # def _find_consecutive_peak_sequences(self, peak_fids: List[int]) -> List[List[int]]:
    #     """
    #     找到真正連續的波峰序列，基於幀間距離判斷是否為連續序列。
        
    #     Args:
    #         peak_fids: 波峰對應的幀ID列表
            
    #     Returns:
    #         List[List[int]]: 連續波峰序列的列表，每個子列表包含一個連續序列的幀ID
    #     """
    #     if not peak_fids:
    #         return []
        
    #     sequences = []
    #     current_sequence = [peak_fids[0]]
        
    #     for i in range(1, len(peak_fids)):
    #         current_peak = peak_fids[i]
    #         previous_peak = peak_fids[i-1]
            
    #         # 使用 gap_tolerance 作為判斷連續性的閾值
    #         # 如果兩個波峰之間的間距超過 gap_tolerance，則認為不連續
    #         if current_peak - previous_peak <= self._gap_tolerance:
    #             # 連續的波峰
    #             current_sequence.append(current_peak)
    #         else:
    #             # 發現間斷，結束當前序列並開始新序列
    #             if len(current_sequence) >= self._min_consecutive_peaks:
    #                 sequences.append(current_sequence)
    #             current_sequence = [current_peak]
        
    #     # 處理最後一個序列
    #     if len(current_sequence) >= self._min_consecutive_peaks:
    #         sequences.append(current_sequence)
        
    #     logger.debug(f"Found {len(sequences)} consecutive peak sequences: {sequences}")
    #     return sequences


class HipOrientationStrategy(SegmentStrategy):
    """
    根據左臀和右臀的相對 X 座標判斷人物朝向 (正面/背面)。
    假設：
    - 使用原本的骨架點 (keypoints)。
    - 面向攝影機時，左臀 X 座標通常小於右臀 X 座標。
    - 背向攝影機時，左臀 X 座標通常大於右臀 X 座標。
    """
    def __init__(
        self,
        min_len: int = 1,
        gap_tolerance: int = 0,
        left_hip_idx: int = 11,  # COCO17 左臀索引
        right_hip_idx: int = 12, # COCO17 右臀索引
    ):
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance
        self._left_hip_idx = left_hip_idx
        self._right_hip_idx = right_hip_idx

    @property
    def name(self) -> str:
        return SegmentType.HIP_ORIENTATION.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        conditions: Dict[int, bool] = {}
        if not track.keypoints or track.first_frame is None or track.last_frame is None:
            logger.warning(
                f"Track {track.track_id}: 'keypoints' is empty or frame range not set. "
                f"Cannot generate '{self.name}' conditions."
            )
            if track.first_frame is not None and track.last_frame is not None:
                return {fid: False for fid in range(track.first_frame, track.last_frame + 1)}
            return {}

        for frame_id in range(track.first_frame, track.last_frame + 1):
            kps = track.keypoints.get(frame_id)
            if kps is not None and kps.shape[0] > max(self._left_hip_idx, self._right_hip_idx):
                left_hip_x = kps[self._left_hip_idx, 0]
                right_hip_x = kps[self._right_hip_idx, 0]

                # 如果 left_hip_x > right_hip_x，則認為是正面。
                # 這裡不設定閾值，直接比較。如果需要更穩健的判斷，可以加入一個小的容差。
                is_facing_front = left_hip_x > right_hip_x
                conditions[frame_id] = is_facing_front
            else:
                # 如果該幀沒有足夠的關鍵點數據，則繼承前一幀的狀態或預設為 False (背向)
                # 這裡選擇預設為 False，表示無法判斷時，不認為是正面。
                conditions[frame_id] = conditions.get(frame_id - 1, False)
        return conditions


class TimeRangeSegment(SegmentStrategy):
    """
    基於指定的時間範圍（start_frame, end_frame）來創建segment。
    適用於從 metadata 中的 start_second 和 end_second 轉換而來的 frame 範圍。
    """
    def __init__(
        self,
        start_frame: int,
        end_frame: Optional[int] = None,
        min_len: int = 1,
        gap_tolerance: int = 0,
    ):
        """
        初始化時間範圍分段策略

        Args:
            start_frame: 開始幀號
            end_frame: 結束幀號，如果為 None 則表示到軌跡結束
            min_len: 分段的最小長度
            gap_tolerance: 分段中允許的短暫間隙
        """
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._min_len = min_len
        self._gap_tolerance = gap_tolerance

    @property
    def name(self) -> str:
        return SegmentType.TIME_RANGE.value

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def gap_tolerance(self) -> int:
        return self._gap_tolerance

    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        """
        根據指定的時間範圍生成幀條件

        Args:
            track: 軌跡記錄
            dependencies: AnalysisResult 物件（此策略不需要依賴其他策略）

        Returns:
            Dict[int, bool]: 幀ID到布林值的映射，True表示在指定時間範圍內
        """
        conditions: Dict[int, bool] = {}

        if track.first_frame is None or track.last_frame is None:
            logger.warning(
                f"Track {track.track_id}: Frame range (first_frame/last_frame) is not set. "
                f"Cannot generate '{self.name}' conditions."
            )
            return conditions

        # 確定實際的結束幀
        actual_end_frame = self._end_frame if self._end_frame is not None else track.last_frame

        # 確保時間範圍在軌跡範圍內
        effective_start = max(self._start_frame, track.first_frame)
        effective_end = min(actual_end_frame, track.last_frame)

        logger.info(
            f"Track {track.track_id}: TimeRangeSegment - "
            f"requested range: {self._start_frame}-{self._end_frame}, "
            f"track range: {track.first_frame}-{track.last_frame}, "
            f"effective range: {effective_start}-{effective_end}"
        )

        # 為軌跡的所有幀設置條件
        for frame_id in range(track.first_frame, track.last_frame + 1):
            # 檢查該幀是否在指定的時間範圍內
            in_time_range = effective_start <= frame_id <= effective_end
            conditions[frame_id] = in_time_range

        return conditions