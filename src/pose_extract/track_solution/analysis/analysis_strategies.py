# a_strategies.py - 新的策略基類
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .analysis_results import AnalysisResult
from src.pose_extract.track_solution import TrackRecord
from src.pose_extract.track_solution.analysis.base import SegmentType

import logging
logger = logging.getLogger(__name__)


class AnalysisStrategy(ABC):
    """所有分析策略的統一基類。"""
    required_metrics: List[str] = []
    required_segments: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """策略的唯一名稱，將用作結果的鍵。"""
        pass

    @abstractmethod
    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> Any:
        """
        執行分析。
        - 接收 TrackRecord 作為原始數據源。
        - 接收 AnalysisResult 獲取先前策略的結果 (依賴項)。
        - **必須回傳**計算結果，**不得**修改傳入的 track 或 dependencies 物件。
        """
        pass

class MetricStrategy(AnalysisStrategy):
    """計算每幀數值指標的策略基類。"""
    @abstractmethod
    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """回傳一個以 frame_id 為索引的 DataFrame。"""
        pass

class SegmentStrategy(AnalysisStrategy):
    """產生分段的策略基類。"""
    @property
    @abstractmethod
    def min_len(self) -> int: pass

    @property
    @abstractmethod
    def gap_tolerance(self) -> int: pass

    @abstractmethod
    def generate_frame_conditions(self, track: TrackRecord, dependencies: AnalysisResult) -> Dict[int, bool]:
        """產生每幀的布林條件字典。"""
        pass

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> Tuple[pd.Series, List[Tuple[int, int]]]:
        """
        標準化的 analyze 實現：
        1. 呼叫 generate_frame_conditions 獲取布林條件。
        2. 將條件轉換為分段。
        3. 回傳 (條件Series, 分段List)。
        """
        frame_conditions_dict = self.generate_frame_conditions(track, dependencies)

        # 轉換為 pandas Series
        if not frame_conditions_dict:
            series_name = self.name
            conditions_series = pd.Series(dtype=bool, name=series_name)
        else:
            conditions_series = pd.Series(frame_conditions_dict, name=self.name).sort_index()

        # 使用靜態方法來計算分段
        segments_true, _ = self._convert_conditions_to_segments(
            frame_conditions=frame_conditions_dict,
            min_len=self.min_len,
            gap_tolerance=self.gap_tolerance
        )
        return conditions_series, segments_true

    @staticmethod
    def _convert_conditions_to_segments(
        frame_conditions: Dict[int, bool],
        min_len: int,
        gap_tolerance: int = 0,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        segments_true: List[Tuple[int, int]] = []
        segments_false: List[Tuple[int, int]] = []

        if not frame_conditions:
            return segments_true, segments_false

        sorted_frames = sorted(frame_conditions.keys())
        if not sorted_frames:
            return segments_true, segments_false

        current_segment_start_frame: Optional[int] = None
        current_segment_is_true: Optional[bool] = None # True if current segment is for True condition, False for False condition
        # last_true_frame_in_potential_segment 用於處理 gap_tolerance
        last_true_frame_in_potential_segment: Optional[int] = None
        gap_count = 0

        for i, frame_id in enumerate(sorted_frames):
            condition_at_frame = frame_conditions[frame_id]

            if current_segment_start_frame is None: # 處理第一個幀
                current_segment_start_frame = frame_id
                current_segment_is_true = condition_at_frame
                if condition_at_frame:
                    last_true_frame_in_potential_segment = frame_id
                gap_count = 0
                continue

            if current_segment_is_true: # 目前正在追蹤一個 True 條件的潛在分段
                if condition_at_frame: # 條件仍然為 True
                    last_true_frame_in_potential_segment = frame_id
                    gap_count = 0
                else: # 條件變為 False
                    gap_count += 1
                    if gap_count > gap_tolerance: # 超出容忍的間斷
                        # 結束 True 分段，結束點是 last_true_frame_in_potential_segment
                        if last_true_frame_in_potential_segment is not None and \
                           last_true_frame_in_potential_segment - current_segment_start_frame + 1 >= min_len:
                            segments_true.append((current_segment_start_frame, last_true_frame_in_potential_segment))

                        # 開始一個新的 False 分段
                        current_segment_start_frame = sorted_frames[i - gap_count + 1] # False 分段從第一個 False 幀開始
                        current_segment_is_true = False
                        last_true_frame_in_potential_segment = None
                        gap_count = 0 # 重置 gap_count for the new False segment
            else: # 目前正在追蹤一個 False 條件的潛在分段
                if not condition_at_frame: # 條件仍然為 False
                    # 不需要 gap_count 或 last_true_frame_in_potential_segment
                    pass
                else: # 條件變為 True
                    # 結束 False 分段，結束點是前一個幀
                    end_false_segment_frame = sorted_frames[i-1]
                    if end_false_segment_frame - current_segment_start_frame + 1 >= min_len:
                        segments_false.append((current_segment_start_frame, end_false_segment_frame))

                    # 開始一個新的 True 分段
                    current_segment_start_frame = frame_id
                    current_segment_is_true = True
                    last_true_frame_in_potential_segment = frame_id
                    gap_count = 0

        # 處理最後一個分段
        if current_segment_start_frame is not None:
            final_frame_in_data = sorted_frames[-1]
            if current_segment_is_true: # 最後是 True 分段
                # 如果 gap_count > 0 且 <= gap_tolerance，表示結尾有一些 False 幀，但不超過容忍範圍
                # 這種情況下，分段的結束點應該是 last_true_frame_in_potential_segment
                effective_end_frame = last_true_frame_in_potential_segment if last_true_frame_in_potential_segment is not None else current_segment_start_frame
                if effective_end_frame >= current_segment_start_frame and \
                   effective_end_frame - current_segment_start_frame + 1 >= min_len:
                    segments_true.append((current_segment_start_frame, effective_end_frame))

        return segments_true, segments_false


class MetricMedianStrategy(MetricStrategy):
    """
    計算指定指標的中位數。
    這個策略依賴於先前的 MetricStrategy 已經計算出的每幀數據。
    """
    def __init__(self,
                 source_metric_name: str,
                 target_columns: List[str],
                 segment_type_filter: Optional[SegmentType] = None):
        """
        Args:
            source_metric_name: 依賴的來源指標名稱 (e.g., 'body_proportion')
            target_columns: 要計算中位數的欄位名稱 (e.g., ['sitting_height_index', 'body_to_head_ratio'])
            segment_type_filter: 可選的 segment 過濾器
        """
        self._source_metric_name = source_metric_name
        self._target_columns = target_columns
        self._segment_type_filter = segment_type_filter

    @property
    def name(self) -> str:
        # 名字可以反映其配置，確保唯一性
        filter_name = f"_{self._segment_type_filter.value}" if self._segment_type_filter else ""
        return f"{self._source_metric_name}_medians{filter_name}"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        # 1. 從依賴項中獲取來源指標的 DataFrame
        source_df = dependencies.get_metric(self._source_metric_name)
        if source_df is None or source_df.empty:
            print(f"  [Warning] 來源指標 '{self._source_metric_name}' 未找到。")
            return pd.DataFrame()

        # 2. 根據 segment 過濾器篩選有效的幀
        target_df = source_df
        if self._segment_type_filter:
            segment_name = self._segment_type_filter.value
            segments = dependencies.get_segments(segment_name)
            conditions = dependencies.get_conditions(segment_name)

            if conditions is not None and not conditions.empty:
                # 只保留 conditions 為 True 的行，先對齊索引
                aligned_conditions = conditions.reindex(source_df.index, fill_value=False)
                target_df = source_df[aligned_conditions]
            else:
                print(f"  [Warning] Segment '{segment_name}' 未找到，將計算整個軌跡的中位數。")

        # 3. 計算中位數
        median_values = {}
        for col in self._target_columns:
            if col in target_df.columns:
                # 使用 np.nanmedian 來忽略 NaN 值
                median_val = np.nanmedian(target_df[col].dropna())
                median_values[f"median_{col}"] = median_val if np.isfinite(median_val) else None
            else:
                median_values[f"median_{col}"] = None

        # 4. 回傳一個只包含一行數據的 DataFrame
        # 這使得結果仍然可以被合併到大的 metrics DataFrame 中，儘管它不是時間序列
        return pd.DataFrame([median_values])


class SegmentSummaryMetricStrategy(MetricStrategy):
    """
    計算所有AnalysisResult中metric strategy計算完的統計數值（中位數、平均值、最大值、最小值）。
    這個策略依賴於先前的所有 MetricStrategy 已經計算出的每幀數據。
    """
    def __init__(self,
                 segment_type_filter: Optional[SegmentType] = None,
                 exclude_metrics: Optional[List[str]] = None):
        """
        Args:
            segment_type_filter: 可選的 segment 過濾器，只計算指定segment類型內的統計值
            exclude_metrics: 可選的排除指標列表，不計算這些指標的統計值
        """
        self._segment_type_filter = segment_type_filter
        self._exclude_metrics = exclude_metrics or []

    @property
    def name(self) -> str:
        # 名字可以反映其配置，確保唯一性
        filter_name = f"_{self._segment_type_filter.value}" if self._segment_type_filter else ""
        return f"segment_summary{filter_name}"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        # 1. 獲取所有可用的指標
        available_metrics = dependencies.metrics
        if not available_metrics:
            print(f"  [Warning] 沒有可用的指標數據。")
            return pd.DataFrame()

        # 2. 根據 segment 過濾器獲取 segments
        segments = []
        if self._segment_type_filter:
            segment_name = self._segment_type_filter.value
            segments = dependencies.get_segments(segment_name)
            if not segments:
                print(f"  [Warning] Segment '{segment_name}' 未找到，將計算整個軌跡的統計值。")
                # 如果沒有 segments，創建一個包含整個軌跡的虛擬 segment
                if available_metrics:
                    # 找到第一個有數據的指標來確定軌跡的幀範圍
                    for metric_df in available_metrics.values():
                        if metric_df is not None and not metric_df.empty:
                            min_frame = metric_df.index.min()
                            max_frame = metric_df.index.max()
                            segments = [(min_frame, max_frame)]
                            break

        if not segments:
            return pd.DataFrame()

        # 3. 為每個 segment 計算統計值
        segment_summaries = []

        for segment_idx, (start_frame, end_frame) in enumerate(segments):
            segment_stats = {
                'segment_index': segment_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'segment_length': end_frame - start_frame + 1
            }

            for metric_name, metric_df in available_metrics.items():
                # 跳過排除的指標
                if metric_name in self._exclude_metrics:
                    continue

                if metric_df is None or metric_df.empty:
                    continue

                # 如果指標包含布林資料，跳過
                if any(metric_df.dtypes == 'bool'):
                    continue

                # 檢查是否為 list 類型數據
                if isinstance(metric_df.iloc[0, 0], list):
                    # 對於 list 類型，list 的順序對應 segment_idx
                    list_data = metric_df.iloc[0, 0]
                    print(f"*********{list_data}**********")
                    if isinstance(list_data, list) and len(list_data) > segment_idx:
                        try:
                            segment_value = list_data[segment_idx]
                            # 如果 segment_value 本身是 list，計算其統計值
                            if isinstance(segment_value, list) and segment_value:
                                col_data = np.array(segment_value)
                                col = metric_df.columns[0]
                                segment_stats.update({
                                    f"{col}_median": np.nanmedian(col_data),
                                    f"{col}_mean": np.nanmean(col_data),
                                    f"{col}_max": np.nanmax(col_data),
                                    f"{col}_min": np.nanmin(col_data),
                                    f"{col}_std": np.nanstd(col_data),
                                    f"{col}_cv": np.nanstd(col_data) / np.nanmean(col_data) if np.nanmean(col_data) else None
                                })
                            else:
                                # 如果是單一數值，直接使用
                                col = metric_df.columns[0]
                                segment_stats[f"{col}_value"] = segment_value
                        except (ValueError, IndexError) as e:
                            logger.warning(f"metric '{metric_name}' 的 list 數據無法處理 segment {segment_idx}: {e}")
                    continue

                # 對於普通數值數據，用 segment_mask 篩選
                segment_mask = (metric_df.index >= start_frame) & (metric_df.index <= end_frame)
                segment_data = metric_df[segment_mask]

                if segment_data.empty:
                    continue

                # 為每個數值列計算統計值
                for col in segment_data.select_dtypes(include=[np.number]).columns:
                    col_data = segment_data[col].dropna()
                    if len(col_data) == 0:
                        continue

                    # 計算統計值
                    median_val = np.nanmedian(col_data)
                    mean_val = np.nanmean(col_data)
                    max_val = np.nanmax(col_data)
                    min_val = np.nanmin(col_data)
                    std_val = np.nanstd(col_data)

                    # 將統計值添加到結果中
                    segment_stats[f"{col}_median"] = median_val if np.isfinite(median_val) else None
                    segment_stats[f"{col}_mean"] = mean_val if np.isfinite(mean_val) else None
                    segment_stats[f"{col}_max"] = max_val if np.isfinite(max_val) else None
                    segment_stats[f"{col}_min"] = min_val if np.isfinite(min_val) else None
                    segment_stats[f"{col}_std"] = std_val if np.isfinite(std_val) else None
                    segment_stats[f"{col}_cv"] = (std_val / mean_val) if mean_val not in (0, None, np.nan) and np.isfinite(std_val) else None

            segment_summaries.append(segment_stats)

        # 4. 回傳包含所有 segment 統計值的 DataFrame
        if not segment_summaries:
            return pd.DataFrame()

        return pd.DataFrame(segment_summaries)
