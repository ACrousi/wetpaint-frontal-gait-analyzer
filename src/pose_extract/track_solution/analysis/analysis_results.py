# a_results.py - 新的分析結果容器
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any

class AnalysisResult:
    """
    一個用來儲存和管理所有分析結果的中央容器 (我們的「共享樂譜」)。
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        # 儲存 MetricStrategy 的結果 (數值指標)
        self.metrics: Dict[str, pd.DataFrame] = {}
        # 儲存 SegmentStrategy 的每幀條件 (布林值)
        self.conditions: Dict[str, pd.Series] = {}
        # 儲存 SegmentStrategy 的最終分段 (start, end)
        self.segments: Dict[str, List[Tuple[int, int]]] = {}
        # 儲存其他可能需要的數據，例如標準化後的關鍵點
        self.artifacts: Dict[str, Any] = {}

    def add_metric_result(self, name: str, df: pd.DataFrame):
        """新增一個 Metric 策略的結果。"""
        if not df.index.name:
            df.index.name = "frame_id"
        self.metrics[name] = df
        print(f"  [Result] 指標 '{name}' 已新增。")

    def add_segment_result(self, name: str, conditions: pd.Series, segments: List[Tuple[int, int]]):
        """新增一個 Segment 策略的結果。"""
        self.conditions[name] = conditions
        self.segments[name] = segments
        print(f"  [Result] 分段 '{name}' 已新增 (含 conditions)。")

    def add_artifact(self, name: str, data: Any):
        """新增非表格類型的結果，例如處理過的 keypoints。"""
        self.artifacts[name] = data
        print(f"  [Result] 產物 '{name}' 已新增。")

    def get_metric(self, name: str) -> Optional[pd.DataFrame]:
        return self.metrics.get(name)

    def get_conditions(self, name: str) -> Optional[pd.Series]:
        return self.conditions.get(name)

    def get_segments(self, name: str) -> Optional[List[Tuple[int, int]]]:
        return self.segments.get(name)

    def get_artifact(self, name: str) -> Any:
        return self.artifacts.get(name)

    def get_all_metrics_df(self) -> pd.DataFrame:
        """將所有數值指標合併成一個大的 DataFrame，方便視覺化或機器學習。"""
        if not self.metrics: return pd.DataFrame()
        # 沿著 column (axis=1) 合併，索引 (frame_id) 會自動對齊
        return pd.concat(list(self.metrics.values()), axis=1)

    def __repr__(self):
        return (f"<AnalysisResult for Track {self.track_id} | "
                f"Metrics: {list(self.metrics.keys())} | "
                f"Segments: {list(self.segments.keys())}>")