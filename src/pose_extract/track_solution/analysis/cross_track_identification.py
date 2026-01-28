# a_identification_strategies.py - 目標識別策略基類
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Tuple, Optional
import logging
from .analysis_results import AnalysisResult
from src.pose_extract.track_solution.repository import TrackRepository

logger = logging.getLogger(__name__)


class TargetIdentificationResult:
    """
    目標識別策略的結果。
    包含保留和建議移除的軌跡ID。
    """
    def __init__(self, kept_track_ids: List[int], removed_track_ids: List[int]):
        self.kept_track_ids = kept_track_ids
        self.removed_track_ids = removed_track_ids

    def __repr__(self):
        return (f"TargetIdentificationResult("
                f"kept: {len(self.kept_track_ids)} tracks, "
                f"removed: {len(self.removed_track_ids)} tracks)")


class TargetIdentificationStrategy(ABC):
    """
    目標識別策略的基類。
    這些策略操作整個軌跡集合，而不是單一軌跡。
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """策略的唯一名稱。"""
        pass

    @abstractmethod
    def identify(self,
                 repository: TrackRepository,
                 all_results: Dict[int, AnalysisResult]
                ) -> TargetIdentificationResult:
        """
        執行目標識別。
        Args:
            repository: 原始軌跡倉庫。
            all_results: 一個字典 {track_id: AnalysisResult}，包含所有軌跡的分析結果。
        Returns:
            TargetIdentificationResult: 包含保留和建議移除的軌跡ID。
        """
        pass


class ChildIdentificationStrategy(TargetIdentificationStrategy):
    """
    重構後的兒童識別策略。
    它接收所有軌跡的分析結果，並回傳決策。
    """
    def __init__(self,
                 median_head_ratio_threshold: float = 6.0,
                 median_sitting_index_threshold: float = 55.0,
                 # 這個策略需要知道它應該讀取哪個中位數結果
                 median_metric_name: str = "body_proportion_medians"):
        self.median_head_ratio_threshold = median_head_ratio_threshold
        self.median_sitting_index_threshold = median_sitting_index_threshold
        self.median_metric_name = median_metric_name  # e.g., "body_proportion_medians_standing"

    @property
    def name(self) -> str:
        return "child_identification"

    def identify(self,
                 repository: TrackRepository,  # 仍然需要 repo 來獲取原始 track 物件
                 all_results: Dict[int, AnalysisResult]
                ) -> TargetIdentificationResult:
        """
        核心決策邏輯。
        Args:
            repository: 原始軌跡倉庫。
            all_results: 一個字典 {track_id: AnalysisResult}，包含所有軌跡的分析結果。
        Returns:
            TargetIdentificationResult: 包含保留和建議移除的軌跡ID。
        """
        logger.info("Starting refactored child identification process...")

        # 1. 找出重疊群組 (現在基於 analysis results)
        overlapping_groups = self._find_overlapping_groups(repository, all_results)

        # 2. 決策
        kept_ids = []
        removed_ids_in_groups = []

        for group in overlapping_groups:
            best_child_id, removed_in_group = self._select_best_child_in_group(group, all_results)
            if best_child_id is not None:
                kept_ids.append(best_child_id)
            removed_ids_in_groups.extend(removed_in_group)

        # 3. 處理獨立軌跡
        all_grouped_ids = {tid for group in overlapping_groups for tid in group}
        for track_id, result in all_results.items():
            if track_id not in all_grouped_ids:
                if self._meets_child_criteria(result):
                    kept_ids.append(track_id)

        return TargetIdentificationResult(kept_ids, removed_ids_in_groups)

    def _find_overlapping_groups(self, repository: TrackRepository, all_results: Dict[int, AnalysisResult]) -> List[Set[int]]:
        """
        找出重疊的軌跡群組。
        現在基於 analysis results 中的 segments 來確定有效幀範圍。
        """
        overlapping_groups: List[Set[int]] = []

        # 獲取所有軌跡ID
        track_ids = list(all_results.keys())

        # 簡單的重疊檢測邏輯 (可以根據需要改進)
        for i, track_id_a in enumerate(track_ids):
            for track_id_b in track_ids[i+1:]:
                if self._tracks_overlap(track_id_a, track_id_b, all_results):
                    # 找到或創建包含這些軌跡的群組
                    existing_group = None
                    for group in overlapping_groups:
                        if track_id_a in group or track_id_b in group:
                            existing_group = group
                            break

                    if existing_group:
                        existing_group.add(track_id_a)
                        existing_group.add(track_id_b)
                    else:
                        overlapping_groups.append({track_id_a, track_id_b})

        return overlapping_groups

    def _tracks_overlap(self, track_id_a: int, track_id_b: int, all_results: Dict[int, AnalysisResult]) -> bool:
        """
        檢查兩個軌跡是否重疊。
        這裡使用一個簡單的邏輯：如果兩個軌跡在時間上有重疊且在空間上有接近，就認為重疊。
        """
        result_a = all_results.get(track_id_a)
        result_b = all_results.get(track_id_b)

        if not result_a or not result_b:
            return False

        # 檢查是否有任何 segment 重疊
        segments_a = result_a.get_segments("standing") or []
        segments_b = result_b.get_segments("standing") or []

        for seg_a in segments_a:
            for seg_b in segments_b:
                # 簡單的重疊檢查：如果時間區間有重疊
                if max(seg_a[0], seg_b[0]) <= min(seg_a[1], seg_b[1]):
                    return True

        return False

    def _select_best_child_in_group(self, group: Set[int], all_results: Dict[int, AnalysisResult]) -> Tuple[Optional[int], List[int]]:
        group_scores = {}

        for track_id in group:
            result = all_results.get(track_id)
            if result and self._meets_child_criteria(result):
                score = self._calculate_child_score(result)
                if score is not None:
                    group_scores[track_id] = score

        if not group_scores:
            return None, list(group)

        best_track_id = min(group_scores, key=group_scores.get)
        removed_track_ids = [tid for tid in group if tid != best_track_id]
        return best_track_id, removed_track_ids

    def _get_medians(self, result: AnalysisResult) -> Tuple[Optional[float], Optional[float]]:
        """從 AnalysisResult 中提取中位數。"""
        median_df = result.get_metric(self.median_metric_name)
        if median_df is None or median_df.empty:
            return None, None

        # 假設 median_df 的欄位是 'median_body_to_head_ratio', 'median_sitting_height_index'
        head_ratio = median_df.iloc[0].get('median_body_to_head_ratio')
        sitting_index = median_df.iloc[0].get('median_sitting_height_index')
        return head_ratio, sitting_index

    def _meets_child_criteria(self, result: AnalysisResult) -> bool:
        head_ratio, sitting_index = self._get_medians(result)
        if head_ratio is None or sitting_index is None:
            return False
        return (head_ratio < self.median_head_ratio_threshold and
                sitting_index > self.median_sitting_index_threshold)

    def _calculate_child_score(self, result: AnalysisResult) -> Optional[float]:
        head_ratio, sitting_index = self._get_medians(result)
        if head_ratio is None or sitting_index is None or sitting_index <= 0:
            return None
        return head_ratio / sitting_index