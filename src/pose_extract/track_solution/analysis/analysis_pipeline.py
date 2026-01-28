from __future__ import annotations
import pandas as pd
from typing import List, Dict, Any, Optional
from .analysis_strategies import AnalysisStrategy, MetricStrategy, SegmentStrategy
from .analysis_results import AnalysisResult
from src.pose_extract.track_solution import TrackRecord
from .base import logger

class AnalysisPipeline:
    """
    我們的「指揮家」，負責按順序執行所有策略。
    """
    def __init__(self, strategies: List[AnalysisStrategy]):
        """
        初始化分析管道

        Args:
            strategies: 要執行的策略列表，應按依賴順序排列
        """
        self.strategies = strategies

    def run(self, track: TrackRecord) -> AnalysisResult:
        """
        對一個 TrackRecord 執行整個分析流程。

        Args:
            track: 要分析的軌跡記錄

        Returns:
            AnalysisResult: 包含所有分析結果的物件
        """
        logger.info(f"--- 開始為 Track {track.track_id} 執行分析流程 ---")
        analysis_result = AnalysisResult(track.track_id)

        for strategy in self.strategies:

            try:
                # 根據策略的類型，處理不同的返回結果
                if isinstance(strategy, MetricStrategy):
                    metric_df = strategy.analyze(track, analysis_result)
                    if not metric_df.empty:
                        analysis_result.add_metric_result(strategy.name, metric_df)
                elif isinstance(strategy, SegmentStrategy):
                    conditions_series, segments_list = strategy.analyze(track, analysis_result)
                    logger.info(f"{strategy.name} for Track {track.track_id}: {segments_list}")
                    analysis_result.add_segment_result(strategy.name, conditions_series, segments_list)
                else:
                    # 處理其他類型的策略，例如回傳 artifact 的策略
                    artifact = strategy.analyze(track, analysis_result)
                    analysis_result.add_artifact(strategy.name, artifact)

            except Exception as e:
                logger.error(f"策略 {strategy.name} 執行失敗: {e}")
                # 繼續執行其他策略，不中斷整個流程
                continue

        logger.info(f"--- 分析流程全部完成 ---")
        return analysis_result

    def get_strategy_names(self) -> List[str]:
        """獲取所有策略的名稱列表"""
        return [strategy.name for strategy in self.strategies]

    def get_strategy_dependencies(self) -> Dict[str, List[str]]:
        """
        分析策略間的依賴關係

        Returns:
            Dict[str, List[str]]: 策略名稱到其依賴項的映射
        """
        dependencies = {}

        for strategy in self.strategies:
            if isinstance(strategy, SegmentStrategy):
                # 對於 SegmentStrategy，我們需要檢查它在 generate_frame_conditions 中使用了哪些依賴項
                # 這裡是一個簡化的實現，實際上可能需要更複雜的靜態分析
                deps = []
                if hasattr(strategy, '_target_segment_type'):
                    # FixedLengthCuttingStrategy 依賴於目標 segment type
                    deps.append(strategy._target_segment_type.value)
                elif hasattr(strategy, 'criteria'):
                    # CombinedAnalysisStrategy 依賴於其 criteria 中的 segment types
                    deps.extend([seg_type.value for seg_type in strategy.criteria.keys()])

                dependencies[strategy.name] = deps
            else:
                dependencies[strategy.name] = []

        return dependencies

    def validate_pipeline(self) -> List[str]:
        """
        驗證管道配置是否有效

        Returns:
            List[str]: 驗證錯誤訊息列表，如果為空則表示配置有效
        """
        errors = []
        strategy_names = self.get_strategy_names()

        # 檢查是否有重複的策略名稱
        if len(strategy_names) != len(set(strategy_names)):
            duplicates = [name for name in strategy_names if strategy_names.count(name) > 1]
            errors.append(f"發現重複的策略名稱: {duplicates}")

        # 檢查依賴關係
        dependencies = self.get_strategy_dependencies()
        for i, strategy_name in enumerate(strategy_names):
            deps = dependencies[strategy_name]
            for dep in deps:
                # 檢查依賴項是否存在於之前的策略中
                if dep not in strategy_names[:i]:
                    errors.append(f"策略 '{strategy_name}' 的依賴項 '{dep}' 必須在之前執行")

        return errors