import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from pathlib import Path

from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution import TrackRecord, TrackState
from src.pose_extract.track_solution.analysis.analysis_strategies import AnalysisStrategy
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult
from src.pose_extract.track_solution.analysis.analysis_pipeline import AnalysisPipeline
from src.pose_extract.track_solution.analysis.analysis_strategies import MetricMedianStrategy, SegmentSummaryMetricStrategy
from src.pose_extract.track_solution.analysis.preprocessing import (
    PreprocessingPipeline,
    KeypointScoreThresholdPreprocessor,
    InterpolateTracksPreprocessor,
    RemoveShortTracksPreprocessor,
    KeypointsNormalizationPreprocessor
)

# 導入所有分析策略
from src.pose_extract.track_solution.analysis.metric_strategies import (
    BodyProportionMetric, StandingMetric, TorsoProportionMetric, LegDistanceMetric, SpineAngleMetric, AnkleAlternationMetric, StepTimeMetric, HipCenterMetric
)
from src.pose_extract.track_solution.analysis.segment_strategies import (
    StandingSegment, CombinedAnalysisStrategy, TorsoRatioSegment,
    WalkingDetectionByAnkleAlternationStrategy, HipOrientationStrategy, TimeRangeSegment,
    FixedLengthCuttingStrategy, SegmentType, InvalidSegmentTypeError 
)
from src.pose_extract.track_solution.analysis.cross_track_identification import (
    ChildIdentificationStrategy, TargetIdentificationResult
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    重構後的分析服務，扮演 VideoAnalysisManager 的角色。

    負責協調整個分析流程：
    1. 建立統一的 Per-Track 分析管線。
    2. 執行管線，為每個軌跡生成一份分析結果。
    3. 執行 Cross-Track 的目標識別。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.per_track_pipeline: Optional[AnalysisPipeline] = None
        self.preprocessing_stats: Optional[Dict[str, Any]] = None
        
    def process(self, track_manager: TrackManager) -> Dict[int, AnalysisResult]:
        """
        執行完整的、高效的兩階段分析流程。
        """
        logger.info("--- 開始執行重構後的分析流程 ---")
        
        # 使用局部變數儲存分析結果（無狀態設計）
        all_results: Dict[int, AnalysisResult] = {}

        # 步驟 0: 預處理
        self._run_preprocessing(track_manager)

        # 步驟 1: 建立統一的 Per-Track 分析管線
        self.per_track_pipeline = self._build_per_track_pipeline()

        # 步驟 2: (階段一) 執行 Per-Track 分析
        self._run_per_track_analysis(track_manager, all_results)

        # 步驟 3: (階段二) 執行目標識別
        self._run_target_identification(track_manager, all_results)

        logger.info("--- 分析流程全部完成 ---")
        return all_results

    def _build_per_track_pipeline(self) -> AnalysisPipeline:
        """
        根據設定檔，建立一個包含所有 Metric 和 Segment 策略的**單一**管線。
        策略的順序至關重要，依賴項必須在前。
        """
        logger.info("建立統一的 Per-Track 分析管線...")
        strategies: List[AnalysisStrategy] = []

        # --- Metric Strategies ---
        strategies.append(BodyProportionMetric())
        logger.info("添加身體比例指標")

        strategies.append(StandingMetric(**self.config.get("standing_metric_params", {})))
        logger.info("添加站立指標")

        strategies.append(TorsoProportionMetric())
        logger.info("添加軀幹比例指標")

        strategies.append(LegDistanceMetric())
        logger.info("添加腿部距離指標")

        strategies.append(SpineAngleMetric())
        logger.info("添加脊椎角度指標")

        strategies.append(HipCenterMetric())
        logger.info("添加髖關節中心點指標")

        strategies.append(AnkleAlternationMetric(**self.config.get("ankle_alternation_metric_params", {})))
        logger.info("添加腳踝交替指標")

        # --- Segment Strategies (依賴上面的 Metric) ---
        if self.config.get("standing_segment_params", {}).get("enabled", True):
            strategies.append(StandingSegment(**self.config.get("standing_segment_params", {})))
            logger.info("添加站立分段策略")

        if self.config.get("torso_ratio_segment_params", {}).get("enabled", True):
            strategies.append(TorsoRatioSegment(**self.config.get("torso_ratio_segment_params", {})))
            logger.info("添加軀幹比例分段策略")

        if self.config.get("walking_segment_params", {}).get("enabled", True):
            strategies.append(WalkingDetectionByAnkleAlternationStrategy(**self.config.get("walking_segment_params", {})))
            logger.info("添加走路分段策略")

        # --- Combined & Cutting Strategies (依賴上面的 Segment) ---
        if self.config.get("combined_segment_params", {}).get("enabled", True):
            strategies.append(CombinedAnalysisStrategy(**self.config.get("combined_segment_params", {})))
            logger.info("添加組合分段策略")
        
        if self.config.get("fixed_length_cutting_params", {}).get("enabled", True):
            strategies.append(FixedLengthCuttingStrategy(**self.config.get("fixed_length_cutting_params", {})))
            logger.info("添加固定長度切割策略")

        if self.config.get("step_time_metric_params", {}).get("enabled", True):
            strategies.append(StepTimeMetric(**self.config.get("step_time_metric_params", {})))
            logger.info("添加步伐時間指標")

        # --- 添加中位數計算策略 (用於目標識別) ---
        child_params = self.config.get("child_identification_params", {})
        if child_params.get("enabled", True):
            # 計算身體比例指標的中位數
            median_strategy = MetricMedianStrategy(
                source_metric_name="body_proportion",
                target_columns=["body_to_head_ratio", "sitting_height_index"],
                segment_type_filter=SegmentType.STANDING  # 只在站立分段中計算中位數
            )
            strategies.append(median_strategy)
            logger.info("添加身體比例中位數計算策略")

        summary_strategy = SegmentSummaryMetricStrategy(
            segment_type_filter=SegmentType.FIXED_LENGTH_CUTTING
        )
        strategies.append(summary_strategy)
        logger.info("添加分段統計摘要策略")

        logger.info(f"管線建立完成，共包含 {len(strategies)} 個 Per-Track 策略。")
        return AnalysisPipeline(strategies)

    def _run_per_track_analysis(self, track_manager: TrackManager, all_results: Dict[int, AnalysisResult]):
        """
        【階段一】只遍歷一次所有軌跡，執行完整的 Per-Track 分析。
        
        Args:
            track_manager: 軌跡管理器
            all_results: 分析結果字典（由呼叫者傳入，會被此方法填入資料）
        """
        if not self.per_track_pipeline:
            logger.error("分析管線未建立，無法執行 Per-Track 分析。")
            return

        logger.info("【階段一】開始執行 Per-Track 分析...")

        tracks_to_analyze = [
            t for t in track_manager.repository.get_all_tracks(removed=False)
            if t.states == TrackState.TRACKED
        ]

        for track in tracks_to_analyze:
            logger.debug(f"分析軌跡 {track.track_id}...")
            try:
                analysis_result = self.per_track_pipeline.run(track)
                all_results[track.track_id] = analysis_result
                logger.debug(f"軌跡 {track.track_id} 分析完成")
            except Exception as e:
                logger.error(f"分析軌跡 {track.track_id} 時發生錯誤: {e}")
                # 創建一個空的 AnalysisResult 以保持一致性
                all_results[track.track_id] = AnalysisResult(track.track_id)

        logger.info(f"【階段一】Per-Track 分析完成，共處理 {len(all_results)} 個軌跡。")

    def _run_target_identification(self, track_manager: TrackManager, all_results: Dict[int, AnalysisResult]):
        """
        【階段二】執行跨軌跡的目標識別。
        
        Args:
            track_manager: 軌跡管理器
            all_results: 分析結果字典
        """
        child_params = self.config.get("child_identification_params", {})
        if not child_params.get("enabled", True):
            logger.info("【階段二】目標識別已停用。")
            return

        logger.info("【階段二】開始執行目標識別...")

        # 建立識別策略
        child_identification_strategy = ChildIdentificationStrategy(
            median_head_ratio_threshold=child_params.get("median_head_ratio_threshold", 4.0),
            median_sitting_index_threshold=child_params.get("median_sitting_index_threshold", 58.0),
            median_metric_name="body_proportion_medians_standing"  # 使用我們在管線中定義的中位數指標名稱
        )

        # 執行目標識別
        result = child_identification_strategy.identify(track_manager.repository, all_results)

        # 應用決策
        if result.removed_track_ids:
            removed_count = track_manager.mark_tracks_removed(result.removed_track_ids)
            logger.info(f"【階段二】應用決策：標記 {removed_count} 個軌跡為 REMOVED。")

        logger.info(f"【階段二】目標識別完成。保留: {result.kept_track_ids} 軌跡，移除: {result.removed_track_ids} 軌跡")
        
    def _run_preprocessing(self, track_manager: TrackManager):
        """執行預處理步驟"""
        logger.info("執行預處理步驟")

        # 建立預處理管道
        preprocessors = []

        # 關鍵點分數過濾
        keypoint_threshold_params = self.config.get("keypoint_score_threshold_params", {})
        if keypoint_threshold_params.get("enabled", True):
            threshold_value = keypoint_threshold_params.get("threshold", 0.5)
            logger.info(f"添加關鍵點分數閾值過濾預處理器: {threshold_value}")
            preprocessors.append(KeypointScoreThresholdPreprocessor(threshold=threshold_value))

        # 插值處理
        preprocess_params = self.config.get("preprocess_params", {})
        interpolate_max_frames = preprocess_params.get("interpolate_max_frames", 30)
        logger.info(f"添加軌跡插值預處理器: max_frames={interpolate_max_frames}")
        preprocessors.append(InterpolateTracksPreprocessor(max_frames=interpolate_max_frames))

        # 移除短軌跡
        min_duration_frames = preprocess_params.get("min_track_duration_frames", 90)
        logger.info(f"添加移除短軌跡預處理器: min_duration={min_duration_frames}")
        preprocessors.append(RemoveShortTracksPreprocessor(min_duration_frames=min_duration_frames))

        # 關鍵點正規化
        normalization_params = self.config.get("keypoints_normalization_params", {})
        if normalization_params.get("enabled", True):
            logger.info("添加關鍵點正規化預處理器")
            preprocessors.append(KeypointsNormalizationPreprocessor(**normalization_params))

        # 執行預處理管道
        if preprocessors:
            preprocessing_pipeline = PreprocessingPipeline(preprocessors)
            stats = preprocessing_pipeline.run(track_manager)

            # 記錄預處理結果
            self.preprocessing_stats = stats
            logger.info("預處理管道執行完成")
        else:
            logger.info("沒有啟用的預處理器")
    
    def _execute_analysis_pipeline(self, track_manager: TrackManager,
                                 strategies: List[AnalysisStrategy],
                                 track_id: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """實際執行分析管道邏輯
        
        Args:
            track_manager: 軌跡管理器，用於獲取資料
            strategies: 分析策略列表
            track_id: 指定軌跡 ID，None 表示分析所有軌跡
            
        Returns:
            分析結果字典
        """
        logger.info(f"開始執行分析管道，策略數量: {len(strategies)}")
        
        # 建立分析管道
        pipeline = AnalysisPipeline(strategies)
        
        # 從 TrackManager 獲取要分析的軌跡
        tracks_to_analyze = track_manager.get_tracks_for_analysis(track_id, removed=False)
        
        # 過濾出 TRACKED 狀態的軌跡
        valid_tracks = []
        for track in tracks_to_analyze:
            if track.states == TrackState.TRACKED:
                valid_tracks.append(track)
            else:
                logger.warning(f"軌跡 {track.track_id} 不在 TRACKED 狀態，跳過分析")
        
        logger.info(f"找到 {len(valid_tracks)} 個有效軌跡進行分析")
        
        # 執行分析
        current_run_results: Dict[int, Dict[str, Any]] = {}
        for track in valid_tracks:
            logger.debug(f"為軌跡 {track.track_id} 執行分析管道...")
            result = pipeline(track)
            current_run_results[track.track_id] = result
            logger.debug(f"軌跡 {track.track_id} 分析完成")
            
        logger.info(f"分析管道執行完成，共分析 {len(current_run_results)} 個軌跡")
        return current_run_results

    def aggregate_case_segment_summary(
        self,
        results_per_video: List[Dict[int, AnalysisResult]],
        segment_type_filter: Union[SegmentType, str, None] = SegmentType.COMBINED,
        exclude_metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        聚合多影片的 SegmentSummary 統計，以每個 segment 片段為主計算統計值。
        與 SegmentSummaryMetricStrategy 保持一致的處理邏輯。

        Args:
            results_per_video: 每支影片的分析結果列表，每個元素是 Dict[int, AnalysisResult]
            segment_type_filter: segment 過濾器，預設 COMBINED
            exclude_metrics: 排除的指標列表

        Returns:
            多行 DataFrame，每行包含一個 segment 的統計值
        """
        exclude_metrics = exclude_metrics or []
        all_segment_summaries = []

        # 逐影片、逐軌跡收集統計值
        for res_map in results_per_video:
            for track_id, res in res_map.items():
                # 取得 segment_summary 指標
                segment_summary_df = res.get_metric('segment_summary_combined')
                if segment_summary_df is None:
                    segment_summary_df = res.get_metric('segment_summary')

                if segment_summary_df is None or segment_summary_df.empty:
                    continue

                # 對於每個 segment，添加軌跡資訊並收集
                for _, segment_row in segment_summary_df.iterrows():
                    segment_data = segment_row.to_dict()
                    # 添加軌跡資訊
                    segment_data['track_id'] = track_id
                    all_segment_summaries.append(segment_data)

        # 返回所有 segment 的統計值
        if all_segment_summaries:
            return pd.DataFrame(all_segment_summaries)
        else:
            return pd.DataFrame()
