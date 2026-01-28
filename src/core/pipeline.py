"""
Pipeline 模組

提供可組合的影片處理管線框架。
允許靈活地組合、跳過、或插入處理階段。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from src.pose_extract.track_solution import TrackManager
from .models import VideoInfo, AnalysisOutput


logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Pipeline 執行上下文
    
    在各個 Stage 之間傳遞資料和狀態。
    """
    video_info: VideoInfo
    config: Dict[str, Any]
    
    # 處理過程中產生的資料
    track_manager: Optional[TrackManager] = None
    analysis_output: Optional[AnalysisOutput] = None
    
    # 狀態追蹤
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 控制流程
    should_stop: bool = False
    
    def add_error(self, error: str):
        """新增錯誤並標記停止"""
        self.errors.append(error)
        self.should_stop = True
        logger.error(error)
    
    def add_warning(self, warning: str):
        """新增警告"""
        self.warnings.append(warning)
        logger.warning(warning)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def is_success(self) -> bool:
        return not self.has_errors and not self.should_stop


class PipelineStage(ABC):
    """
    Pipeline 階段基類
    
    每個階段負責處理 context 的特定部分。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """階段名稱"""
        pass
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        執行此階段的處理
        
        Args:
            context: Pipeline 上下文
            
        Returns:
            更新後的 context
        """
        pass
    
    def should_skip(self, context: PipelineContext) -> bool:
        """
        判斷是否應跳過此階段
        
        預設不跳過。子類別可覆寫此方法實現條件跳過。
        """
        return False
    
    def on_error(self, context: PipelineContext, error: Exception) -> PipelineContext:
        """
        錯誤處理鉤子
        
        預設記錄錯誤並標記停止。子類別可覆寫實現自訂處理。
        """
        context.add_error(f"{self.name} 發生錯誤: {str(error)}")
        return context


class VideoProcessingPipeline:
    """
    可組合的影片處理管線
    
    使用方式:
        pipeline = (VideoProcessingPipeline()
            .add_stage(SkeletonExtractionStage())
            .add_stage(AnalysisStage())
            .add_stage(ExportStage()))
        
        result = pipeline.run(context)
    """
    
    def __init__(self, name: str = "VideoProcessingPipeline"):
        self.name = name
        self.stages: List[PipelineStage] = []
        self._before_stage_hooks: List[Callable[[PipelineContext, PipelineStage], None]] = []
        self._after_stage_hooks: List[Callable[[PipelineContext, PipelineStage], None]] = []
    
    def add_stage(self, stage: PipelineStage) -> 'VideoProcessingPipeline':
        """
        新增處理階段（流式 API）
        
        Returns:
            self，支援鏈式呼叫
        """
        self.stages.append(stage)
        return self
    
    def add_before_stage_hook(self, hook: Callable[[PipelineContext, PipelineStage], None]) -> 'VideoProcessingPipeline':
        """新增階段執行前的鉤子"""
        self._before_stage_hooks.append(hook)
        return self
    
    def add_after_stage_hook(self, hook: Callable[[PipelineContext, PipelineStage], None]) -> 'VideoProcessingPipeline':
        """新增階段執行後的鉤子"""
        self._after_stage_hooks.append(hook)
        return self
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        執行完整的 Pipeline
        
        Args:
            context: 初始 context
            
        Returns:
            處理完成的 context
        """
        logger.info(f"開始執行 Pipeline: {self.name}")
        
        for stage in self.stages:
            # 檢查是否應停止
            if context.should_stop:
                logger.info(f"Pipeline 已停止，跳過後續階段")
                break
            
            # 檢查是否應跳過此階段
            if stage.should_skip(context):
                logger.debug(f"跳過階段: {stage.name}")
                continue
            
            # 執行前鉤子
            for hook in self._before_stage_hooks:
                hook(context, stage)
            
            # 執行階段
            logger.info(f"執行階段: {stage.name}")
            try:
                context = stage.process(context)
            except Exception as e:
                logger.error(f"階段 {stage.name} 發生例外: {e}", exc_info=True)
                context = stage.on_error(context, e)
            
            # 執行後鉤子
            for hook in self._after_stage_hooks:
                hook(context, stage)
        
        status = "成功" if context.is_success else "失敗"
        logger.info(f"Pipeline 執行{status}: {self.name}")
        
        return context


# ============================================================
# 預定義的 Pipeline Stages
# ============================================================

class CacheCheckStage(PipelineStage):
    """快取檢查階段 - 檢查並載入已存在的骨架快取"""
    
    @property
    def name(self) -> str:
        return "CacheCheck"
    
    def __init__(self, cache_dir: Path, check_existing: bool = True):
        self.cache_dir = Path(cache_dir)
        self.check_existing = check_existing
    
    def should_skip(self, context: PipelineContext) -> bool:
        return not self.check_existing
    
    def process(self, context: PipelineContext) -> PipelineContext:
        import json
        from src.pose_extract.track_solution import TrackManager
        
        video_name = context.video_info.video_name
        base_name = Path(video_name).stem
        cache_path = self.cache_dir / f"{base_name}.json"
        
        context.metadata['from_cache'] = False
        
        if not cache_path.exists():
            logger.debug(f"快取不存在: {cache_path}")
            return context
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'tracks' not in data or not data['tracks']:
                logger.warning(f"快取資料格式不正確: {cache_path}")
                return context
            
            track_manager = TrackManager()
            track_manager.load_from_coco_data(data)
            
            if track_manager.has_sufficient_skeleton_data():
                context.track_manager = track_manager
                context.metadata['from_cache'] = True
                logger.info(f"成功從快取載入骨架資料: {cache_path}")
            else:
                logger.info(f"快取資料不完整，將重新追蹤: {cache_path}")
                
        except Exception as e:
            logger.warning(f"載入快取失敗: {cache_path} - {e}")
        
        return context


class SkeletonExtractionStage(PipelineStage):
    """骨架提取階段"""
    
    @property
    def name(self) -> str:
        return "SkeletonExtraction"
    
    def __init__(self, video_processor):
        self.video_processor = video_processor
    
    def should_skip(self, context: PipelineContext) -> bool:
        # 如果已從快取載入，跳過提取
        return context.track_manager is not None
    
    def process(self, context: PipelineContext) -> PipelineContext:
        video_info_dict = context.video_info.to_dict()
        context.track_manager = self.video_processor.extract_skeletons(video_info_dict)
        return context


class RawSkeletonExportStage(PipelineStage):
    """原始骨架導出階段 - 作為快取使用"""
    
    @property
    def name(self) -> str:
        return "RawSkeletonExport"
    
    def __init__(self, export_service):
        self.export_service = export_service
    
    def should_skip(self, context: PipelineContext) -> bool:
        # 如果不需要儲存，或從快取載入，跳過
        if not context.metadata.get('save_results', True):
            return True
        if context.metadata.get('from_cache', False):
            return True
        return context.track_manager is None
    
    def process(self, context: PipelineContext) -> PipelineContext:
        video_info_dict = context.video_info.to_dict()
        try:
            self.export_service.export_to_raw_skeleton(context.track_manager, video_info_dict)
        except Exception as e:
            context.add_warning(f"儲存原始骨架資料失敗: {e}")
        return context


class AnalysisStage(PipelineStage):
    """分析階段"""
    
    @property
    def name(self) -> str:
        return "Analysis"
    
    def __init__(self, analysis_service):
        self.analysis_service = analysis_service
    
    def should_skip(self, context: PipelineContext) -> bool:
        return context.track_manager is None
    
    def process(self, context: PipelineContext) -> PipelineContext:
        results = self.analysis_service.process(context.track_manager)
        context.analysis_output = AnalysisOutput(results=results)
        return context


class SegmentExportStage(PipelineStage):
    """分段 JSON 導出階段"""
    
    @property
    def name(self) -> str:
        return "SegmentExport"
    
    def __init__(self, export_service, config: Dict[str, Any]):
        self.export_service = export_service
        self.config = config
    
    def should_skip(self, context: PipelineContext) -> bool:
        if not context.metadata.get('save_results', True):
            return True
        if context.analysis_output is None:
            return True
        # 檢查是否有活躍軌跡
        if context.track_manager is None:
            return True
        active_tracks = context.track_manager.get_all_tracks(removed=False)
        return len(active_tracks) == 0
    
    def process(self, context: PipelineContext) -> PipelineContext:
        import numpy as np
        
        video_info_dict = context.video_info.to_dict()
        analysis_results = context.analysis_output.results
        track_manager = context.track_manager
        
        # 收集所有軌跡的完整資料結構
        complete_data_list = []
        active_track_ids = {track.track_id for track in track_manager.get_all_tracks(removed=False)}
        segment_type = self.config.get("export_segment_type", "combined")
        
        for track_id, analysis_result in analysis_results.items():
            if track_id not in active_track_ids:
                continue
            try:
                segments = analysis_result.get_segments(segment_type)
                if not segments:
                    continue
                
                for segment_index, (start_frame, end_frame) in enumerate(segments):
                    frames_data = track_manager.get_frames_data(
                        track_id=track_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        use_normalized_keypoints=True,
                        output_format="soa"
                    )
                    
                    if frames_data is None:
                        continue
                    
                    # 提取統計特徵
                    features = {}
                    summary_df = analysis_result.get_metric(f"segment_summary_{segment_type}")
                    if summary_df is not None and not summary_df.empty and segment_index < len(summary_df):
                        segment_row = summary_df.iloc[segment_index]
                        features = segment_row.to_dict()
                    
                    # 組合 metadata
                    metadata = dict(video_info_dict)
                    for key, value in metadata.items():
                        if hasattr(value, 'item'):
                            metadata[key] = value.item()
                        elif isinstance(value, np.integer):
                            metadata[key] = int(value)
                        elif isinstance(value, np.floating):
                            metadata[key] = float(value)
                    
                    if 'video_name' not in metadata:
                        metadata['video_name'] = context.video_info.video_name
                    
                    metadata.update({
                        "track_id": track_id,
                        "segment_index": segment_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    })
                    
                    complete_data_list.append({
                        "frames": frames_data,
                        "metadata": metadata,
                        "features": features
                    })
                    
            except Exception as e:
                logger.error(f"處理軌跡 {track_id} 資料時發生錯誤: {e}", exc_info=True)
                continue
        
        # 導出
        if complete_data_list:
            exported_paths = self.export_service.export_segments_by_type(complete_data_list, video_info_dict)
            context.metadata['exported_segments'] = len(complete_data_list)
            logger.info(f"成功導出 {len(complete_data_list)} 個 {segment_type} 的 SOA JSON 資料")
        
        return context


class VisualizationStage(PipelineStage):
    """視覺化階段"""
    
    @property
    def name(self) -> str:
        return "Visualization"
    
    def __init__(self, visualization_service, config: Dict[str, Any]):
        self.visualization_service = visualization_service
        self.config = config
    
    def should_skip(self, context: PipelineContext) -> bool:
        if not context.metadata.get('save_results', True):
            return True
        if not self.config.get("visualization", {}).get("enabled", False):
            return True
        return context.analysis_output is None
    
    def process(self, context: PipelineContext) -> PipelineContext:
        video_info_dict = context.video_info.to_dict()
        target_segment_type = self.config.get("visualization", {}).get("segment_type")
        
        self.visualization_service.visualize_analysis_segments(
            track_manager=context.track_manager,
            analysis_results=context.analysis_output.results,
            video_info=video_info_dict,
            target_segment_type=target_segment_type
        )
        return context

