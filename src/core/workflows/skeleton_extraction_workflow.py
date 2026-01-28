import logging
import time
import numpy as np
from typing import Dict, Any, List, Union
from ..services.video_processing_service import VideoProcessingService
from ..services.analysis_service import AnalysisService
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult
from ..services.export_service import ExportService
from ..services.visualization_service import SkeletonVisualizationService

# 匯入新的 DTO 模型
from ..models import VideoInfo, AnalysisOutput, WorkflowResult, ensure_video_info

# 匯入 Pipeline
from ..pipeline import (
    PipelineContext,
    VideoProcessingPipeline,
    CacheCheckStage,
    SkeletonExtractionStage,
    RawSkeletonExportStage,
    AnalysisStage,
    SegmentExportStage,
    VisualizationStage,
)

from pathlib import Path

logger = logging.getLogger(__name__)

class SkeletonExtractionWorkflow:
    """骨架提取、分析和導出整合工作流程
    
    使用 Pipeline 模式組織處理流程，支援：
    - 快取檢查與載入
    - 骨架提取
    - 原始骨架導出
    - 分析處理
    - 分段 JSON 導出
    - 視覺化
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各服務 (VideoSource 現在使用 FFmpeg Pipe，無需 TranscodeService)
        self.video_processor = VideoProcessingService(config.get("video_processing", {}))
        self.analysis_service = AnalysisService(config.get("analysis", {}))
        self.export_service = ExportService(config.get("export", {}))
        self.visualization_service = SkeletonVisualizationService(config.get("visualization", {}))
        
        # 快取設定
        self._raw_skeleton_config = config.get("export", {}).get("raw_skeleton", {})
        self._cache_dir = Path(self._raw_skeleton_config.get("output_dir", "./outputs/raw_skeleton"))
        
        # 建立 Pipeline
        self._pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> VideoProcessingPipeline:
        """建立處理 Pipeline"""
        check_existing = self._raw_skeleton_config.get("check_existing", True)
        
        return (VideoProcessingPipeline("SkeletonExtractionPipeline")
            .add_stage(CacheCheckStage(self._cache_dir, check_existing))
            .add_stage(SkeletonExtractionStage(self.video_processor))
            .add_stage(RawSkeletonExportStage(self.export_service))
            .add_stage(AnalysisStage(self.analysis_service))
            .add_stage(SegmentExportStage(self.export_service, self.config))
            .add_stage(VisualizationStage(self.visualization_service, self.config))
        )

    def extract_analyze_and_export(
        self, 
        video_info: Union[VideoInfo, Dict[str, Any]], 
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        整合的骨架提取、分析和導出工作流程

        Args:
            video_info: 影片資訊（支援 VideoInfo 或舊格式 dict）
            save_results: 是否儲存結果到檔案

        Returns:
            包含處理結果的字典（向後相容格式）
        """
        # 確保使用 VideoInfo 物件（向後相容）
        video_info_obj = ensure_video_info(video_info) if isinstance(video_info, dict) else video_info
        
        filename = video_info_obj.video_name
        logger.info(f"開始整合處理流程: {filename}")
        
        # 建立 Pipeline Context
        context = PipelineContext(
            video_info=video_info_obj,
            config=self.config
        )
        context.metadata['save_results'] = save_results
        
        # 執行 Pipeline
        start_time = time.time()
        context = self._pipeline.run(context)
        total_time = time.time() - start_time
        
        logger.info(f"整合處理流程完成: {filename} (總耗時: {total_time:.2f} 秒)")
        
        # 轉換為向後相容的結果格式
        return self._context_to_result(context)
    
    def _context_to_result(self, context: PipelineContext) -> Dict[str, Any]:
        """將 PipelineContext 轉換為向後相容的結果字典"""
        if context.is_success:
            result = WorkflowResult(
                success=True,
                video_info=context.video_info,
                analysis_output=context.analysis_output,
                track_manager=context.track_manager
            )
        else:
            error_msg = "; ".join(context.errors) if context.errors else "Unknown error"
            result = WorkflowResult.failure(error_msg, context.video_info)
        
        return result.to_dict()

