"""
Core 模組

提供骨架提取、分析和導出的核心功能。

使用範例:
    from src.core import SkeletonExtractionWorkflow, VideoInfo
    from src.core.config import load_config
    
    config = load_config("config/config.yaml", "skeleton_extraction")
    workflow = SkeletonExtractionWorkflow(config_dict)
    result = workflow.extract_analyze_and_export(video_info)
"""

# 從子模組匯出主要類別
from .config import (
    load_config,
    ConfigManager,
    SkeletonExtractionConfig,
)

from .services import (
    AnalysisService,
    ExportService,
    VideoProcessingService,
    SkeletonVisualizationService,
)

from .infrastructure import (
    VideoSource,
    VideoTranscodeService,
)

from .workflows import (
    SkeletonExtractionWorkflow,
    PredictionWorkflow
)

# DTO 和基礎設施
from .models import (
    VideoInfo,
    AnalysisOutput,
    WorkflowResult,
    VideoProcessingResult,
    ensure_video_info,
)

from .factory import (
    ServiceFactory,
    WorkflowFactory,
    create_workflow,
    create_workflow_from_yaml,
)

from .pipeline import (
    PipelineContext,
    PipelineStage,
    VideoProcessingPipeline,
)

from src.exceptions import (
    CoreServiceError,
    VideoProcessingError,
    AnalysisError,
    ExportError,
    ConfigurationError,
)


__all__ = [
    # 配置
    'load_config',
    'ConfigManager',
    'SkeletonExtractionConfig',
    # 服務
    'AnalysisService',
    'ExportService',
    'VideoProcessingService',
    'SkeletonVisualizationService',
    # 基礎設施
    'VideoSource',
    'VideoTranscodeService',
    # 工作流程
    'SkeletonExtractionWorkflow',
    'ResGCNTrainingWorkflow',
    # DTO
    'VideoInfo',
    'AnalysisOutput',
    'WorkflowResult',
    'VideoProcessingResult',
    'ensure_video_info',
    # 工廠
    'ServiceFactory',
    'WorkflowFactory',
    'create_workflow',
    'create_workflow_from_yaml',
    # Pipeline
    'PipelineContext',
    'PipelineStage',
    'VideoProcessingPipeline',
    # 例外
    'CoreServiceError',
    'VideoProcessingError',
    'AnalysisError',
    'ExportError',
    'ConfigurationError',
]
