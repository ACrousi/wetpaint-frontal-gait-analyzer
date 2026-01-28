"""
向後相容模組

此模組從 src.exceptions 重新導出所有例外類別，
以支援現有的 from src.core.exceptions import 語法。

建議使用新的 import 路徑：
    from src.exceptions import ConfigLoadError, VideoProcessingError
"""

# 從新位置重新導出所有例外
from src.exceptions import (
    # 基底類別
    CoreServiceError,
    # 影片處理
    VideoProcessingError,
    VideoReadError,
    TrackingError,
    SkeletonDataError,
    VideoTranscodeError,
    # 分析
    AnalysisError,
    SegmentationError,
    IdentificationError,
    MetricCalculationError,
    # 導出
    ExportError,
    FileWriteError,
    SerializationError,
    # 視覺化
    VisualizationError,
    VideoWriteError,
    # 配置
    ConfigurationError,
    ConfigValidationError,
    ConfigLoadError,
    # 訓練
    TrainingError,
    SubprocessError,
    DataGenerationError,
    # 預測
    PredictionError,
    ModelLoadError,
    InferenceError,
    InvalidInputError,
    # 元數據
    MetadataError,
    MetadataLoadError,
    VideoNotFoundError,
    # CLI
    CLIError,
    ArgumentError,
)

__all__ = [
    'CoreServiceError',
    'VideoProcessingError',
    'VideoReadError',
    'TrackingError',
    'SkeletonDataError',
    'VideoTranscodeError',
    'AnalysisError',
    'SegmentationError',
    'IdentificationError',
    'MetricCalculationError',
    'ExportError',
    'FileWriteError',
    'SerializationError',
    'VisualizationError',
    'VideoWriteError',
    'ConfigurationError',
    'ConfigValidationError',
    'ConfigLoadError',
    'TrainingError',
    'SubprocessError',
    'DataGenerationError',
    'PredictionError',
    'ModelLoadError',
    'InferenceError',
    'InvalidInputError',
    'MetadataError',
    'MetadataLoadError',
    'VideoNotFoundError',
    'CLIError',
    'ArgumentError',
]