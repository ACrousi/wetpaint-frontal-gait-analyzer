"""
例外層次結構

提供統一的例外類別，便於錯誤處理和除錯。
此模組獨立於其他業務邏輯，避免循環引入問題。

使用範例:
    from src.exceptions import ConfigLoadError, VideoProcessingError
    
    try:
        result = video_processor.extract_skeletons(video_info)
    except VideoReadError as e:
        logger.error(f"無法讀取影片: {e}")
    except TrackingError as e:
        logger.error(f"追蹤失敗: {e}")
    except VideoProcessingError as e:
        logger.error(f"影片處理錯誤: {e}")
"""


# ============================================================
# 基底類別
# ============================================================

class CoreServiceError(Exception):
    """所有 core services 錯誤的基類"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


# ============================================================
# 影片處理服務例外
# ============================================================

class VideoProcessingError(CoreServiceError):
    """影片處理服務相關錯誤的基類"""
    pass


class VideoReadError(VideoProcessingError):
    """讀取影片或影格失敗"""
    pass


class TrackingError(VideoProcessingError):
    """執行姿態追蹤時失敗"""
    pass


class SkeletonDataError(VideoProcessingError):
    """骨架資料無效或損毀"""
    pass


class VideoTranscodeError(VideoProcessingError):
    """影片轉碼失敗"""
    pass


# ============================================================
# 分析服務例外
# ============================================================

class AnalysisError(CoreServiceError):
    """分析服務相關錯誤的基類"""
    pass


class SegmentationError(AnalysisError):
    """分段處理失敗"""
    pass


class IdentificationError(AnalysisError):
    """目標識別失敗"""
    pass


class MetricCalculationError(AnalysisError):
    """指標計算失敗"""
    pass


# ============================================================
# 導出服務例外
# ============================================================

class ExportError(CoreServiceError):
    """導出服務相關錯誤的基類"""
    pass


class FileWriteError(ExportError):
    """檔案寫入失敗"""
    pass


class SerializationError(ExportError):
    """序列化失敗（JSON/Pickle）"""
    pass


# ============================================================
# 視覺化服務例外
# ============================================================

class VisualizationError(CoreServiceError):
    """視覺化服務相關錯誤的基類"""
    pass


class VideoWriteError(VisualizationError):
    """影片寫入失敗"""
    pass


# ============================================================
# 配置例外
# ============================================================

class ConfigurationError(CoreServiceError):
    """配置相關錯誤的基類"""
    pass


class ConfigValidationError(ConfigurationError):
    """配置驗證失敗"""
    pass


class ConfigLoadError(ConfigurationError):
    """配置載入失敗"""
    pass


# ============================================================
# 訓練相關例外
# ============================================================

class TrainingError(CoreServiceError):
    """訓練相關錯誤的基類"""
    pass


class SubprocessError(TrainingError):
    """子進程執行失敗"""
    pass


class DataGenerationError(TrainingError):
    """資料生成失敗"""
    pass


# ============================================================
# 預測相關例外
# ============================================================

class PredictionError(CoreServiceError):
    """預測相關錯誤的基類"""
    pass


class ModelLoadError(PredictionError):
    """模型載入失敗"""
    pass


class InferenceError(PredictionError):
    """推論執行失敗"""
    pass


class InvalidInputError(PredictionError):
    """輸入資料無效"""
    pass


# ============================================================
# 元數據相關例外
# ============================================================

class MetadataError(CoreServiceError):
    """元數據相關錯誤的基類"""
    pass


class MetadataLoadError(MetadataError):
    """元數據載入失敗"""
    pass


class VideoNotFoundError(MetadataError):
    """影片檔案不存在"""
    pass


# ============================================================
# CLI 相關例外
# ============================================================

class CLIError(CoreServiceError):
    """命令行相關錯誤的基類"""
    pass


class ArgumentError(CLIError):
    """命令行參數錯誤"""
    pass
