"""
配置模型 (Pydantic)

使用 Pydantic 定義配置模型，提供：
1. 型別驗證
2. 預設值
3. 自動錯誤訊息
4. IDE 自動補全

使用範例:
    from src.core.config_models import SkeletonExtractionConfig
    
    config = SkeletonExtractionConfig.model_validate(raw_dict)
    print(config.video_processing.fps)
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal, Tuple
from pathlib import Path


# ============================================================
# 影片處理配置
# ============================================================

class RTMOConfig(BaseModel):
    """RTMO 姿態估計模型配置"""
    weight_path: str = Field(default="./models/rtmo-l/end2end.onnx")
    input_size: List[int] = Field(default=[640, 640])
    
    @field_validator('input_size')
    @classmethod
    def validate_input_size(cls, v):
        if len(v) != 2:
            raise ValueError('input_size 必須包含 [width, height] 兩個值')
        if any(s <= 0 for s in v):
            raise ValueError('input_size 的值必須為正整數')
        return v


class BoTSORTConfig(BaseModel):
    """BoT-SORT 追蹤器配置"""
    track_high_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.7, ge=0.0, le=1.0)
    track_buffer: int = Field(default=15, ge=1)
    with_reid: bool = False
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    proximity_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    appearance_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    cmc_method: str = 'none'
    name: str = 'exp'
    ablation: bool = False
    mot20: bool = False
    fast_reid_config: str = ""
    fast_reid_weights: str = ""
    device: str = 'cuda'


class TranscodeConfig(BaseModel):
    """影片轉碼配置"""
    enabled: bool = True
    output_dir: str = "./outputs/transcoded"
    delete_after_processing: bool = False
    check_existing: bool = True
    use_nvenc: bool = False
    video_codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "fast"
    cq: int = Field(default=23, ge=0, le=51)
    target_fps: int = Field(default=30, ge=1, le=120)
    target_width: int = Field(default=1080, ge=1)
    target_height: int = Field(default=1920, ge=1)


class VideoProcessingConfig(BaseModel):
    """影片處理服務配置"""
    fps: int = Field(default=30, ge=1, le=120, description="目標 FPS")
    batch_size: int = Field(default=8, ge=1, le=128, description="RTMO 推論批次大小")
    orientation: Literal['portrait', 'landscape'] = Field(
        default='portrait',
        description="影片方向模式: portrait(直向) 或 landscape(橫向)"
    )
    rtmo: RTMOConfig = Field(default_factory=RTMOConfig)
    BoTSORT: BoTSORTConfig = Field(default_factory=BoTSORTConfig)
    transcode: TranscodeConfig = Field(default_factory=TranscodeConfig)


# ============================================================
# 分析配置
# ============================================================

class StandingMetricParams(BaseModel):
    """站立指標參數"""
    angle_threshold: float = Field(default=120, ge=0, le=180)


class AnkleAlternationMetricParams(BaseModel):
    """腳踝交替指標參數"""
    min_peak_distance: int = Field(default=7, ge=1)
    peak_prominence: float = Field(default=0.1, ge=0)
    height: float = Field(default=0.1, ge=0)
    min_alternating_cycles: int = Field(default=3, ge=1)
    gap_tolerance: int = Field(default=45, ge=0)
    smoothing_window: int = Field(default=5, ge=1)
    left_ankle_idx: int = Field(default=15, ge=0)
    right_ankle_idx: int = Field(default=16, ge=0)
    use_normalized: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class SegmentParams(BaseModel):
    """通用分段參數"""
    min_len: int = Field(default=10, ge=1)
    gap_tolerance: int = Field(default=30, ge=0)


class FixedLengthCuttingParams(BaseModel):
    """固定長度切割參數"""
    target_segment_type: str = "combined"
    fixed_length: int = Field(default=128, ge=1)
    min_len: int = Field(default=128, ge=1)
    gap_tolerance: int = Field(default=0, ge=0)


class ChildIdentificationParams(BaseModel):
    """目標識別參數"""
    enabled: bool = True
    median_head_ratio_threshold: float = Field(default=4.0, ge=0)
    median_sitting_index_threshold: float = Field(default=58.0, ge=0)


class PreprocessParams(BaseModel):
    """預處理參數"""
    interpolate_max_frames: int = Field(default=30, ge=0)
    min_track_duration_frames: int = Field(default=90, ge=1)


class KeypointsNormalizationParams(BaseModel):
    """關鍵點正規化參數"""
    image_width: int = Field(default=1920, ge=1)
    image_height: int = Field(default=1080, ge=1)
    center_keypoint: str = "shoulder_center"
    scale_method: str = "fixed_torso_length"
    reference_scale: float = Field(default=2.0, gt=0)
    enable_smoothing: bool = True
    gauss_sigma: float = Field(default=0.5, gt=0)


class AnalysisConfig(BaseModel):
    """分析服務配置"""
    standing_metric_params: StandingMetricParams = Field(default_factory=StandingMetricParams)
    ankle_alternation_metric_params: AnkleAlternationMetricParams = Field(default_factory=AnkleAlternationMetricParams)
    standing_segment_params: SegmentParams = Field(default_factory=SegmentParams)
    torso_ratio_segment_params: Dict[str, Any] = Field(default_factory=dict)
    walking_segment_params: Dict[str, Any] = Field(default_factory=dict)
    combined_segment_params: Dict[str, Any] = Field(default_factory=dict)
    fixed_length_cutting_params: FixedLengthCuttingParams = Field(default_factory=FixedLengthCuttingParams)
    step_time_metric_params: Dict[str, Any] = Field(default_factory=dict)
    child_identification_params: ChildIdentificationParams = Field(default_factory=ChildIdentificationParams)
    keypoint_score_threshold_params: Dict[str, Any] = Field(default_factory=dict)
    preprocess_params: PreprocessParams = Field(default_factory=PreprocessParams)
    keypoints_normalization_params: KeypointsNormalizationParams = Field(default_factory=KeypointsNormalizationParams)
    
    model_config = {"extra": "allow"}  # 允許額外欄位


# ============================================================
# 導出配置
# ============================================================

class SegSkeletonConfig(BaseModel):
    """分段骨架導出配置"""
    enabled: bool = True
    use_normalized_keypoints: bool = True
    segment_type: str = "fixed_length_cutting"
    output_dir: str = "./outputs/json"
    output_metadata_name: str = "analysis_metadata.csv"
    check_existing: bool = True


class RawSkeletonConfig(BaseModel):
    """原始骨架導出配置"""
    enabled: bool = True
    output_dir: str = "./outputs/raw_skeleton"
    check_existing: bool = True


class ExportConfig(BaseModel):
    """導出服務配置"""
    seg_skeleton: SegSkeletonConfig = Field(default_factory=SegSkeletonConfig)
    raw_skeleton: RawSkeletonConfig = Field(default_factory=RawSkeletonConfig)
    
    model_config = {"extra": "allow"}


# ============================================================
# 視覺化配置
# ============================================================

class DrawOptions(BaseModel):
    """繪圖選項"""
    show_interpolated: bool = True
    line_thickness: int = Field(default=2, ge=1)
    keypoint_radius: int = Field(default=4, ge=1)


class VideoWriterConfig(BaseModel):
    """影片寫入器配置"""
    fps: int = Field(default=30, ge=1, le=120)
    codec: str = "mp4v"


class VisualizationConfig(BaseModel):
    """視覺化服務配置"""
    enabled: bool = False
    output_dir: str = "./outputs/skeleton_videos"
    segment_type: Optional[str] = None
    overwrite: bool = True
    draw_options: DrawOptions = Field(default_factory=DrawOptions)
    video_writer: VideoWriterConfig = Field(default_factory=VideoWriterConfig)


# ============================================================
# Metadata 配置
# ============================================================

class MetadataConfig(BaseModel):
    """Metadata 配置"""
    use_csv: bool = True
    deduplicate: bool = False
    label_column: str = "age_months"
    metadata_path: str = ""


# ============================================================
# 頂層配置
# ============================================================

class SkeletonExtractionConfig(BaseModel):
    """骨架提取模式的完整配置"""
    video_processing: VideoProcessingConfig = Field(default_factory=VideoProcessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    export_segment_type: str = "fixed_length_cutting"
    
    model_config = {"extra": "allow"}  # 允許額外欄位以支援擴展
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkeletonExtractionConfig':
        """從字典建立配置（等同於 model_validate）"""
        return cls.model_validate(data)
