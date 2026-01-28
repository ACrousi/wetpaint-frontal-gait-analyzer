"""
Core Data Transfer Objects (DTO) 資料傳輸物件

提供型別安全的資料結構，取代 Dict[str, Any] 的使用。
支援向後相容：可從舊格式字典建立。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Optional, Any, List, Union
from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult


@dataclass
class VideoInfo:
    """
    影片資訊資料類別
    
    儲存影片處理所需的所有元資料。
    支援從舊格式 dict 建立以保持向後相容。
    """
    video_path: Path
    original_video: Optional[Path] = None
    case_id: Optional[str] = None
    fps: Optional[float] = None
    target_fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    effective_frame_count: Optional[int] = None
    
    # 額外元資料
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def video_name(self) -> str:
        """影片檔名"""
        return self.video_path.name
    
    @property
    def video_stem(self) -> str:
        """影片檔名（不含副檔名）"""
        return self.video_path.stem
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式（向後相容）"""
        result = {
            'video_path': str(self.video_path),
            'video_name': self.video_name,
        }
        if self.original_video:
            result['original_video'] = str(self.original_video)
        if self.case_id:
            result['case_id'] = self.case_id
        if self.fps is not None:
            result['fps'] = self.fps
        if self.target_fps is not None:
            result['target_fps'] = self.target_fps
        if self.width is not None:
            result['width'] = self.width
        if self.height is not None:
            result['height'] = self.height
        if self.effective_frame_count is not None:
            result['effective_frame_count'] = self.effective_frame_count
        
        # 合併額外元資料
        result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoInfo':
        """
        從字典建立 VideoInfo（向後相容）
        
        Args:
            data: 舊格式的 video_info 字典
            
        Returns:
            VideoInfo 實例
        """
        # 提取已知欄位
        known_fields = {
            'video_path', 'original_video', 'case_id', 'fps', 
            'target_fps', 'width', 'height', 'effective_frame_count'
        }
        
        # 處理 video_path
        video_path = data.get('video_path')
        if video_path is None:
            # 嘗試從 original_video 取得
            video_path = data.get('original_video')
        if video_path is None:
            raise ValueError("video_path 或 original_video 必須提供")
        
        # 收集額外欄位
        extra = {k: v for k, v in data.items() if k not in known_fields and k != 'video_name'}
        
        return cls(
            video_path=Path(video_path),
            original_video=Path(data['original_video']) if data.get('original_video') else None,
            case_id=data.get('case_id'),
            fps=data.get('fps'),
            target_fps=data.get('target_fps'),
            width=data.get('width'),
            height=data.get('height'),
            effective_frame_count=data.get('effective_frame_count'),
            extra=extra
        )


@dataclass
class AnalysisOutput:
    """
    分析輸出資料類別
    
    封裝分析服務的輸出結果。
    """
    results: Dict[int, AnalysisResult]
    active_track_ids: Set[int] = field(default_factory=set)
    
    def __post_init__(self):
        """初始化後自動填入 active_track_ids"""
        if not self.active_track_ids and self.results:
            self.active_track_ids = set(self.results.keys())
    
    def get_result(self, track_id: int) -> Optional[AnalysisResult]:
        """取得指定軌跡的分析結果"""
        return self.results.get(track_id)
    
    def get_segments(self, track_id: int, segment_type: str) -> List[tuple]:
        """取得指定軌跡和類型的所有片段"""
        result = self.results.get(track_id)
        if result is None:
            return []
        return result.get_segments(segment_type) or []
    
    @property
    def track_count(self) -> int:
        """分析的軌跡數量"""
        return len(self.results)
    
    @classmethod
    def from_dict(cls, results: Dict[int, AnalysisResult]) -> 'AnalysisOutput':
        """從分析結果字典建立"""
        return cls(results=results)


@dataclass
class VideoProcessingResult:
    """
    影片處理結果
    
    封裝骨架提取階段的輸出。
    """
    track_manager: TrackManager
    video_info: VideoInfo
    processing_time: float
    from_cache: bool = False
    
    @property
    def track_count(self) -> int:
        """追蹤的軌跡數量"""
        return len(self.track_manager.get_all_tracks(removed=False))


@dataclass  
class WorkflowResult:
    """
    工作流程結果
    
    封裝整個處理流程的最終輸出。
    """
    success: bool
    video_info: Optional[VideoInfo] = None
    analysis_output: Optional[AnalysisOutput] = None
    track_manager: Optional[TrackManager] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式（向後相容）"""
        result = {
            'success': self.success,
        }
        if self.video_info:
            result['video_info'] = self.video_info.to_dict()
        if self.analysis_output:
            result['analysis_results'] = self.analysis_output.results
        if self.track_manager:
            result['track_manager'] = self.track_manager
        if self.error:
            result['error'] = self.error
        return result
    
    @classmethod
    def failure(cls, error: str, video_info: Optional[VideoInfo] = None) -> 'WorkflowResult':
        """建立失敗結果的便利方法"""
        return cls(success=False, error=error, video_info=video_info)
    
    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> 'WorkflowResult':
        """從舊格式結果字典建立（向後相容）"""
        video_info = None
        if 'video_info' in data and data['video_info']:
            video_info = VideoInfo.from_dict(data['video_info'])
        
        analysis_output = None
        if 'analysis_results' in data and data['analysis_results']:
            analysis_output = AnalysisOutput.from_dict(data['analysis_results'])
        
        return cls(
            success=data.get('success', False),
            video_info=video_info,
            analysis_output=analysis_output,
            track_manager=data.get('track_manager'),
            error=data.get('error')
        )


# Type alias for backward compatibility
VideoInfoDict = Dict[str, Any]


def ensure_video_info(value: Union[VideoInfo, Dict[str, Any]]) -> VideoInfo:
    """
    確保輸入為 VideoInfo 物件（向後相容工具函數）
    
    Args:
        value: VideoInfo 實例或舊格式字典
        
    Returns:
        VideoInfo 實例
    """
    if isinstance(value, VideoInfo):
        return value
    return VideoInfo.from_dict(value)
