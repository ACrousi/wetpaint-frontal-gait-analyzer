"""檔名生成工具模組

此模組負責根據影片資訊和分段資料生成標準化的檔名格式。
"""

from pathlib import Path
from typing import Dict, Any
from src.pose_extract.track_solution import TrackRecord


def generate_filename_from_segment(
    video_info: Dict[str, Any],
    segment: tuple,  # (start_frame, end_frame)
    segment_type: str,
    segment_index: int,
    track_id: int,
    processing_flags: Dict[str, Any] = None
) -> str:
    """
    根據影片資訊和分段資料生成檔名
    
    生成格式：{video_base_name}_id{track_id}_{segment_type}_seg{index}_{start}-{end}_{flags}
    例如：20240427_050412000_iOS_id2_combined_seg0_447-578_norm
    
    Args:
        video_info: 影片資訊字典，包含 video_name 等
        segment: 分段資訊 (start_frame, end_frame)
        segment_type: 分段類型，如 "combined"
        segment_index: 分段索引
        track_id: 軌跡 ID
        processing_flags: 處理過程的旗標，如 {"normalized": True}
        
    Returns:
        生成的檔名（不含副檔名）
    """
    # 從 video_info 取得基本名稱（支援多種來源）
    if 'video_name' in video_info and video_info['video_name']:
        base_name = Path(video_info['video_name']).stem
    elif 'video_path' in video_info and video_info['video_path']:
        base_name = Path(video_info['video_path']).stem
    elif 'original_video' in video_info and video_info['original_video']:
        base_name = Path(video_info['original_video']).stem
    else:
        base_name = 'unknown'
    
    # 分段資訊
    start_frame, end_frame = segment
    
    # 組合基本檔名
    filename_parts = [
        base_name,
        f"id{track_id}",
        segment_type,
        f"seg{segment_index}",
        f"{start_frame}-{end_frame}"
    ]
    
    # 添加處理旗標
    if processing_flags:
        flag_parts = []
        if processing_flags.get("normalized", False):
            flag_parts.append("norm")
        if processing_flags.get("standardized", False):
            flag_parts.append("std")
        
        if flag_parts:
            filename_parts.extend(flag_parts)
    
    return "_".join(filename_parts)


def generate_filename_from_track_segment(
    video_info: Dict[str, Any],
    track: TrackRecord,
    segment_type: str,
    segment_index: int,
    segment: tuple,
    use_normalized: bool = False
) -> str:
    """
    從軌跡和分段生成檔名的便利函數
    
    Args:
        video_info: 影片資訊
        track: 軌跡記錄
        segment_type: 分段類型
        segment_index: 分段索引
        segment: 分段範圍 (start_frame, end_frame)
        use_normalized: 是否使用正規化關鍵點
        
    Returns:
        生成的檔名
    """
    processing_flags = {"normalized": use_normalized}
    
    return generate_filename_from_segment(
        video_info=video_info,
        segment=segment,
        segment_type=segment_type,
        segment_index=segment_index,
        track_id=track.track_id,
        processing_flags=processing_flags
    )