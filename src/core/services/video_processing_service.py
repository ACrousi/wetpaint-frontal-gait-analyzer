import logging
from pathlib import Path
from typing import Dict, Any, Union

from ..infrastructure.video_source import VideoSource
from src.pose_extract.track_solution import TrackManager
from src.pose_extract.wrapper import RTMOWrapper
from src.pose_extract.wrapper import BoTSORTWrapper

# 導入例外類別
from src.exceptions import VideoReadError, TrackingError

# 導入新的 DTO 模型
from ..models import VideoInfo, ensure_video_info

logger = logging.getLogger(__name__)


class VideoProcessingService:
    """
    服務：負責從影片中提取原始的姿態與追蹤資料。
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理資源
        if hasattr(self, 'pose_model') and self.pose_model:
            self.pose_model.close()
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.close()
        return False

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化模型
        self.pose_model = RTMOWrapper(self.config.get("rtmo", {}))

    def extract_skeletons(self, video_info: Union[VideoInfo, Dict[str, Any]]) -> TrackManager:
        """
        主要公開方法：從影片中提取骨架並回傳 TrackManager。
        如果失敗，會拋出 VideoProcessingError 的子類別例外。
        
        Args:
            video_info: 影片資訊（支援 VideoInfo 或舊格式 dict）
        
        Note: 快取檢查由 SkeletonExtractionWorkflow 負責，此方法專注於純骨架提取。
        """
        # 支援 VideoInfo 和 dict 兩種輸入
        video_info_obj = ensure_video_info(video_info) if isinstance(video_info, dict) else video_info
        video_info_dict = video_info_obj.to_dict() if isinstance(video_info, VideoInfo) else video_info
        
        video_path = video_info_obj.video_path
        filename = video_info_obj.video_name

        # 初始化Tracker
        self.tracker = BoTSORTWrapper(self.config.get("BoTSORT", {}))
        track_manager = TrackManager()

        # 執行追蹤
        logger.info(f"開始執行骨架提取追蹤: {filename}")
        try:
            self._run_tracking_on_video(track_manager, video_info_dict)
        except (VideoReadError, TrackingError):
            raise
        except Exception as e:
            logger.error(f"執行追蹤時發生未預期錯誤: {e}", exc_info=True)
            raise TrackingError(f"追蹤影片 '{filename}' 時發生未預期錯誤") from e

        return track_manager

    def _run_tracking_on_video(self, track_manager: TrackManager, video_info: Union[VideoInfo, Dict[str, Any]]):
        """
        內部方法：執行影片追蹤的核心邏輯。
        使用 batch 處理提升 RTMO 推論效率，tracking 仍按順序執行。
        """
        try:
            video_source = VideoSource(
                str(Path(video_info['video_path'])),
                target_fps=video_info.get('target_fps', self.config.get("fps", 30))
            )
            video_source.open()
        except Exception as e:
            raise VideoReadError(f"無法開啟影片資源: {video_info['video_path']}") from e

        batch_size = self.config.get("batch_size", 8)
        frame_id = 0
        
        while True:
            # 1. 讀取 batch_size 個幀
            frames = []
            frame_ids = []
            for _ in range(batch_size):
                ret, frame = video_source.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_ids.append(frame_id)
                frame_id += 1
            
            if not frames:
                break
            
            try:
                # 2. Batch 推論 RTMO
                batch_detections = self.pose_model.batch_inference(frames)
                
                # 3. 逐幀進行 tracking (tracking 必須順序執行以維持時序一致性)
                for i, f_id in enumerate(frame_ids):
                    bboxes, keypoints, box_scores, kpt_scores = batch_detections[i]
                    tracks_output = self.tracker.update(
                        frames[i], bboxes, box_scores, keypoints, kpt_scores
                    )
                    track_manager.update_from_tracker(tracks_output, f_id)
            except Exception as e:
                raise TrackingError(f"在處理幀 {frame_ids[0]}-{frame_ids[-1]} 時失敗") from e

        logger.info(f"追蹤完成，共處理 {frame_id} 幀。")