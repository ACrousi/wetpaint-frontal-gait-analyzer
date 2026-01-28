import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal

from src.pose_extract.track_solution.record import TrackRecord, TrackState
from src.pose_extract.track_solution.repository import TrackRepository
from src.utils import setup_logging

logger = setup_logging(level=logging.DEBUG)


class TrackManager:
    """追蹤管理器
    
    專注於軌跡資料的管理和操作，包括：
    1. 透過 Repository 儲存和查詢軌跡資料
    2. 提供軌跡資料的過濾、插值等操作
    3. 從不同來源（如 COCO 格式）載入軌跡
    """
    
    def __init__(self):
        self.repository = TrackRepository()
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

    def update_from_tracker(self, tracker_output: List[Any], frame_id: int):
        """從追蹤器輸出更新軌跡資料庫"""
        for track_obj in tracker_output:
            self.repository.add_track(
                track_obj.track_id,
                frame_id,
                np.array([
                    track_obj.tlwh[0], 
                    track_obj.tlwh[1], 
                    track_obj.tlwh[0] + track_obj.tlwh[2], 
                    track_obj.tlwh[1] + track_obj.tlwh[3]
                ]),
                TrackState.TRACKED,
                track_obj.score,
                track_obj.mean,
                track_obj.keypoints,
                track_obj.keypoint_scores
            )

    def interpolate_track(self, track_id: int, max_frames: int = 30) -> Union[bool, int]:
        """對追蹤記錄進行插值補償"""
        return self.repository.interpolate_track(track_id, max_frames)

    def remove_short_tracks(self, min_duration: int) -> int:
        """移除短軌跡"""
        return self.repository.filter_short_tracks(min_duration)

    def load_from_coco_data(self, coco_data: dict):
        """從 COCO 格式資料載入追蹤記錄
            start_frame, end_fframe 沒有意義
        """
        if 'tracks' not in coco_data:
            logger.warning("COCO 資料中沒有 tracks 欄位")
            return

        info = coco_data.get('info', {})

        if self.start_frame is None:
            self.start_frame = 0
            logger.info(f"設定 start_frame 為: {self.start_frame}")

        total_analyze_frame = info.get('total_analyze_frame')
        if total_analyze_frame is not None:
            self.end_frame = total_analyze_frame
            logger.info(f"設定 end_frame 為: {self.end_frame}")
        else:
            logger.warning("COCO 資料中沒有 total_analyze_frame 資訊，且未提供 video_source")

        for track_data in coco_data['tracks']:
            track_id = track_data['track_id']
            keypoints_data = track_data.get('keypoints', {})
            bounding_boxes_data = track_data.get('bounding_boxes', {})
            scores_data = track_data.get('scores', {})
            keypoint_scores_data = track_data.get('keypoint_scores', {})

            # 取得所有幀 ID
            all_frame_ids = set()
            all_frame_ids.update(keypoints_data.keys())
            all_frame_ids.update(bounding_boxes_data.keys())
            all_frame_ids.update(scores_data.keys())
            all_frame_ids.update(keypoint_scores_data.keys())

            # 重建 TrackRecord
            for frame_id_str in all_frame_ids:
                frame_id = int(frame_id_str)

                # 載入關鍵點
                if frame_id_str not in keypoints_data:
                    continue
                keypoints_array = np.array(keypoints_data[frame_id_str])

                # 載入邊界框
                if frame_id_str in bounding_boxes_data:
                    bbox = np.array(bounding_boxes_data[frame_id_str])
                else:
                    bbox = np.array([0, 0, 100, 100])

                # 載入分數
                if frame_id_str in scores_data:
                    score = float(scores_data[frame_id_str])
                else:
                    score = 1.0

                # 載入關鍵點分數
                if frame_id_str in keypoint_scores_data:
                    kp_scores = np.array(keypoint_scores_data[frame_id_str])
                else:
                    kp_scores = np.ones(len(keypoints_array))

                self.repository.add_track(
                    track_id,
                    frame_id,
                    bbox,
                    TrackState.TRACKED,
                    score,
                    None,  # mean
                    keypoints_array,
                    kp_scores
                )

        logger.info(f"從 COCO 資料載入了 {len(coco_data['tracks'])} 個追蹤記錄")

    def has_sufficient_skeleton_data(self) -> bool:
        """檢查是否有足夠的骨架資料"""
        active_tracks = self.repository.get_all_tracks(removed=False)

        if not active_tracks:
            return False

        # 檢查是否至少有一個軌跡有關鍵點資料
        for track in active_tracks:
            if track.keypoints and len(track.keypoints) > 0:
                return True

        return False

    def get_track(self, track_id: int) -> Optional[TrackRecord]:
        """取得特定軌跡"""
        return self.repository.get_track(track_id)

    def _convert_frames_to_soa(self, frames) -> Optional[np.ndarray]:
        """
        將 Array of Structures (AoS) 的 frames 轉換為 Structure of Arrays (SoA) 的 numpy array

        Args:
            frames: 可以是 list of dicts (AoS) 或已經是 numpy array (SoA)

        Returns:
            numpy array 格式 (T, V, C)，其中 T=時間, V=關節點, C=通道(x,y,score)
        """
        if isinstance(frames, np.ndarray):
            return frames  # 已經是 SoA 格式

        if not isinstance(frames, list) or len(frames) == 0:
            return np.array([]).reshape(0, 0, 0)

        try:
            T = len(frames)
            # 假設所有 frames 有相同數量的 keypoints
            V = len(frames[0]['keypoints'])
            C = 3  # x, y, score

            soa = np.zeros((T, V, C))

            for t, frame in enumerate(frames):
                keypoints = frame['keypoints']
                scores = frame['keypoint_scores']

                if len(keypoints) != V or len(scores) != V:
                    logger.warning(f"Frame {t} 的 keypoints 或 scores 長度不一致")
                    continue

                for v in range(V):
                    soa[t, v, 0] = keypoints[v][0]  # x
                    soa[t, v, 1] = keypoints[v][1]  # y
                    soa[t, v, 2] = scores[v]        # score

            return soa

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"轉換 frames 時發生錯誤: {e}")
            return None

    def get_frames_data(self, track_id: int, start_frame: int, end_frame: int,
                       use_normalized_keypoints: bool = True,
                       output_format: Literal["aos", "soa"] = "aos") -> Dict[str, Any]:
        """
        根據給定的 track_id 和幀範圍，取得該範圍內的骨架資料。

        Args:
            track_id: 軌跡 ID
            start_frame: 開始幀
            end_frame: 結束幀
            use_normalized_keypoints: 是否使用正規化關鍵點
            output_format: 輸出格式，"aos" (Array of Structures) 或 "soa" (Structure of Arrays)

        Returns:
            如果 output_format="aos": {
                "frames": [ { frame資料... }, ... ]
            }

            如果 output_format="soa": {
                "frames": numpy array (T, V, C)
            }
        """
        track = self.repository.get_track(track_id)
        if not track:
            return [] if output_format == "aos" else np.array([]).reshape(0, 0, 0)

        def to_list(val):
            try:
                if hasattr(val, "tolist"):
                    return val.tolist()
            except Exception:
                pass
            return val

        if use_normalized_keypoints:
            keypoints_source_attr = 'keypoints_normalized'
        else:
            keypoints_source_attr = 'keypoints'

        if output_format == "soa":
            # SOA 格式：收集所有 frame 資料，然後轉換為 numpy array
            frames_list = []
            keypoints_data_dict = getattr(track, keypoints_source_attr, None)

            for frame_id in range(start_frame, end_frame + 1):
                kp = keypoints_data_dict.get(frame_id)
                kp_scores = track.keypoint_scores.get(frame_id)

                if kp is not None and kp_scores is not None:
                    frame_entry = {
                        "keypoints": to_list(kp),
                        "keypoint_scores": to_list(kp_scores)
                    }
                    frames_list.append(frame_entry)

            # 轉換為 SOA 格式
            if frames_list:
                soa_data = self._convert_frames_to_soa(frames_list)
                return soa_data
            else:
                return np.array([]).reshape(0, 0, 0)

        else:  # output_format == "aos"
            # AOS 格式：保持原有結構
            frames = []
            keypoints_data_dict = getattr(track, keypoints_source_attr, None)

            for frame_id in range(start_frame, end_frame + 1):
                kp = keypoints_data_dict.get(frame_id)
                kp_scores = track.keypoint_scores.get(frame_id)
                bbox = track.positions.get(frame_id)
                score = track.scores.get(frame_id)
                is_interp = track.is_interpolated.get(frame_id, False)

                frame_entry = {
                    "frame_id": frame_id,
                    "keypoints": to_list(kp) if kp is not None else None,
                    "keypoint_scores": to_list(kp_scores) if kp_scores is not None else None,
                    "bbox": to_list(bbox) if bbox is not None else None,
                    "score": float(score) if score is not None else None,
                    "is_interpolated": bool(is_interp)
                }

                frames.append(frame_entry)

            return frames

    
    def get_all_tracks(self, removed: bool = False) -> List[TrackRecord]:
        """取得所有軌跡"""
        return self.repository.get_all_tracks(removed=removed)

    def get_track_ids(self, removed: bool = False) -> List[int]:
        """取得所有軌跡 ID"""
        return self.repository.get_track_ids(removed=removed)

    def get_active_tracks_at_frame(self, frame_id: int, removed: bool = False) -> Dict[int, TrackRecord]:
        """取得特定幀的活躍軌跡"""
        return self.repository.get_active_tracks_at_frame(frame_id, removed)

    def set_frame_range(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """設定處理的幀範圍"""
        if start_frame is not None:
            self.start_frame = max(0, start_frame)
        if end_frame is not None:
            self.end_frame = end_frame
        
        logger.info(f"設定幀範圍: {self.start_frame} - {self.end_frame}")

    def get_track_statistics(self) -> Dict[str, Any]:
        """取得軌跡統計資訊"""
        all_tracks = self.repository.get_all_tracks(removed=True)
        active_tracks = self.repository.get_all_tracks(removed=False)
        
        stats = {
            "total_tracks": len(all_tracks),
            "active_tracks": len(active_tracks),
            "removed_tracks": len(all_tracks) - len(active_tracks),
            "track_durations": []
        }
        
        for track in active_tracks:
            if track.first_frame is not None and track.last_frame is not None:
                duration = track.last_frame - track.first_frame + 1
                stats["track_durations"].append(duration)
        
        if stats["track_durations"]:
            stats["avg_duration"] = sum(stats["track_durations"]) / len(stats["track_durations"])
            stats["max_duration"] = max(stats["track_durations"])
            stats["min_duration"] = min(stats["track_durations"])
        else:
            stats["avg_duration"] = 0
            stats["max_duration"] = 0
            stats["min_duration"] = 0
            
        return stats

    # ===== 軌跡標記和管理方法 =====
    
    def mark_tracks_removed(self, track_ids: List[int]) -> int:
        """標記指定軌跡為已移除狀態
        
        Args:
            track_ids: 要標記為移除的軌跡 ID 列表
            
        Returns:
            實際標記為移除的軌跡數量
        """
        return self.repository.remove_tracks(track_ids)
    
    def get_tracks_for_analysis(self, track_id: Optional[int] = None,
                              removed: bool = False) -> List[TrackRecord]:
        """獲取用於分析的軌跡資料
        
        Args:
            track_id: 指定軌跡 ID，None 表示獲取所有軌跡
            removed: 是否包含已移除的軌跡
            
        Returns:
            軌跡記錄列表
        """
        if track_id is not None:
            track = self.repository.get_track(track_id)
            return [track] if track else []
        else:
            return self.repository.get_all_tracks(removed=removed)

    # ===== 臨時保留的方法（用於視覺化服務） =====
    
    # def create_track_video(self, output_path: str, start_frame: Optional[int] = None,
    #                       end_frame: Optional[int] = None, visualization_options: Optional[Dict[str, Any]] = None,
    #                       normalized: bool = False, target_canvas_size: Optional[Tuple[int, int]] = None) -> List[str]:
    #     """創建軌跡視頻（臨時保留，應由 VisualizationService 處理）"""
    #     logger.warning("create_track_video 方法已廢棄，請使用 VisualizationService")
    #     return []

    # def export_segments_to_csv(self, output_dir: str, base_filename: str,
    #                           track_ids: Optional[List[int]] = None,
    #                           segment_types_to_export: Optional[List[Any]] = None,
    #                           min_segment_len: int = 1,
    #                           use_normalized_keypoints: bool = False) -> List[str]:
    #     """導出分段到 CSV（臨時保留，應由 ExportService 處理）"""
    #     logger.warning("export_segments_to_csv 方法已廢棄，請使用 ExportService")
    #     return []

    # def plot_ankle_difference_for_tracks(self, output_base_dir: str, base_filename: str,
    #                                    track_ids: Optional[List[int]] = None) -> List[str]:
    #     """繪製腳踝差異圖（臨時保留，應由 VisualizationService 處理）"""
    #     logger.warning("plot_ankle_difference_for_tracks 方法已廢棄，請使用 VisualizationService")
    #     return []

    # def draw_tracks_on_frame(self, frame: np.ndarray, frame_id: int, 
    #                        options: Optional[Dict[str, Any]] = None,
    #                        target_canvas_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    #     """在幀上繪製軌跡（臨時保留，應由 VisualizationService 處理）"""
    #     logger.warning("draw_tracks_on_frame 方法已廢棄，請使用 VisualizationService")
    #     return frame