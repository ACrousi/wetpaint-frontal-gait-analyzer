# pose_extract/track_solution/track_repository.py
from .record import TrackRecord, TrackState # SegmentType 也可能需要，取決於 filter_by_segment_type 這類方法的實現
import numpy as np
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class TrackRepository:
    """管理所有追蹤記錄的倉庫"""
    def __init__(self):
        self.tracks: Dict[int, TrackRecord] = {}  # {track_id: TrackRecord}
        self.max_frame = 0

    def add_track(self, track_id, frame_id, position, state, score, mean, keypoints=None, keypoints_score=None):
        """添加或更新一個追蹤記錄"""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackRecord(track_id)

        self.tracks[track_id].add_detection(frame_id, position, state, score, mean, keypoints, keypoints_score)
        self.max_frame = max(self.max_frame, frame_id)

    def get_track(self, track_id) -> TrackRecord | None:
        """獲取特定ID的追蹤記錄"""
        return self.tracks.get(track_id)

    def get_all_tracks(self, removed=False) -> List[TrackRecord]:
        """獲取所有追蹤記錄"""
        if removed:
            return list(self.tracks.values())
        else:
            return [track for track in self.tracks.values() if track.states == TrackState.TRACKED]

    def get_track_ids(self, removed=False) -> List[int]:
        """獲取所有軌跡ID列表"""
        if removed:
            return list(self.tracks.keys())
        else:
            return [track_id for track_id, track in self.tracks.items() if track.states == TrackState.TRACKED]

    def get_active_tracks_at_frame(self, frame_id, removed=False) -> Dict[int, TrackRecord]:
        """獲取特定幀中的活躍追蹤"""
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if frame_id in track.positions:
                if removed or track.states != TrackState.REMOVED:
                    active_tracks[track_id] = track
        return active_tracks

    def _linear_interpolate(self, start_val, end_val, ratio):
        """通用線性插值函數，返回 NumPy 數組結果"""
        if start_val is None or end_val is None:
            return None

        if isinstance(start_val, (int, float)):
            return start_val + (end_val - start_val) * ratio
        elif isinstance(start_val, (list, tuple, np.ndarray)):
            # 確保輸入轉換為 NumPy 數組
            start_array = np.array(start_val, dtype=float) # 確保為 float 進行計算
            end_array = np.array(end_val, dtype=float)

            if start_array.shape != end_array.shape:
                logger.warning(f"Interpolation shape mismatch: {start_array.shape} vs {end_array.shape}")
                return None
            return start_array + (end_array - start_array) * ratio
        return None

    def interpolate_track(self, track_id, max_frames=15) -> bool:
        """對指定追蹤記錄進行插值補償"""
        track = self.get_track(track_id)
        if not track:
            logger.warning(f"Track {track_id} not found for interpolation.")
            return False

        if track.first_frame is None or track.last_frame is None:
            logger.debug(f"Track {track_id} has no first/last frame, skipping interpolation.")
            return False

        frame_ids = sorted(track.positions.keys())
        if len(frame_ids) <= 1:
            logger.debug(f"Track {track_id} has insufficient frames ({len(frame_ids)}) for interpolation.")
            return False

        changes_made = False
        for i in range(len(frame_ids) - 1):
            current_frame = frame_ids[i]
            next_frame = frame_ids[i + 1]
            gap = next_frame - current_frame

            if 1 < gap <= max_frames:
                for missing_frame in range(current_frame + 1, next_frame):
                    ratio = (missing_frame - current_frame) / float(gap)

                    interpolated_pos = self._linear_interpolate(track.positions.get(current_frame), track.positions.get(next_frame), ratio)
                    interpolated_score = self._linear_interpolate(track.scores.get(current_frame), track.scores.get(next_frame), ratio)
                    interpolated_mean = self._linear_interpolate(track.mean.get(current_frame), track.mean.get(next_frame), ratio)
                    interpolated_keypoints = self._linear_interpolate(track.keypoints.get(current_frame), track.keypoints.get(next_frame), ratio)
                    interpolated_kp_scores = self._linear_interpolate(track.keypoint_scores.get(current_frame), track.keypoint_scores.get(next_frame), ratio)

                    if interpolated_pos is not None and interpolated_score is not None and interpolated_mean is not None:
                        track.add_detection(
                            missing_frame,
                            tuple(interpolated_pos), # type: ignore
                            TrackState.TRACKED,
                            float(interpolated_score), # type: ignore
                            list(interpolated_mean), # type: ignore
                            interpolated_keypoints, # type: ignore
                            interpolated_kp_scores, # type: ignore
                            is_interpolated=True
                        )
                        changes_made = True
                if changes_made:
                    logger.debug(f"Interpolated frames between {current_frame} and {next_frame} for track {track_id}")


        return changes_made

    # def interpolate_all_tracks(self, max_frames=30) -> int:
    #     """對所有追蹤記錄進行插值補償"""
    #     interpolated_count = 0
    #     for track_id in list(self.tracks.keys()): # Iterate over a copy of keys if tracks can be modified
    #         if self.interpolate_track(track_id, max_frames):
    #             interpolated_count += 1
    #     logger.info(f"Interpolated {interpolated_count} tracks.")
        # return interpolated_count

    def filter_short_tracks(self, min_duration: int) -> int:
        """
        Mark tracks that have a duration (last_frame - first_frame + 1) less than the minimum threshold as removed.
        """
        removed_count = 0
        for track in self.tracks.values():
            if track.first_frame is not None and track.last_frame is not None:
                duration = track.last_frame - track.first_frame + 1
                if duration < min_duration:
                    track.states = TrackState.REMOVED
                    removed_count += 1
        if removed_count > 0:
            logger.info(f"Removed {removed_count} short tracks (duration < {min_duration} frames).")
        return removed_count

    def remove_tracks(self, track_ids_to_remove: List[int]):
        """Marks specified tracks as REMOVED."""
        count = 0
        for track_id in track_ids_to_remove:
            track = self.get_track(track_id)
            if track:
                track.states = TrackState.REMOVED
                count +=1
        return count
