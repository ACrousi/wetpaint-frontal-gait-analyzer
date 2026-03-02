from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import scipy.signal as sig


from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution import TrackRecord
from .base import logger

class Preprocessor(ABC):
    """預處理器的基類，可以修改 TrackRecord"""

    @property
    @abstractmethod
    def name(self) -> str:
        """預處理器的唯一名稱"""
        pass

    @abstractmethod
    def process(self, track: TrackRecord) -> Dict[str, Any]:
        """
        執行預處理操作，可以修改 TrackRecord

        Args:
            track: 要處理的軌跡記錄

        Returns:
            處理結果的統計信息
        """
        pass

class KeypointScoreThresholdPreprocessor(Preprocessor):
    """
    根據指定的 threshold 過濾 keypoint_scores，
    當某個 frame 中所有 keypoint 的 score 都低於 threshold 時，
    將該 frame 的 keypoints、keypoint_scores、bounding box 和 scores 都刪除。
    """

    def __init__(self, threshold: float = 0.5):
        """
        初始化 KeypointScoreThresholdPreprocessor

        Args:
            threshold: keypoint score 的閾值，當 frame 中所有 keypoint score 都低於此值時，刪除該 frame 的所有資料
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "keypoint_score_threshold"

    def process(self, track_manager: TrackManager) -> Dict[str, Any]:
        all_tracks = track_manager.get_all_tracks(removed=False)
        total_removed_frames = 0
        total_processed_frames = 0

        for track in all_tracks:
            # 這裡的邏輯與您原本的 process 方法完全相同
            # 它會直接修改 track 物件
            stats = self._process_single_track(track)
            total_processed_frames += stats["processed_frames"]
            total_removed_frames += stats["removed_frames"]

        logger.info(f"KeypointScoreThreshold 處理完成。總共刪除 {total_removed_frames} 個幀。")
        return {
            "processed_tracks": len(all_tracks),
            "removed_frames": total_removed_frames
        }

    def _process_single_track(self, track: TrackRecord) -> Dict[str, Any]:
        """
        分析 track 中每一幀的 keypoint_scores，
        當該 frame 中所有 keypoint score 都低於 threshold 時，
        將該 frame 的 keypoints、keypoint_scores、bounding box 和 scores 都刪除。

        Args:
            track: 要處理的 TrackRecord

        Returns:
            處理統計信息
        """
        if not track.keypoints or not track.keypoint_scores:
            logger.warning(f"Track {track.track_id}: 沒有 keypoints 或 keypoint_scores 資料可供過濾。")
            return {"processed_frames": 0, "removed_frames": 0}

        processed_frames = 0
        removed_frames = []

        # 需要檢查的 frame 列表
        frames_to_check = list(track.keypoints.keys())

        for fid in frames_to_check:
            # 檢查該幀是否有 keypoints 和 keypoint_scores
            keypoints_for_frame = track.keypoints.get(fid)
            keypoint_scores_for_frame = track.keypoint_scores.get(fid)

            if keypoints_for_frame is None or keypoint_scores_for_frame is None:
                logger.debug(f"Track {track.track_id} Frame {fid}: 缺少 keypoints 或 keypoint_scores，跳過處理。")
                continue

            # 確保 keypoints 是 numpy array 並且有正確的形狀
            import numpy as np
            if not isinstance(keypoints_for_frame, np.ndarray):
                logger.debug(f"Track {track.track_id} Frame {fid}: keypoints 不是 numpy array，跳過處理。")
                continue

            # 確保 keypoints 有正確的形狀 (17, 2) 或 (17, 3)
            if keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: keypoints 形狀不正確 {keypoints_for_frame.shape}，跳過處理。")
                continue

            # 處理 keypoint_scores，可能是 numpy array 或其他格式
            scores = self._extract_scores(keypoint_scores_for_frame, fid, track.track_id)
            if scores is None:
                continue

            # 確保 scores 的長度與 keypoints 數量一致
            if len(scores) != keypoints_for_frame.shape[0]:
                logger.warning(f"Track {track.track_id} Frame {fid}: keypoint_scores 數量 ({len(scores)}) 與 keypoints 數量 ({keypoints_for_frame.shape[0]}) 不一致。")
                continue

            # 檢查是否所有 keypoint scores 都低於 threshold
            all_scores_below_threshold = all(score < self.threshold for score in scores)

            if all_scores_below_threshold:
                # 刪除該 frame 的所有相關資料
                if fid in track.keypoints:
                    del track.keypoints[fid]
                if fid in track.keypoint_scores:
                    del track.keypoint_scores[fid]
                if fid in track.positions:
                    del track.positions[fid]
                if fid in track.scores:
                    del track.scores[fid]
                if fid in track.mean:
                    del track.mean[fid]
                if fid in track.is_interpolated:
                    del track.is_interpolated[fid]

                removed_frames.append(fid)
                logger.debug(f"Track {track.track_id} Frame {fid}: 所有 keypoint scores 都低於 threshold ({self.threshold})，刪除該 frame 的所有資料。")

            processed_frames += 1

        # 更新 track 的 first_frame 和 last_frame
        if track.keypoints:
            remaining_frames = sorted(track.keypoints.keys())
            track.first_frame = remaining_frames[0]
            track.last_frame = remaining_frames[-1]
        else:
            track.first_frame = None
            track.last_frame = None

        logger.info(f"Track {track.track_id}: KeypointScoreThreshold 處理完成。處理了 {processed_frames} 幀，刪除了 {len(removed_frames)} 個 frames (threshold={self.threshold})。")

        return {
            "processed_frames": processed_frames,
            "removed_frames": len(removed_frames)
        }

    def _extract_scores(self, keypoint_scores_data: Any, frame_id: int, track_id: int) -> Optional[List[float]]:
        """
        從 keypoint_scores_data 中提取 scores 列表

        Args:
            keypoint_scores_data: keypoint_scores 資料，可能是 numpy array、list 或其他格式
            frame_id: 幀 ID，用於 log
            track_id: track ID，用於 log

        Returns:
            scores 列表，如果無法提取則返回 None
        """
        try:
            # 如果是 numpy array
            if isinstance(keypoint_scores_data, np.ndarray):
                if keypoint_scores_data.ndim == 1:
                    return keypoint_scores_data.tolist()
                elif keypoint_scores_data.ndim == 2 and keypoint_scores_data.shape[1] == 1:
                    return keypoint_scores_data.flatten().tolist()
                else:
                    logger.warning(f"Track {track_id} Frame {frame_id}: keypoint_scores numpy array 形狀不支援: {keypoint_scores_data.shape}")
                    return None

            # 如果是 list
            elif isinstance(keypoint_scores_data, list):
                # 確保所有元素都是數字
                try:
                    return [float(score) for score in keypoint_scores_data]
                except (ValueError, TypeError):
                    logger.warning(f"Track {track_id} Frame {frame_id}: keypoint_scores list 包含非數字元素。")
                    return None

            # 如果是其他可迭代格式，嘗試轉換
            else:
                try:
                    scores_list = list(keypoint_scores_data)
                    return [float(score) for score in scores_list]
                except (ValueError, TypeError, AttributeError):
                    logger.warning(f"Track {track_id} Frame {frame_id}: 無法從 keypoint_scores 提取 scores，類型: {type(keypoint_scores_data)}")
                    return None

        except Exception as e:
            logger.error(f"Track {track_id} Frame {frame_id}: 提取 keypoint_scores 時發生錯誤: {e}")
            return None

class InterpolateTracksPreprocessor(Preprocessor):
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames

    @property
    def name(self) -> str:
        return "interpolate_tracks"

    def process(self, track_manager: TrackManager) -> Dict[str, Any]:
        logger.info(f"開始執行軌跡插值 (max_frames={self.max_frames})...")
        
        # 1. 從 Manager 獲取所有需要處理的軌跡 ID
        track_ids_to_process = track_manager.get_track_ids(removed=False)
        
        interpolated_count = 0
        # 2. Preprocessor 自己執行遍歷
        for track_id in track_ids_to_process:
            # 3. 在迴圈中，調用 Manager 提供的 "單一操作" 工具
            if track_manager.interpolate_track(track_id, max_frames=self.max_frames):
                interpolated_count += 1
        
        logger.info(f"軌跡插值完成，共插值 {interpolated_count} 個軌跡。")
        return {"interpolated_tracks": interpolated_count}

class RemoveShortTracksPreprocessor(Preprocessor):
    """移除短軌跡預處理器"""
    def __init__(self, min_duration_frames: int = 90):
        self.min_duration_frames = min_duration_frames

    @property
    def name(self) -> str:
        return "remove_short_tracks"

    def process(self, track_manager: TrackManager) -> Dict[str, Any]:
        """找出並標記移除短軌跡"""
        all_tracks = track_manager.get_all_tracks(removed=False)
        short_track_ids = []
        for track in all_tracks:
            if track.first_frame is not None and track.last_frame is not None:
                duration = track.last_frame - track.first_frame + 1
                if duration < self.min_duration_frames:
                    short_track_ids.append(track.track_id)

        removed_count = 0
        if short_track_ids:
            removed_count = track_manager.mark_tracks_removed(short_track_ids)

        logger.info(f"移除短軌跡完成，共移除 {removed_count} 個軌跡。")
        return {"tracks_removed": removed_count}

class KeypointsNormalizationPreprocessor(Preprocessor):
    """
    關鍵點正規化預處理器（標準化流程）：
    1. 平移標準化：以肩部中心為原點進行置中
    2. 尺度標準化：基於軀幹長度進行尺度正規化，並歸一化到 [-1, 1] 範圍

    適合作為機器學習模型的輸入預處理，消除位置和尺度的影響
    """

    def __init__(self,
                 center_keypoint: str = "shoulder_center",
                 scale_method: str = "fixed_torso_length",
                 reference_scale: float = 2.0,
                 **kwargs):
        """
        初始化關鍵點正規化預處理器

        Args:
            center_keypoint: 中心關鍵點選擇 ("shoulder_center", "hip_center", "torso_center")
            scale_method: 尺度計算方法 ("median_torso_length", "fixed_torso_length")
            reference_scale: 參考尺度值，用於尺度標準化
        """
        self.center_keypoint = center_keypoint
        self.scale_method = scale_method
        self.reference_scale = reference_scale

    @property
    def name(self) -> str:
        return "keypoints_normalization"

    def process(self, track_manager: TrackManager) -> Dict[str, Any]:
        """
        對所有軌跡進行五步驟關鍵點正規化：
        1. 旋轉標準化（將身體軀幹垂直於地面，對齊Y軸）
        2. 平移標準化（以肩部中心為原點置中）
        3. 尺度標準化（基於軀幹長度正規化）
        4. 高斯平滑（對時間序列應用高斯濾波器進行平滑化）
        5. 解析度標準化（正規化到 [-1,1] 範圍）

        Args:
            track_manager: TrackManager 實例

        Returns:
            處理統計信息
        """
        all_tracks = track_manager.get_all_tracks(removed=False)
        processed_tracks = 0
        total_frames_processed = 0

        for track in all_tracks:
            if not track.keypoints:
                logger.debug(f'Track {track.track_id}: 沒有關鍵點資料可供正規化。')
                continue

            # 步驟 0: 提取有效的關鍵點資料
            valid_keypoints = self._extract_valid_keypoints(track)
            if not valid_keypoints:
                logger.debug(f'Track {track.track_id}: 沒有有效的關鍵點資料。')
                continue

            # 步驟 1: 旋轉標準化（將身體軀幹垂直於地面，對齊Y軸）
            # rotation_normalized = self._apply_rotation_normalization(valid_keypoints, track.track_id)

            # 步驟 2: 平移標準化（以肩部中心為原點置中）
            translation_normalized = self._apply_translation_normalization(valid_keypoints, track.track_id)

            # 步驟 3: 尺度標準化（基於軀幹長度正規化）
            scale_normalized = self._apply_scale_normalization(translation_normalized, track.track_id)

            # 步驟 4: 高斯平滑（對時間序列應用高斯濾波器進行平滑化）
            # smooth_normalized = self._apply_gaussian_smoothing(scale_normalized, track.track_id) if self.enable_smoothing else scale_normalized

            # 儲存正規化結果到 track 物件
            track.keypoints_normalized = scale_normalized

            processed_tracks += 1
            total_frames_processed += len(scale_normalized)

        logger.info(f'關鍵點正規化完成（平移 + 尺度標準化），處理了 {processed_tracks} 個軌跡，共 {total_frames_processed} 個幀。')

        return {
            "processed_tracks": processed_tracks,
            "total_frames_processed": total_frames_processed
        }

    def _extract_valid_keypoints(self, track: TrackRecord) -> Dict[int, np.ndarray]:
        """提取有效的關鍵點資料"""
        valid_keypoints = {}

        for fid, kp in track.keypoints.items():
            if kp is None or not isinstance(kp, np.ndarray):
                continue

            # 確保關鍵點有正確的形狀
            if kp.ndim == 1:
                kp = kp.reshape(-1, 2)
            elif kp.ndim == 2 and kp.shape[1] > 2:
                kp = kp[:, :2]  # 只取 x, y 座標

            if kp.shape[0] != 17 or kp.shape[1] != 2:
                logger.debug(f'Track {track.track_id} Frame {fid}: 關鍵點形狀不正確 {kp.shape}，跳過。')
                continue

            valid_keypoints[fid] = kp.copy()

        return valid_keypoints



    def _apply_translation_normalization(self, keypoints_dict: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
        """
        平移標準化 - 以指定的身體中心為原點進行置中

        Args:
            keypoints_dict: 旋轉標準化後的關鍵點字典 {frame_id: keypoints}
            track_id: 軌跡ID，用於日誌

        Returns:
            平移標準化後的關鍵點字典
        """
        translated_keypoints = {}

        for fid, kp in keypoints_dict.items():
            center_point = self._calculate_center_point(kp)
            if center_point is None:
                logger.debug(f'Track {track_id} Frame {fid}: 無法計算身體中心，跳過平移標準化。')
                continue

            # 應用平移標準化
            translated_kp = kp.copy()
            valid_mask = ~np.all(kp == 0, axis=1)
            translated_kp[valid_mask] = kp[valid_mask] - center_point

            translated_keypoints[fid] = translated_kp

        logger.debug(f'Track {track_id}: 平移標準化完成，處理了 {len(translated_keypoints)} 個幀。')
        return translated_keypoints



    def _apply_scale_normalization(self, keypoints_dict: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
        """
        尺度標準化 - 基於指定的尺度因子進行正規化

        Args:
            keypoints_dict: 平移標準化後的關鍵點字典 {frame_id: keypoints}
            track_id: 軌跡ID，用於日誌

        Returns:
            尺度標準化後的關鍵點字典
        """
        scale_normalized_keypoints = {}

        # 如果使用軀幹長度作為尺度因子，先計算所有幀的軀幹長度
        if self.scale_method == "median_torso_length":
            torso_lengths = []
            for fid, kp in keypoints_dict.items():
                torso_length = self._calculate_torso_length(kp)
                if torso_length is not None and torso_length > 1e-6:
                    torso_lengths.append(torso_length)

            if not torso_lengths:
                logger.warning(f'Track {track_id}: 無法計算軀幹長度，跳過尺度標準化。')
                return keypoints_dict

            # 使用中位數軀幹長度作為參考尺度
            median_torso_length = np.median(torso_lengths)
            reference_scale_factor = median_torso_length
        elif self.scale_method == "fixed_torso_length":
            reference_scale_factor = self.reference_scale  # 固定尺度，不進行縮放
        else:
            reference_scale_factor = self.reference_scale

        logger.debug(f'Track {track_id}: 使用參考尺度因子: {reference_scale_factor:.6f}')

        for fid, kp in keypoints_dict.items():
            current_scale = self._calculate_torso_length(kp)

            if current_scale is None or current_scale < 1e-6:
                logger.debug(f'Track {track_id} Frame {fid}: 無法計算尺度因子，跳過。')
                continue

            # 應用尺度標準化，並除以 reference_scale * 2 將座標歸一化到 [-1, 1] 範圍
            norm_div = reference_scale_factor * 2
            scale_factor = reference_scale_factor / current_scale / norm_div
            scaled_kp = kp.copy()
            valid_mask = ~np.all(kp == 0, axis=1)
            scaled_kp[valid_mask] = kp[valid_mask] * scale_factor

            scale_normalized_keypoints[fid] = scaled_kp

        logger.debug(f'Track {track_id}: 尺度標準化完成，處理了 {len(scale_normalized_keypoints)} 個幀。')
        return scale_normalized_keypoints

    def _calculate_center_point(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        根據設定計算身體中心點

        Args:
            keypoints: 關鍵點數組 (17, 2)

        Returns:
            身體中心座標，如果無法計算則返回 None
        """
        if self.center_keypoint == "shoulder_center":
            # 計算肩部中心 (COCO: 5=左肩, 6=右肩)
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            if not (np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0)):
                return (left_shoulder + right_shoulder) / 2

        elif self.center_keypoint == "hip_center":
            # 計算髖部中心 (COCO: 11=左髖, 12=右髖)
            left_hip, right_hip = keypoints[11], keypoints[12]
            if not (np.allclose(left_hip, 0) or np.allclose(right_hip, 0)):
                return (left_hip + right_hip) / 2

        elif self.center_keypoint == "torso_center":
            # 計算軀幹中心（肩部中心和髖部中心的中點）
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            left_hip, right_hip = keypoints[11], keypoints[12]

            shoulder_center = None
            if not (np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0)):
                shoulder_center = (left_shoulder + right_shoulder) / 2

            hip_center = None
            if not (np.allclose(left_hip, 0) or np.allclose(right_hip, 0)):
                hip_center = (left_hip + right_hip) / 2

            if shoulder_center is not None and hip_center is not None:
                return (shoulder_center + hip_center) / 2
            elif hip_center is not None:
                return hip_center
            elif shoulder_center is not None:
                return shoulder_center

        return None

    def _calculate_torso_length(self, keypoints: np.ndarray) -> Optional[float]:
        """
        計算軀幹長度（肩部中心到髖部中心的距離）

        Args:
            keypoints: 關鍵點數組 (17, 2)

        Returns:
            軀幹長度，如果無法計算則返回 None
        """
        # 計算肩部中心 (COCO: 5=左肩, 6=右肩)
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        if np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0):
            return None
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # 計算髖部中心 (COCO: 11=左髖, 12=右髖)
        left_hip, right_hip = keypoints[11], keypoints[12]
        if np.allclose(left_hip, 0) or np.allclose(right_hip, 0):
            return None
        hip_center = (left_hip + right_hip) / 2

        # 計算距離
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        return torso_length if torso_length > 1e-6 else None

    def _apply_rotation_normalization(self, keypoints_dict: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
        """
        旋轉標準化 - 將身體軀幹垂直於地面（對齊Y軸）

        通過計算軀幹方向（肩部中心到髖部中心的向量）並將其對齊到Y軸正方向來實現旋轉標準化。
        在原始像素座標上執行，保持像素空間的準確性。

        Args:
            keypoints_dict: 原始關鍵點字典 {frame_id: keypoints}
            track_id: 軌跡ID，用於日誌

        Returns:
            旋轉標準化後的關鍵點字典
        """
        rotated_keypoints = {}

        for fid, kp in keypoints_dict.items():
            # 計算軀幹方向向量（從肩部中心到髖部中心）
            torso_vector = self._calculate_torso_vector(kp)
            if torso_vector is None:
                logger.debug(f'Track {track_id} Frame {fid}: 無法計算軀幹方向，跳過旋轉標準化。')
                rotated_keypoints[fid] = kp.copy()  # 保留原始資料
                continue

            target_vector = np.array([0.0, 1.0])  # Y軸正方向（向下）
            rotation_angle = self._calculate_rotation_angle(torso_vector, target_vector)

            # 應用旋轉變換
            rotated_kp = self._apply_rotation_transform(kp, rotation_angle)
            rotated_keypoints[fid] = rotated_kp

        logger.debug(f'Track {track_id}: 旋轉標準化完成，處理了 {len(rotated_keypoints)} 個幀。')
        return rotated_keypoints

    def _calculate_torso_vector(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        計算軀幹方向向量（從肩部中心到髖部中心）

        Args:
            keypoints: 關鍵點數組 (17, 2)

        Returns:
            軀幹方向向量，如果無法計算則返回 None
        """
        # 計算肩部中心 (COCO: 5=左肩, 6=右肩)
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        if np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0):
            return None
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # 計算髖部中心 (COCO: 11=左髖, 12=右髖)
        left_hip, right_hip = keypoints[11], keypoints[12]
        if np.allclose(left_hip, 0) or np.allclose(right_hip, 0):
            return None
        hip_center = (left_hip + right_hip) / 2

        # 計算從肩部中心到髖部中心的向量
        torso_vector = hip_center - shoulder_center

        # 檢查向量長度是否有效
        torso_length = np.linalg.norm(torso_vector)
        if torso_length < 1e-6:
            return None

        # 正規化向量
        return torso_vector / torso_length

    def _calculate_rotation_angle(self, current_vector: np.ndarray, target_vector: np.ndarray) -> float:
        """
        計算將 current_vector 旋轉到 target_vector 所需的角度

        Args:
            current_vector: 當前向量（已正規化）
            target_vector: 目標向量（已正規化）

        Returns:
            旋轉角度（弧度）
        """
        current_norm = current_vector / np.linalg.norm(current_vector)
        target_norm = target_vector / np.linalg.norm(target_vector)

        dot_product = np.dot(current_norm, target_norm)
        cross_product = np.cross(current_norm, target_norm)

        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        if cross_product < 0:
            angle = -angle

        return angle

    def _apply_rotation_transform(self, keypoints: np.ndarray, angle: float) -> np.ndarray:
        """
        對關鍵點應用旋轉變換

        Args:
            keypoints: 關鍵點數組 (17, 2)
            angle: 旋轉角度（弧度）

        Returns:
            旋轉後的關鍵點數組
        """
        rotated_kp = keypoints.copy()

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        valid_mask = ~np.all(keypoints == 0, axis=1)
        if np.any(valid_mask):
            rotated_kp[valid_mask] = (rotation_matrix @ keypoints[valid_mask].T).T

        return rotated_kp

    def _apply_gaussian_smoothing(self, keypoints_dict: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
        """
        高斯平滑 - 對關鍵點時間序列應用高斯濾波器進行平滑化

        Args:
            keypoints_dict: 尺度標準化後的關鍵點字典 {frame_id: keypoints}
            track_id: 軌跡ID，用於日誌

        Returns:
            高斯平滑後的關鍵點字典
        """
        if self.gauss_sigma <= 0 or not keypoints_dict:
            logger.debug(f'Track {track_id}: 高斯平滑被禁用或無資料，跳過平滑化。')
            return keypoints_dict

        frame_ids = sorted(keypoints_dict.keys())
        if len(frame_ids) < 3:
            logger.debug(f'Track {track_id}: 幀數太少 ({len(frame_ids)})，跳過高斯平滑。')
            return keypoints_dict

        smoothed_keypoints = {}

        window_size = int(self.gauss_sigma * 6 + 1) | 1
        gaussian_kernel = sig.windows.gaussian(window_size, std=self.gauss_sigma)
        gaussian_kernel /= gaussian_kernel.sum()

        for kp_idx in range(17):
            for coord_idx in range(2):
                time_series = []
                valid_mask = []

                for fid in frame_ids:
                    kp = keypoints_dict[fid]
                    if not np.allclose(kp[kp_idx], 0):
                        time_series.append(kp[kp_idx, coord_idx])
                        valid_mask.append(True)
                    else:
                        time_series.append(0.0)
                        valid_mask.append(False)

                time_series = np.array(time_series)
                valid_mask = np.array(valid_mask)

                if valid_mask.sum() < 3:
                    continue

                try:
                    valid_values = time_series[valid_mask]
                    pad_len = min(len(valid_values) - 1, window_size // 2)
                    if pad_len < 0:
                        pad_len = 0

                    if len(valid_values) > pad_len * 2 + 1:
                        smoothed_valid = sig.filtfilt(
                            gaussian_kernel, [1.0], valid_values,
                            padlen=pad_len, method='pad'
                        )

                        valid_idx = 0
                        for i, fid in enumerate(frame_ids):
                            if valid_mask[i]:
                                if fid not in smoothed_keypoints:
                                    smoothed_keypoints[fid] = keypoints_dict[fid].copy()
                                smoothed_keypoints[fid][kp_idx, coord_idx] = smoothed_valid[valid_idx]
                                valid_idx += 1
                    else:
                        logger.debug(f'Track {track_id} 關鍵點 {kp_idx} 座標 {coord_idx}: 資料點不足，跳過平滑化。')

                except Exception as e:
                    logger.debug(f'Track {track_id} 關鍵點 {kp_idx} 座標 {coord_idx}: 高斯平滑失敗 - {e}')
                    continue

        for fid, kp in keypoints_dict.items():
            if fid not in smoothed_keypoints:
                smoothed_keypoints[fid] = kp.copy()

        logger.debug(f'Track {track_id}: 高斯平滑完成（sigma={self.gauss_sigma}），處理了 {len(smoothed_keypoints)} 個幀。')
        return smoothed_keypoints


class PreprocessingPipeline:
    """
    預處理管道，負責按順序執行所有預處理器
    """

    def __init__(self, preprocessors: List[Preprocessor]):
        """
        初始化預處理管道

        Args:
            preprocessors: 要執行的預處理器列表，應按順序排列
        """
        self.preprocessors = preprocessors

    def run(self, track_manager: TrackManager) -> Dict[str, Any]:
        """
        對 TrackManager 執行完整的預處理流程。
        這個方法是通用的，它按順序執行所有預處理器，
        而不需要知道任何關於預處理器的具體細節。
        """
        logger.info("--- 開始執行通用預處理流程 ---")
        pipeline_stats = {}

        for preprocessor in self.preprocessors:
            logger.info(f"-> 正在執行預處理器: {preprocessor.name}")
            try:
                # 無論 preprocessor 是哪種類型，我們都用同樣的方式調用它。
                stats = preprocessor.process(track_manager)
                
                # 將該步驟的統計結果合併到總結果中
                pipeline_stats[preprocessor.name] = stats

            except Exception as e:
                logger.error(f"預處理器 {preprocessor.name} 執行失敗: {e}", exc_info=True)
                continue

        logger.info("--- 預處理流程全部完成 ---")
        logger.info(f"處理結果統計: {pipeline_stats}")

        return pipeline_stats