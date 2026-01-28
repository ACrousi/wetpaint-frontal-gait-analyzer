from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from typing import Dict, List, Tuple, Optional, Any
import math
import scipy.signal as sig
from scipy.signal import find_peaks
from .base import logger
from typing import Dict, Tuple, Optional
from .base import logger
from src.pose_extract.track_solution import TrackRecord

from .analysis_strategies import MetricStrategy
from .analysis_results import AnalysisResult
from .base import SegmentType

def _angle(p1, p2, p3) -> float | None:
    v1, v2 = np.asarray(p1) - np.asarray(p2), np.asarray(p3) - np.asarray(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


class BodyProportionMetric(MetricStrategy):
    @property
    def name(self) -> str:
        return "body_proportion"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        Analyzes the track to calculate body proportion metrics for each frame
        and returns them as a DataFrame.
        """
        # 收集每幀的數據
        per_frame_data = []

        for fid, keypoints_for_frame in track.keypoints.items():
            if keypoints_for_frame is None:
                logger.debug(f"Track {track.track_id} Frame {fid}: No keypoints, skipping body proportion.")
                # 即使是空數據，也創建一個記錄，以保持索引完整
                result_dict = {
                    'frame_id': fid,
                    'sitting_height_index': np.nan,
                    'body_to_head_ratio': np.nan,
                    'sitting_height_raw': np.nan,
                    'total_height_raw': None,
                    'head_height_raw': np.nan
                }
            else:
                position_bbox = track.positions.get(fid)
                if position_bbox is None:
                    logger.debug(f"Track {track.track_id} Frame {fid}: No position bbox, skipping body proportion.")
                    result_dict = {
                        'frame_id': fid,
                        'sitting_height_index': np.nan,
                        'body_to_head_ratio': np.nan,
                        'sitting_height_raw': np.nan,
                        'total_height_raw': None,
                        'head_height_raw': np.nan
                    }
                else:
                    # Calculate proportions for this single frame
                    # The position_bbox from track.positions is (x1, y1, x2, y2).
                    # calculate_single_frame_body_proportions expects this format and uses x1, y1.
                    frame_metrics = self.calculate_single_frame_body_proportions(keypoints_for_frame, position_bbox)

                    if frame_metrics:
                        # 收集所有計算的指標
                        val_sitting_idx = frame_metrics.get("sitting_height_index")
                        val_b2h_ratio = frame_metrics.get("body_to_head_ratio")
                        val_sit_h_raw = frame_metrics.get("sitting_height_raw")
                        val_head_h_raw = frame_metrics.get("head_height_raw")

                        result_dict = {
                            'frame_id': fid,
                            'sitting_height_index': val_sitting_idx if val_sitting_idx is not None else np.nan,
                            'body_to_head_ratio': val_b2h_ratio if val_b2h_ratio is not None else np.nan,
                            'sitting_height_raw': val_sit_h_raw if val_sit_h_raw is not None else np.nan,
                            'total_height_raw': frame_metrics.get("total_height_raw"),
                            'head_height_raw': val_head_h_raw if val_head_h_raw is not None else np.nan
                        }
                    else: # frame_metrics is None, calculation failed for this frame
                        logger.debug(f"Track {track.track_id} Frame {fid}: Body proportion calculation returned None.")
                        result_dict = {
                            'frame_id': fid,
                            'sitting_height_index': np.nan,
                            'body_to_head_ratio': np.nan,
                            'sitting_height_raw': np.nan,
                            'total_height_raw': None,
                            'head_height_raw': np.nan
                        }

            per_frame_data.append(result_dict)

        if not per_frame_data:
            return pd.DataFrame()

        # 將結果列表轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        logger.debug(f"Track {track.track_id} body proportion per-frame analysis completed.")
        return df

    @staticmethod
    def calculate_single_frame_body_proportions(keypoints: np.ndarray, position_bbox: Tuple[float, float, float, float]) -> Optional[Dict[str, float]]:
        """
        Calculates body proportions for a single frame.
        Args:
            keypoints: Numpy array of shape (17, 2) or (17, 3) for keypoints.
            position_bbox: Tuple (x1, y1, x2, y2) of the bounding box. Only x1, y1 are used for head_top_y.
        Returns:
            A dictionary with 'sitting_height_index', 'body_to_head_ratio',
            'sitting_height_raw', 'total_height_raw', 'head_height_raw',
            or None if critical keypoints are missing or calculation is not possible.
            'total_height_raw' can be None if full height cannot be determined.
        """
        if keypoints is None or keypoints.shape[0] != 17 or keypoints.ndim < 2 : #ndim check handles (17,) if only one coord
            logger.debug("Invalid keypoints shape or type for body proportion calculation.")
            return None

        kp = keypoints[:, :2] # Use only x, y coordinates

        # position_bbox is (x1, y1, x2, y2)
        x1_bbox, y1_bbox, _, _ = position_bbox 

        # COCO 17 points: 0:Nose, 5:LShoulder, 6:RShoulder, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        l_shoulder, r_shoulder = kp[5], kp[6]
        l_hip, r_hip = kp[11], kp[12]
        l_knee, r_knee = kp[13], kp[14]
        l_ankle, r_ankle = kp[15], kp[16]
        nose = kp[0]

        # Check for invalid (all zero) critical keypoints for basic calculations
        if np.allclose(l_shoulder, 0) or np.allclose(r_shoulder, 0) or \
           np.allclose(l_hip, 0) or np.allclose(r_hip, 0) or \
           np.allclose(l_knee, 0) or np.allclose(r_knee, 0) or \
           np.allclose(l_ankle, 0) or np.allclose(r_ankle, 0):
            logger.debug("Missing critical keypoints (shoulders, hips) for body proportion.")
            return None

        shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
        hip_mid = np.mean([l_hip, r_hip], axis=0)

        # Head top estimation
        head_top_y = y1_bbox # Use bbox top as default
        if not np.allclose(nose, 0): # If nose keypoint is valid
            head_top_y = min(y1_bbox, nose[1]) # y-coordinates increase downwards

        shoulder_to_top_dist = abs(shoulder_mid[1] - head_top_y)
        if shoulder_to_top_dist <= 1e-3: # Avoid division by zero or meaningless small values
            logger.debug("Shoulder to top distance is too small or zero.")
            # Still possible to return raw heights without ratios
            hip_to_shoulder_dist = np.linalg.norm(hip_mid - shoulder_mid)
            sitting_h_raw = hip_to_shoulder_dist + shoulder_to_top_dist
            return {
                "sitting_height_index": None, # Cannot calculate index
                "body_to_head_ratio": None, # Cannot calculate ratio
                "sitting_height_raw": sitting_h_raw,
                "total_height_raw": None, # Cannot reliably calculate total height either
                "head_height_raw": shoulder_to_top_dist,
            }

        hip_to_shoulder_dist = np.linalg.norm(hip_mid - shoulder_mid)
        sitting_h_raw = hip_to_shoulder_dist + shoulder_to_top_dist
        head_height_raw = shoulder_to_top_dist

        # For total height, ensure other points are valid
        knee_mid = np.mean([l_knee, r_knee], axis=0)
        ankle_mid = np.mean([l_ankle, r_ankle], axis=0)

        # Initialize metrics that depend on total height
        total_h_raw = None
        sitting_height_index = None
        body_to_head_ratio = None

        if np.allclose(knee_mid, 0) or np.allclose(ankle_mid, 0) :
             logger.debug("Missing knee or ankle keypoints for full height calculation. Sitting height and head height are still available.")
             # total_h_raw remains None, as do the ratios depending on it.
        else:
            ankle_to_knee_dist = np.linalg.norm(ankle_mid - knee_mid)
            knee_to_hip_dist = np.linalg.norm(knee_mid - hip_mid)
            
            current_total_h = ankle_to_knee_dist + knee_to_hip_dist + hip_to_shoulder_dist + shoulder_to_top_dist

            if current_total_h <= 1e-3:
                logger.debug("Calculated total height is too small.")
                # total_h_raw remains None
            else:
                total_h_raw = current_total_h
                sitting_height_index = 100 * sitting_h_raw / total_h_raw
                body_to_head_ratio = total_h_raw / head_height_raw # head_height_raw is shoulder_to_top_dist

        return {
            "sitting_height_index": sitting_height_index,
            "body_to_head_ratio": body_to_head_ratio,
            "sitting_height_raw": sitting_h_raw,
            "total_height_raw": total_h_raw,
            "head_height_raw": head_height_raw,
        }

class StandingMetric(MetricStrategy):
    def __init__(self, angle_threshold: float = 160):
        self.angle_threshold = angle_threshold

    @property
    def name(self) -> str:
        return "standing_metric"

    def _leg_angles(self, kp: np.ndarray) -> Tuple[Optional[float], Optional[float]]: # kp 是 (17,2) 或 (17,3)
        # COCO 17點: 11:LHip, 13:LKnee, 15:LAnkle, 12:RHip, 14:RKnee, 16:RAnkle
        lhip, lknee, lank = kp[11,:2], kp[13,:2], kp[15,:2]
        rhip, rknee, rank = kp[12,:2], kp[14,:2], kp[16,:2]

        # 檢查是否有無效的 (0,0) 點，這些點不應用於角度計算
        left_valid = not (np.allclose(lhip, 0) or np.allclose(lknee, 0) or np.allclose(lank, 0))
        right_valid = not (np.allclose(rhip, 0) or np.allclose(rknee, 0) or np.allclose(rank, 0))

        left_angle = _angle(lhip, lknee, lank) if left_valid else None
        right_angle = _angle(rhip, rknee, rank) if right_valid else None
        return left_angle, right_angle

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        standing_by_frame = []
        for fid, kp in track.keypoints.items():
            if kp is None or kp.shape[0] != 17 : # 至少需要17個點
                is_standing = False
            else:
                l_angle, r_angle = self._leg_angles(kp)
                is_standing = (
                    l_angle is not None and r_angle is not None and
                    l_angle > self.angle_threshold and r_angle > self.angle_threshold
                )

            standing_by_frame.append({'frame_id': fid, 'is_standing': is_standing})

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(standing_by_frame).set_index('frame_id')
        logger.debug("Track %s standing metric computed.", track.track_id)
        return df

class TorsoProportionMetric(MetricStrategy):
    @property
    def name(self) -> str:
        return "torso_proportion"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        Analyzes the track to calculate torso proportion metrics for each frame
        and returns them as a DataFrame.
        """
        # 收集每幀的數據
        per_frame_data = []

        for fid, keypoints_for_frame in track.keypoints.items():
            if keypoints_for_frame is None or keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: Invalid keypoints, skipping torso proportion.")
                result_dict = {
                    'frame_id': fid,
                    'hip_width': np.nan,
                    'torso_length': np.nan,
                    'torso_to_hip_ratio': np.nan
                }
            else:
                kp = keypoints_for_frame[:, :2]  # Use only x, y coordinates

                # COCO 17 points: 5:LShoulder, 6:RShoulder, 11:LHip, 12:RHip
                l_shoulder, r_shoulder = kp[5], kp[6]
                l_hip, r_hip = kp[11], kp[12]

                # Calculate hip width (or shoulder width if hips are invalid)
                hip_width = np.nan
                if not (np.allclose(l_hip, 0) or np.allclose(r_hip, 0)):
                    hip_width = np.linalg.norm(l_hip - r_hip)
                elif not (np.allclose(l_shoulder, 0) or np.allclose(r_shoulder, 0)):
                    hip_width = np.linalg.norm(l_shoulder - r_shoulder)
                    logger.debug(f"Track {track.track_id} Frame {fid}: Using shoulder width as hip width.")
                else:
                    logger.debug(f"Track {track.track_id} Frame {fid}: Cannot calculate hip or shoulder width.")

                # Calculate torso length
                torso_length = np.nan
                if not (np.allclose(l_shoulder, 0) or np.allclose(r_shoulder, 0) or \
                        np.allclose(l_hip, 0) or np.allclose(r_hip, 0)):
                    shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
                    hip_mid = np.mean([l_hip, r_hip], axis=0)
                    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
                else:
                    logger.debug(f"Track {track.track_id} Frame {fid}: Cannot calculate torso length due to missing keypoints.")

                # Calculate torso-to-hip ratio
                torso_to_hip_ratio = np.nan
                if np.isfinite(torso_length) and np.isfinite(hip_width) and hip_width > 1e-6:
                    torso_to_hip_ratio = torso_length / hip_width

                result_dict = {
                    'frame_id': fid,
                    'hip_width': hip_width,
                    'torso_length': torso_length,
                    'torso_to_hip_ratio': torso_to_hip_ratio
                }

            per_frame_data.append(result_dict)

        if not per_frame_data:
            return pd.DataFrame()

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        logger.debug(f"Track {track.track_id} torso proportion per-frame analysis completed.")
        return df


class LegDistanceMetric(MetricStrategy):
    @property
    def name(self) -> str:
        return "leg_distance"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        Analyzes the track to calculate leg distance metrics (knee and ankle) for each frame
        and returns them as a DataFrame.
        """
        # 收集每幀的數據
        per_frame_data = []

        for fid, keypoints_for_frame in track.keypoints.items():
            if keypoints_for_frame is None or keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: Invalid keypoints, skipping leg distance.")
                result_dict = {
                    'frame_id': fid,
                    'knee_distance': np.nan,
                    'ankle_distance': np.nan
                }
            else:
                kp = keypoints_for_frame[:, :2]  # Use only x, y coordinates

                # COCO 17 points: 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
                l_knee, r_knee = kp[13], kp[14]
                l_ankle, r_ankle = kp[15], kp[16]

                # Calculate knee distance
                knee_distance = np.nan
                if not (np.allclose(l_knee, 0) or np.allclose(r_knee, 0)):
                    knee_distance = np.linalg.norm(l_knee - r_knee)
                else:
                    logger.debug(f"Track {track.track_id} Frame {fid}: Cannot calculate knee distance due to missing knee keypoints.")

                # Calculate ankle distance
                ankle_distance = np.nan
                if not (np.allclose(l_ankle, 0) or np.allclose(r_ankle, 0)):
                    ankle_distance = np.linalg.norm(l_ankle - r_ankle)
                else:
                    logger.debug(f"Track {track.track_id} Frame {fid}: Cannot calculate ankle distance due to missing ankle keypoints.")

                result_dict = {
                    'frame_id': fid,
                    'knee_distance': knee_distance,
                    'ankle_distance': ankle_distance
                }

            per_frame_data.append(result_dict)

        if not per_frame_data:
            return pd.DataFrame()

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        logger.debug(f"Track {track.track_id} leg distance per-frame analysis completed.")
        return df

# class KeypointsStandardizationMetric(MetricAnalysisStrategy):
#     def __init__(self,
#                  nominal_torso_length : Optional[float]  = 540.0, # Default fallback for torso length (changed from hip width)
#                  nominal_ratio: Optional[float]  = 1.5,  # Default fallback for torso_length / hip_width (deprecated, not used anymore)
#                  output_wh    : Tuple[int,int] = (1080, 1920),
#                  theta_smooth_win: int = 5,
#                  max_scale    : float  = 6.0,
#                  gauss_sigma  : float  = 0.7   # 平滑強度；0 = 關閉
#                  ):
#         self.nominal_torso_length = nominal_torso_length
#         self.default_nominal_ratio = nominal_ratio # Deprecated: not used in new torso-based scaling
#         self.output_wh = output_wh # (width, height)
#         self.theta_smooth_win = theta_smooth_win
#         self.max_scale = max_scale
#         self.gauss_sigma = gauss_sigma

#     def _calculate_and_smooth_theta(self, track_record: TrackRecord) -> Optional[Dict[int, float]]:
#         """Calculates and smooths theta for each valid frame."""
#         theta_seq, fids_valid = [], []
#         for fid, kp in track_record.keypoints.items():
#             if not isinstance(kp, np.ndarray): continue
#             pts = kp.reshape(-1, 2) if kp.ndim != 2 else kp
#             if pts.shape[0] < 17: continue
#             hip_ok = np.sum(pts[11]) > 0 and np.sum(pts[12]) > 0
#             sh_ok = np.sum(pts[5]) > 0 and np.sum(pts[6]) > 0
#             if not hip_ok and not sh_ok: continue
#             vec = (pts[11] - pts[12]) if hip_ok else (pts[5] - pts[6])
#             if np.linalg.norm(vec) < 1e-6: continue
#             theta_seq.append(math.atan2(vec[1], vec[0]))
#             fids_valid.append(fid)

#         if not fids_valid:
#             logger.info(f'Track {track_record.track_id}: no valid frames for theta calculation.')
#             return None

#         theta_seq_np = np.array(theta_seq)
#         if len(theta_seq_np) >= self.theta_smooth_win:
#             k = np.ones(self.theta_smooth_win) / self.theta_smooth_win
#             theta_seq_np = np.convolve(theta_seq_np, k, mode='same')
        
#         return dict(zip(fids_valid, theta_seq_np))

#     def _transform_to_relative_coordinates(self,
#                                            track_record: TrackRecord,
#                                            theta_map: Dict[int, float],
#                                            nominal_torso_length: float
#                                            ) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[int, np.ndarray]]]:
#         """Transforms keypoints to relative coordinates based on theta_map using torso length as scaling reference."""
#         rel_dict = {}
#         root_ok_m = {}
        
#         if nominal_torso_length <= 0:
#             logger.error(f"Track {track_record.track_id}: nominal_torso_length ({nominal_torso_length}) is invalid. Skipping transform.")
#             return None, None

#         for fid, kp_raw in track_record.keypoints.items():
#             if not isinstance(kp_raw, np.ndarray):
#                 continue
#             pts = kp_raw.reshape(-1, 2) if kp_raw.ndim != 2 else kp_raw.copy()
#             if pts.shape[0] < 17: continue
#             theta = theta_map.get(fid)
#             if theta is None: continue

#             hip_ok = np.sum(pts[11]) > 0 and np.sum(pts[12]) > 0
#             sh_ok = np.sum(pts[5]) > 0 and np.sum(pts[6]) > 0
#             if not hip_ok and not sh_ok: continue
#             root = (pts[11] + pts[12]) / 2 if hip_ok else (pts[5] + pts[6]) / 2

#             msk = np.sum(pts, 1) > 0
#             pts[msk] -= root
#             c, s = math.cos(-theta), math.sin(-theta)
#             R = np.array([[c, -s], [s, c]])
#             # Apply rotation to align body orientation
#             # pts[msk] = (R @ pts[msk].T).T
            
#             # Calculate observed torso length as scaling reference
#             if hip_ok and sh_ok:
#                 sh_mid = (pts[5] + pts[6]) / 2
#                 hip_mid = (pts[11] + pts[12]) / 2
#                 torso_obs = np.linalg.norm(sh_mid - hip_mid)
#             else:
#                 logger.debug(f"Track {track_record.track_id} Frame {fid}: Missing shoulder or hip keypoints for torso length calculation.")
#                 continue
            
#             torso_obs = max(torso_obs, 1e-6) # Avoid division by zero
            
#             # Scale based on torso length instead of hip width
#             scale = np.clip(nominal_torso_length / torso_obs, 1 / self.max_scale, self.max_scale)
#             pts[msk] *= scale

#             # No additional torso length adjustment needed since we're already scaling by torso length
            
#             rel_dict[fid] = pts
#             root_ok_m[fid] = msk
        
#         if not rel_dict:
#             logger.info(f'Track {track_record.track_id}: no frames processed for relative coordinates.')
#             return None, None
#         return rel_dict, root_ok_m

#     def _apply_temporal_gaussian_smoothing(self, track_record: TrackRecord, rel_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
#         """Applies temporal Gaussian smoothing to relative keypoints."""
#         if self.gauss_sigma <= 0 or not rel_dict:
#             return rel_dict

#         fids = sorted(rel_dict.keys())
#         if not fids:
#             logger.info(f'Track {track_record.track_id}: rel_dict is empty before Gaussian smoothing.')
#             return rel_dict

#         first_frame_kp = rel_dict[fids[0]]
#         n_kp = first_frame_kp.shape[0]

#         win = int(self.gauss_sigma * 6 + 1) | 1
#         b = sig.windows.gaussian(win, std=self.gauss_sigma)
#         b /= b.sum()
#         a = [1.0]

#         for k_idx in range(n_kp):
#             traj_list = []
#             valid_fids_for_kp = []
#             for fid_val in fids: # Iterate directly over sorted fids
#                 if k_idx < rel_dict[fid_val].shape[0]:
#                     traj_list.append(rel_dict[fid_val][k_idx])
#                     valid_fids_for_kp.append(fid_val)
            
#             if not traj_list: continue

#             traj = np.array(traj_list)
#             valid_mask = np.any(traj != 0, axis=1)

#             if valid_mask.sum() < 3:
#                 continue

#             for dim in (0, 1):
#                 vals = traj[valid_mask, dim]
#                 padlen_val = min(len(vals) - 1, win // 2)
#                 if padlen_val < 0: padlen_val = 0
                
#                 if len(vals) > padlen_val * 2 + 1:
#                     traj[valid_mask, dim] = sig.filtfilt(b, a, vals, padlen=padlen_val, method='pad')
#                 else:
#                     logger.debug(f"Track {track_record.track_id}, kp {k_idx}, dim {dim}: Not enough data points ({len(vals)}) for filtfilt with padlen {padlen_val}. Skipping.")

#             for i, fid_for_kp in enumerate(valid_fids_for_kp):
#                 if valid_mask[i]: # Ensure we only write back if it was part of the valid trajectory
#                     rel_dict[fid_for_kp][k_idx] = traj[i]
#         return rel_dict

#     def _finalize_pixel_coordinates(self, rel_dict: Dict[int, np.ndarray], root_ok_m: Dict[int, np.ndarray], canvas_c: np.ndarray, out_w: int, out_h: int) -> Dict[int, np.ndarray]:
#         """Converts relative keypoints to pixel coordinates, applies clipping."""
#         pixel_dict = {}
#         for fid, pts_rel in rel_dict.items():
#             msk = root_ok_m[fid]
#             pts = pts_rel.copy()
#             pts[msk] += canvas_c
#             pts[msk, 0] = np.clip(pts[msk, 0], 0, out_w - 1)
#             pts[msk, 1] = np.clip(pts[msk, 1], 0, out_h - 1)
#             pixel_dict[fid] = pts
#         return pixel_dict

#     def analyze(self, track_record: TrackRecord) -> Dict[int, np.ndarray]:
#         """
#         Standardizes keypoints for the track record and stores them as pixel coordinates.
#         Returns the standardized keypoints in pixel coordinates.
#         """
#         if not track_record.keypoints:
#             logger.warning(f'Track {track_record.track_id}: no keypoints for standardization.')
#             track_record.keypoints_standardized = {}
#             return track_record.keypoints_standardized

#         # Determine current nominal torso length
#         nominal_torso_length = self.nominal_torso_length
#         if hasattr(track_record, 'median_torso_length') and track_record.median_torso_length is not None and track_record.median_torso_length > 0:
#             nominal_torso_length = track_record.median_torso_length
#             logger.info(f"Track {track_record.track_id}: Using median_torso_length ({nominal_torso_length:.2f}) for scaling.")
#         elif self.nominal_torso_length is not None and self.nominal_torso_length > 0:
#              logger.warning(f"Track {track_record.track_id}: median_torso_length not available or invalid. Consider running TorsoProportionMetric first. Using default nominal_torso_length ({self.nominal_torso_length:.2f}).")
#         else:
#             logger.error(f"Track {track_record.track_id}: median_torso_length not available (consider running TorsoProportionMetric) and no valid default_nominal_torso_length set. Cannot proceed with standardization.")
#             track_record.keypoints_standardized = {}
#             return track_record.keypoints_standardized
        
#         # Ensure the determined value is not None before passing
#         if nominal_torso_length is None:
#             logger.error(f"Track {track_record.track_id}: Failed to determine valid nominal_torso_length. Skipping standardization.")
#             track_record.keypoints_standardized = {}
#             return track_record.keypoints_standardized

#         theta_map = self._calculate_and_smooth_theta(track_record)
#         if theta_map is None:
#             track_record.keypoints_standardized = {}
#             return track_record.keypoints_standardized

#         rel_dict, root_ok_m = self._transform_to_relative_coordinates(
#             track_record,
#             theta_map,
#             nominal_torso_length
#         )
#         if rel_dict is None or root_ok_m is None:
#             track_record.keypoints_standardized = {}
#             return track_record.keypoints_standardized
            
#         rel_dict_smoothed = self._apply_temporal_gaussian_smoothing(track_record, rel_dict)

#         out_w, out_h = self.output_wh
#         canvas_c = np.array([out_w / 2, out_h / 2])
#         pixel_dict = self._finalize_pixel_coordinates(rel_dict_smoothed, root_ok_m, canvas_c, out_w, out_h)

#         if not pixel_dict:
#             logger.info(f'Track {track_record.track_id}: pixel_dict is empty, no standardized keypoints (pixel) to store.')
#             track_record.keypoints_standardized = {}
#         else:
#             track_record.keypoints_standardized = pixel_dict

#         logger.info(f"Track {track_record.track_id} keypoints standardized (pixel coordinates) and stored.")
#         return track_record.keypoints_standardized


class KeypointScoreThresholdMetric(MetricStrategy):
    """
    根據指定的 threshold 過濾 keypoint_scores，
    當某個 frame 中所有 keypoint 的 score 都低於 threshold 時，
    將該 frame 的 keypoints、keypoint_scores、bounding box 和 scores 都刪除。
    """

    def __init__(self, threshold: float = 0.5):
        """
        初始化 KeypointScoreThresholdMetric

        Args:
            threshold: keypoint score 的閾值，當 frame 中所有 keypoint score 都低於此值時，刪除該 frame 的所有資料
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "keypoint_score_threshold"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        分析 track 中每一幀的 keypoint_scores，
        當該 frame 中所有 keypoint score 都低於 threshold 時，
        將該 frame 的 keypoints、keypoint_scores、bounding box 和 scores 都刪除。

        Args:
            track: 要分析的 TrackRecord
            dependencies: AnalysisResult 物件（此策略不需要依賴其他策略）

        Returns:
            包含過濾結果的 DataFrame
        """
        if not track.keypoints or not track.keypoint_scores:
            logger.warning(f"Track {track.track_id}: 沒有 keypoints 或 keypoint_scores 資料可供過濾。")
            return pd.DataFrame()

        processed_frames = 0
        removed_frames = []

        # 需要檢查的 frame 列表
        frames_to_check = list(track.keypoints.keys())

        # 收集每幀的處理結果
        per_frame_data = []

        for fid in frames_to_check:
            # 檢查該幀是否有 keypoints 和 keypoint_scores
            keypoints_for_frame = track.keypoints.get(fid)
            keypoint_scores_for_frame = track.keypoint_scores.get(fid)

            if keypoints_for_frame is None or keypoint_scores_for_frame is None:
                logger.debug(f"Track {track.track_id} Frame {fid}: 缺少 keypoints 或 keypoint_scores，跳過處理。")
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'missing_data'
                })
                continue

            # 確保 keypoints 是 numpy array 並且有正確的形狀
            if not isinstance(keypoints_for_frame, np.ndarray):
                logger.debug(f"Track {track.track_id} Frame {fid}: keypoints 不是 numpy array，跳過處理。")
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'invalid_keypoints'
                })
                continue

            # 確保 keypoints 有正確的形狀 (17, 2) 或 (17, 3)
            if keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: keypoints 形狀不正確 {keypoints_for_frame.shape}，跳過處理。")
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'invalid_shape'
                })
                continue

            # 處理 keypoint_scores，可能是 numpy array 或其他格式
            scores = self._extract_scores(keypoint_scores_for_frame, fid, track.track_id)
            if scores is None:
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'invalid_scores'
                })
                continue

            # 確保 scores 的長度與 keypoints 數量一致
            if len(scores) != keypoints_for_frame.shape[0]:
                logger.warning(f"Track {track.track_id} Frame {fid}: keypoint_scores 數量 ({len(scores)}) 與 keypoints 數量 ({keypoints_for_frame.shape[0]}) 不一致。")
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'mismatch_count'
                })
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

                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': True,
                    'reason': 'below_threshold'
                })
            else:
                per_frame_data.append({
                    'frame_id': fid,
                    'filtered': False,
                    'reason': 'above_threshold'
                })

            processed_frames += 1

        # 更新 track 的 first_frame 和 last_frame
        if track.keypoints:
            remaining_frames = sorted(track.keypoints.keys())
            track.first_frame = remaining_frames[0]
            track.last_frame = remaining_frames[-1]
        else:
            track.first_frame = None
            track.last_frame = None

        logger.info(f"Track {track.track_id}: KeypointScoreThreshold 分析完成。處理了 {processed_frames} 幀，刪除了 {len(removed_frames)} 個 frames (threshold={self.threshold})。")

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        return df

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


# class KeypointsPreprocessingMetric(MetricAnalysisStrategy):
#     """
#     根據指定的資料預處理流程對關鍵點進行標準化處理：
#     1. 視覺中心置中 (Visual Center Centering)：計算整個軌跡的視覺中心並進行置中
#     2. 平滑化 (Smoothing)：使用一維高斯濾波器對時間序列進行平滑化處理
#     """
    
#     def __init__(self, gauss_sigma: float = 1.0):
#         """
#         初始化 KeypointsPreprocessingMetric
        
#         Args:
#             gauss_sigma: 高斯濾波器的標準差，用於平滑化處理
#         """
#         self.gauss_sigma = gauss_sigma
    
#     def analyze(self, track: TrackRecord) -> Dict[int, np.ndarray]:
#         """
#         對 track 進行關鍵點預處理
        
#         Args:
#             track: 要處理的 TrackRecord
            
#         Returns:
#             預處理後的關鍵點字典 {frame_id: preprocessed_keypoints}
#         """
#         if not track.keypoints:
#             logger.warning(f'Track {track.track_id}: 沒有關鍵點資料可供預處理。')
#             track.keypoints_standardized = {}
#             return track.keypoints_standardized
        
#         # Step 1: 提取有效的關鍵點資料
#         frame_data = self._extract_valid_keypoints(track)
#         if not frame_data:
#             logger.warning(f'Track {track.track_id}: 沒有有效的關鍵點資料。')
#             track.keypoints_standardized = {}
#             return track.keypoints_standardized
        
#         # Step 2: 計算視覺中心並進行置中
#         centered_data = self._apply_visual_center_centering(frame_data, track.track_id)
        
#         # Step 3: 平滑化 (Smoothing) - 使用一維高斯濾波器
#         smoothed_data = self._apply_smoothing(centered_data, track.track_id)
        
#         # 儲存預處理結果
#         track.keypoints_standardized = smoothed_data
#         logger.info(f'Track {track.track_id}: 關鍵點預處理完成，處理了 {len(centered_data)} 個幀。')
        
#         return track.keypoints_standardized
    
#     def _extract_valid_keypoints(self, track: TrackRecord) -> Dict[int, np.ndarray]:
#         """提取有效的關鍵點資料"""
#         valid_data = {}
        
#         for fid, kp in track.keypoints.items():
#             if kp is None or not isinstance(kp, np.ndarray):
#                 continue
            
#             # 確保關鍵點有正確的形狀
#             if kp.ndim == 1:
#                 kp = kp.reshape(-1, 2)
#             elif kp.ndim == 2 and kp.shape[1] > 2:
#                 kp = kp[:, :2]  # 只取 x, y 座標
            
#             if kp.shape[0] != 17 or kp.shape[1] != 2:
#                 logger.debug(f'Track {track.track_id} Frame {fid}: 關鍵點形狀不正確 {kp.shape}，跳過。')
#                 continue
            
#             valid_data[fid] = kp.copy()
        
#         return valid_data
    
#     def _apply_visual_center_centering(self, frame_data: Dict[int, np.ndarray], track_id: int, margin: int = 50) -> Dict[int, np.ndarray]:
#         """步驟1: 視覺中心置中 - 將人物中心作為畫布視覺中心，確保所有座標為正值"""
#         if not frame_data:
#             return {}
        
#         # 收集所有有效關鍵點的座標來計算視覺中心
#         all_x_coords = []
#         all_y_coords = []
        
#         for fid, kp in frame_data.items():
#             # 只考慮非零的關鍵點
#             valid_mask = ~np.all(kp == 0, axis=1)
#             valid_keypoints = kp[valid_mask]
            
#             if len(valid_keypoints) > 0:
#                 all_x_coords.extend(valid_keypoints[:, 0])
#                 all_y_coords.extend(valid_keypoints[:, 1])
        
#         if not all_x_coords or not all_y_coords:
#             logger.warning(f'Track {track_id}: 沒有有效的關鍵點座標，無法計算視覺中心。')
#             return frame_data
        
#         # 計算人物運動範圍
#         x_min, x_max = min(all_x_coords), max(all_x_coords)
#         y_min, y_max = min(all_y_coords), max(all_y_coords)
        
#         # 直接使用人物中心作為新的原點
#         person_center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
        
#         # 計算偏移量，確保所有點都為正值
#         # 找出變換後的最小座標
#         min_x_after = x_min - person_center[0]
#         min_y_after = y_min - person_center[1]
        
#         # 如果會產生負座標，加上額外的偏移量
#         additional_offset = np.array([0.0, 0.0])
#         if min_x_after < margin:
#             additional_offset[0] = margin - min_x_after
#         if min_y_after < margin:
#             additional_offset[1] = margin - min_y_after
        
#         logger.debug(f'Track {track_id}: 計算得到人物中心: ({person_center[0]:.2f}, {person_center[1]:.2f})')
#         logger.debug(f'Track {track_id}: X範圍: {x_min:.2f} ~ {x_max:.2f}, Y範圍: {y_min:.2f} ~ {y_max:.2f}')
#         logger.debug(f'Track {track_id}: 額外偏移量: ({additional_offset[0]:.2f}, {additional_offset[1]:.2f})')
        
#         # 對每一幀進行視覺中心置中
#         centered_data = {}
#         for fid, kp in frame_data.items():
#             centered_kp = kp.copy()
#             # 只對非零點進行置中處理
#             valid_mask = ~np.all(kp == 0, axis=1)
#             # 將人物中心移到原點，然後加上偏移量確保正座標
#             centered_kp[valid_mask] = (kp[valid_mask] - person_center) + additional_offset
            
#             centered_data[fid] = centered_kp
        
#         logger.debug(f'Track {track_id}: 以人物中心為畫布中心的置中處理完成，處理了 {len(centered_data)} 個幀。')
#         return centered_data
    
    
#     def _apply_smoothing(self, frame_data: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
#         """步驟2: 平滑化 - 使用一維高斯濾波器"""
#         if self.gauss_sigma <= 0 or not frame_data:
#             return frame_data
        
#         # 獲取排序的幀 ID
#         frame_ids = sorted(frame_data.keys())
#         if len(frame_ids) < 3:  # 至少需要3個點才能進行有效的平滑化
#             logger.debug(f'Track {track_id}: 幀數太少 ({len(frame_ids)})，跳過平滑化。')
#             return frame_data
        
#         smoothed_data = {}
        
#         # 建立高斯濾波器
#         window_size = int(self.gauss_sigma * 6 + 1) | 1  # 確保是奇數
#         gaussian_kernel = sig.windows.gaussian(window_size, std=self.gauss_sigma)
#         gaussian_kernel /= gaussian_kernel.sum()
        
#         # 對每個關鍵點的每個座標進行平滑化
#         for kp_idx in range(17):  # COCO 有 17 個關鍵點
#             for coord_idx in range(2):  # x, y 座標
#                 # 收集時間序列資料
#                 time_series = []
#                 valid_mask = []
                
#                 for fid in frame_ids:
#                     kp = frame_data[fid]
#                     # 檢查該關鍵點是否有效
#                     if not np.allclose(kp[kp_idx], 0):
#                         time_series.append(kp[kp_idx, coord_idx])
#                         valid_mask.append(True)
#                     else:
#                         time_series.append(0.0)  # 填入0，但標記為無效
#                         valid_mask.append(False)
                
#                 time_series = np.array(time_series)
#                 valid_mask = np.array(valid_mask)
                
#                 # 只對有效點進行平滑化
#                 if valid_mask.sum() < 3:
#                     continue
                
#                 try:
#                     # 提取有效值進行平滑化
#                     valid_values = time_series[valid_mask]
                    
#                     # 計算填充長度
#                     pad_len = min(len(valid_values) - 1, window_size // 2)
#                     if pad_len < 0:
#                         pad_len = 0
                    
#                     # 應用濾波器
#                     if len(valid_values) > pad_len * 2 + 1:
#                         smoothed_valid = sig.filtfilt(
#                             gaussian_kernel, [1.0], valid_values, 
#                             padlen=pad_len, method='pad'
#                         )
                        
#                         # 將平滑化的值放回原位置
#                         valid_idx = 0
#                         for i, fid in enumerate(frame_ids):
#                             if valid_mask[i]:
#                                 if fid not in smoothed_data:
#                                     smoothed_data[fid] = frame_data[fid].copy()
#                                 smoothed_data[fid][kp_idx, coord_idx] = smoothed_valid[valid_idx]
#                                 valid_idx += 1
#                     else:
#                         logger.debug(f'Track {track_id} 關鍵點 {kp_idx} 座標 {coord_idx}: 資料點不足，跳過平滑化。')
                        
#                 except Exception as e:
#                     logger.debug(f'Track {track_id} 關鍵點 {kp_idx} 座標 {coord_idx}: 平滑化失敗 - {e}')
#                     continue
        
#         # 對於沒有被平滑化的幀，保留原始資料
#         for fid, kp in frame_data.items():
#             if fid not in smoothed_data:
#                 smoothed_data[fid] = kp.copy()
        
#         logger.debug(f'Track {track_id}: 高斯平滑化完成，處理了 {len(smoothed_data)} 個幀。')
#         return smoothed_data


class KeypointsNormalizationMetric(MetricStrategy):
    """
    關鍵點正規化策略（五步驟標準化流程）：
    1. 旋轉標準化：將身體軀幹垂直於地面（對齊Y軸）
    2. 平移標準化：以肩部中心為原點進行置中
    3. 尺度標準化：基於軀幹長度進行尺度正規化
    4. 高斯平滑：對時間序列應用高斯濾波器進行平滑化
    5. 解析度標準化：將座標正規化到 [0,1] 範圍

    適合作為機器學習模型的輸入預處理，消除方向、位置、尺度、噪聲和解析度的影響
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 center_keypoint: str = "shoulder_center",
                 scale_method: str = "torso_length",
                 reference_scale: float = 0.2,
                 enable_smoothing: bool = True,
                 gauss_sigma: float = 1.0):
        """
        初始化關鍵點正規化策略

        Args:
            image_width: 影像寬度，用於解析度標準化
            image_height: 影像高度，用於解析度標準化
            center_keypoint: 中心關鍵點選擇 ("shoulder_center", "hip_center", "torso_center")
            scale_method: 尺度計算方法 ("torso_length", "max_distance", "bounding_box")
            reference_scale: 參考尺度值，用於尺度標準化
            enable_smoothing: 是否啟用高斯平滑功能
            gauss_sigma: 高斯濾波器的標準差，用於平滑化處理（當 enable_smoothing=True 時生效）
        """
        self.image_width = image_width
        self.image_height = image_height
        self.center_keypoint = center_keypoint
        self.scale_method = scale_method
        self.reference_scale = reference_scale
        self.enable_smoothing = enable_smoothing
        self.gauss_sigma = gauss_sigma

    @property
    def name(self) -> str:
        return "keypoints_normalization"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        對軌跡進行五步驟關鍵點正規化：
        1. 旋轉標準化（將身體軀幹垂直於地面，對齊Y軸）
        2. 平移標準化（以肩部中心為原點置中）
        3. 尺度標準化（基於軀幹長度正規化）
        4. 高斯平滑（對時間序列應用高斯濾波器進行平滑化）
        5. 解析度標準化（正規化到 [-1,1] 範圍）

        Args:
            track: 要處理的 TrackRecord
            dependencies: AnalysisResult 物件（此策略不需要依賴其他策略）

        Returns:
            包含正規化結果的 DataFrame
        """
        if not track.keypoints:
            logger.warning(f'Track {track.track_id}: 沒有關鍵點資料可供正規化。')
            return pd.DataFrame()

        # 步驟 0: 提取有效的關鍵點資料
        valid_keypoints = self._extract_valid_keypoints(track)
        if not valid_keypoints:
            logger.warning(f'Track {track.track_id}: 沒有有效的關鍵點資料。')
            return pd.DataFrame()

        # 步驟 1: 旋轉標準化（將身體軀幹垂直於地面，對齊Y軸）
        # rotation_normalized = self._apply_rotation_normalization(valid_keypoints, track.track_id)

        # 步驟 2: 平移標準化（以肩部中心為原點置中）
        translation_normalized = self._apply_translation_normalization(valid_keypoints, track.track_id)

        # 步驟 3: 尺度標準化（基於軀幹長度正規化）
        scale_normalized = self._apply_scale_normalization(translation_normalized, track.track_id)

        # 步驟 4: 解析度標準化（正規化到 [-1,1] 範圍）
        resolution_normalized = self._apply_resolution_normalization(scale_normalized, track.track_id)

        # 步驟 5: 高斯平滑（對時間序列應用高斯濾波器進行平滑化）
        smooth_normalized = self._apply_gaussian_smoothing(resolution_normalized, track.track_id) if self.enable_smoothing else resolution_normalized

        # 解析度標準化後再以肩部中心為原點進行平移
        final_normalized = self._apply_translation_normalization(smooth_normalized, track.track_id)

        # 儲存正規化結果到 track 物件（保持向後相容性）
        track.keypoints_normalized = final_normalized

        # 收集每幀的處理結果
        per_frame_data = []
        for fid, kp in final_normalized.items():
            per_frame_data.append({
                'frame_id': fid,
                'normalized_keypoints': kp.tolist() if isinstance(kp, np.ndarray) else kp
            })

        smoothing_status = "含高斯平滑" if self.enable_smoothing else "不含平滑"
        logger.info(f'Track {track.track_id}: 五步驟關鍵點正規化完成（{smoothing_status}，解析度正規化後再平移）')

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        return df
    
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
    
    def _apply_resolution_normalization(self, keypoints_dict: Dict[int, np.ndarray], track_id: int) -> Dict[int, np.ndarray]:
        """
        解析度標準化 - 將座標正規化，除以影像寬高
        
        這是最後一步，將經過所有預處理的座標除以影像寬高進行正規化。
        
        Args:
            keypoints_dict: 高斯平滑後的關鍵點字典 {frame_id: keypoints}
            track_id: 軌跡ID，用於日誌
            
        Returns:
            解析度正規化後的關鍵點字典
        """
        if not keypoints_dict:
            return {}
        
        normalized_keypoints = {}
        
        for fid, kp in keypoints_dict.items():
            normalized_kp = kp.copy()
            
            # 只對非零點進行正規化
            valid_mask = ~np.all(kp == 0, axis=1)
            
            # 統一除以影像寬高最大值進行正規化
            norm_div = max(self.image_width, self.image_height)
            normalized_kp[valid_mask, 0] = kp[valid_mask, 0] / norm_div
            normalized_kp[valid_mask, 1] = kp[valid_mask, 1] / norm_div

            normalized_keypoints[fid] = normalized_kp
        
        logger.debug(f'Track {track_id}: 解析度標準化完成，處理了 {len(normalized_keypoints)} 個幀。')
        return normalized_keypoints
    
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
            
            target_vector = np.array([0.0, 1.0])  # Y軸正方向（向上）
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
        # 確保向量已正規化
        current_norm = current_vector / np.linalg.norm(current_vector)
        target_norm = target_vector / np.linalg.norm(target_vector)
        
        # 計算點積和叉積
        dot_product = np.dot(current_norm, target_norm)
        cross_product = np.cross(current_norm, target_norm)
        
        # 限制點積範圍避免數值誤差
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 計算角度
        angle = np.arccos(dot_product)
        
        # 根據叉積的符號決定旋轉方向
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
        
        # 建立2D旋轉矩陣
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # 只對非零關鍵點應用旋轉
        valid_mask = ~np.all(keypoints == 0, axis=1)
        if np.any(valid_mask):
            rotated_kp[valid_mask] = (rotation_matrix @ keypoints[valid_mask].T).T
        
        return rotated_kp
    
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
        if self.scale_method == "torso_length":
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
            
        else:
            reference_scale_factor = self.reference_scale
        
        logger.debug(f'Track {track_id}: 使用參考尺度因子: {reference_scale_factor:.6f}')
        
        for fid, kp in keypoints_dict.items():
            if self.scale_method == "torso_length":
                current_scale = self._calculate_torso_length(kp)
            elif self.scale_method == "max_distance":
                current_scale = self._calculate_max_distance(kp)
            elif self.scale_method == "bounding_box":
                current_scale = self._calculate_bounding_box_size(kp)
            else:
                current_scale = reference_scale_factor
            
            if current_scale is None or current_scale < 1e-6:
                logger.debug(f'Track {track_id} Frame {fid}: 無法計算尺度因子，跳過。')
                continue
            
            # 應用尺度標準化
            scale_factor = reference_scale_factor / current_scale
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
    
    def _calculate_max_distance(self, keypoints: np.ndarray) -> Optional[float]:
        """
        計算所有關鍵點到原點的最大距離
        
        Args:
            keypoints: 關鍵點數組 (17, 2)
            
        Returns:
            最大距離，如果無法計算則返回 None
        """
        valid_mask = ~np.all(keypoints == 0, axis=1)
        valid_keypoints = keypoints[valid_mask]
        
        if len(valid_keypoints) == 0:
            return None
        
        distances = np.linalg.norm(valid_keypoints, axis=1)
        max_distance = np.max(distances)
        return max_distance if max_distance > 1e-6 else None
    
    def _calculate_bounding_box_size(self, keypoints: np.ndarray) -> Optional[float]:
        """
        計算關鍵點包圍框的對角線長度
        
        Args:
            keypoints: 關鍵點數組 (17, 2)
            
        Returns:
            包圍框對角線長度，如果無法計算則返回 None
        """
        valid_mask = ~np.all(keypoints == 0, axis=1)
        valid_keypoints = keypoints[valid_mask]
        
        if len(valid_keypoints) == 0:
            return None
        
        x_min, x_max = np.min(valid_keypoints[:, 0]), np.max(valid_keypoints[:, 0])
        y_min, y_max = np.min(valid_keypoints[:, 1]), np.max(valid_keypoints[:, 1])
        diagonal_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        return diagonal_length if diagonal_length > 1e-6 else None
    
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
        
        # 獲取排序的幀 ID
        frame_ids = sorted(keypoints_dict.keys())
        if len(frame_ids) < 3:  # 至少需要3個點才能進行有效的平滑化
            logger.debug(f'Track {track_id}: 幀數太少 ({len(frame_ids)})，跳過高斯平滑。')
            return keypoints_dict
        
        smoothed_keypoints = {}
        
        # 建立高斯濾波器
        window_size = int(self.gauss_sigma * 6 + 1) | 1  # 確保是奇數
        gaussian_kernel = sig.windows.gaussian(window_size, std=self.gauss_sigma)
        gaussian_kernel /= gaussian_kernel.sum()
        
        # 對每個關鍵點的每個座標進行平滑化
        for kp_idx in range(17):  # COCO 有 17 個關鍵點
            for coord_idx in range(2):  # x, y 座標
                # 收集時間序列資料
                time_series = []
                valid_mask = []
                
                for fid in frame_ids:
                    kp = keypoints_dict[fid]
                    # 檢查該關鍵點是否有效（非零值）
                    if not np.allclose(kp[kp_idx], 0):
                        time_series.append(kp[kp_idx, coord_idx])
                        valid_mask.append(True)
                    else:
                        time_series.append(0.0)  # 填入0，但標記為無效
                        valid_mask.append(False)
                
                time_series = np.array(time_series)
                valid_mask = np.array(valid_mask)
                
                # 只對有效點進行平滑化
                if valid_mask.sum() < 3:
                    continue
                
                try:
                    # 提取有效值進行平滑化
                    valid_values = time_series[valid_mask]
                    
                    # 計算填充長度
                    pad_len = min(len(valid_values) - 1, window_size // 2)
                    if pad_len < 0:
                        pad_len = 0
                    
                    # 應用濾波器
                    if len(valid_values) > pad_len * 2 + 1:
                        smoothed_valid = sig.filtfilt(
                            gaussian_kernel, [1.0], valid_values,
                            padlen=pad_len, method='pad'
                        )
                        
                        # 將平滑化的值放回原位置
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
        
        # 對於沒有被平滑化的幀，保留原始資料
        for fid, kp in keypoints_dict.items():
            if fid not in smoothed_keypoints:
                smoothed_keypoints[fid] = kp.copy()
        logger.debug(f'Track {track_id}: 高斯平滑完成（sigma={self.gauss_sigma}），處理了 {len(smoothed_keypoints)} 個幀。')
        return smoothed_keypoints
    
    
class AnkleAlternationMetric(MetricStrategy):
    """
    計算腳踝 Y 座標差值的交替模式（波峰-波谷序列）來偵測走路片段。
    偵測連續的波峰 (左腳高) 和波谷 (右腳高) 作為步態交替的標記。
    """

    def __init__(
        self,
        min_peak_distance: int = 7,     # 波峰/波谷之間的最小幀距離 (大致對應半個步態週期)
        peak_prominence: float = 0.01,      # 波峰/波谷的顯著性 (相對於標準化座標)
        height: float = 0.01,            # 波峰/波谷的高度閾值 (相對於標準化座標)
        min_alternating_cycles: int = 7, # 判斷走路所需的最小交替週期數 (e.g., 3 = 波峰-波谷-波峰)
        gap_tolerance: int = 15,         # 交替序列中允許的短暫間隙
        smoothing_window: int = 5,       # 平滑視窗大小 (奇數)
        left_ankle_idx: int = 15,        # COCO17 左腳踝索引
        right_ankle_idx: int = 16,       # COCO17 右腳踝索引
        use_normalized: bool = True,     # 使用 keypoints_normalized
        confidence_threshold: float = 0.5,  # 腳踝關鍵點的信心分數閾值
    ):
        if min_peak_distance < 1:
            raise ValueError("min_peak_distance must be at least 1.")
        if min_alternating_cycles < 1:
            raise ValueError("min_alternating_cycles must be at least 1.")
        if smoothing_window % 2 == 0 or smoothing_window < 3:
            raise ValueError("smoothing_window must be an odd integer >= 3.")

        self._min_peak_distance = min_peak_distance
        self._peak_prominence = peak_prominence
        self._height = height
        self._min_alternating_cycles = min_alternating_cycles
        self._gap_tolerance = gap_tolerance
        self._smoothing_window = smoothing_window
        self._left_ankle_idx = left_ankle_idx
        self._right_ankle_idx = right_ankle_idx
        self._use_normalized = use_normalized
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "ankle_alternation"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        計算腳踝交替序列並返回包含序列列表的 DataFrame。
        """
        # 獲取 keypoints 數據
        if self._use_normalized:
            # 使用 preprocessor 正規化後的 keypoints
            keypoints_data = track.keypoints_normalized
            if not keypoints_data:
                logger.warning(f"Track {track.track_id}: Normalized keypoints not found, falling back to original keypoints")
                keypoints_data = track.keypoints
        else:
            keypoints_data = track.keypoints

        keypoints_name = "keypoints_normalized" if self._use_normalized else "keypoints"

        if not keypoints_data or track.first_frame is None or track.last_frame is None:
            logger.warning(f"Track {track.track_id}: Insufficient data for ankle alternation calculation.")
            return pd.DataFrame({'alternating_sequences': [[]]})

        ankle_diff_series = []
        valid_fids_for_diff = []
        for fid in range(track.first_frame, track.last_frame + 1):
            kps = keypoints_data.get(fid)
            if (kps is not None and
                kps.shape[0] > max(self._left_ankle_idx, self._right_ankle_idx) and
                kps.shape[1] >= 2):
                left_ankle_y = kps[self._left_ankle_idx, 1]
                right_ankle_y = kps[self._right_ankle_idx, 1]
                ankle_diff_series.append(left_ankle_y - right_ankle_y)
                valid_fids_for_diff.append(fid)

        if len(ankle_diff_series) < self._min_peak_distance * 2:  # 至少需要足夠的點來找交替
            logger.debug(f"Track {track.track_id}: Not enough valid data points ({len(ankle_diff_series)}).")
            return pd.DataFrame({'alternating_sequences': [[]]})

        diff_series = np.array(ankle_diff_series)

        # 中位數中心化，減去中位數以減少偏移對峰/谷偵測的影響
        if diff_series.size > 0:
            median_val = float(np.nanmedian(diff_series))
        else:
            median_val = 0.0
        diff_series_centered = diff_series - median_val

        peak_indices, _ = find_peaks(diff_series_centered, distance=self._min_peak_distance, prominence=self._peak_prominence, height=self._height)
        valley_indices, _ = find_peaks(-diff_series_centered, distance=self._min_peak_distance, prominence=self._peak_prominence, height=self._height)

        logger.info(f"Track {track.track_id}: Found {len(peak_indices)} peaks and {len(valley_indices)} valleys (median centered).")

        if len(peak_indices) + len(valley_indices) < self._min_alternating_cycles:
            return pd.DataFrame({'alternating_sequences': [[]]})

        # 映射到幀ID
        peak_fids = [valid_fids_for_diff[i] for i in peak_indices if i < len(valid_fids_for_diff)]
        valley_fids = [valid_fids_for_diff[i] for i in valley_indices if i < len(valid_fids_for_diff)]

        # 找到連續的交替序列 (波峰-波谷-波峰...)
        alternating_sequences = self._find_alternating_peak_valley_sequences(peak_fids, valley_fids)

        # 儲存用於繪圖的數據 (擴展原有的)
        track.temp_ankle_diff_data = {
            'valid_fids_for_peaks': valid_fids_for_diff,
            'input_series_for_peaks': diff_series_centered,
            'peak_indices': peak_indices,
            'valley_indices': valley_indices,
            'track_id': track.track_id
        }

        return pd.DataFrame({'alternating_sequences': [alternating_sequences]})

    def _find_alternating_peak_valley_sequences(self, peak_fids: List[int], valley_fids: List[int]) -> List[List[int]]:
        """
        找到連續的波峰-波谷交替序列。
        使用簡單的 flag 方法：當是波峰時，下一個應該是波谷；若距離大於_gap_tolerance則視為新的序列。
        """
        if not peak_fids and not valley_fids:
            return []

        # 合併所有點並排序，按幀ID排序
        all_points = [(fid, 'peak') for fid in peak_fids] + [(fid, 'valley') for fid in valley_fids]
        all_points.sort(key=lambda x: x[0])

        sequences = []
        current_sequence = []
        expecting_peak = None  # None 表示尚未確定，True 表示下一個應該是波峰，False 表示下一個應該是波谷
        last_fid = None

        for fid, ptype in all_points:
            if not current_sequence:
                # 第一個點
                current_sequence.append(fid)
                # 根據第一個點的類型設置期望
                expecting_peak = (ptype == 'valley')  # 如果第一個是波峰，則下一個應該是波谷
                last_fid = fid
            else:
                # 檢查是否符合交替規則
                if (ptype == 'peak' and expecting_peak) or (ptype == 'valley' and not expecting_peak):
                    # 符合交替規則
                    if last_fid is not None and (fid - last_fid) <= self._gap_tolerance:
                        current_sequence.append(fid)
                        expecting_peak = not expecting_peak  # 切換期望類型
                        last_fid = fid
                    else:
                        # 距離太大，視為新序列
                        if len(current_sequence) >= self._min_alternating_cycles:
                            sequences.append(current_sequence)
                        current_sequence = [fid]
                        expecting_peak = (ptype == 'valley')  # 重新設定期望類型
                        last_fid = fid
                else:
                    # 不符合交替規則
                    if len(current_sequence) >= self._min_alternating_cycles:
                        sequences.append(current_sequence)
                    current_sequence = [fid]
                    expecting_peak = (ptype == 'valley')  # 重新設定期望類型
                    last_fid = fid

        # 處理最後一個序列
        if len(current_sequence) >= self._min_alternating_cycles:
            sequences.append(current_sequence)

        logger.debug(f"Found {len(sequences)} alternating sequences: {sequences}")
        return sequences


class SpineAngleMetric(MetricStrategy):
    @property
    def name(self) -> str:
        return "spine_angle"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        Analyzes the track to calculate spine line angle with y-axis for each frame
        and returns them as a DataFrame.
        """
        # 收集每幀的數據
        per_frame_data = []

        for fid, keypoints_for_frame in track.keypoints.items():
            if keypoints_for_frame is None or keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: Invalid keypoints, skipping spine angle.")
                result_dict = {
                    'frame_id': fid,
                    'spine_angle': np.nan
                }
            else:
                kp = keypoints_for_frame[:, :2]  # Use only x, y coordinates

                # COCO 17 points: 5:LShoulder, 6:RShoulder, 11:LHip, 12:RHip
                l_shoulder, r_shoulder = kp[5], kp[6]
                l_hip, r_hip = kp[11], kp[12]

                # Check if critical keypoints are valid
                if (np.allclose(l_shoulder, 0) or np.allclose(r_shoulder, 0) or
                    np.allclose(l_hip, 0) or np.allclose(r_hip, 0)):
                    logger.debug(f"Track {track.track_id} Frame {fid}: Missing critical keypoints for spine angle.")
                    result_dict = {
                        'frame_id': fid,
                        'spine_angle': np.nan
                    }
                else:
                    # Calculate shoulder center and hip center
                    shoulder_center = np.mean([l_shoulder, r_shoulder], axis=0)
                    hip_center = np.mean([l_hip, r_hip], axis=0)

                    # Spine vector: from shoulder to hip
                    spine_vector = hip_center - shoulder_center

                    # Calculate angle with y-axis
                    if np.linalg.norm(spine_vector) < 1e-6:
                        angle = np.nan
                        logger.debug(f"Track {track.track_id} Frame {fid}: Spine vector too small.")
                    else:
                        # Angle with y-axis: atan2(vy, vx) - pi/2
                        angle = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]) - np.pi/2)

                    result_dict = {
                        'frame_id': fid,
                        'spine_angle': angle
                    }

            per_frame_data.append(result_dict)

        if not per_frame_data:
            return pd.DataFrame()

        # 將結果列表轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        logger.debug(f"Track {track.track_id} spine angle per-frame analysis completed.")
        return df
    
    
class HipCenterMetric(MetricStrategy):
    """
    計算每幀左右髖關節的中心點座標。
    使用 COCO keypoints: 左髖(11), 右髖(12)
    """

    @property
    def name(self) -> str:
        return "hip_center"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        計算每幀的左右髖關節中心點。

        Args:
            track: TrackRecord 物件
            dependencies: AnalysisResult 物件（此策略不需要依賴其他策略）

        Returns:
            包含每幀髖關節中心點座標的 DataFrame
        """
        # 收集每幀的數據
        per_frame_data = []

        for fid, keypoints_for_frame in track.keypoints.items():
            if keypoints_for_frame is None or keypoints_for_frame.shape[0] != 17 or keypoints_for_frame.ndim < 2:
                logger.debug(f"Track {track.track_id} Frame {fid}: Invalid keypoints, skipping hip center calculation.")
                result_dict = {
                    'frame_id': fid,
                    'hip_center_x': np.nan,
                    'hip_center_y': np.nan
                }
            else:
                kp = keypoints_for_frame[:, :2]  # Use only x, y coordinates

                # COCO keypoints: 11=LHip, 12=RHip
                l_hip, r_hip = kp[11], kp[12]

                # 檢查關鍵點是否有效
                if np.allclose(l_hip, 0) or np.allclose(r_hip, 0):
                    logger.debug(f"Track {track.track_id} Frame {fid}: Missing hip keypoints.")
                    result_dict = {
                        'frame_id': fid,
                        'hip_center_x': np.nan,
                        'hip_center_y': np.nan
                    }
                else:
                    # 計算髖關節中心點
                    hip_center = np.mean([l_hip, r_hip], axis=0)

                    result_dict = {
                        'frame_id': fid,
                        'hip_center_x': hip_center[0],
                        'hip_center_y': hip_center[1]
                    }

            per_frame_data.append(result_dict)

        if not per_frame_data:
            return pd.DataFrame()

        # 將結果轉換為 DataFrame 並回傳
        df = pd.DataFrame(per_frame_data).set_index('frame_id')
        logger.debug(f"Track {track.track_id} hip center per-frame analysis completed.")
        return df


class StepTimeMetric(MetricStrategy):
    """
    根據 AnkleAlternationMetric 的交替序列計算每一步的時間（秒）。
    每一步的時間 = (後一個交替點幀 - 前一個交替點幀) / fps
    """

    def __init__(self, fps: float = 30.0, segment_type_filter: Optional[str] = None):
        """
        初始化 StepTimeMetric

        Args:
            fps: 影片幀率，用於將幀數轉換為時間
            segment_type_filter: 可選的 segment 過濾器，只計算指定segment類型內的統計值
        """
        self.fps = fps
        self.segment_type_filter = segment_type_filter

    @property
    def name(self) -> str:
        filter_name = f"_{self.segment_type_filter}" if self.segment_type_filter else ""
        return f"step_time{filter_name}"

    def analyze(self, track: TrackRecord, dependencies: AnalysisResult) -> pd.DataFrame:
        """
        根據 AnkleAlternationMetric 的結果計算每一步的時間。

        Args:
            track: TrackRecord 物件
            dependencies: AnalysisResult 包含 AnkleAlternationMetric 的結果

        Returns:
            包含每一步時間列表的 DataFrame
        """
        # 從 dependencies 中獲取 AnkleAlternationMetric 的結果
        ankle_df = dependencies.get_metric("ankle_alternation")
        if ankle_df is None or ankle_df.empty:
            logger.warning(f"Track {track.track_id}: AnkleAlternationMetric result not found.")
            return pd.DataFrame({'step_times': [[]]})

        # 提取 alternating_sequences
        alternating_sequences = ankle_df.iloc[0].get('alternating_sequences', [])
        if not alternating_sequences:
            logger.debug(f"Track {track.track_id}: No alternating sequences found.")
            return pd.DataFrame({'step_times': [[]]})

        # 如果有 segment 過濾器，獲取有效的幀
        valid_frames = None
        if self.segment_type_filter:
            segment_name = self.segment_type_filter
            conditions = dependencies.get_conditions(segment_name)
            if conditions is not None and not conditions.empty:
                # 獲取條件為 True 的幀
                valid_frames = set(conditions[conditions].index)
            else:
                logger.warning(f"Track {track.track_id}: Segment '{segment_name}' not found, calculating for entire track.")

        all_step_times = []
        for sequence in alternating_sequences:
            if len(sequence) < 2:  # 至少需要兩個點
                continue

            # 如果有 segment 過濾器，只考慮在有效幀內的交替點
            if valid_frames is not None:
                filtered_sequence = [fid for fid in sequence if fid in valid_frames]
                if len(filtered_sequence) < 2:
                    continue
                sequence = filtered_sequence

            # 計算每一步的時間：每兩個連續點之間的時間
            sequence_step_times = []
            for i in range(len(sequence) - 1):
                frame_diff = sequence[i + 1] - sequence[i]
                if frame_diff > 0:
                    step_time = frame_diff / self.fps
                    sequence_step_times.append(step_time)

            # 保留每個序列的步行時間作為子列表
            all_step_times.append(sequence_step_times)

        logger.debug(f"Track {track.track_id}: Calculated {len(all_step_times)} individual step times.")
        logger.debug(all_step_times)

        return pd.DataFrame({'step_times': [all_step_times]})
    