import cv2
import numpy as np
import logging # 導入 logging 模組
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from vendor.rtmlib.rtmlib.visualization.draw import draw_skeleton, draw_bbox
from src.pose_extract.track_solution.record import TrackRecord
from src.pose_extract.track_solution.analysis.base import SegmentType


class TrackVisualizer:
    """用於可視化軌跡的類"""
    def __init__(self, target_canvas_size=None):
        self.width = 0
        self.height = 0
        self.target_canvas_size = target_canvas_size  # (width, height)

    def draw_tracks(self, frame, tracks, frame_id, options=None, target_canvas_size=None):
        """在幀上繪製追蹤，支援目標畫布尺寸"""
        options = options or {}
        show_interpolated = options.get('show_interpolated', True)
        normalized = options.get('normalized', False)

        all_bboxes = []
        all_keypoints = []
        all_keypoint_scores = []
        valid_tracks = []

        # 檢查並設定寬高
        if self.width == 0 or self.height == 0:
            self.height, self.width, _ = frame.shape
            logging.info(f"設定 TrackVisualizer 寬高為: {self.width}x{self.height}")

        # 決定目標畫布尺寸的優先順序
        canvas_size = (
            target_canvas_size or
            self.target_canvas_size or
            options.get('target_canvas_size') or
            (self.width, self.height)
        )
        
        # 如果目標畫布尺寸與原始frame尺寸不同，則進行縮放
        if not normalized and canvas_size != (self.width, self.height):
            frame = cv2.resize(frame, canvas_size)

        # 收集所有軌跡的正規化關鍵點，用於計算全局世界邊界
        all_normalized_keypoints = {}
        if normalized:
            for track in tracks:
                if hasattr(track, 'keypoints_normalized') and track.keypoints_normalized:
                    all_normalized_keypoints.update(track.keypoints_normalized)

        for track in tracks:
            # 若 caller 指定了 draw_track_ids，則只處理那些 ID
            draw_track_ids = options.get('draw_track_ids') if options else None
            if draw_track_ids is not None and track.track_id not in draw_track_ids:
                continue

            if frame_id not in track.positions:
                continue

            # 檢查是否為插值數據
            is_interpolated = track.is_interpolated.get(frame_id, False)

            if is_interpolated and not show_interpolated:
                continue

            if track.positions[frame_id] is not None:
                bbox = track.positions[frame_id]
                if not normalized and canvas_size != (self.width, self.height):
                    x1, y1, x2, y2 = bbox
                    scaled_bbox = (
                        int(x1 * (canvas_size[0] / self.width)),
                        int(y1 * (canvas_size[1] / self.height)),
                        int(x2 * (canvas_size[0] / self.width)),
                        int(y2 * (canvas_size[1] / self.height))
                    )
                    all_bboxes.append(scaled_bbox)
                else:
                    all_bboxes.append(bbox)
                valid_tracks.append(track)
            if normalized:
                keypoints_dict = track.keypoints_normalized
                is_norm = True
                display_scale = options.get('display_scale', 0.8)
            else:
                keypoints_dict = track.keypoints
                is_norm = False
                if canvas_size != (self.width, self.height):
                    scale_x = canvas_size[0] / self.width
                    scale_y = canvas_size[1] / self.height
                    scale_factor = min(scale_x, scale_y)  # 保持長寬比
                    display_scale = options.get('display_scale', scale_factor)
                else:
                    display_scale = options.get('display_scale', 1.0)

            if frame_id in keypoints_dict and keypoints_dict[frame_id] is not None:
                raw_keypoints = keypoints_dict[frame_id]
                if is_norm or canvas_size != (self.width, self.height):
                    converted_keypoints = self._convert_to_target_canvas(
                        raw_keypoints, canvas_size,
                        display_scale=display_scale,
                        is_normalized=is_norm,
                        all_normalized_keypoints=all_normalized_keypoints if is_norm else None
                    )
                    all_keypoints.append(converted_keypoints)
                else:
                    all_keypoints.append(raw_keypoints)
                if frame_id in track.keypoint_scores:
                    all_keypoint_scores.append(track.keypoint_scores[frame_id])
                else:
                    default_scores = np.ones((17,)) * 0.9
                    all_keypoint_scores.append(default_scores)

        # 繪製邊界框
        if all_bboxes and not normalized:
            frame = draw_bbox(frame, all_bboxes)

        # 繪製骨架
        if all_keypoints:
            all_keypoints = np.array(all_keypoints)
            all_keypoint_scores = np.array(all_keypoint_scores)
            frame = draw_skeleton(frame, all_keypoints, all_keypoint_scores, openpose_skeleton=False)
        else:
            logging.info(f"frame_id: {frame_id} 無軌跡")

        # 繪製文字信息
        frame = self.draw_tracks_info(frame, valid_tracks, frame_id, options)

        return frame

    def _convert_to_target_canvas(self, keypoints, target_size, display_scale=0.8, is_normalized=False, all_normalized_keypoints=None):
        """將關鍵點轉換為目標畫布的像素座標，使用全局世界邊界和縮放因子
        
        Args:
            keypoints: 當前幀的關鍵點座標
            target_size: 目標畫布尺寸 (width, height)
            display_scale: keypoint顯示範圍相對於畫布尺寸的比例 (預設0.8)
            is_normalized: 是否為正規化的關鍵點 (0-1範圍)
            all_normalized_keypoints: 所有幀的正規化關鍵點字典，用於計算世界邊界
        """
        if keypoints is None:
            return None
        
        canvas_width, canvas_height = target_size
        converted_kp = keypoints.copy()
        valid_mask = ~np.all(keypoints == 0, axis=1)
        
        if not np.any(valid_mask):
            return converted_kp
        
        if is_normalized and all_normalized_keypoints is not None:
            # 步驟 1: 計算整個序列的世界邊界
            all_points = []
            for kp in all_normalized_keypoints.values():
                if kp is not None:
                    all_points.append(kp)
            
            if all_points:
                all_points = np.vstack(all_points)
                global_valid_mask = ~np.all(all_points == 0, axis=1)
                
                if np.any(global_valid_mask):
                    valid_points = all_points[global_valid_mask]
                    
                    min_x, min_y = np.min(valid_points, axis=0)
                    max_x, max_y = np.max(valid_points, axis=0)
                    
                    world_center_x = (min_x + max_x) / 2
                    world_center_y = (min_y + max_y) / 2
                    
                    # 步驟 2: 計算全局縮放因子
                    world_width = max_x - min_x
                    world_height = max_y - min_y
                    
                    padding = (1 - display_scale) / 2  # 從 display_scale 計算 padding
                    
                    if world_width > 1e-6 and world_height > 1e-6:
                        # 分別計算寬和高的縮放比例
                        scale_ratio_x = (canvas_width * (1 - 2 * padding)) / world_width
                        scale_ratio_y = (canvas_height * (1 - 2 * padding)) / world_height
                        
                        scale_factor = min(scale_ratio_x, scale_ratio_y)  # 保持長寬比
                        
                        # 步驟 3: 轉換當前幀的關鍵點
                        valid_keypoints = converted_kp[valid_mask]
                        
                        # 使用世界中心和縮放因子進行座標轉換
                        canvas_x = ((valid_keypoints[:, 0] - world_center_x) * scale_factor) + canvas_width / 2
                        canvas_y = ((valid_keypoints[:, 1] - world_center_y) * scale_factor) + canvas_height / 2
                        
                        valid_keypoints[:, 0] = canvas_x
                        valid_keypoints[:, 1] = canvas_y
                        
                        converted_kp[valid_mask] = valid_keypoints
        
        elif is_normalized:
            # 簡單的正規化座標處理（當沒有提供全局資訊時）
            valid_keypoints = converted_kp[valid_mask]
            valid_keypoints[:, 0] = valid_keypoints[:, 0] * canvas_width
            valid_keypoints[:, 1] = valid_keypoints[:, 1] * canvas_height
            converted_kp[valid_mask] = valid_keypoints
        
        else:
            # 對於原始像素座標的處理
            current_w, current_h = self.width, self.height
            if current_w > 0 and current_h > 0:
                valid_keypoints = converted_kp[valid_mask]
                # 轉換為相對座標 (0-1)
                valid_keypoints[:, 0] = valid_keypoints[:, 0] / current_w
                valid_keypoints[:, 1] = valid_keypoints[:, 1] / current_h
                
                # 縮放到目標畫布尺寸
                valid_keypoints[:, 0] = valid_keypoints[:, 0] * canvas_width
                valid_keypoints[:, 1] = valid_keypoints[:, 1] * canvas_height
                
                converted_kp[valid_mask] = valid_keypoints
        
        # 裁剪到目標畫布範圍
        converted_kp[valid_mask, 0] = np.clip(converted_kp[valid_mask, 0], 0, canvas_width - 1)
        converted_kp[valid_mask, 1] = np.clip(converted_kp[valid_mask, 1], 0, canvas_height - 1)
        
        return converted_kp


    def draw_tracks_info(self, frame, tracks:list, frame_id, options=None):
        """繪製單個追蹤的詳細信息在邊界框的左上角"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = options.get('font_scale', 0.5) if options else 0.5
        thickness = 2
        line_height = 25 # 每行文字的高度
        normalized = options.get('normalized', False) if options else False

        # 獲取要顯示的分段類型過濾清單
        segment_types_to_show = options.get('segment_types', []) if options else []
        draw_track_ids = options.get('draw_track_ids') if options else None

        target_canvas_size = options.get('target_canvas_size', (self.width, self.height)) if options else (self.width, self.height)
        target_w, target_h = target_canvas_size

        for track in tracks:
            if frame_id not in track.positions:
                continue

            # 如果指定了特定的軌跡ID，只繪製這些軌跡
            if draw_track_ids is not None and track.track_id not in draw_track_ids:
                continue

            if normalized:
                x1, y1 = (int(target_w/4), int(target_h/4))  # 根據目標畫布尺寸調整文字位置
            else:
                x1, y1, x2, y2 = track.positions[frame_id]

            frame_count = len(track.positions)

            # 計算文字位置 - 邊界框的左上角
            text_x = int(x1)
            base_text_y = int(y1 - 5)  # 略微在邊界框上方

            # 如果文字會超出圖像上邊界，則放在邊界框內部上方
            if base_text_y < 15:
                base_text_y = int(y1 + 15)

            current_y = base_text_y

            # --- 繪製基本信息 ---
            text = f"ID:{track.track_id} F:{frame_count}"
            
            # 只有在有相關數據時才顯示比例信息
            if hasattr(track, 'body_to_head_ratio_at_frame') and track.body_to_head_ratio_at_frame:
                h_r_value = track.body_to_head_ratio_at_frame.get(frame_id, 'ERR')
                if h_r_value != 'ERR':
                    text += f" H_R:{h_r_value:.2f}"
            
            if hasattr(track, 'sitting_height_index_at_frame') and track.sitting_height_index_at_frame:
                s_h_value = track.sitting_height_index_at_frame.get(frame_id, 'ERR')
                if s_h_value != 'ERR':
                    text += f" S/H:{s_h_value:.2f}"

            cv2.putText(frame, text, (text_x, current_y),
                    font, font_scale, (0, 0, 255), thickness) # 基本信息用紅色
            current_y += line_height

            # --- 繪製 frame_conditions 中的分段狀態信息（根據過濾清單） ---
            if hasattr(track, 'frame_conditions') and track.frame_conditions:
                # 如果沒有指定要顯示的分段類型，則顯示所有
                conditions_to_process = (
                    {k: v for k, v in track.frame_conditions.items() if k in segment_types_to_show}
                    if segment_types_to_show else track.frame_conditions
                )

                for condition_key, frame_data in conditions_to_process.items():
                    if frame_id in frame_data:
                        state = frame_data[frame_id] # bool: True or False
                        prefix_display = condition_key # 默認顯示原始鍵名
                        status_display = str(state)   # 默認顯示 True/False
                        try:
                            # 假設 condition_key 是 SegmentType 的 value
                            st_enum_member = SegmentType(condition_key)
                            prefix_display = st_enum_member.name # 例如 "MOVING"
                            # 使用 label_map 來獲取狀態的文本描述
                            # 例如，SegmentType.MOVING.label_map[True] -> "移動"
                            if hasattr(st_enum_member, 'label_map') and isinstance(st_enum_member.label_map, dict):
                                status_display = st_enum_member.label_map.get(state, str(state))
                        except ValueError:
                            # 如果 condition_key 不是標準 SegmentType 的 value，則直接使用原始鍵和狀態
                            pass
                        state_text = f"{prefix_display}: {status_display}"
                        state_color = (0, 255, 0) if state else (0, 0, 255)  # state 布林狀態顏色 (True: 綠色, False: 紅色)

                        # 繪製文字
                        cv2.putText(frame, state_text, (text_x, current_y),
                                font, font_scale, state_color, thickness)
                        current_y += line_height

        return frame

    # def plot_ankle_difference_and_peaks(self, track: TrackRecord, output_path: str = None):
    #     """
    #     繪製腳踝 Y 座標差值的變化以及偵測到的峰值和波谷。
    #     適配新版 WalkingDetectionByAnkleAlternationStrategy 的資料結構。
    #     """
    #     if not hasattr(track, 'temp_ankle_diff_data') or not track.temp_ankle_diff_data:
    #         logging.warning(f"Track {track.track_id}: No temp_ankle_diff_data found. Cannot plot ankle difference.")
    #         return None

    #     data = track.temp_ankle_diff_data
    #     valid_fids_for_peaks = data.get('valid_fids_for_peaks', [])
    #     input_series_for_peaks = data.get('input_series_for_peaks', [])
    #     peak_indices = data.get('peak_indices', [])
    #     valley_indices = data.get('valley_indices', [])
        
    #     if len(valid_fids_for_peaks) != len(input_series_for_peaks):
    #         logging.error(f"Track {track.track_id}: Mismatch in lengths of valid_fids_for_peaks and input_series_for_peaks. Cannot plot.")
    #         return None

    #     def _plot_extremes(indices, marker, color, label_prefix):
    #         """繪製極值點的通用函數"""
    #         if len(indices) > 0:
    #             valid_indices = [idx for idx in indices if 0 <= idx < len(input_series_for_peaks)]
    #             if valid_indices:
    #                 fids = [valid_fids_for_peaks[i] for i in valid_indices]
    #                 values = [input_series_for_peaks[i] for i in valid_indices]
    #                 plt.plot(fids, values, marker, color=color, markersize=8,
    #                         label=f'{label_prefix} ({len(valid_indices)})')

    #     # 建立單一序列圖
    #     plt.figure(figsize=(15, 6))
    #     plt.plot(valid_fids_for_peaks, input_series_for_peaks, 'b-', linewidth=1,
    #             label='Ankle Y Difference (Left - Right)')
        
    #     # 繪製峰值和波谷
    #     _plot_extremes(peak_indices, "^", 'red', 'Peaks (Left higher)')
    #     _plot_extremes(valley_indices, "v", 'green', 'Valleys (Right higher)')
        
    #     plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    #     plt.xlabel("Frame ID")
    #     plt.ylabel("Ankle Y Difference")
    #     plt.title(f"Track {track.track_id}: Ankle Alternation Pattern")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
        
    #     # 儲存圖表
    #     if output_path:
    #         try:
    #             output_path = Path(output_path)
    #             output_path.parent.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #             logging.info(f"Track {track.track_id}: Ankle difference plot saved to {output_path}")
    #         except Exception as e:
    #             logging.error(f"Track {track.track_id}: Failed to save plot to {output_path}: {e}")
        
    #     return plt.gcf()