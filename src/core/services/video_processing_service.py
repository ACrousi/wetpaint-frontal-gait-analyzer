import logging
import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional, Any, Union
import numpy as np

from ..infrastructure.video_source import VideoSource
from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution.analysis.base import SegmentType, InvalidSegmentTypeError
from src.pose_extract.wrapper import RTMOWrapper
from src.pose_extract.wrapper import BoTSORTWrapper

# 導入例外類別
from src.exceptions import VideoReadError, TrackingError, SkeletonDataError

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
        logger.info(f"開始提取骨架: {filename}")

        # 初始化Tracker
        self.tracker = BoTSORTWrapper(self.config.get("BoTSORT", {}))
        track_manager = TrackManager()

        # 執行追蹤
        logger.info(f"開始執行即時追蹤: {filename}")
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

    # def run_tracking(self, track_manager: TrackManager, video_info: Dict[str, Any]):
    #     """執行影片追蹤"""
    #     video_source = VideoSource(str(Path(video_info['video_path'])), target_fps=video_info.get('target_fps', 30))
    #     video_source.open()

    #     video_info['width'] = video_source.target_width
    #     video_info['height'] = video_source.target_height
    #     video_info['effective_frame_count'] = video_source.effective_frame_count
    #     video_info['fps'] = video_source.target_fps

    #     # 設定幀範圍
    #     start_frame = 0
    #     end_frame = video_source.effective_frame_count
    #     video_source.set_frame(start_frame)

    #     # 執行影片處理循環
    #     frame_id = start_frame
    #     while frame_id < end_frame:
    #         ret, frame = video_source.read()
    #         if not ret:
    #             logger.info(f"影片讀取失敗或結束於幀 {frame_id}")
    #             break

    #         # 執行檢測和追蹤
    #         detections = self.pose_model(frame)
    #         bonding_boxes, keypoints, boxes_scores, keypoint_scores = detections
    #         tracks_output = self.tracker.update(frame, bonding_boxes, boxes_scores, keypoints, keypoint_scores)

    #         # 更新 TrackManager
    #         track_manager.update_from_tracker(tracks_output, frame_id)

    #         frame_id += 1
    #         if frame_id % 100 == 0:
    #             logger.info(f"已處理幀 {frame_id}/{end_frame}")
    
    def check_skeleton_data_exists(self, video_filename: str, data_dir: str = None) -> tuple:
        """
        檢查指定影片是否已有骨架資料（JSON 或 pickle 格式）
        
        Args:
            video_filename: 影片檔名
            data_dir: 資料目錄路徑，如果為 None 則使用預設路徑
            
        Returns:
            (bool, str, dict): (是否存在, 檔案路徑, 資料內容或None)
        """
        if data_dir is None:
            data_dir = "../outputs/coco_format"
        
        base_name = Path(video_filename).stem
        data_dir_path = Path(data_dir)
        
        # 檢查 JSON 檔案
        json_path = data_dir_path / f"{base_name}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_skeleton_data(data):
                        return True, str(json_path), data
            except Exception as e:
                logging.warning(f"無法讀取 JSON 檔案 {json_path}: {e}")
        
        # 檢查 pickle 檔案
        pickle_path = data_dir_path / f"{base_name}.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    if self._validate_skeleton_data(data):
                        return True, str(pickle_path), data
            except Exception as e:
                logging.warning(f"無法讀取 pickle 檔案 {pickle_path}: {e}")
        
        return False, None, None
    
    def _validate_skeleton_data(self, data: dict) -> bool:
        """
        驗證骨架資料是否完整
        
        Args:
            data: 要驗證的資料字典
            
        Returns:
            bool: 資料是否有效
        """
        if not isinstance(data, dict):
            return False
        
        # 檢查基本結構
        if 'tracks' not in data:
            return False
        
        tracks = data['tracks']
        if not isinstance(tracks, list) or len(tracks) == 0:
            return False
        
        # 檢查至少一個軌跡有關鍵點資料
        for track in tracks:
            if 'keypoints' in track and track['keypoints']:
                # 進一步檢查是否有額外的資料欄位（向後兼容）
                has_basic_data = True
                
                # 如果有 bounding_boxes 或 scores，則檢查其格式
                if 'bounding_boxes' in track:
                    if not isinstance(track['bounding_boxes'], dict):
                        logging.warning(f"Track {track.get('track_id', 'unknown')} 的 bounding_boxes 格式不正確")
                        has_basic_data = False
                
                if 'scores' in track:
                    if not isinstance(track['scores'], dict):
                        logging.warning(f"Track {track.get('track_id', 'unknown')} 的 scores 格式不正確")
                        has_basic_data = False
                
                if 'keypoint_scores' in track:
                    if not isinstance(track['keypoint_scores'], dict):
                        logging.warning(f"Track {track.get('track_id', 'unknown')} 的 keypoint_scores 格式不正確")
                        has_basic_data = False
                
                if has_basic_data:
                    return True
        
        return False
        
    # def run_tracking(self, track_manager: TrackManager, video_info: Dict[str, Any]):
    #     """執行影片追蹤"""
    #     video_source = VideoSource(str(Path(video_info['video_path'])), target_fps=video_info.get('target_fps', 30))
    #     video_source.open()

    #     video_info['width'] = video_source.target_width
    #     video_info['height'] = video_source.target_height
    #     video_info['effective_frame_count'] = video_source.effective_frame_count
    #     video_info['fps'] = video_source.target_fps
        
    #     # 設定幀範圍
    #     start_frame = 0
    #     end_frame = video_source.effective_frame_count
    #     video_source.set_frame(start_frame)

    #     # 執行影片處理循環
    #     frame_id = start_frame
    #     while frame_id < end_frame:
    #         ret, frame = video_source.read()
    #         if not ret:
    #             logger.info(f"影片讀取失敗或結束於幀 {frame_id}")
    #             break

    #         # 執行檢測和追蹤
    #         detections = self.pose_model(frame)
    #         bonding_boxes, keypoints, boxes_scores, keypoint_scores = detections
    #         tracks_output = self.tracker.update(frame, bonding_boxes, boxes_scores, keypoints, keypoint_scores)
            
    #         # 更新 TrackManager
    #         track_manager.update_from_tracker(tracks_output, frame_id)

    #         frame_id += 1
    #         if frame_id % 100 == 0:
    #             logger.info(f"已處理幀 {frame_id}/{end_frame}")
        
##############################################################

    def _run_post_processing(self, track_manager: TrackManager, video_info: Dict[str, Any], save=False):
        """執行後續處理，如分析和導出"""
        filename = Path(video_info['video_path']).name
        logger.info(f"開始對 {filename} 進行後續處理")

        # 執行分析
        self._run_analysis(track_manager, video_info)
        
        # 導出結果
        if save:
            self._export_results(track_manager, video_info)

    def _run_analysis(self, track_manager: TrackManager, video_info: Dict[str, Any]):
        """執行所有分析步驟"""
        filename = Path(video_info['video_path']).name
        logger.info(f"開始分析處理：{filename}")

        # 使用新的統一分析流程（階段一 + 階段二）
        analysis_results = self.analysis_service.process(track_manager)

        # 將分析結果存儲到 video_info 中，以便後續使用
        video_info['analysis_results'] = analysis_results

        logger.info(f"分析處理完成：{filename}，共分析了 {len(analysis_results)} 個軌跡")
        
    def _export_results(self, track_manager: TrackManager, video_info: Dict[str, Any]):
        """導出所有結果"""
        filename = Path(video_info['video_path']).name
        logger.info(f"開始導出結果：{filename}")
        
        # 檢查是否還有活躍軌跡
        active_tracks = track_manager.get_all_tracks(removed=False)
        if not active_tracks:
            logger.info(f"沒有活躍軌跡，跳過導出：{filename}")
            return
            
        # 導出 COCO 格式
        # self._export_coco_format(track_manager, video_info)
        
        # 導出 CSV
        self._export_csv(track_manager, video_info)
        
        # 創建視覺化
        # self._create_visualizations(track_manager, video_info)
        
        logger.info(f"結果導出完成：{filename}")
        
    # def _export_coco_format(self, track_manager: TrackManager, video_info: Dict[str, Any]):
    #     """導出 COCO 格式"""
    #     coco_config = self.config.get("export", {}).get("coco", {})
    #     if not coco_config.get("enabled", True):
    #         return
            
    #     coco_output_dir = self.output_base / "coco"
    #     coco_output_dir.mkdir(parents=True, exist_ok=True)
        
    #     saved_path = self.export_service.export_to_coco(
    #         track_manager,
    #         video_info
    #     )
    #     logger.info(f"已儲存 COCO 格式資料：{saved_path}")
        
    def _export_csv(self, track_manager: TrackManager, video_info: Dict[str, Any]):
        """導出 CSV 格式（逐軌跡、逐片段）"""
        csv_config = self.config.get("export", {}).get("csv", {})
        if not csv_config.get("enabled", True):
            return

        csv_output_dir = self.output_base / "csv"
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 從設定或預設決定要的 segment 類型
        segment_type_cfg = csv_config.get("segment_type", "combined")
        try:
            if isinstance(segment_type_cfg, str):
                segment_type_filter = SegmentType.from_string(segment_type_cfg)
            else:
                segment_type_filter = segment_type_cfg
        except InvalidSegmentTypeError as e:
            logger.warning(f"{e} Using default 'combined'")
            segment_type_filter = SegmentType.COMBINED

        # 取得所有有效軌跡
        valid_tracks = track_manager.get_all_tracks(removed=False)
        if not valid_tracks:
            logging.info("沒有找到任何有效的軌跡，跳過 CSV 匯出。")
            return

        total_segments_exported = 0
        # 從 video_info 取得基本名稱（支援多種來源）
        if video_info.get('video_name'):
            base_name = Path(video_info['video_name']).stem
        elif video_info.get('video_path'):
            base_name = Path(video_info['video_path']).stem
        elif video_info.get('original_video'):
            base_name = Path(video_info['original_video']).stem
        else:
            base_name = 'video'
        for track in valid_tracks:
            # 取得該 track 的所有符合條件的 segment 資料（包含 frames 與 start/end/index）
            all_segment_data = track_manager.get_track_data_for_segment(
                track_id=track.track_id,
                segment_type=segment_type_filter,
                use_normalized_keypoints=csv_config.get("use_normalized_keypoints", True)
            )

            if not all_segment_data:
                continue

            for segment_info in all_segment_data:
                segment_data = segment_info.get("frames")
                start_frame = segment_info.get("start_frame")
                end_frame = segment_info.get("end_frame")
                segment_index = segment_info.get("segment_index")

                if not segment_data:
                    logging.warning(f"Track {track.track_id} 的 segment {segment_index} 沒有骨架資料，跳過。")
                    continue

                segment_type_str = segment_type_filter.value if isinstance(segment_type_filter, SegmentType) else str(segment_type_filter)
                processing_flag = "norm"  # Example，可改為從 config 決定

                filename = (
                    f"{base_name}_track{track.track_id}_{segment_type_str}_"
                    f"seg{segment_index}_{start_frame}-{end_frame}_{processing_flag}"
                )

                # 呼叫導出服務匯出單一 segment（由 ExportService 負責寫入檔案）
                self.export_service.export_segment_to_json_aos(
                    data=segment_data,
                    output_dir=str(csv_output_dir),
                    base_filename=filename,
                    start_frame=start_frame,
                    track_id=track.track_id,
                    segment_index=segment_index,
                    use_normalized=csv_config.get("use_normalized_keypoints", True)
                )
                total_segments_exported += 1

        logging.info(f"CSV 匯出完成，共匯出 {total_segments_exported} 個片段。")
        
    def _create_visualizations(self, track_manager: TrackManager, video_info: Dict[str, Any]):
        """創建視覺化"""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("enabled", True):
            return

        video_path = video_info['video_path']
        
        # 創建視頻
        video_config = viz_config.get("video", {})
        if video_config.get("enabled", True):
            video_output_dir = self.output_base / "videos"
            video_output_dir.mkdir(parents=True, exist_ok=True)
            self.visualization_service.create_track_video(
                track_manager=track_manager,
                video_path=video_path,
                output_path=str(video_output_dir / f"{Path(video_path).stem}_tracked.mp4"),
            )
            
        # 創建圖表
        plot_config = viz_config.get("plots", {})
        if plot_config.get("enabled", True):
            plot_output_dir = self.output_base / "plots"
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            self.visualization_service.create_plots(
                track_manager=track_manager,
                output_dir=str(plot_output_dir),
                base_filename=Path(video_path).stem
            )

    # def _run_resgcn_prediction(self, track_manager: TrackManager, video_info: Dict[str, Any]):
    #     """對所有有效軌跡的所有片段執行 ResGCN 動作識別，回傳每個片段的 predicted_class 列表"""
    #     logger.info("開始執行 ResGCN 動作識別")
 
    #     predictions: List[int] = []  # 儲存每個片段的 predicted_class（int）
 
    #     # 取得所有有效軌跡
    #     valid_tracks = track_manager.get_all_tracks(removed=False)
    #     if not valid_tracks:
    #         logger.info("沒有有效軌跡，跳過 ResGCN 預測")
    #         return predictions
 
    #     video_name = video_info.get('video_name', Path(video_info.get('video_path', 'unknown')).name)
        
    #     # 遍歷所有有效軌跡
    #     for track in valid_tracks:
    #         # 取得該 track 的所有 COMBINED segment 資料
    #         segment_data_list = track_manager.get_track_data_for_segment(
    #             track_id=track.track_id,
    #             segment_type=SegmentType.COMBINED,
    #             use_normalized_keypoints=True  # 根據 ResGCN 的需求調整
    #         )
 
    #         if not segment_data_list:
    #             logger.debug(f"Track {track.track_id} 沒有 COMBINED segment 資料，跳過預測")
    #             continue
 
    #         # 遍歷該軌跡的所有 segment
    #         for segment_info in segment_data_list:
    #             skeleton_data = segment_info.get("frames")  # (T, V, C)
    #             segment_index = segment_info.get("segment_index")
 
    #             if skeleton_data is None or skeleton_data.shape[0] == 0:
    #                 logger.warning(f"Track {track.track_id} 的 segment {segment_index} 資料為空，跳過預測")
    #                 continue
 
    #             # 轉換資料格式 (T, V, C) -> (C, T, V, M)，此處 M=1（單人）
    #             data = np.transpose(skeleton_data, (2, 0, 1))  # -> (C, T, V)
    #             data = np.expand_dims(data, axis=-1)          # -> (C, T, V, M)
 
    #             # 執行預測
    #             try:
    #                 prediction = self.resgcn_model.predict(data)
    #                 predicted_class = int(np.argmax(prediction))
    #                 predictions.append(predicted_class)
    #                 logger.info(
    #                     f"ResGCN 預測結果 - 影片: {video_name}, "
    #                     f"軌跡: {track.track_id}, 片段: {segment_index}, 預測類別: {predicted_class}"
    #                 )
    #             except Exception as e:
    #                 logger.error(
    #                     f"執行 ResGCN 預測時發生錯誤 (軌跡: {track.track_id}, 片段: {segment_index}): {e}",
    #                     exc_info=True
    #                 )
 
    #     return predictions