import logging
import json
import pickle
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal, Tuple, Set
from datetime import datetime
from collections import defaultdict

from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution import TrackRecord
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult
from src.pose_extract.track_solution.analysis.base import SegmentType, InvalidSegmentTypeError

# 導入新的 DTO 模型
from ..models import VideoInfo, ensure_video_info

logger = logging.getLogger(__name__)


class ExportService:
    """導出服務
    
    負責將分析結果導出為各種格式，包括：
    1. COCO 格式（JSON/Pickle）
    2. CSV 格式
    3. 其他自定義格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def export_to_raw_skeleton(self, track_manager: TrackManager, video_info: Union[VideoInfo, Dict[str, Any]]) -> str:
        """將 TrackManager 的資料轉換為原始骨架格式並儲存"""
        # 支援 VideoInfo 和 dict 兩種輸入
        video_info_obj = ensure_video_info(video_info) if isinstance(video_info, dict) else video_info
        video_filename = video_info_obj.video_name
        logger.info(f"轉換為原始骨架格式: {video_filename}")

        raw_skeleton_data = {
            "info": {
                "video_filename": video_filename,
                "creation_date": datetime.now().isoformat(),
                "total_tracks": len(track_manager.get_all_tracks(removed=False)),
                "total_analyze_frame": video_info_obj.effective_frame_count,
                "target_fps": video_info_obj.fps
            },
            "tracks": []
        }
        
        active_tracks = track_manager.get_all_tracks(removed=False)
        
        for track in active_tracks:
            track_data = self.convert_track_to_coco_format(track)
            raw_skeleton_data["tracks"].append(track_data)

        logger.info(f"原始骨架格式轉換完成，軌跡數: {len(active_tracks)}")

        # 儲存檔案
        raw_skeleton_config = self.config.get("raw_skeleton", {})
        output_dir = Path(raw_skeleton_config.get("output_dir", "outputs/raw_skeleton"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(video_filename).stem}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_skeleton_data, f, ensure_ascii=False, indent=2)

        logger.info(f"原始骨架資料已儲存至JSON: {output_path}")
        return str(output_path)
        
    def convert_track_to_coco_format(self, track: TrackRecord) -> Dict[str, Any]:
        """將單個軌跡轉換為 COCO 格式"""
        track_data = {
            "track_id": track.track_id,
            "keypoints": {},
            "bounding_boxes": {},
            "scores": {},
            "keypoint_scores": {},
            "metadata": {
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "duration": track.last_frame - track.first_frame + 1 if track.last_frame and track.first_frame else 0
            }
        }
        
        # 轉換 keypoints 格式
        for frame_id, keypoints in track.keypoints.items():
            if keypoints is not None:
                track_data["keypoints"][str(frame_id)] = keypoints.tolist()
                
        # 轉換 bounding_boxes 格式
        for frame_id, bbox in track.positions.items():
            if bbox is not None:
                track_data["bounding_boxes"][str(frame_id)] = list(bbox)
                
        # 轉換 scores 格式
        for frame_id, score in track.scores.items():
            if score is not None:
                track_data["scores"][str(frame_id)] = float(score)
                
        # 轉換 keypoint_scores 格式
        for frame_id, kp_scores in track.keypoint_scores.items():
            if kp_scores is not None:
                track_data["keypoint_scores"][str(frame_id)] = kp_scores.tolist()
                
        return track_data
        
    def save_coco_data(self, data: Dict[str, Any], output_path: str, 
                      format_type: str = 'both') -> List[str]:
        """儲存 COCO 格式資料"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        if format_type in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved_paths.append(str(json_path))
            logger.info(f"儲存 JSON 格式: {json_path}")
            
        if format_type in ['pickle', 'both']:
            pickle_path = output_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            saved_paths.append(str(pickle_path))
            logger.info(f"儲存 pickle 格式: {pickle_path}")
            
        return saved_paths

    def get_all_tracks_segment_data(
        self,
        track_manager: TrackManager,
        analysis_results: Dict[int, AnalysisResult],
        segment_type: Union[SegmentType, str],
        use_normalized_keypoints: bool = True,
        removed: bool = False,
        output_format: Literal["aos", "soa"] = "aos"
    ) -> List[Dict[str, Any]]:
        """
        批量獲取所有軌跡的片段資料

        Args:
            track_manager: TrackManager 實例
            analysis_results: 分析結果字典 Dict[track_id, AnalysisResult]
            segment_type: 片段類型（SegmentType 或字串）
            use_normalized_keypoints: 是否使用正規化關鍵點
            removed: 是否包含已移除的軌跡
            output_format: "aos" (Array of Structures) 或 "soa" (Structure of Arrays)

        Returns:
            List[Dict]: 每個元素包含:
                {
                    "track_id": int,
                    "track": TrackRecord,
                    "segments": List[{
                        "segment_index": int,
                        "start_frame": int,
                        "end_frame": int,
                        "frames": Any   # 依 output_format：aos 為 list[dict]；soa 為 np.ndarray (T, V, C)
                    }]
                }
        """
        # 正規化 segment_type
        segment_type_key = segment_type.value if isinstance(segment_type, SegmentType) else str(segment_type)
        if isinstance(segment_type, str):
            try:
                segment_type = SegmentType.from_string(segment_type)
            except InvalidSegmentTypeError as e:
                logger.warning(f"{e} Skipping due to invalid segment type")
                return []

        valid_tracks = track_manager.get_all_tracks(removed=removed)
        if not valid_tracks:
            return []

        results: List[Dict[str, Any]] = []

        for track in valid_tracks:
            try:
                track_id = track.track_id
                logger.debug(f"收集軌跡 {track_id} 的片段資料")

                # 從分析結果取得該軌跡所有片段範圍
                if track_id not in analysis_results:
                    logger.debug(f"軌跡 {track_id} 無分析結果")
                    continue
                    
                segments: List[Tuple[int, int]] = analysis_results[track_id].get_segments(segment_type_key)
                if not segments:
                    logger.debug(f"軌跡 {track_id} 無 {segment_type.value} 片段")
                    continue

                segment_data_list: List[Dict[str, Any]] = []
                for seg_idx, (start_frame, end_frame) in enumerate(segments):
                    # 取得該 segment 的幀資料
                    frames_data = track_manager.get_frames_data(
                        track_id=track_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        use_normalized_keypoints=use_normalized_keypoints,
                        output_format=output_format
                    )

                    if frames_data is None:
                        continue

                    # 打包片段資料
                    segment_entry = {
                        "segment_index": seg_idx,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frames": frames_data
                    }
                    segment_data_list.append(segment_entry)

                if segment_data_list:
                    results.append({
                        "track_id": track_id,
                        "track": track,
                        "segments": segment_data_list
                    })
            except Exception as e:
                logger.error(f"收集軌跡 {getattr(track, 'track_id', 'unknown')} 片段資料失敗: {e}", exc_info=True)
                continue

        return results
    def export_segments_by_type(self, complete_data_list: List[Dict[str, Any]], video_info: Union[VideoInfo, Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        根據完整資料結構導出所有 segments 為 SOA JSON 格式

        Args:
            complete_data_list: 包含 frames, metadata, features 的完整資料結構列表
            video_info: 影片資訊（支援 VideoInfo 或舊格式 dict）

        Returns:
            一個字典，鍵為 track_id，值為成功匯出 segment 的路徑列表
        """
        seg_skeleton_config = self.config.get("seg_skeleton", {})
        if not seg_skeleton_config.get("enabled", True):
            return {}

        seg_skeleton_output_dir = Path(seg_skeleton_config.get("output_dir", "outputs/json"))
        seg_skeleton_output_dir.mkdir(parents=True, exist_ok=True)

        total_segments_exported = 0
        exported_paths = defaultdict(list)  # 儲存 track_id 到路徑列表的映射

        for complete_data in complete_data_list:
            try:
                metadata = complete_data["metadata"]
                track_id = metadata["track_id"]
                segment_index = metadata["segment_index"]
                case_id = metadata.get("case_id", "unknown")

                # case_id 已經包含日期 (例如 WETP000814_20250926)，直接使用
                base_name = case_id
                filename = f"{base_name}_track{track_id}_combined_seg{segment_index}_{metadata['start_frame']}-{metadata['end_frame']}_norm"

                # 呼叫導出服務匯出單一 segment SOA JSON
                output_path = self.export_segment_to_json_soa(
                    complete_data=complete_data,
                    output_dir=str(seg_skeleton_output_dir),
                    base_filename=filename
                )

                if output_path:  # 只有成功匯出時才計數
                    total_segments_exported += 1
                    exported_paths[track_id].append(output_path)

            except Exception as e:
                logger.error(f"匯出 segment 時發生錯誤: {e}", exc_info=True)
                continue

        # 匯出簡化的 metadata CSV（不含統計特徵）
        if exported_paths:
            exported_track_ids = set(exported_paths.keys())
            self.export_analysis_metadata_csv(complete_data_list, exported_track_ids, exported_paths)

        logger.info(f"SOA JSON 匯出完成，共匯出 {total_segments_exported} 個片段，來自 {len(exported_paths)} 個軌跡。")
        return dict(exported_paths)

    def export_segment_to_json_soa(self, complete_data: Dict[str, Any], output_dir: str, base_filename: str) -> Optional[str]:
        """
        導出單一 segment 為 SOA JSON 格式
        complete_data 包含 frames (SOA), metadata, features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立檔名
        final_filename = f"{base_filename}.json"
        output_path = output_dir / final_filename

        try:
            # 處理 frames，如果是 ndarray 轉為 list
            frames = complete_data["frames"]
            if hasattr(frames, 'tolist'):  # 如果是 numpy array
                frames = frames.tolist()

            # 處理 features，將任何 ndarray 轉為 list 或基本類型
            features = complete_data["features"]
            if isinstance(features, dict):
                processed_features = {}
                for key, value in features.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        processed_features[key] = value.tolist()
                    elif isinstance(value, np.integer):  # numpy int
                        processed_features[key] = int(value)
                    elif isinstance(value, np.floating):  # numpy float
                        processed_features[key] = float(value)
                    else:
                        processed_features[key] = value
                features = processed_features

            # 使用完整的 SOA JSON 格式
            json_data = {
                "frames": frames,  # SOA 格式: [T, V, C]
                "metadata": complete_data["metadata"],
                "features": features
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            metadata = complete_data["metadata"]
            logger.info(f"導出 segment SOA JSON: {output_path} (track={metadata['track_id']}, seg_idx={metadata['segment_index']}, start={metadata['start_frame']})")
            return str(output_path)
        except Exception as e:
            logger.error(f"導出 segment SOA JSON 失敗: {e}", exc_info=True)
            return None

    def export_segment_to_json_aos(self, data: List[Dict[str, Any]], output_dir: str,
                                    base_filename: str, start_frame: int,
                                    track_id: int, segment_index: int,
                                    use_normalized: bool = False) -> Optional[str]:
        """
        導出單一 segment 為 JSON AOS 格式
        data 為由 TrackManager.get_frames_data 回傳的 frames 列表（AOS 格式）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立檔名
        final_filename = f"{base_filename}.json"
        output_path = output_dir / final_filename

        try:
            # 準備 JSON 資料結構
            json_data = {
                "frames": data
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            logger.info(f"導出 segment JSON AOS: {output_path} (track={track_id}, seg_idx={segment_index}, start={start_frame})")
            return str(output_path)
        except Exception as e:
            logger.error(f"導出 segment JSON AOS 失敗: {e}", exc_info=True)
            return None

    def export_analysis_metadata_csv(self, complete_data_list: List[Dict[str, Any]],
                                        active_track_ids: Set[int], exported_paths: Dict[int, List[str]] = None) -> Optional[str]:
        """
        將指定軌跡的簡化 metadata 匯出為 CSV 檔案（不含統計特徵）。

        Args:
            complete_data_list: 包含完整資料的列表，每個元素包含 metadata
            active_track_ids: 只匯出這些 track_id 的 metadata
            exported_paths: 匯出的檔案路徑字典

        Returns:
            成功時回傳輸出檔案路徑，失敗時回傳 None
        """
        if not active_track_ids or not exported_paths or not complete_data_list:
            logger.warning("沒有活躍軌跡、匯出路徑或完整資料可匯出")
            return None

        seg_skeleton_config = self.config.get("seg_skeleton", {})
        if not seg_skeleton_config.get("enabled", True):
            return None

        seg_skeleton_output_dir = Path(seg_skeleton_config.get("output_dir", "outputs/json"))
        seg_skeleton_output_dir.parent.mkdir(parents=True, exist_ok=True)

        seg_skeleton_output_metadata_name = seg_skeleton_config.get("output_metadata_name", "analysis_metadata.csv")
        output_path = seg_skeleton_output_dir / seg_skeleton_output_metadata_name

        # 建立 track_id 到 metadata 的映射
        metadata_by_track_segment = {}
        for complete_data in complete_data_list:
            metadata = complete_data["metadata"]
            track_id = metadata["track_id"]
            segment_index = metadata["segment_index"]
            metadata_by_track_segment[(track_id, segment_index)] = metadata

        # 收集新資料列
        rows = []

        for track_id in active_track_ids:
            if track_id not in exported_paths:
                continue

            # 為每個 segment 創建一行資料
            for segment_idx, exported_path in enumerate(exported_paths[track_id]):
                # 從 metadata 獲取資料
                metadata = metadata_by_track_segment.get((track_id, segment_idx))
                if not metadata:
                    logger.warning(f"找不到 track_id={track_id}, segment_index={segment_idx} 的 metadata")
                    continue

                row_data = dict(metadata)  # 複製所有 metadata
                row_data.update({
                    'keypoint_filename': Path(exported_path).name,
                    'exported_paths': exported_path
                })

                # 嘗試從檔案名稱提取 start_frame 和 end_frame（如果 metadata 中沒有）
                if row_data.get('start_frame') is None or row_data.get('end_frame') is None:
                    filename = Path(exported_path).stem
                    try:
                        # 從檔案名稱如 "WETP000001_20240101_track1_combined_seg0_100-200_norm" 提取
                        parts = filename.split('_')
                        for i, part in enumerate(parts):
                            if part.startswith('seg') and i + 1 < len(parts):
                                frame_range = parts[i + 1]  # "100-200"
                                if '-' in frame_range:
                                    start_frame, end_frame = frame_range.split('-')
                                    row_data['start_frame'] = int(start_frame)
                                    row_data['end_frame'] = int(end_frame)
                                    break
                    except (ValueError, IndexError):
                        logger.warning(f"無法從檔案名稱提取幀範圍: {filename}")

                rows.append(row_data)

        if not rows:
            logger.warning("沒有有效的資料列可匯出")
            return None

        try:
            # 將新資料轉為 DataFrame
            new_df = pd.DataFrame(rows)

            # 如果檔案已存在，先讀取現有資料
            if output_path.exists():
                existing_df = pd.read_csv(output_path, encoding='utf-8-sig')
                # 合併現有和新資料
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            # 存檔
            combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            logger.info(f"簡化 metadata CSV 已匯出至: {output_path} (共 {len(combined_df)} 筆記錄)")
            return str(output_path)
        except Exception as e:
            logger.error(f"匯出簡化 metadata CSV 失敗: {e}", exc_info=True)
            return None
            
    def export_case_segment_summary(self, case_id: str, df: pd.DataFrame, output_dir: Optional[str] = None) -> Optional[str]:
        """
        導出個案級 SegmentSummary 統計到 CSV。

        Args:
            case_id: 個案 ID
            df: 統計 DataFrame
            output_dir: 輸出目錄，預設 "outputs"

        Returns:
            輸出檔案路徑，若無資料則返回 None
        """
        if df is None or df.empty:
            logger.info(f"個案 {case_id} 無統計資料，跳過導出")
            return None

        out_dir = Path(output_dir or "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{case_id}_segment_summary.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"個案 {case_id} SegmentSummary 已導出到: {out_path}")
        return str(out_path)
        