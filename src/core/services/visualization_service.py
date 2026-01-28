import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from ..infrastructure.video_source import VideoSource
from src.pose_extract.track_solution import TrackManager
from src.pose_extract.track_solution.visualization import TrackVisualizer
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult

# 導入新的 DTO 模型
from ..models import VideoInfo, ensure_video_info

logger = logging.getLogger(__name__)

class SkeletonVisualizationService:
    """
    負責將骨架分析結果可視化為影片片段的服務。
    職責：
    1. 讀取原始影片
    2.根據 AnalysisResult 中的分段資訊，裁切並繪製對應片段
    3. 輸出影片檔案
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 視覺化配置，例如:
                {
                    "output_dir": "outputs/visualizations",
                    "draw_options": {
                        "show_interpolated": True,
                        "normalized": False,
                        "line_thickness": 2,
                        "keypoint_radius": 4
                    },
                    "video_writer": {
                         "fps": 30,
                         "codec": "mp4v"
                    }
                }
        """
        self.config = config
        self.output_base_dir = Path(config.get("output_dir", "../outputs/visualizations"))
        self.visualizer = TrackVisualizer() # 這裡初始化底層繪圖器

    def visualize_analysis_segments(
        self,
        track_manager: TrackManager,
        analysis_results: Dict[int, AnalysisResult],
        video_info: Union[VideoInfo, Dict[str, Any]],
        target_segment_type: str = None
    ) -> List[str]:
        """
        主要入口：根據分析結果生成可視化影片片段。

        Args:
            track_manager: 包含軌跡資料的管理器
            analysis_results: 分析結果字典 {track_id: AnalysisResult}
            video_info: 原始影片資訊（支援 VideoInfo 或舊格式 dict）
            target_segment_type: 指定要匯出的分段類型 (例如 "walking", "crawling")。
                                 若為 None，則處理所有類型 (視需求而定，目前建議指定)。

        Returns:
            List[str]: 生成的影片檔案路徑列表
        """
        # 支援 VideoInfo 和 dict 兩種輸入
        video_info_obj = ensure_video_info(video_info) if isinstance(video_info, dict) else video_info
        
        video_path = video_info_obj.video_path
        if not video_path.exists():
            logger.error(f"找不到原始影片: {video_path}")
            return []
        
        # 準備輸出目錄
        video_stem = video_path.stem
        output_dir = self.output_base_dir / video_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 1. 收集所有需要處理的任務 (Segment Task)
        # 格式: (track_id, segment_type, segment_index, start_frame, end_frame)
        tasks = []
        
        # 確定要處理的分段類型
        # 如果 caller 沒指定，我們可以從 config 拿，或者遍歷 result 中所有的 keys
        # 這裡簡單起見，若 target_segment_type 為 None，則不執行 (還是說要全部? 依據 User Request "如果有給予segment_type目標")
        # User: "如果有給予segment_type目標 其會根據analysis_results 的有效片段繪製"
        
        types_to_process = [target_segment_type] if target_segment_type else []
        # 如果需要支援 "全部"，可以遍歷 analysis_results[tid].segments.keys()
        
        for track_id, result in analysis_results.items():
            current_types = types_to_process if types_to_process else result.segments.keys()
            
            for seg_type in current_types:
                segments = result.get_segments(seg_type)
                if not segments:
                    continue
                
                for idx, (start, end) in enumerate(segments):
                    tasks.append({
                        "track_id": track_id,
                        "segment_type": seg_type,
                        "segment_index": idx,
                        "start_frame": start,
                        "end_frame": end
                    })

        if not tasks:
            logger.info(f"沒有找到符合類型 {target_segment_type} 的分段需要視覺化。")
            return []

        logger.info(f"準備生成 {len(tasks)} 個視覺化片段...")

        # 2. 為了效率，我們可以依據 'start_frame' 排序任務，這樣如果我們共享 VideoSource 可以減少 seek (但這裡我們可能每次生成一個獨立檔案)
        # 考慮到可能要生成獨立檔案，這裡逐一處理最簡單。
        
        # 注意：由於使用 FFmpeg Pipe，VideoSource 不支援 seek
        # 因此每個片段都需要重新創建 VideoSource

        fps = video_info_obj.fps or 30

        # 繪圖選項
        draw_options = self.config.get("draw_options", {})
        # 確保顯示指定的 segment type 狀態
        if target_segment_type:
             # 如果尚未設定，將 target_segment_type 加入顯示列表
            current_types = draw_options.get('segment_types', [])
            if target_segment_type not in current_types:
                 # 複製一份以免修改到 self.config
                 draw_options = draw_options.copy()
                 draw_options['segment_types'] = current_types + [target_segment_type]

        for task in tasks:
            t_id = task["track_id"]
            s_type = task["segment_type"]
            s_idx = task["segment_index"]
            start_f = task["start_frame"]
            end_f = task["end_frame"]

            # 定義輸出檔名
            # User 要求: 內容包含 骨架 bondingbox track id
            out_name = f"{video_stem}_T{t_id}_{s_type}_{s_idx:03d}_F{start_f}-{end_f}.mp4"
            out_path = output_dir / out_name
            
            if out_path.exists() and not self.config.get("overwrite", True):
                logger.info(f"檔案已存在跳過: {out_name}")
                continue

            # 每個片段創建新的 VideoSource（FFmpeg Pipe 不支援 seek）
            try:
                video_source = VideoSource(str(video_path))
                video_source.open()
                
                width = video_info.get('width', int(video_source.target_width))
                height = video_info.get('height', int(video_source.target_height))
            except Exception as e:
                logger.error(f"無法開啟影片資源: {e}")
                continue

            success = self._render_clip(
                video_source=video_source,
                track_manager=track_manager,
                track_id=t_id,
                start_frame=start_f,
                end_frame=end_f,
                output_path=str(out_path),
                fps=fps,
                width=width,
                height=height,
                draw_options=draw_options
            )
            
            # 釋放 VideoSource
            video_source.release()
            
            if success:
                generated_files.append(str(out_path))
        logger.info(f"視覺化完成，共生成 {len(generated_files)} 個檔案。")
        return generated_files

    def _render_clip(
        self,
        video_source: VideoSource,
        track_manager: TrackManager,
        track_id: int,
        start_frame: int,
        end_frame: int,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        draw_options: Dict[str, Any]
    ) -> bool:
        """
        生成單一影片片段
        """
        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*self.config.get("video_writer", {}).get("codec", "mp4v"))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"無法建立 VideoWriter: {output_path}")
            return False

        try:
            # 設定讀取位置
            # 注意: VideoSource.set_frame 是 0-indexed
            video_source.set_frame(start_frame)
            
            # 準備 TrackVisualizer 需要的 options
            # 我們只想畫特定的 track_id
            viz_options = draw_options.copy()
            viz_options["draw_track_ids"] = {track_id} 
            
            current_frame_idx = start_frame
            while current_frame_idx <= end_frame:
                ret, frame = video_source.read()
                if not ret:
                    logger.warning(f"在幀 {current_frame_idx} 讀取失敗或是影片結束")
                    break
                
                # 從 TrackManager 獲取當前幀的 active tracks
                # 注意: 因為我們只關注特定 track_id，所以即使 get_tracks_by_frame 回傳多個，visualizer 也會過濾
                # 但為了效能，我們可以只傳入我們關心的 track instance
                
                track = track_manager.get_track(track_id)
                if not track:
                     logger.warning(f"Track {track_id} 不存在")
                     break

                # 呼叫 TrackVisualizer 繪製
                # draw_tracks 接受 tracks list
                frame_viz = self.visualizer.draw_tracks(
                    frame=frame,
                    tracks=[track], # 只傳入該 track
                    frame_id=current_frame_idx,
                    options=viz_options
                )
                
                out.write(frame_viz)
                current_frame_idx += 1
                
        except Exception as e:
            logger.error(f"生成片段失敗 {output_path}: {e}", exc_info=True)
            return False
        finally:
            out.release()
            
        return True
