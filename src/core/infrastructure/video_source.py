"""
VideoSource - 影片讀取來源

使用 FFmpeg CLI + subprocess pipe 技術，
直接將處理後的 raw frames 傳給 Python，無需寫入硬碟。
"""

import subprocess
import shutil
import logging
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoSource:
    """
    FFmpeg Pipe Video Source
    
    使用 FFmpeg CLI 處理影片（fps 轉換、縮放、黑邊填充），
    並透過 pipe 直接輸出 raw frames 給 Python。
    
    特點：
    1. 零磁碟 I/O - 不寫入轉碼檔案
    2. 零重編碼 - 直接輸出 raw pixels
    3. 自動旋轉 - FFmpeg 會根據 metadata 自動處理
    """
    
    def __init__(
        self,
        source_path: Optional[str] = None,
        source_type: str = "file",
        target_fps: Optional[int] = 30,
        input_size: Optional[Tuple[int, int]] = (1080, 1920)
    ):
        """
        Args:
            source_path: 影片路徑
            source_type: 來源類型 ("file" or "camera")
            target_fps: 目標幀率 (e.g., 30)
            input_size: 目標尺寸 (width, height) (e.g., 1080, 1920)
        """
        self.source_path = source_path
        self.source_type = source_type
        self.process: Optional[subprocess.Popen] = None
        
        self.frame_count = 0
        self.origin_width = 0
        self.origin_height = 0
        self.original_fps = 0.0
        self.target_fps = target_fps or 30
        self.input_size = input_size
        self.target_width = input_size[0] if input_size else 1080
        self.target_height = input_size[1] if input_size else 1920
        
        self.frame_step = 1  # 由 FFmpeg 處理 fps 轉換
        self.current_frame_idx = 0
        self.effective_frame_count = 0
        
        # 每幀的 bytes 數 (BGR24 格式)
        self._frame_size = self.target_width * self.target_height * 3
        
        # 找到 FFmpeg 路徑
        self._ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"

    def _get_video_info(self) -> bool:
        """使用 ffprobe 獲取影片資訊"""
        try:
            ffprobe_path = shutil.which("ffprobe") or "ffprobe"
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of", "csv=p=0",
                str(self.source_path)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    self.origin_width = int(parts[0])
                    self.origin_height = int(parts[1])
                    
                    # 解析 fps (格式如 "30/1" 或 "30000/1001")
                    fps_parts = parts[2].split('/')
                    if len(fps_parts) == 2:
                        self.original_fps = float(fps_parts[0]) / float(fps_parts[1])
                    else:
                        self.original_fps = float(fps_parts[0])
                    
                    # 獲取總幀數
                    if parts[3] and parts[3] != 'N/A':
                        self.frame_count = int(parts[3])
                    elif len(parts) >= 5 and parts[4]:
                        # 如果沒有 nb_frames，用 duration * fps 估算
                        duration = float(parts[4])
                        self.frame_count = int(duration * self.original_fps)
                    
                    # 計算有效幀數（轉換到目標 FPS 後）
                    if self.original_fps > 0 and self.target_fps > 0:
                        self.effective_frame_count = int(
                            self.frame_count * self.target_fps / self.original_fps
                        )
                    else:
                        self.effective_frame_count = self.frame_count
                    
                    return True
            
            logger.warning(f"無法獲取影片資訊: {result.stderr}")
            return False
            
        except Exception as e:
            logger.error(f"ffprobe 執行失敗: {e}")
            return False

    def open(self) -> bool:
        """啟動 FFmpeg pipe 進程"""
        try:
            logger.info(f"FFmpeg Pipe 模式: target_fps={self.target_fps}, target_size={self.target_width}x{self.target_height}")
            
            # 組建 video filter
            vf = (
                f"fps={self.target_fps},"
                f"scale={self.target_width}:{self.target_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.target_width}:{self.target_height}:"
                f"(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1"
            )
            
            # FFmpeg 命令 - 輸出 raw video 到 pipe
            cmd = [
                self._ffmpeg_path,
                "-i", str(self.source_path),
                "-vf", vf,
                "-f", "rawvideo",      # 輸出原始像素
                "-pix_fmt", "bgr24",   # BGR 格式（OpenCV/numpy 標準）
                "-an",                  # 無音訊
                "pipe:1"               # 輸出到 stdout
            ]
            
            logger.debug(f"FFmpeg Pipe 命令: {' '.join(cmd)}")
            
            # 啟動 FFmpeg 進程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # 忽略 stderr 以避免阻塞
                bufsize=self._frame_size * 10  # 緩衝區大小
            )
            
            return True
            
        except Exception as e:
            logger.error(f"啟動 FFmpeg Pipe 失敗: {e}")
            return False

    def get_frame_index(self, logical_frame_id: int) -> int:
        """將邏輯幀ID轉換為實際視頻中的幀索引"""
        return logical_frame_id * self.frame_step

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        從 FFmpeg pipe 讀取一幀
        
        Returns:
            ret (bool): 是否成功
            img (numpy array): BGR 格式的標準化圖片 (target_width x target_height)
        """
        if self.process is None or self.process.stdout is None:
            return False, None

        try:
            # 從 pipe 讀取一幀的 raw bytes
            raw_data = self.process.stdout.read(self._frame_size)
            
            if len(raw_data) != self._frame_size:
                # 資料不完整，可能是影片結束
                return False, None
            
            # 轉換為 numpy array，使用 copy() 讓 array 可寫入
            # （np.frombuffer 產生的 array 預設是 read-only）
            frame = np.frombuffer(raw_data, dtype=np.uint8).copy()
            frame = frame.reshape((self.target_height, self.target_width, 3))
            
            self.current_frame_idx += 1
            return True, frame
            
        except Exception as e:
            logger.error(f"讀取幀時出錯: {e}")
            return False, None

    def set_frame(self, logical_frame_id: int):
        """
        設置到指定邏輯幀
        
        注意：FFmpeg pipe 模式下，seek 需要重啟進程
        """
        if logical_frame_id == 0:
            # 如果是回到開頭，重新啟動進程
            self.release()
            self.open()
            self.current_frame_idx = 0
        else:
            # 對於其他位置，跳過幀直到到達目標
            while self.current_frame_idx < logical_frame_id:
                ret, _ = self.read()
                if not ret:
                    break

    def release(self):
        """釋放資源"""
        if self.process is not None:
            try:
                self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None

    def __enter__(self):
        """Context manager 支援"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 支援"""
        self.release()
        return False