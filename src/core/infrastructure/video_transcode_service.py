"""
VideoTranscodeService - 影片轉碼服務

負責將任意格式的影片轉碼為標準化的 H.264 MP4 格式，
並自動處理旋轉校正（根據 DISPLAYMATRIX）。
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from ..exceptions import VideoTranscodeError

logger = logging.getLogger(__name__)


class VideoTranscodeService:
    """
    影片轉碼服務
    
    職責：
    1. 將任意格式影片轉碼為 H.264 MP4
    2. 自動應用旋轉校正（FFmpeg 會根據 DISPLAYMATRIX 自動處理）
    3. 統一輸出尺寸和幀率
    """
    
    # 硬編碼的子目錄名稱（基於 workspace_root）
    TRANSCODED_DIR = "transcoded"
    
    def __init__(self, config: Dict[str, Any], workspace_root: Optional[Path] = None):
        """
        初始化轉碼服務
        
        Args:
            config: 轉碼配置，例如:
                {
                    "enabled": True,
                    "delete_after_processing": False,
                    "check_existing": True,
                    "video_codec": "libx264",
                    "crf": 18,
                    "preset": "fast",
                    "target_fps": 30,
                    "target_width": 1080,
                    "target_height": 1920
                }
            workspace_root: 工作目錄根路徑（可選）
        """
        self.config = config
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self.enabled = config.get("enabled", True)
        # 必須使用 workspace_root 推算路徑
        if not self._workspace_root:
            raise ValueError("VideoTranscodeService 需要 workspace_root 參數才能決定輸出路徑")
        self.output_dir = self._workspace_root / self.TRANSCODED_DIR
        self.delete_after_processing = config.get("delete_after_processing", False)
        self.check_existing = config.get("check_existing", True)
        
        # NVENC 硬體編碼設定
        self.use_nvenc = config.get("use_nvenc", False)
        
        # 編碼參數
        if self.use_nvenc:
            # NVENC 使用 h264_nvenc
            self.video_codec = "h264_nvenc"
            # NVENC 使用 -cq (常量質量) 而非 crf
            self.cq = config.get("cq", 23)  # 0-51, 越小越好
            self.preset = config.get("preset", "p4")  # p1(最快)-p7(最好)
            logger.info(f"NVENC 硬體編碼已啟用 (codec={self.video_codec}, cq={self.cq}, preset={self.preset})")
        else:
            # 軟體編碼 libx264
            self.video_codec = config.get("video_codec", "libx264")
            self.crf = config.get("crf", 18)
            self.preset = config.get("preset", "fast")
        
        # 目標參數
        self.target_fps = config.get("target_fps", 30)
        self.target_width = config.get("target_width", 1080)
        self.target_height = config.get("target_height", 1920)
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 檢查 FFmpeg 是否可用
        self._ffmpeg_path = self._find_ffmpeg()
    
    def _find_ffmpeg(self) -> str:
        """找到 FFmpeg 執行檔路徑"""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            logger.warning("FFmpeg 未安裝或不在 PATH 中")
            return "ffmpeg"  # 仍然嘗試使用，讓錯誤在實際執行時發生
        return ffmpeg_path
    
    def transcode(self, input_path: str, video_info: Optional[Dict[str, Any]] = None) -> str:
        """
        將影片轉碼為標準化 MP4 格式
        
        Args:
            input_path: 原始影片路徑
            video_info: 影片資訊（可選，用於生成輸出檔名）
            
        Returns:
            str: 轉碼後的影片路徑
            
        Raises:
            VideoTranscodeError: 轉碼失敗時拋出
        """
        if not self.enabled:
            logger.info("轉碼功能已停用，直接返回原始路徑")
            return input_path
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise VideoTranscodeError(f"輸入影片不存在: {input_path}")
        
        # 檢查是否已經是 MP4 且不需要處理
        # （這裡我們選擇總是轉碼以確保格式一致性）
        
        # 生成輸出路徑
        output_path = self._get_output_path(input_path, video_info)
        
        # 檢查是否已存在轉碼檔案
        if self.check_existing and output_path.exists():
            logger.info(f"已存在轉碼檔案，跳過轉碼: {output_path}")
            return str(output_path)
        
        # 執行轉碼
        logger.info(f"開始轉碼: {input_path} -> {output_path}")
        
        try:
            self._run_ffmpeg(input_path, output_path)
            logger.info(f"轉碼完成: {output_path}")
            return str(output_path)
        except Exception as e:
            # 如果轉碼失敗，清理可能的不完整檔案
            if output_path.exists():
                output_path.unlink()
            raise VideoTranscodeError(f"轉碼失敗: {e}") from e
    
    def _get_output_path(self, input_path: Path, video_info: Optional[Dict[str, Any]] = None) -> Path:
        """生成輸出檔案路徑（使用原始影片檔名）"""
        # 使用原始檔名（去除擴展名）+ .mp4
        # 一個 id 可能有多部影片，所以使用原始檔名
        stem = input_path.stem
        
        return self.output_dir / f"{stem}.mp4"
    
    def _run_ffmpeg(self, input_path: Path, output_path: Path):
        """
        執行 FFmpeg 轉碼命令
        
        FFmpeg 參數說明：
        - fps={fps}: 統一幀率
        - scale=w:h:force_original_aspect_ratio=decrease: 縮放並保持長寬比
        - pad=w:h:(ow-iw)/2:(oh-ih)/2:black: letterbox 填充至目標尺寸
        - setsar=1: 設定 SAR 為 1:1
        - -c:v libx264: H.264 編碼
        - -crf: 品質參數
        - -preset: 編碼速度/品質權衡
        - -an: 移除音訊（骨架提取不需要）
        """
        # 組建 video filter
        vf = (
            f"fps={self.target_fps},"
            f"scale={self.target_width}:{self.target_height}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={self.target_width}:{self.target_height}:"
            f"(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )
        
        # 基本命令
        cmd = [
            self._ffmpeg_path,
            "-y",  # 覆蓋輸出檔案
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", self.video_codec,
        ]
        
        # 根據編碼器添加不同的質量參數
        if self.use_nvenc:
            # NVENC 使用 -cq 和 -preset
            cmd.extend(["-cq", str(self.cq), "-preset", self.preset])
        else:
            # libx264 使用 -crf 和 -preset
            cmd.extend(["-crf", str(self.crf), "-preset", self.preset])
        
        # 移除音訊並指定輸出
        cmd.extend(["-an", str(output_path)])
        
        logger.debug(f"FFmpeg 命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 處理無法解碼的字元
                check=True
            )
            logger.debug(f"FFmpeg stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 執行失敗:\nstderr: {e.stderr}")
            raise VideoTranscodeError(f"FFmpeg 執行失敗: {e.stderr}") from e
        except FileNotFoundError:
            raise VideoTranscodeError(
                "找不到 FFmpeg，請確認已安裝並加入 PATH 環境變數"
            )
    
    def cleanup(self, transcoded_path: str):
        """
        清理轉碼檔案（如果設定為處理後刪除）
        
        Args:
            transcoded_path: 轉碼後的檔案路徑
        """
        if self.delete_after_processing:
            path = Path(transcoded_path)
            if path.exists() and path.parent == self.output_dir:
                logger.info(f"清理轉碼檔案: {path}")
                path.unlink()
