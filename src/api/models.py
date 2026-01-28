from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VideoProcessRequest(BaseModel):
    """
    影片處理請求模型
    """
    case_id: str
    videopath: str
    months: int

class VideoProcessResponse(BaseModel):
    """
    影片處理響應模型
    """
    case_id: str
    label: str = "15-18"
    prob: float = 1.0
    result: str = "normal"
    timestamp: str

class VideoProcessingError(Exception):
    """影片處理相關錯誤"""
    def __init__(self, message: str, error_code: int = 500):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class VideoFileNotFoundError(VideoProcessingError):
    """影片檔案不存在錯誤"""
    def __init__(self, filepath: str):
        super().__init__(f"影片檔案不存在: {filepath}", 404)

class InvalidRequestError(VideoProcessingError):
    """無效請求錯誤"""
    def __init__(self, message: str):
        super().__init__(f"無效請求: {message}", 400)