from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class VideoProcessRequest(BaseModel):
    """
    影片處理請求模型
    """
    case_id: str
    videopath: str
    months: int  # 實際月齡

class VideoProcessResponse(BaseModel):
    """
    影片處理響應模型 - 對應 ResGCN 預測輸出格式
    
    Fields:
        case_id: 個案 ID
        predicted_age: 預測月齡 (期望值)
        predicted_class: 預測類別 index (對應 bin_centers)
        confidence: 預測信心度 (最高機率值)
        prob_distribution: 各 bin 的機率分布
        actual_age: 實際月齡 (來自請求的 months)
        age_difference: 預測與實際的差異
        development_status: 發展評估狀態 (正常/邊緣/遲緩)
        num_segments: 使用的片段數量
        timestamp: 處理時間戳
    """
    case_id: str
    predicted_age: float                         # 期望月齡
    predicted_class: int                         # 最高機率 bin index
    confidence: float                            # 信心度
    prob_distribution: List[float]               # 機率分布
    actual_age: Optional[float] = None           # 實際月齡
    age_difference: Optional[float] = None       # 預測 - 實際
    development_status: str = "未知"              # 發展狀態
    num_segments: int = 1                        # 片段數
    timestamp: str                               # ISO 時間戳

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