# 錯誤處理和日誌記錄設計

## 錯誤類型定義

### 自定義異常類
```python
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
```

## HTTP 狀態碼處理

### 成功響應
- 200 OK: 處理成功

### 客戶端錯誤
- 400 Bad Request: 請求參數無效
- 404 Not Found: 影片檔案不存在
- 422 Unprocessable Entity: 請求體驗證失敗

### 伺服器錯誤
- 500 Internal Server Error: 處理過程中發生未預期錯誤
- 503 Service Unavailable: 服務暫時不可用

## 錯誤響應格式
```json
{
    "error": {
        "type": "error_type",
        "message": "錯誤描述",
        "code": 400
    }
}
```

## 日誌記錄策略

### 日誌級別
- DEBUG: 詳細的調試信息
- INFO: 一般信息，如請求接收、處理完成
- WARNING: 警告信息，如檔案不存在但可以處理
- ERROR: 錯誤信息，如處理失敗
- CRITICAL: 嚴重錯誤，如系統崩潰

### 日誌格式
```
[時間] [級別] [模組] 訊息
```

### 日誌記錄點
1. API 請求接收
2. 影片處理開始
3. 影片處理完成
4. 錯誤發生
5. 系統啟動/關閉

### 日誌配置示例
```python
import logging
from datetime import datetime

# 創建 logger
logger = logging.getLogger("video_api")
logger.setLevel(logging.INFO)

# 創建格式化器
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 創建文件處理器
file_handler = logging.FileHandler(
    f"logs/api_{datetime.now().strftime('%Y%m%d')}.log",
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)