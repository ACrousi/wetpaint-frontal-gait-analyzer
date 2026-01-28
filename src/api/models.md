# API 資料模型定義

## VideoProcessRequest 模型

### 描述
用於接收影片處理請求的資料模型

### 屬性
- `case_id` (str): 案例識別碼
- `videopath` (str): 影片檔案路徑
- `months` (int): 月份數

### Pydantic 定義
```python
from pydantic import BaseModel
from typing import Optional

class VideoProcessRequest(BaseModel):
    case_id: str
    videopath: str
    months: int
```

## VideoProcessResponse 模型

### 描述
用於回傳影片處理結果的資料模型

### 屬性
- `case_id` (str): 案例識別碼
- `score` (float): 評分結果（暫時用 1 代替）
- `prob` (float): 機率結果（暫時用 1 代替）
- `timestamp` (str): ISO 8601 格式時間戳

### Pydantic 定義
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VideoProcessResponse(BaseModel):
    case_id: str
    score: float = 1.0
    prob: float = 1.0
    timestamp: str