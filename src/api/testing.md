# API 測試策略

## 測試類型

### 單元測試
- 測試資料模型驗證
- 測試路由處理函數
- 測試錯誤處理邏輯

### 整合測試
- 測試完整的 API 請求/響應流程
- 測試與 VideoProcessingWorkflow 的整合
- 測試檔案存在/不存在的情況

### 性能測試
- 測試 API 響應時間
- 測試併發請求處理能力

## 測試工具

### 推薦工具
- pytest: Python 測試框架
- httpx: 用於測試 FastAPI 應用
- pytest-cov: 代碼覆蓋率報告

### 測試環境設置
```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_process_video_success():
    response = client.post("/process_video", json={
        "case_id": "test_case_001",
        "videopath": "test_video.mp4",
        "months": 12
    })
    assert response.status_code == 200
    data = response.json()
    assert "case_id" in data
    assert "score" in data
    assert "prob" in data
    assert "timestamp" in data
```

## 測試用例

### 正常情況測試
1. 發送有效的影片處理請求
2. 驗證響應格式和內容

### 錯誤情況測試
1. 發送無效的請求參數
2. 請求不存在的影片檔案
3. 模擬處理過程中的錯誤

### 邊界情況測試
1. 空字串參數
2. 超長字串參數
3. 負數月份值

## 測試數據準備

### 測試影片
- 準備小型測試影片檔案
- 確保測試環境中有可用的影片

### 測試配置
- 使用專門的測試配置文件
- 設置獨立的輸出路徑

## 自動化測試

### CI/CD 集成
- 在 GitHub Actions 中運行測試
- 設置測試覆蓋率門檻

### 測試報告
- 生成 HTML 格式的測試報告
- 記錄測試執行時間和結果