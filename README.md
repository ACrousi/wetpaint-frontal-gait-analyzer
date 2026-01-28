# 影片處理 API

基於 FastAPI 的影片處理服務，用於接收影片處理請求並執行分析。

## 功能特性

- 接收影片處理請求
- 執行影片追蹤分析
- 返回標準化處理結果
- 完整的錯誤處理和日誌記錄
- 整合的機器學習預測架構（Baseline MLP、XGBoost、Fusion）
- 統一的特徵提取和模型訓練流程

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 啟動服務

### 開發模式

```bash
python start_api.py --reload
```

### 生產模式

```bash
python start_api.py
```

### 自定義配置

```bash
python start_api.py --host 127.0.0.1 --port 8080
```

## API 端點

### 健康檢查

- **GET** `/`
- 用於檢查服務是否正常運行

### 處理影片

- **POST** `/process_video`
- 接收影片處理請求並返回結果

#### 請求格式

```json
{
  "case_id": "string",
  "videopath": "string",
  "months": 0
}
```

#### 響應格式

```json
{
  "case_id": "string",
  "score": 1.0,
  "prob": 1.0,
  "timestamp": "2025-04-01T12:00:00.000Z"
}
```

## 錯誤處理

API 會返回適當的 HTTP 狀態碼和錯誤信息：

- 400: 請求參數無效
- 404: 影片檔案不存在
- 500: 伺服器內部錯誤

## 日誌記錄

服務會將日誌記錄到 `api.log` 文件中，同時輸出到控制台。

## 架構整合

專案已完成簡潔的四模型預測架構，支援直接從 metadata.csv + skeleton.csv 進行批次推論：

### 核心組件

- **Predictor Factory** (`src/core/predictor_factory.py`): 統一建構四種預測器
- **Dataset Inference Runner** (`src/core/dataset_inference_runner.py`): 從 CSV 載入資料並批次預測
- **Batch Prediction Workflow** (`src/core/batch_prediction_workflow.py`): 統一批次預測工作流
- **Unified Dataset** (`src/core/models/dataset.py`): 訓練用資料載入
- **Unified Trainer** (`src/core/unified_trainer.py`): MLP 訓練器

### 支援的模型類型

1. **gait_mlp**: 僅步態特徵 + MLP
2. **gait_xgboost**: 僅步態特徵 + XGBoost
3. **resgcn**: 關節點 ResGCN
4. **fusion**: 步態特徵 + 關節點 ResGCN + 融合特徵 MLP

### 設定架構

參考 `config_schema.yaml` 進行配置：

```yaml
model:
  type: gait_mlp  # 選擇模型類型

data:
  metadata_csv_paths: ["outputs/metadata.csv"]
  csv_path_column: "exported_paths"
  gait_start_column: "segment_length"
  label_column: "age_months_range"

infer:
  confidence_threshold: 0.0

# 模型特定參數...
```

### 快速開始

```python
from src.core.batch_prediction_workflow import BatchPredictionWorkflow

config = {
    "model": {"type": "gait_mlp"},
    "data": {"metadata_csv_paths": ["metadata.csv"]},
    # ... 其他設定
}

workflow = BatchPredictionWorkflow(config)
result = workflow.run()
print(f"預測 {len(result['predictions'])} 個樣本")
```

### 特徵提取

- **Gait Extractor** (`src/prediction/features/gait_extractor.py`): 步態特徵提取
- **Metrics Extractor** (`src/prediction/features/metrics_extractor.py`): 計算指標提取

## 測試

可以使用以下命令測試 API 是否正常運行：

```bash
python test_api.py
```

也可以測試整合架構：

```bash
python test_simple_architecture.py
```

## API 文檔

啟動服務後，可以訪問以下地址查看自動生成的 API 文檔：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`