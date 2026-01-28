# API 路由設計

## 主要端點

### POST /process_video

#### 描述
接收影片處理請求並執行影片追蹤分析

#### 請求
- **方法**: POST
- **路徑**: /process_video
- **請求體**: VideoProcessRequest 模型

#### 響應
- **成功**: 200 OK + VideoProcessResponse 模型
- **錯誤**: 400 Bad Request 或 500 Internal Server Error

#### 處理流程
1. 驗證請求參數
2. 檢查影片檔案是否存在
3. 初始化 VideoProcessingWorkflow
4. 建立 video_info 字典
5. 執行 run_tracking
6. 生成響應結果

#### 整合現有工作流程
```python
# 初始化配置
config_manager = ConfigManager("config.yaml")
config = config_manager.config

# 創建工作流程實例
workflow = VideoProcessingWorkflow(config)

# 構造 video_info
video_info = {
    "video_path": request.videopath,
    "video_name": Path(request.videopath).name,
    "target_fps": 30,  # 默認值或從配置讀取
    "case_id": request.case_id,
    "months": request.months
}

# 執行處理
result = workflow.run_tracking(track_manager, video_info)

# 生成響應
response = VideoProcessResponse(
    case_id=request.case_id,
    score=1.0,  # 暫時用 1 代替
    prob=1.0,   # 暫時用 1 代替
    timestamp=datetime.utcnow().isoformat() + "Z"
)