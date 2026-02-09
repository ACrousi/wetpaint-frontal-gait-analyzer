from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
import os
from datetime import datetime

# 添加項目根目錄到 Python 路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.models import VideoProcessRequest, VideoProcessResponse, VideoProcessingError, VideoFileNotFoundError
from config import ConfigManager
from src.core.workflows.prediction_workflow import PredictionWorkflow

# 設置日誌
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = log_dir / f"api_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 全局變量
config = None
workspace_root = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理器"""
    global config, workspace_root
    try:
        logger.info("正在初始化應用...")
        
        # 初始化配置管理器
        config_manager = ConfigManager("./config/config.yaml")
        config = config_manager.config
        workspace_root = config_manager.workspace_root
        logger.info(f"配置加載完成 (workspace: {workspace_root})")
        
        # 注意：PredictionWorkflow 現在在每次請求時創建，以確保 GPU 記憶體可被釋放
        logger.info("應用初始化完成（workflow 將在每次請求時創建）")
        
        yield
        
        # 關閉時的清理工作
        logger.info("應用正在關閉...")
        
    except Exception as e:
        logger.error(f"應用啟動失敗: {e}", exc_info=True)
        raise

# 創建 FastAPI 應用實例
app = FastAPI(
    title="Video Processing API",
    description="影片處理 API，用於接收影片處理請求並執行分析",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路徑，用於健康檢查"""
    return {"message": "Video Processing API 正在運行", "status": "ok"}

@app.post("/process_video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """
    處理影片分析請求
    
    Args:
        request: 影片處理請求
        
    Returns:
        VideoProcessResponse: 處理結果
    """
    global config, workspace_root
    
    try:
        # 每次請求創建新的 workflow，確保預測後釋放 GPU 記憶體
        workflow = PredictionWorkflow(config, workspace_root=workspace_root)
        logger.info(f"收到處理請求 - Case ID: {request.case_id}, Video: {request.videopath}")
        
        # 檢查影片檔案是否存在
        video_path = Path(request.videopath)
        if not video_path.exists():
            logger.error(f"影片檔案不存在: {request.videopath}")
            raise VideoFileNotFoundError(str(video_path))
        
        # 執行骨架提取和預測
        prediction_result = workflow.predict_from_videos(
            video_paths=[str(video_path)],
            case_id=request.case_id,
            actual_age=float(request.months)
        )

        # 生成響應 - 使用 PredictionResult 的結構
        response = VideoProcessResponse(
            case_id=request.case_id,
            predicted_age=prediction_result.predicted_age,
            predicted_class=prediction_result.segment_predictions[0].predicted_class if prediction_result.segment_predictions else 0,
            confidence=prediction_result.confidence,
            prob_distribution=prediction_result.prob_distribution,
            actual_age=prediction_result.actual_age,
            age_difference=prediction_result.age_difference,
            development_status=prediction_result.development_status,
            num_segments=prediction_result.num_segments,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"處理完成 - Case ID: {request.case_id}, 預測月齡: {prediction_result.predicted_age:.2f}, 狀態: {prediction_result.development_status}")
        return response
        
    except VideoFileNotFoundError as e:
        logger.error(f"檔案不存在錯誤: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except VideoProcessingError as e:
        logger.error(f"處理錯誤: {e.message}")
        raise HTTPException(
            status_code=e.error_code,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"未預期的錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"伺服器內部錯誤: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
