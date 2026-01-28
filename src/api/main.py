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
from src.core.workflow import VideoProcessingWorkflow
from src.core.prediction_workflow import ResGCNPredictionWorkflow

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局變量
workflow = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理器"""
    global workflow, config
    try:
        logger.info("正在初始化應用...")
        
        # 初始化配置管理器
        config_manager = ConfigManager("./config/config.yaml")
        config = config_manager.config
        logger.info("配置加載完成")
        
        # 創建影片處理工作流程
        workflow = ResGCNPredictionWorkflow(config)
        logger.info("影片處理工作流程初始化完成")
        
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
    global workflow, config
    
    try:
        logger.info(f"收到處理請求 - Case ID: {request.case_id}, Video: {request.videopath}")
        
        # 檢查影片檔案是否存在
        video_path = Path(request.videopath)
        if not video_path.exists():
            logger.error(f"影片檔案不存在: {request.videopath}")
            raise VideoFileNotFoundError(str(video_path))
        
        # 構造 video_info 字典
        video_info = {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "target_fps": 30,  # 默認值或從配置讀取
            "case_id": request.case_id,
            "months": request.months
        }
        
        # 執行影片處理
        logger.info(f"開始處理影片: {request.videopath}")
        result = workflow.run_prediction(video_info)

        if not result.get("success", False):
            error_msg = result.get("error", "未知錯誤")
            logger.error(f"影片處理失敗: {error_msg}")
            raise VideoProcessingError(f"影片處理失敗: {error_msg}")

        # 解析預測結果
        predictions = result.get("predictions", [])
        if not predictions:
            logger.warning("沒有預測結果")
            raise VideoProcessingError("沒有找到有效的預測結果")

        # 計算整體預測結果（這裡可以根據業務邏輯調整）
        # 例如：取最高信心度的預測，或計算平均值等
        if predictions:
            # 簡單的策略：取第一個預測結果
            best_prediction = predictions[0]
            predicted_class = best_prediction.get("predicted_class", 0)
            confidence = best_prediction.get("confidence", 0.0)

            # 將預測類別轉換為業務標籤（這裡需要根據實際的類別映射調整）
            label_map = {
                0: "0-3",
                1: "3-6",
                2: "6-9",
                3: "9-12",
                4: "12-15",
                5: "15-18"
            }
            label = label_map.get(predicted_class, "unknown")

            # 根據信心度決定結果
            result_status = "normal" if confidence > 0.5 else "uncertain"
        else:
            label = "unknown"
            confidence = 0.0
            result_status = "no_prediction"

        # 生成響應
        response = VideoProcessResponse(
            case_id=request.case_id,
            label=label,
            prob=confidence,
            result=result_status,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"處理完成 - Case ID: {request.case_id}")
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