"""
預測工作流程

整合骨架提取、ResGCN 推論的完整預測流程
透過 subprocess 調用 ResGCNv1 保持一致性
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.core.workflows.skeleton_extraction_workflow import SkeletonExtractionWorkflow
from src.models.resgcn_predictor import ResGCNPredictor
from src.models.prediction_result import PredictionResult
from src.exceptions import InvalidInputError, PredictionError

logger = logging.getLogger(__name__)


class PredictionWorkflow:
    """預測工作流程
    
    處理流程：
    1. 從影片列表提取骨架（生成 JSON）
    2. 透過 subprocess 調用 ResGCN 推論
    3. 整合多片段結果（平均策略）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化預測工作流程
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 骨架提取工作流程
        skeleton_config = config.get("skeleton_extraction", {})
        self.skeleton_workflow = SkeletonExtractionWorkflow(skeleton_config)
        
        # ResGCN 預測器（subprocess 模式）
        predict_config = config.get("predict", {})
        self.predictor = ResGCNPredictor(predict_config)
        
        # 輸出設定
        export_config = config.get("skeleton_extraction", {}).get("export", config.get("export", {}))
        self.json_output_dir = Path(export_config.get("seg_skeleton", {}).get("output_dir", "./outputs/json"))
    
    def predict_from_videos(
        self,
        video_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None,
        skip_extraction: bool = False
    ) -> PredictionResult:
        """從影片列表執行預測
        
        Args:
            video_paths: 影片路徑列表
            case_id: 個案 ID（可選）
            actual_age: 實際月齡（可選，用於評估）
            skip_extraction: 是否跳過骨架提取（假設已有 JSON）
            
        Returns:
            整合後的 PredictionResult
        """
        logger.info(f"開始預測流程，共 {len(video_paths)} 支影片")
        
        json_paths = []
        
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).stem
            logger.info(f"處理影片 ({i+1}/{len(video_paths)}): {video_name}")
            
            try:
                if skip_extraction:
                    # 假設已有 JSON，直接找檔案
                    json_path = self._find_json_for_video(video_path)
                    if json_path:
                        json_paths.append(json_path)
                    else:
                        logger.warning(f"找不到 {video_name} 的 JSON 檔案，跳過")
                else:
                    # 執行骨架提取
                    extracted_jsons = self._extract_skeleton(video_path)
                    if extracted_jsons:
                        json_paths.extend(extracted_jsons)
                    else:
                        logger.warning(f"骨架提取失敗: {video_name}，跳過")
                        
            except Exception as e:
                logger.error(f"處理影片失敗: {video_name} - {e}")
                continue
        
        if not json_paths:
            raise InvalidInputError(
                "沒有成功處理任何影片，無法進行預測",
                details={"video_count": len(video_paths)}
            )
        
        # 執行 ResGCN 推論（透過 subprocess）
        logger.info(f"執行 ResGCN 推論，共 {len(json_paths)} 個 JSON 片段")
        result = self.predictor.predict_from_jsons(
            json_paths=json_paths,
            case_id=case_id,
            actual_age=actual_age
        )
        
        logger.info(f"預測完成：期望月齡={result.predicted_age:.2f}, 信心度={result.confidence:.3f}")
        
        return result
    
    def predict_from_jsons(
        self,
        json_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None
    ) -> PredictionResult:
        """從 JSON 檔案列表執行預測（跳過骨架提取）
        
        Args:
            json_paths: JSON 骨架檔案路徑列表
            case_id: 個案 ID
            actual_age: 實際月齡
            
        Returns:
            整合後的 PredictionResult
        """
        logger.info(f"從 JSON 執行預測，共 {len(json_paths)} 個檔案")
        
        return self.predictor.predict_from_jsons(
            json_paths=json_paths,
            case_id=case_id,
            actual_age=actual_age
        )
    
    def _extract_skeleton(self, video_path: str) -> List[str]:
        """提取影片骨架並返回生成的 JSON 檔案路徑
        
        Args:
            video_path: 影片路徑
            
        Returns:
            生成的 JSON 檔案路徑列表
        """
        from src.core.models import VideoInfo
        
        video_info = VideoInfo(
            video_path=Path(video_path),
            original_video=Path(video_path),
            case_id=Path(video_path).stem
        )
        
        result = self.skeleton_workflow.extract_analyze_and_export(video_info)
        
        if not result.get('success'):
            return []
        
        # 找到生成的 JSON 檔案
        json_paths = self._find_jsons_for_video(video_path)
        return json_paths
    
    def _find_json_for_video(self, video_path: str) -> Optional[str]:
        """根據影片路徑找到對應的單一 JSON 骨架檔案"""
        jsons = self._find_jsons_for_video(video_path)
        return jsons[0] if jsons else None
    
    def _find_jsons_for_video(self, video_path: str) -> List[str]:
        """根據影片路徑找到所有對應的 JSON 骨架檔案
        
        Args:
            video_path: 影片路徑
            
        Returns:
            JSON 檔案絕對路徑列表（確保 subprocess 在不同目錄執行時也能找到）
        """
        video_name = Path(video_path).stem
        
        # 搜尋可能的 JSON 檔案
        if self.json_output_dir.exists():
            matches = list(self.json_output_dir.glob(f"{video_name}*.json"))
            if matches:
                # 轉換為絕對路徑，確保 subprocess 能正確找到
                return [str(m.resolve()) for m in sorted(matches)]
        
        return []
