"""
預測工作流程

整合骨架提取、ResGCN 推論的完整預測流程
透過 subprocess 調用 ResGCNv1 保持一致性
使用臨時目錄存放 JSON，預測完成後自動清理
"""

import logging
import os
import tempfile
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
    1. 從影片列表提取骨架（生成 JSON 到臨時目錄）
    2. 透過 subprocess 調用 ResGCN 推論
    3. 整合多片段結果（平均策略）
    4. 自動清理臨時檔案
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
        
        # 輸出設定（用於 skip_extraction 模式）
        export_config = config.get("skeleton_extraction", {}).get("export", config.get("export", {}))
        self.json_output_dir = Path(export_config.get("seg_skeleton", {}).get("output_dir", "./outputs/json"))
    
    def predict_from_videos(
        self,
        video_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None,
        skip_extraction: bool = False,
        save_json: bool = False
    ) -> PredictionResult:
        """從影片列表執行預測
        
        Args:
            video_paths: 影片路徑列表
            case_id: 個案 ID（可選）
            actual_age: 實際月齡（可選，用於評估）
            skip_extraction: 是否跳過骨架提取（假設已有 JSON）
            save_json: 是否儲存 JSON 檔案（骨架和片段），預設 False 使用臨時檔案
            
        Returns:
            整合後的 PredictionResult
        """
        logger.info(f"開始預測流程，共 {len(video_paths)} 支影片 (save_json={save_json})")
        
        if skip_extraction:
            json_paths = []
            for video_path in video_paths:
                found = self._find_jsons_for_video(video_path)
                if found:
                    json_paths.extend(found)
                else:
                    logger.warning(f"找不到 {Path(video_path).stem} 的 JSON 檔案，跳過")
            
            if not json_paths:
                raise InvalidInputError(
                    "沒有找到任何 JSON 檔案",
                    details={"video_count": len(video_paths)}
                )
            
            return self._predict_from_json_paths(json_paths, case_id, actual_age)
        
        if save_json:
            # 儲存模式：JSON 檔案存到正常輸出目錄
            return self._predict_with_save(video_paths, case_id, actual_age)
        else:
            # 臨時模式：使用臨時目錄，預測後自動清理
            return self._predict_with_temp(video_paths, case_id, actual_age)
    
    def _predict_with_save(
        self,
        video_paths: List[str],
        case_id: Optional[str],
        actual_age: Optional[float]
    ) -> PredictionResult:
        """使用永久儲存模式進行預測
        
        JSON 檔案（骨架 + 片段）會儲存到配置的輸出目錄
        """
        logger.info("使用永久儲存模式")
        
        json_paths = []
        
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).stem
            logger.info(f"處理影片 ({i+1}/{len(video_paths)}): {video_name}")
            
            try:
                extracted_jsons = self._extract_skeleton_and_save(video_path)
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
        
        return self._predict_from_json_paths(json_paths, case_id, actual_age)
    
    def _predict_with_temp(
        self,
        video_paths: List[str],
        case_id: Optional[str],
        actual_age: Optional[float]
    ) -> PredictionResult:
        """使用臨時目錄模式進行預測
        
        JSON 檔案會存到臨時目錄，預測完成後自動刪除
        """
        with tempfile.TemporaryDirectory(prefix="wetpaint_predict_") as tmpdir:
            logger.info(f"使用臨時目錄: {tmpdir}")
            
            json_paths = []
            
            for i, video_path in enumerate(video_paths):
                video_name = Path(video_path).stem
                logger.info(f"處理影片 ({i+1}/{len(video_paths)}): {video_name}")
                
                try:
                    # 執行骨架提取，JSON 輸出到臨時目錄
                    extracted_jsons = self._extract_skeleton_to_temp(video_path, tmpdir)
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
            
            result = self._predict_from_json_paths(json_paths, case_id, actual_age)
            
            logger.info(f"臨時目錄將被自動清理: {tmpdir}")
            return result
        # 離開 with 區塊後，臨時目錄自動刪除
    
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
    
    def _predict_from_json_paths(
        self,
        json_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None
    ) -> PredictionResult:
        """從 JSON 路徑列表執行 ResGCN 預測
        
        Args:
            json_paths: JSON 檔案路徑列表
            case_id: 個案 ID
            actual_age: 實際月齡
            
        Returns:
            PredictionResult
        """
        logger.info(f"執行 ResGCN 推論，共 {len(json_paths)} 個 JSON 片段")
        
        result = self.predictor.predict_from_jsons(
            json_paths=json_paths,
            case_id=case_id,
            actual_age=actual_age
        )
        
        logger.info(f"預測完成：期望月齡={result.predicted_age:.2f}, 信心度={result.confidence:.3f}")
        return result
    
    def _extract_skeleton_and_save(self, video_path: str) -> List[str]:
        """提取影片骨架並儲存到配置的輸出目錄
        
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
        
        # save_results=True 表示儲存所有結果（骨架 + 片段 + metadata）
        result = self.skeleton_workflow.extract_analyze_and_export(
            video_info, 
            save_results=True
        )
        
        if not result.get('success'):
            return []
        
        # 從結果中直接獲取導出的 JSON 路徑
        exported_paths = result.get('exported_json_paths', {})
        if exported_paths:
            json_list = []
            for paths in exported_paths.values():
                json_list.extend(paths)
            return json_list
        
        # 備用：從配置的輸出目錄搜尋
        return self._find_jsons_for_video(video_path)
    
    def _extract_skeleton_to_temp(self, video_path: str, temp_dir: str) -> List[str]:
        """提取影片骨架並輸出 JSON 到臨時目錄
        
        Args:
            video_path: 影片路徑
            temp_dir: 臨時目錄路徑
            
        Returns:
            生成的 JSON 檔案路徑列表
        """
        from src.core.models import VideoInfo
        
        video_info = VideoInfo(
            video_path=Path(video_path),
            original_video=Path(video_path),
            case_id=Path(video_path).stem
        )
        
        # 傳入 temp_output_dir，JSON 將輸出到臨時目錄
        result = self.skeleton_workflow.extract_analyze_and_export(
            video_info, 
            save_results=False,  # 不儲存 metadata CSV 等
            temp_output_dir=temp_dir  # 輸出到臨時目錄
        )
        
        if not result.get('success'):
            return []
        
        # 從結果中直接獲取導出的 JSON 路徑
        exported_paths = result.get('exported_json_paths', {})
        if exported_paths:
            # exported_paths 是 Dict[track_id, List[path]]，展平為路徑列表
            json_list = []
            for paths in exported_paths.values():
                json_list.extend(paths)
            return json_list
        
        # 備用：從臨時目錄搜尋
        video_name = Path(video_path).stem
        temp_path = Path(temp_dir)
        if temp_path.exists():
            matches = list(temp_path.glob(f"{video_name}*.json"))
            return [str(m.resolve()) for m in sorted(matches)]
        
        return []
    
    def _find_jsons_for_video(self, video_path: str) -> List[str]:
        """根據影片路徑找到所有對應的 JSON 骨架檔案（用於 skip_extraction 模式）
        
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

