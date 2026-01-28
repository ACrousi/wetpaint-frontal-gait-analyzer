"""
ResGCN 預測器

透過 subprocess 調用 ResGCNv1/main.py --predict 進行推論
"""

import logging
import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.models.prediction_result import SegmentPrediction, PredictionResult
from src.exceptions import InvalidInputError, InferenceError

logger = logging.getLogger(__name__)


class ResGCNPredictor:
    """ResGCN 預測器
    
    透過 subprocess 調用 ResGCNv1/main.py --predict 進行推論，
    與訓練模式保持一致的調用方式。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化預測器
        
        Args:
            config: 配置字典，包含：
                - resgcn:
                    - config: ResGCN 配置名稱 (如 'resgcn_coco_2')
                    - pretrained_path: 模型權重路徑
        """
        self.config = config
        
        # 支援兩種格式: 
        # 1. predict.resgcn.config (推薦)
        # 2. predict.config (舊格式相容)
        resgcn_config = config.get('resgcn', config)
        self.resgcn_config_name = resgcn_config.get('config', 'resgcn_coco_2')
        self.pretrained_path = resgcn_config.get('pretrained_path', '')
        
        # ResGCN 路徑
        self.project_root = Path(__file__).parent.parent.parent
        self.resgcn_dir = self.project_root / 'vendor' / 'ResGCNv1'
    
    def predict_from_jsons(
        self,
        json_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None
    ) -> PredictionResult:
        """從 JSON 檔案列表執行預測
        
        Args:
            json_paths: JSON 骨架檔案路徑列表
            case_id: 個案 ID
            actual_age: 實際月齡
            
        Returns:
            整合後的 PredictionResult
        """
        if not json_paths:
            raise InvalidInputError(
                "未提供 JSON 檔案",
                details={"json_paths": []}
            )
        
        # 調用 ResGCNv1/main.py --predict
        cmd = [
            sys.executable, "-u", "main.py",
            "--config", self.resgcn_config_name,
            "--predict",
            "--input_json"
        ] + [str(p) for p in json_paths]
        
        # 加入模型權重路徑（如有指定）
        if self.pretrained_path:
            cmd.extend(["--pretrained_path", str(self.pretrained_path)])
        
        logger.info(f"執行 ResGCN 預測: {' '.join(cmd[:6])}... ({len(json_paths)} files)")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.resgcn_dir),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        if result.returncode != 0:
            raise InferenceError(
                f"ResGCN 預測失敗",
                details={
                    "returncode": result.returncode,
                    "stderr": result.stderr[:500] if result.stderr else None,
                    "json_count": len(json_paths)
                }
            )
        
        # 解析輸出結果
        stdout = result.stdout
        predictions = self._parse_prediction_output(stdout)
        
        if not predictions:
            raise InferenceError(
                "無法解析預測結果",
                details={"stdout_preview": stdout[:500]}
            )
        
        # 轉換為 SegmentPrediction 列表
        segments = []
        for i, pred in enumerate(predictions):
            if 'error' in pred:
                logger.warning(f"Prediction error for {pred.get('file')}: {pred['error']}")
                continue
            
            segments.append(SegmentPrediction(
                segment_id=Path(pred['file']).stem,
                predicted_age=pred['predicted_age'],
                predicted_class=pred['predicted_class'],
                confidence=pred['confidence'],
                prob_distribution=pred['prob_distribution']
            ))
        
        if not segments:
            raise InferenceError(
                "沒有有效的預測結果",
                details={"predictions_count": len(predictions)}
            )
        
        # 整合結果
        return PredictionResult.from_segments(
            segments=segments,
            case_id=case_id,
            actual_age=actual_age
        )
    
    def _parse_prediction_output(self, stdout: str) -> List[Dict]:
        """解析 subprocess 輸出中的預測結果 JSON
        
        Args:
            stdout: subprocess 的標準輸出
            
        Returns:
            預測結果列表
        """
        # 尋找標記之間的 JSON
        start_marker = '===PREDICTION_RESULTS_START==='
        end_marker = '===PREDICTION_RESULTS_END==='
        
        start_idx = stdout.find(start_marker)
        end_idx = stdout.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            logger.error(f"Cannot find prediction markers in output")
            return []
        
        json_str = stdout[start_idx + len(start_marker):end_idx].strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []
    
    def predict_and_aggregate(
        self,
        json_paths: List[str],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None
    ) -> PredictionResult:
        """預測並整合結果（別名方法，保持 API 相容）"""
        return self.predict_from_jsons(json_paths, case_id, actual_age)
