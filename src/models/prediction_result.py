"""
預測結果資料模型

用於封裝單一片段或整合多片段的預測結果
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class SegmentPrediction:
    """單一片段的預測結果"""
    segment_id: str                    # 片段識別碼
    predicted_age: float               # 期望月齡 (expectation)
    predicted_class: int               # 最高機率的 bin index
    confidence: float                  # 信心度 (max probability)
    prob_distribution: List[float]     # 各 bin 的機率分布
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "predicted_age": self.predicted_age,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "prob_distribution": self.prob_distribution
        }


@dataclass
class PredictionResult:
    """整合預測結果（可能包含多個片段）"""
    
    # 最終預測結果（多片段平均後）
    predicted_age: float               # 期望月齡 (平均)
    confidence: float                  # 平均信心度
    prob_distribution: List[float]     # 平均機率分布
    
    # 各片段的個別結果
    segment_predictions: List[SegmentPrediction] = field(default_factory=list)
    
    # 元資料
    case_id: Optional[str] = None      # 個案 ID
    actual_age: Optional[float] = None # 實際月齡 (如有提供)
    
    @property
    def num_segments(self) -> int:
        """片段數量"""
        return len(self.segment_predictions)
    
    @property
    def age_difference(self) -> Optional[float]:
        """預測與實際的差異（如有實際月齡）"""
        if self.actual_age is not None:
            return self.predicted_age - self.actual_age
        return None
    
    @property
    def development_status(self) -> str:
        """發展評估狀態
        
        根據預測月齡與實際月齡的差異判斷：
        - 差異 <= -3: 遲緩
        - 差異 在 -3 到 -1 之間: 邊緣
        - 差異 > -1: 正常
        """
        diff = self.age_difference
        if diff is None:
            return "未知"
        
        if diff <= -3:
            return "遲緩"
        elif diff < -1:
            return "邊緣"
        else:
            return "正常"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "predicted_age": self.predicted_age,
            "confidence": self.confidence,
            "prob_distribution": self.prob_distribution,
            "num_segments": self.num_segments,
            "segment_predictions": [s.to_dict() for s in self.segment_predictions]
        }
        
        if self.case_id:
            result["case_id"] = self.case_id
        if self.actual_age is not None:
            result["actual_age"] = self.actual_age
            result["age_difference"] = self.age_difference
            result["development_status"] = self.development_status
        
        return result
    
    @classmethod
    def from_segments(
        cls, 
        segments: List[SegmentPrediction],
        case_id: Optional[str] = None,
        actual_age: Optional[float] = None
    ) -> "PredictionResult":
        """從多個片段預測結果建立整合結果（平均策略）"""
        if not segments:
            raise ValueError("At least one segment prediction is required")
        
        # 計算平均
        avg_age = np.mean([s.predicted_age for s in segments])
        avg_confidence = np.mean([s.confidence for s in segments])
        
        # 平均機率分布
        prob_arrays = [np.array(s.prob_distribution) for s in segments]
        avg_prob = np.mean(prob_arrays, axis=0).tolist()
        
        return cls(
            predicted_age=float(avg_age),
            confidence=float(avg_confidence),
            prob_distribution=avg_prob,
            segment_predictions=segments,
            case_id=case_id,
            actual_age=actual_age
        )
