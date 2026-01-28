"""
工作流程模組

提供高層協調邏輯。
"""

from .skeleton_extraction_workflow import SkeletonExtractionWorkflow
from .prediction_workflow import PredictionWorkflow

__all__ = [
    'SkeletonExtractionWorkflow',
    'PredictionWorkflow',
]
