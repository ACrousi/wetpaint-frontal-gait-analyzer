"""
Factory 模組

提供建立 Workflow 和服務實例的工廠方法。
實現依賴注入模式，提升可測試性。

使用範例:
    # 從 YAML 檔案建立 Workflow（推薦）
    workflow = WorkflowFactory.create_from_yaml("config/config.yaml", "skeleton_extraction")
    
    # 從字典建立（向後相容）
    workflow = WorkflowFactory.create_from_config(config_dict)
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

from .services.video_processing_service import VideoProcessingService
from .services.analysis_service import AnalysisService
from .services.export_service import ExportService
from .services.visualization_service import SkeletonVisualizationService
from .workflows.skeleton_extraction_workflow import SkeletonExtractionWorkflow
from .config.models import SkeletonExtractionConfig
from .config.loader import load_config, get_raw_config


class ServiceFactory:
    """
    服務工廠
    
    負責建立各種服務實例，支援依賴注入和測試替換。
    """
    
    def __init__(self, config: Union[SkeletonExtractionConfig, Dict[str, Any]]):
        """
        初始化服務工廠
        
        Args:
            config: 配置物件或字典
        """
        self.config = config
        self._is_validated = isinstance(config, SkeletonExtractionConfig)
    
    def create_video_processor(self) -> VideoProcessingService:
        """建立影片處理服務"""
        if self._is_validated:
            return VideoProcessingService(self.config.video_processing.model_dump())
        return VideoProcessingService(self.config.get("video_processing", {}))
    
    def create_analysis_service(self) -> AnalysisService:
        """建立分析服務"""
        if self._is_validated:
            return AnalysisService(self.config.analysis.model_dump())
        return AnalysisService(self.config.get("analysis", {}))
    
    def create_export_service(self) -> ExportService:
        """建立導出服務"""
        if self._is_validated:
            return ExportService(self.config.export.model_dump())
        return ExportService(self.config.get("export", {}))
    
    def create_visualization_service(self) -> SkeletonVisualizationService:
        """建立視覺化服務"""
        if self._is_validated:
            return SkeletonVisualizationService(self.config.visualization.model_dump())
        return SkeletonVisualizationService(self.config.get("visualization", {}))


class WorkflowFactory:
    """
    Workflow 工廠
    
    提供建立 SkeletonExtractionWorkflow 的便利方法。
    支援：
    1. 從 YAML 檔案建立（推薦，會自動驗證配置）
    2. 從字典建立（向後相容）
    3. 注入自訂服務（測試用）
    """
    
    @staticmethod
    def create_from_yaml(
        config_path: Union[str, Path],
        mode: str = "skeleton_extraction"
    ) -> SkeletonExtractionWorkflow:
        """
        從 YAML 檔案建立 Workflow（推薦方式）
        
        會自動驗證配置格式。
        
        Args:
            config_path: 配置檔案路徑
            mode: 配置模式
            
        Returns:
            配置完成的 SkeletonExtractionWorkflow 實例
            
        Raises:
            ConfigLoadError: 載入失敗
            ConfigValidationError: 驗證失敗
        """
        # 載入並驗證配置
        validated_config = load_config(config_path, mode)
        
        # 轉換為字典傳給 Workflow（目前 Workflow 仍使用 dict）
        raw_config = get_raw_config(config_path, mode)
        return SkeletonExtractionWorkflow(raw_config)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> SkeletonExtractionWorkflow:
        """
        從設定字典建立 Workflow（向後相容）
        
        Args:
            config: 設定字典
            
        Returns:
            SkeletonExtractionWorkflow 實例
        """
        return SkeletonExtractionWorkflow(config)
    
    @staticmethod
    def create_with_services(
        config: Dict[str, Any],
        video_processor: Optional[VideoProcessingService] = None,
        analysis_service: Optional[AnalysisService] = None,
        export_service: Optional[ExportService] = None,
        visualization_service: Optional[SkeletonVisualizationService] = None
    ) -> SkeletonExtractionWorkflow:
        """
        建立 Workflow 並注入自訂服務（主要用於測試）
        
        Args:
            config: 設定字典
            video_processor: 可選的自訂影片處理服務
            analysis_service: 可選的自訂分析服務
            export_service: 可選的自訂導出服務
            visualization_service: 可選的自訂視覺化服務
            
        Returns:
            SkeletonExtractionWorkflow 實例
        """
        workflow = SkeletonExtractionWorkflow(config)
        
        # 替換服務（如果提供）
        if video_processor is not None:
            workflow.video_processor = video_processor
        if analysis_service is not None:
            workflow.analysis_service = analysis_service
        if export_service is not None:
            workflow.export_service = export_service
        if visualization_service is not None:
            workflow.visualization_service = visualization_service
        
        return workflow


# 便利函數
def create_workflow(config: Dict[str, Any]) -> SkeletonExtractionWorkflow:
    """
    建立 Workflow 的便利函數
    
    等同於 WorkflowFactory.create_from_config(config)
    """
    return WorkflowFactory.create_from_config(config)


def create_workflow_from_yaml(
    config_path: Union[str, Path],
    mode: str = "skeleton_extraction"
) -> SkeletonExtractionWorkflow:
    """
    從 YAML 建立 Workflow 的便利函數
    
    等同於 WorkflowFactory.create_from_yaml(config_path, mode)
    """
    return WorkflowFactory.create_from_yaml(config_path, mode)
