"""
配置載入器

提供配置檔案的載入、驗證和解析功能。

使用範例:
    from src.core.config import load_config
    
    config = load_config("config/config.yaml", "skeleton_extraction")
    print(config.video_processing.fps)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import yaml

from .models import SkeletonExtractionConfig
from src.exceptions import ConfigLoadError, ConfigValidationError

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    載入 YAML 配置檔案
    
    Args:
        path: 配置檔案路徑
        
    Returns:
        解析後的配置字典
        
    Raises:
        ConfigLoadError: 檔案不存在或解析失敗
    """
    path = Path(path)
    
    if not path.exists():
        raise ConfigLoadError(f"配置檔案不存在: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ConfigLoadError(f"配置檔案格式不正確，預期為字典: {path}")
        
        return data
        
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"YAML 解析失敗: {e}", details={"path": str(path)})


def load_config(path: Union[str, Path], mode: str) -> SkeletonExtractionConfig:
    """
    載入並驗證配置
    
    Args:
        path: 配置檔案路徑
        mode: 配置模式，例如 "skeleton_extraction"
        
    Returns:
        驗證後的配置物件
        
    Raises:
        ConfigLoadError: 載入失敗
        ConfigValidationError: 驗證失敗
    """
    logger.info(f"載入配置: {path} (mode={mode})")
    
    # 載入 YAML
    raw_config = load_yaml(path)
    
    # 取得指定模式的配置
    mode_config = raw_config.get(mode)
    if mode_config is None:
        raise ConfigLoadError(
            f"配置中找不到模式: {mode}",
            details={"available_modes": list(raw_config.keys())}
        )
    
    # 驗證配置
    try:
        config = SkeletonExtractionConfig.model_validate(mode_config)
        logger.info(f"配置驗證成功: video_processing.fps={config.video_processing.fps}")
        return config
        
    except Exception as e:
        raise ConfigValidationError(
            f"配置驗證失敗: {e}",
            details={"mode": mode}
        )


def get_raw_config(path: Union[str, Path], mode: str) -> Dict[str, Any]:
    """
    取得原始配置字典（不經過 Pydantic 驗證）
    
    用於向後相容，允許服務使用傳統的 dict 配置。
    
    Args:
        path: 配置檔案路徑
        mode: 配置模式
        
    Returns:
        原始配置字典
    """
    raw_config = load_yaml(path)
    return raw_config.get(mode, {})
