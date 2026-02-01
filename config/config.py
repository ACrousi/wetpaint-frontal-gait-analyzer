import os
import yaml
from pathlib import Path
from typing import Any, Dict, List
from src.exceptions import ConfigLoadError


class ConfigManager:
    """
    配置管理器
    
    支援 workspace 路徑解析：
    - 環境變數 WETPAINT_WORKSPACE 覆蓋 config 中的 workspace_root
    - resolve_path() 將相對路徑轉為 workspace 下的絕對路徑
    - 自動將 config 中的相對路徑轉為絕對路徑
    """
    
    WORKSPACE_ENV = "WETPAINT_WORKSPACE"
    PROJECT_ROOT_ENV = "WETPAINT_PROJECT_ROOT"
    
    # 需要轉換為絕對路徑的 key 名稱（包含這些字串的 key）
    PATH_KEYS = ['_path', '_dir', 'output_dir', 'input_path']
    
    # 基於專案根目錄解析的 key（模型權重、配置檔案）
    PROJECT_ROOT_KEYS = ['config_path', 'pretrained_path', 'weight_path', 'model_path']
    
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = config_file
        self._raw_config = self._load_config()
        self._project_root = self._init_project_root()
        self._workspace_root = self._init_workspace()
        # 解析後的配置（路徑已轉為絕對路徑）
        self.config = self._resolve_config_paths(self._raw_config)

    def _load_config(self) -> dict:
        config_path = Path(self.config_file)
        if not config_path.exists():
            raise ConfigLoadError(
                f"配置文件 '{self.config_file}' 不存在",
                details={"path": str(config_path.absolute())}
            )
        with config_path.open("r", encoding="utf8") as file:
            return yaml.safe_load(file)

    def _init_project_root(self) -> Path:
        """
        初始化專案根目錄
        
        優先順序：
        1. 環境變數 WETPAINT_PROJECT_ROOT
        2. config 檔案所在目錄的父目錄
        """
        env_path = os.environ.get(self.PROJECT_ROOT_ENV)
        if env_path:
            root = Path(env_path)
        else:
            # config/config.yaml -> 專案根目錄
            root = Path(self.config_file).resolve().parent.parent
        
        return root.resolve()

    def _init_workspace(self) -> Path:
        """
        初始化 workspace 根目錄
        
        優先順序：
        1. 環境變數 WETPAINT_WORKSPACE
        2. config.yaml 中的 workspace_root（相對於專案根目錄解析）
        3. 預設值 "./outputs"
        """
        env_path = os.environ.get(self.WORKSPACE_ENV)
        if env_path:
            root = Path(env_path)
        else:
            ws_path = self._raw_config.get("workspace_root", "./outputs")
            # 相對路徑從專案根目錄解析
            if ws_path.startswith('./') or ws_path.startswith('../'):
                root = self._project_root / ws_path.lstrip('./')
            else:
                root = Path(ws_path)
        
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _is_project_root_path(self, key: str) -> bool:
        """判斷是否為專案根目錄路徑（模型、配置）"""
        return any(pk in key.lower() for pk in self.PROJECT_ROOT_KEYS)

    def _is_relative_path(self, value: str) -> bool:
        """判斷是否為相對路徑"""
        if not isinstance(value, str):
            return False
        if value.startswith('./') or value.startswith('../'):
            return True
        if not os.path.isabs(value) and value:
            return True
        return False

    def _should_resolve_path(self, key: str, value: str) -> bool:
        """判斷是否需要轉換為 workspace 路徑"""
        if not isinstance(value, str):
            return False
        
        # 專案根目錄路徑不走這裡
        if self._is_project_root_path(key):
            return False
        
        # 檢查是否為路徑相關的 key
        is_path_key = any(pk in key.lower() for pk in self.PATH_KEYS)
        if not is_path_key:
            return False
        
        return self._is_relative_path(value)

    def _resolve_config_paths(self, config: Any, parent_key: str = "") -> Any:
        """
        遞迴解析配置中的所有路徑
        
        將相對路徑轉為 workspace 下的絕對路徑
        """
        if isinstance(config, dict):
            resolved = {}
            for key, value in config.items():
                resolved[key] = self._resolve_config_paths(value, key)
            return resolved
        
        elif isinstance(config, list):
            return [self._resolve_config_paths(item, parent_key) for item in config]
        
        elif isinstance(config, str):
            # 專案根目錄路徑（模型、配置）
            if self._is_project_root_path(parent_key) and self._is_relative_path(config):
                clean_path = config.lstrip('./')
                return str(self._project_root / clean_path)
            
            # Workspace 路徑（資料 I/O）
            elif self._should_resolve_path(parent_key, config):
                clean_path = config.lstrip('./')
                return str(self._workspace_root / clean_path)
        
        return config

    @property
    def project_root(self) -> Path:
        """取得專案根目錄的絕對路徑"""
        return self._project_root

    @property
    def workspace_root(self) -> Path:
        """取得 workspace 根目錄的絕對路徑"""
        return self._workspace_root

    def resolve_path(self, relative_path: str) -> Path:
        """
        將相對路徑轉為 workspace 下的絕對路徑
        
        Args:
            relative_path: 相對於 workspace 的路徑
            
        Returns:
            絕對路徑
            
        Example:
            >>> cm = ConfigManager()
            >>> cm.resolve_path("json")
            Path("D:/WetPaint_Motor_Development/workspace/json")
        """
        return self._workspace_root / relative_path

    def ensure_dir(self, relative_path: str) -> Path:
        """
        確保目錄存在並返回絕對路徑
        
        Args:
            relative_path: 相對於 workspace 的路徑
            
        Returns:
            已建立的目錄絕對路徑
        """
        path = self.resolve_path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get(self, *keys, default=None):
        """
        取得嵌套配置中的值，例如：
            value = config_manager.get("data", "metadata", "metadata_path")
        如果鍵路徑不存在則返回 default。
        
        注意：回傳的路徑已自動轉為絕對路徑
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_raw(self, *keys, default=None):
        """
        取得原始配置值（未經路徑轉換）
        """
        value = self._raw_config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
