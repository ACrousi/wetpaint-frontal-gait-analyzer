import yaml
from pathlib import Path
from src.exceptions import ConfigLoadError

class ConfigManager:
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = Path(self.config_file)
        if not config_path.exists():
            raise ConfigLoadError(
                f"配置文件 '{self.config_file}' 不存在",
                details={"path": str(config_path.absolute())}
            )
        with config_path.open("r", encoding="utf8") as file:
            return yaml.safe_load(file)

    def get(self, *keys, default=None):
        """
        取得嵌套配置中的值，例如：
            value = config_manager.get("data", "metadata", "metadata_path")
        如果鍵路徑不存在則返回 default。
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
