import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        if self._config is None:
            if config_path is None:
                config_path = self._find_config()
            self._config_path = Path(config_path)
            self._load()

    @staticmethod
    def _find_config() -> str:
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "config" / "config.yaml"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError("config/config.yaml not found in project tree")

    def _load(self) -> None:
        with open(self._config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def all(self) -> Dict[str, Any]:
        return self._config

    def project_root(self) -> Path:
        return self._config_path.parent.parent

    def resolve_path(self, relative_path: str) -> str:
        return str(self.project_root() / relative_path)

    def reload(self) -> None:
        self._load()


def get_config(config_path: str = None) -> ConfigLoader:
    return ConfigLoader(config_path)
