"""
Config loader — single source of truth for paths, features, and hyperparams.

Usage:
    from src.config import Config
    cfg = Config()
    cfg.get('paths.data_raw')          # "data/raw"
    cfg.get('lstm.hidden_size')        # 32
    cfg.get('features.columns')        # [...]
"""

import os
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Loads and provides access to config/config.yaml."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")
        with open(self._config_path) as f:
            self._data: dict = yaml.safe_load(f)

    # ── Dot-notation access ──────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Access nested keys via dot notation: cfg.get('lstm.hidden_size')"""
        parts = key.split(".")
        node = self._data
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part)
                if node is None:
                    return default
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Key not found: {key}")
        return val

    # ── Property shortcuts ───────────────────────────────────────────────

    @property
    def feature_columns(self) -> list[str]:
        return self.get("features.columns", [])

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.get("paths.data_raw", "data/raw"))

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.get("paths.data_processed", "data/processed"))

    @property
    def models_dir(self) -> Path:
        return Path(self.get("paths.models", "models"))

    @property
    def results_dir(self) -> Path:
        return Path(self.get("paths.results", "results"))

    @property
    def reports_dir(self) -> Path:
        return Path(self.get("paths.reports", "results/reports"))

    def ensure_dirs(self):
        """Create all configured directories if they don't exist."""
        for key in ["paths.data_raw", "paths.data_processed", "paths.models",
                     "paths.results", "paths.reports"]:
            path = Path(self.get(key))
            path.mkdir(parents=True, exist_ok=True)

    # ── Resolve absolute paths ───────────────────────────────────────────

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path (from config) against project root."""
        project_root = self._config_path.parent.parent.resolve()
        return project_root / relative_path

    def __repr__(self) -> str:
        return f"Config({self._config_path})"


# Module-level singleton for convenience
_config_instance: Config | None = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
