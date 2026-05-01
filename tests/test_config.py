"""
Tests for Config module.
"""

import pytest
from pathlib import Path
from src.config import Config


class TestConfig:
    def test_loads(self):
        cfg = Config()
        assert cfg.get("lstm.hidden_size") == 32
        assert cfg.get("data.label_threshold") == 0.002

    def test_feature_columns(self):
        cfg = Config()
        cols = cfg.feature_columns
        assert isinstance(cols, list)
        assert "log_return" in cols
        assert len(cols) >= 15

    def test_ensure_dirs(self):
        cfg = Config()
        cfg.ensure_dirs()
        for key in ["paths.data_raw", "paths.data_processed", "paths.models", "paths.results", "paths.reports"]:
            assert Path(cfg.get(key)).is_dir()

    def test_resolve(self):
        cfg = Config()
        resolved = cfg.resolve("data/raw/test.csv")
        assert str(resolved).endswith("data/raw/test.csv")
        assert resolved.is_absolute()

    def test_key_error(self):
        cfg = Config()
        assert cfg.get("nonexistent.key") is None

    def test_singleton(self):
        from src.config import get_config
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_new_scaler_paths(self):
        cfg = Config()
        assert cfg.get("files.lstm_scaler") == "models/scaler_lstm.pkl"
        assert cfg.get("files.xgboost_scaler") == "models/scaler_xgboost.pkl"
        assert cfg.get("files.lstm_split") == "models/split_lstm.json"
        assert cfg.get("files.xgboost_split") == "models/split_xgboost.json"
