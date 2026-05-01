"""
Tests for model configuration from config.yaml.
"""

import pytest
from src.config import Config


class TestModelConfig:
    def test_lstm_params(self):
        cfg = Config()
        assert cfg.get("lstm.window_size") == 30
        assert cfg.get("lstm.hidden_size") == 32
        assert cfg.get("lstm.num_layers") == 1
        assert cfg.get("lstm.dropout") == 0.3
        assert cfg.get("lstm.epochs") == 50

    def test_xgboost_params(self):
        cfg = Config()
        assert cfg.get("xgboost.max_depth") == 6
        assert cfg.get("xgboost.learning_rate") == 0.1
        assert cfg.get("xgboost.n_estimators") == 100
        assert cfg.get("xgboost.random_state") == 42

    def test_model_paths(self):
        cfg = Config()
        assert cfg.get("files.lstm_model") == "models/lstm_model.h5"
        assert cfg.get("files.xgboost_model") == "models/xgboost_baseline.joblib"
        assert cfg.get("files.lstm_scaler") == "models/scaler_lstm.pkl"
        assert cfg.get("files.xgboost_scaler") == "models/scaler_xgboost.pkl"
        assert cfg.get("files.metrics") == "results/metrics.json"

    def test_split_params(self):
        cfg = Config()
        assert cfg.get("split.train_val_ratio") == 0.85
        assert cfg.get("split.test_ratio") == 0.15
        assert cfg.get("split.xgboost_train_ratio") == 0.8
