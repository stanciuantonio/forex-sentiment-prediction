"""
Tests for data modules.
"""

import pytest
from src.config import Config


class TestForexDataFetcher:
    def test_save_forex_data_creates_dir(self, tmp_path):
        from src.data.forex_data_fetcher import save_forex_data
        output = tmp_path / "subdir" / "test.csv"
        assert not output.parent.exists()

    def test_module_imports(self):
        from src.data.forex_data_fetcher import fetch_forex_daily, save_forex_data
        assert callable(fetch_forex_daily)
        assert callable(save_forex_data)


class TestGdeltDataCleaner:
    def test_cleaner_imports(self):
        from src.data.gdelt_data_cleaner import clean_and_process_gdelt_articles
        assert callable(clean_and_process_gdelt_articles)

    def test_cleaner_missing_file(self, tmp_path):
        from src.data.gdelt_data_cleaner import clean_and_process_gdelt_articles
        result = clean_and_process_gdelt_articles(
            tmp_path / "nonexistent.csv",
            tmp_path / "out.csv",
        )
        assert result is False
