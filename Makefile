# =============================================================================
# forex-ml — Makefile
# Pipeline completo: data → features → train → evaluate
# =============================================================================

.PHONY: all data features sentiment train-lstm train-xgboost evaluate clean install test

# ── Full pipeline ──────────────────────────────────────────────────────────

all: data features sentiment train-lstm train-xgboost evaluate
	@echo "✅ Pipeline complete — check results/metrics.json"

# ── Data ───────────────────────────────────────────────────────────────────

data/raw/eurusd_daily.csv:
	python -m src.data.forex_data_fetcher

data/raw/gdelt_forex_news.csv:
	python -m src.data.gdelt_news_extractor

data/processed/gdelt_articles_text_cleaned.csv: data/raw/gdelt_forex_news.csv
	python -m src.data.article_content_processor

data: data/raw/eurusd_daily.csv data/raw/gdelt_forex_news.csv data/processed/gdelt_articles_text_cleaned.csv
	@echo "✅ Data stage complete"

# ── Sentiment + Features ───────────────────────────────────────────────────

data/processed/gdelt_eurusd_with_sentiment.csv: data/raw/eurusd_daily.csv data/processed/gdelt_articles_text_cleaned.csv
	python -m src.features.sentiment_analyzer

sentiment: data/processed/gdelt_eurusd_with_sentiment.csv
	@echo "✅ Sentiment stage complete"

data/processed/eurusd_final_processed.csv: data/processed/gdelt_eurusd_with_sentiment.csv
	python -m src.features.feature_engineering

features: data/processed/eurusd_final_processed.csv
	@echo "✅ Features stage complete"

# ── Training ───────────────────────────────────────────────────────────────

models/lstm_model.h5: data/processed/eurusd_final_processed.csv
	python -m src.models.lstm

train-lstm: models/lstm_model.h5
	@echo "✅ LSTM trained"

models/xgboost_baseline.joblib: data/processed/eurusd_final_processed.csv
	python -m src.models.baseline

train-xgboost: models/xgboost_baseline.joblib
	@echo "✅ XGBoost trained"

# ── Evaluation ─────────────────────────────────────────────────────────────

results/metrics.json: models/lstm_model.h5 models/xgboost_baseline.joblib
	python -m src.evaluate

evaluate: results/metrics.json
	@echo "✅ Evaluation complete"

# ── Utilities ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

clean:
	rm -rf models/*.pkl models/*.h5 models/*.joblib models/*.json
	rm -rf results/metrics.json results/reports/
	@echo "🧹 Cleaned artifacts (kept data/)"

clean-all: clean
	rm -rf data/raw/*.csv data/processed/*.csv
	@echo "🧹 Cleaned everything (kept docs/)"

# Show current state
status:
	@echo "=== Data files ==="
	@ls -lh data/raw/ 2>/dev/null || echo "  (empty)"
	@ls -lh data/processed/ 2>/dev/null || echo "  (empty)"
	@echo "=== Models ==="
	@ls -lh models/ 2>/dev/null || echo "  (empty)"
	@echo "=== Results ==="
	@ls -lh results/ 2>/dev/null || echo "  (empty)"
	@[ -f results/metrics.json ] && echo "=== metrics.json ===" && cat results/metrics.json || true
