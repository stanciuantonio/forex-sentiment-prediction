# forex-ml

Machine learning pipeline for EUR/USD forex direction prediction using
news sentiment (FinBERT) + technical indicators.

## Quick Start

```bash
pip install -r requirements.txt
make all    # Full pipeline: data → features → train → evaluate
```

## Pipeline Stages

| Target | Description |
|--------|-------------|
| `make data` | Fetch forex OHLC + GDELT news + extract articles |
| `make sentiment` | Score news with FinBERT, merge with price data |
| `make features` | Compute technical indicators + target labels |
| `make train-lstm` | Train LSTM sequence model **(saves scaler)** |
| `make train-xgboost` | Train XGBoost baseline **(saves scaler)** |
| `make evaluate` | Evaluate all models → `results/metrics.json` |
| `make all` | Full pipeline |

## Project Structure

```
forex-ml/
├── config/config.yaml       # All paths, features, hyperparams
├── data/raw/                # Raw forex + news data
├── data/processed/          # Feature-engineered dataset
├── models/                  # Saved artifacts (scaler.pkl, *.h5, *.joblib)
├── results/                 # metrics.json + evaluation plots
├── docs/                    # Reports (moved out of repo root)
├── src/
│   ├── config.py            # Config loader (dot-notation access)
│   ├── data/                # Data fetching & cleaning
│   ├── features/            # Sentiment analysis + feature engineering
│   ├── models/              # LSTM + XGBoost training
│   └── evaluate.py          # Evaluation with loaded scaler
├── tests/                   # Pytest suite
└── Makefile                 # Pipeline orchestration
```

## Key Fix: Scaler Save/Load

**Before:** `evaluate_model.py` refitted the scaler on eval data (data leakage).
**After:** Training saves the scaler (`models/scaler.pkl`), evaluation loads it.

## Configuration

Edit `config/config.yaml` to change paths, features, hyperparameters, or split ratios.

```bash
# Test config loads
python -c "from src.config import Config; cfg = Config(); print(cfg.get('lstm.hidden_size'))"
```

## Tests

```bash
make test
```
