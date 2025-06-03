# Description of the idea

Our project aims to build a hybrid pipeline that merges historical price data with daily sentiment scores derived from Forex news to predict EUR/USD movements. We utilize price data (e.g., from `eurusd_daily.csv`, with Alpha Vantage as a potential source for updates) to obtain OHLC history and compute log-returns.

In parallel, we leverage the GDELT dataset as our primary source for news articles relevant to Forex markets. A dedicated pipeline extracts and filters these articles, aiming for a manageable number per day (e.g., 1-3) by relevance and tone. Each relevant news item's content is then processed using FinBERT, a pre-trained language model specialized in financial text, to obtain a continuous sentiment score. These individual sentiment scores are aggregated daily (e.g., by averaging) to create a daily sentiment time series that aligns with the price data.

As a baseline, we will train Gaussian Process Regression and XGBoost models (as initially planned, and potentially implemented in `src/models/baseline.py`) on this combined feature set using scikit-learn's TimeSeriesSplit for rolling-window cross-validation. We will then train an LSTM (Long Short-Term Memory) network (implemented in `src/models/lstm.py`). The model will use 30-day lookback windows, incorporating both historical price-derived features (like log-returns) and the daily sentiment scores as inputs. For training, each day will be labeled based on a threshold method applied to the next-day log-return (e.g., > +0.2% for BUY, < -0.2% for SELL, otherwise HOLD), a technique supported by financial literature. Rigorous backtesting and reporting of cumulative profit metrics will be conducted throughout the project to ensure the reproducibility and academic validity of our findings.

We chose to focus on Forex markets due to their high liquidity and 24/5 operation, which provide a robust environment for time-series modeling. By integrating price action with sentiment data derived from a broad source like GDELT, and employing deep learning models alongside classical benchmarks, this project seeks to determine if sentiment information significantly enhances price prediction accuracy. This approach aims to provide a clear, research-backed methodology for testing our hypotheses on real-world market data.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Running the Code](#running-the-code)
  - [Data Acquisition and Initial Processing](#1--data-acquisition-and-initial-processing-output-to-dataraw-and-datainterim)
  - [Sentiment Analysis](#2--sentiment-analysis-output-to-datainterim-or-dataprocessed)
  - [Final Feature Engineering](#3--final-feature-engineering-output-to-dataprocessedeurusd_final_processedcsv)
  - [Model Training](#4--model-training)
  - [Model Evaluation](#5--model-evaluation)
- [Project Structure](#project-structure)
- [Loading Model Weights](#loading-model-weights)
- [Contributing](#contributing)
- [License](#license)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd forex-sentiment-prediction
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\\Scripts\\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file lists all necessary Python packages.

## Running the Code

The primary dataset for modeling, `data/processed/eurusd_final_processed.csv`, is included in this repository. However, if you wish to regenerate all datasets from scratch, follow these steps:

1.  **Data Acquisition and Initial Processing (Output to `data/raw/` and `data/interim/`):**
    Execute the scripts in `src/data/` in the following order:

    ```bash
    python src/data/forex_data_fetcher.py
    python src/data/gdelt_news_extractor.py
    python src/data/article_content_processor.py # Processes news content
    python src/data/gdelt_data_cleaner.py       # Cleans GDELT data
    ```

    _These scripts will produce intermediate files. Ensure any necessary input files or API keys are correctly configured (e.g., via a `.env` file if used)._

2.  **Sentiment Analysis (Output to `data/interim/` or `data/processed/`):**
    This script likely takes data produced by the previous steps and adds sentiment scores.

    ```bash
    python src/features/sentiment_analyzer.py
    ```

    _Verify its input/output paths and dependencies on prior scripts._

3.  **Final Feature Engineering (Output to `data/processed/eurusd_final_processed.csv`):**
    This script creates the final dataset used for modeling.

    ```bash
    python src/features/feature_engineering.py
    ```

    _This script should use data from the previous steps and produce `data/processed/eurusd_final_processed.csv`._

4.  **Model Training:**
    Train the models using scripts in `src/models/`. Ensure the input data path points to `data/processed/eurusd_final_processed.csv` (or provide a different path via arguments).
    Trained models will be saved in the `results/models/` directory.

    - **LSTM Model:**
      To train the LSTM model, run `src/models/lstm.py`. You can customize hyperparameters using command-line arguments. For example:

      ```bash
      python src/models/lstm.py --epochs 150 --batch_size 64 --learning_rate 0.0005 --hidden_size 128 --num_layers 3 --dropout_rate 0.3
      ```

      Common arguments:

      - `--data_path`: Path to the input CSV data (default: `data/processed/eurusd_final_processed.csv`).
      - `--model_save_path`: Path to save the trained model (default: `results/models/lstm_model.h5`).
      - `--epochs`: Number of training epochs (default: 100).
      - `--batch_size`: Batch size (default: 32).
      - `--learning_rate`: Initial learning rate (default: 0.001).
      - `--hidden_size`: LSTM hidden layer size (default: 64).
      - `--num_layers`: Number of LSTM layers (default: 2).
      - `--dropout_rate`: Dropout rate (default: 0.2).
      - `--window_size`: Lookback window for sequences (default: 30).
      - `--early_stopping_patience`: Patience for early stopping (default: 15).

      Run `python src/models/lstm.py --help` for a full list of options.

    - **Baseline Models (XGBoost):**
      To train the XGBoost baseline model, run `src/models/baseline.py`. You can customize hyperparameters. For example:

      ```bash
      python src/models/baseline.py --n_estimators 150 --max_depth 8 --learning_rate 0.05
      ```

      Common arguments:

      - `--data_path`: Path to the input CSV data (default: `data/processed/eurusd_final_processed.csv`).
      - `--model_save_path`: Path to save the trained model (default: `results/models/xgboost_baseline.joblib`).
      - `--window_size`: Lookback window for features (default: 30).
      - `--max_depth`: Maximum tree depth for XGBoost (default: 6).
      - `--learning_rate`: Learning rate for XGBoost (default: 0.1).
      - `--n_estimators`: Number of trees (boosting rounds) for XGBoost (default: 100).
      - `--random_state`: Random state for reproducibility (default: 42).

      Run `python src/models/baseline.py --help` for a full list of options.

    _Trained models should ideally be saved to the `results/models/` directory. Specify any command-line arguments for model configuration if applicable (e.g., epochs, learning rate)._

5.  **Model Evaluation:**
    After training, models can be evaluated using the `src/evaluation/evaluate_model.py` script. This script loads a trained model, processes the evaluation data, and generates performance metrics, a confusion matrix plot, and a classification report plot.

    **Example Usage:**

    - **Evaluate a trained LSTM model:**
      Make sure to provide the same LSTM architecture parameters that were used during training.

      ```bash
      python src/evaluation/evaluate_model.py --model_type lstm --model_path results/models/lstm_model.h5 --lstm_hidden_size 128 --lstm_num_layers 3 --lstm_dropout_rate 0.3
      ```

    - **Evaluate a trained XGBoost model:**
      ```bash
      python src/evaluation/evaluate_model.py --model_type xgboost --model_path results/models/xgboost_baseline.joblib --data_path data/processed/eurusd_final_processed.csv --reports_dir results/reports
      ```

    **Command-line Arguments for `evaluate_model.py`:**

    - `--model_path` (required): Path to the trained model file (`.h5` for LSTM, `.joblib` for XGBoost).
    - `--model_type` (required): Type of the model, either `lstm` or `xgboost`.
    - `--data_path`: Path to the evaluation data CSV file (default: `data/processed/eurusd_final_processed.csv`).
    - `--reports_dir`: Directory to save evaluation reports and plots (default: `results/reports`).
    - `--window_size`: Lookback window size used during training (default: 30).
    - `--lstm_hidden_size`: Hidden size of LSTM layers (default: 64, from `lstm.py`). **Required if `model_type` is `lstm` and non-default was used for training.**
    - `--lstm_num_layers`: Number of LSTM layers (default: 2, from `lstm.py`). **Required if `model_type` is `lstm` and non-default was used for training.**
    - `--lstm_dropout_rate`: Dropout rate for LSTM (default: 0.2, from `lstm.py`). **Required if `model_type` is `lstm` and non-default was used for training.**

    **Outputs:**
    The script will print accuracy and a classification report to the console. It will also save the following files to the specified `--reports_dir` (e.g., `results/reports/`):

    - `classification_metrics_<model_type>.txt`: Text file containing accuracy and detailed classification report.
    - `confusion_matrix_<model_type>.png`: Plot of the confusion matrix.
    - `classification_report_<model_type>.png`: Plot of the classification report (heatmap).

    Run `python src/evaluation/evaluate_model.py --help` for a full list of options.

## Project Structure

The project is organized as follows:

- `data/`: Contains raw, interim, and processed datasets.
  - `data/raw/`: Original, immutable data (generated by scripts, ignored by Git).
    - Example scripts creating data here: `src/data/forex_data_fetcher.py`, `src/data/gdelt_news_extractor.py`
  - `data/interim/`: Intermediate data (generated by scripts, ignored by Git).
    - Example scripts creating data here: `src/data/article_content_processor.py`, `src/data/gdelt_data_cleaner.py`
  - `data/processed/`: The final, canonical data sets for modeling.
    - `eurusd_final_processed.csv`: Main dataset used for modeling (versioned in Git). Generated by `src/features/feature_engineering.py`.
- `src/`: Contains all source code.
  - `src/data/`: Scripts for downloading and preparing raw/interim data.
    - `forex_data_fetcher.py`: Fetches historical Forex data.
    - `gdelt_news_extractor.py`: Extracts news articles from GDELT.
    - `article_content_processor.py`: Processes content from extracted articles.
    - `gdelt_data_cleaner.py`: Cleans the GDELT news data.
  - `src/features/`: Scripts for feature engineering.
    - `sentiment_analyzer.py`: Calculates sentiment scores from news data and adds them to a dataset.
    - `feature_engineering.py`: Combines price data and sentiment scores, performs other feature engineering, and creates the final `data/processed/eurusd_final_processed.csv`.
  - `src/models/`: Scripts for training models and (potentially) making predictions.
    - `lstm.py`: Trains the LSTM model.
    - `baseline.py`: Trains baseline models (e.g., XGBoost, Gaussian Process Regression).
    - _Note: A script for making predictions with trained models (e.g., `predict_model.py`) is not currently present in this directory._
  - `src/evaluation/`: Scripts for evaluating model performance.
    - _Note: This directory is currently empty. Scripts for model evaluation (e.g., `evaluate_model.py`) need to be added here._
- `results/`: Contains model outputs.

  - `results/models/`: Saved model weights/checkpoints (e.g., from `lstm.py`, `baseline.py`).
  - _Note: Consider adding a `results/reports/` directory for evaluation metrics, plots, etc._

- `requirements.txt`: Project Python dependencies.
- `README.md`: This file.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Loading Model Weights

The training scripts `src/models/lstm.py` and `src/models/baseline.py` save the trained models to the `results/models/` directory by default:

- **LSTM Model:** Saved as `results/models/lstm_model.h5`.
- **XGBoost Model:** Saved as `results/models/xgboost_baseline.joblib`.

You can find these files in `results/models/` after running the respective training scripts.

These models can be loaded and used for evaluation (or prediction) using the `src/evaluation/evaluate_model.py` script. Please refer to the ["Model Evaluation"](#5--model-evaluation) section for instructions on how to run this script.

If you intend to load these models manually in your own scripts:

- **For the LSTM model (`lstm_model.h5`):** Ensure you instantiate the model with the same architecture parameters (hidden size, number of layers, dropout rate) that were used during its training. The `evaluate_model.py` script accepts these parameters as command-line arguments (`--lstm_hidden_size`, `--lstm_num_layers`, `--lstm_dropout_rate`) to correctly load the model.

  ```python
  from tensorflow.keras.models import load_model
  # Example:
  # lstm_model = load_model('results/models/lstm_model.h5')
  # Ensure your data preprocessing matches that used in training/evaluation.
  ```

- **For the XGBoost model (`xgboost_baseline.joblib`):**
  ```python
  import joblib
  # Example:
  # xgboost_model = joblib.load('results/models/xgboost_baseline.joblib')
  # Ensure your data preprocessing matches that used in training/evaluation.
  ```

If these model files are not present, you will need to train them first by following the steps in the ["Model Training"](#4--model-training) section.
