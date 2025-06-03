"""
Modelo Baseline XGBoost simplificado para predicción EUR/USD
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os
import argparse

# Define constants for default values
DEFAULT_DATA_PATH = 'data/processed/eurusd_final_processed.csv'
DEFAULT_MODEL_SAVE_PATH = 'results/models/xgboost_baseline.joblib'
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42

def train_baseline(data_path, model_save_path, window_size, max_depth, learning_rate, n_estimators, random_state):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create features from rolling windows
    features = []
    targets = []

    if len(df) <= window_size:
        print(f"Error: DataFrame has insufficient data (rows: {len(df)}) to create sequences with window size {window_size}.")
        return None

    for i in range(window_size, len(df)):
        window = df.iloc[i-window_size:i]
        feature_row = window[['log_return', 'gdelt_sentiment']].values.flatten()
        features.append(feature_row)
        targets.append(df.iloc[i]['label'])

    if not features:
        print("Error: No features were created. Check window_size and data length.")
        return None

    X = np.array(features)
    y = np.array(targets)
    y = y + 1 # Convert labels from -1,0,1 to 0,1,2 for XGBoost

    # Temporal split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data for train/test split.")
        return None

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y)), # Dynamic num_class
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        use_label_encoder=False # Suppress warning
    )

    model.fit(X_train, y_train)

    # Save the trained model
    try:
        joblib.dump(model, model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nXGBoost Baseline Training finished.") # Updated message
    # Evaluation on test set is now handled by evaluate_model.py
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Baseline Model for Forex Prediction")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to the processed data CSV file')
    parser.add_argument('--model_save_path', type=str, default=DEFAULT_MODEL_SAVE_PATH, help='Path to save the trained XGBoost model')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE, help='Size of the lookback window for features')
    parser.add_argument('--max_depth', type=int, default=DEFAULT_MAX_DEPTH, help='Maximum depth of a tree in XGBoost')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for XGBoost')
    parser.add_argument('--n_estimators', type=int, default=DEFAULT_N_ESTIMATORS, help='Number of boosting rounds (trees) for XGBoost')
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE, help='Random state for reproducibility')

    args = parser.parse_args()

    train_baseline(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        window_size=args.window_size,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
