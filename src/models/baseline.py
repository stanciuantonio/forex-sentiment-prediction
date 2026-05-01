"""
XGBoost baseline — flattened-window classifier for EUR/USD direction prediction.

Scaler is saved to disk after training so evaluation can load it without leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import json
from pathlib import Path

from src.config import Config


def train_baseline(cfg: Config) -> dict:
    """Train XGBoost baseline using parameters from config. Returns metrics."""
    data_path = cfg.resolve(cfg.get("files.final_processed"))
    model_save_path = cfg.resolve(cfg.get("files.xgboost_model"))
    scaler_save_path = cfg.resolve(cfg.get("files.xgboost_scaler"))
    split_save_path = cfg.resolve(cfg.get("files.xgboost_split"))
    feature_columns = cfg.feature_columns

    # Hyperparams
    window_size = cfg.get("xgboost.window_size")
    max_depth = cfg.get("xgboost.max_depth")
    learning_rate = cfg.get("xgboost.learning_rate")
    n_estimators = cfg.get("xgboost.n_estimators")
    random_state = cfg.get("xgboost.random_state")
    train_ratio = cfg.get("split.xgboost_train_ratio", 0.8)

    cfg.ensure_dirs()
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data not found at {data_path}")
        return {"error": "data_not_found"}
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Create flattened window features
    features, targets = [], []
    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size : i]
        features.append(window[feature_columns].values.flatten())
        targets.append(df.iloc[i]["label"])

    if not features:
        print("Error: no features created")
        return {"error": "no_features"}

    X = np.array(features)
    y = np.array(targets) + 1  # -1,0,1 → 0,1,2

    # Temporal split
    split_idx = int(train_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: insufficient data for train/test split")
        return {"error": "insufficient_data"}

    # Scale (fit on train ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler (own copy per model)
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved: {scaler_save_path}")

    # Save split indices for consistent evaluation
    split_info = {
        "train_ratio": train_ratio,
        "n_total": len(X),
        "n_train": split_idx,
        "n_test": len(X) - split_idx,
    }
    with open(split_save_path, "w") as f:
        json.dump(split_info, f)
    print(f"Split info saved: {split_save_path}")

    # Train
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric="mlogloss",
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    eval_results = model.evals_result()

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["SELL", "HOLD", "BUY"], output_dict=True)

    print(f"\nXGBoost Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["SELL", "HOLD", "BUY"]))

    # Save
    joblib.dump(model, model_save_path)
    print(f"Model saved: {model_save_path}")

    # Save history as JSON
    history_path = model_save_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(eval_results, f)

    return {"accuracy": accuracy, "model_path": str(model_save_path)}


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    train_baseline(cfg)
