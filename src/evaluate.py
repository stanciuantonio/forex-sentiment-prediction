"""
Evaluation — load trained model + saved scaler, run on test set, produce metrics + plots.

FIX: Scaler is loaded from disk (saved during training), NOT refit on eval data.
     This eliminates the data leakage that existed in the old evaluate_model.py.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import Config
from src.models.lstm import LSTMModel

FEATURE_COLUMNS = Config().feature_columns
CLASS_NAMES = ["SELL", "HOLD", "BUY"]


# ── Plotting helpers ───────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot: {save_path}")


def _plot_classification_report(report_dict, model_name, save_path):
    plot_data = {}
    for cls_name in CLASS_NAMES:
        if cls_name in report_dict and isinstance(report_dict[cls_name], dict):
            plot_data[cls_name] = {
                k: report_dict[cls_name].get(k, 0)
                for k in ("precision", "recall", "f1-score")
            }
    if not plot_data:
        return
    report_df = pd.DataFrame(plot_data).T
    plt.figure(figsize=(8, max(3, len(CLASS_NAMES) * 0.8)))
    sns.heatmap(report_df, annot=True, cmap="viridis", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Per-Class Metrics — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot: {save_path}")


def _plot_summary_metrics(report_dict, model_name, save_path):
    metrics = {}
    if "accuracy" in report_dict:
        metrics["Accuracy"] = report_dict["accuracy"]
    for avg in ["macro avg", "weighted avg"]:
        if avg in report_dict:
            for m in ["precision", "recall", "f1-score"]:
                metrics[f"{avg} {m}"] = report_dict[avg][m]
    if not metrics:
        return
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Score", y="Metric", data=df, palette="mako", orient="h")
    plt.title(f"Summary Metrics — {model_name}")
    plt.xlim(0, 1)
    for i, v in enumerate(df["Score"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot: {save_path}")


def _plot_lstm_history(history, model_name, save_path):
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(10, 6))
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker="x")
    plt.title(f"LSTM Training History — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot: {save_path}")


# ── Evaluators ─────────────────────────────────────────────────────────────

def evaluate_lstm(cfg: Config) -> dict:
    """Evaluate saved LSTM model using the saved scaler (no data leakage)."""
    data_path = cfg.resolve(cfg.get("files.final_processed"))
    model_path = cfg.resolve(cfg.get("files.lstm_model"))
    scaler_path = cfg.resolve(cfg.get("files.lstm_scaler"))
    split_path = cfg.resolve(cfg.get("files.lstm_split"))
    history_path = cfg.resolve(cfg.get("files.lstm_history"))
    reports_dir = cfg.reports_dir
    window_size = cfg.get("lstm.window_size")
    hidden_size = cfg.get("lstm.hidden_size")
    num_layers = cfg.get("lstm.num_layers")
    dropout = cfg.get("lstm.dropout")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load split indices saved during training
    if not split_path.exists():
        print(f"ERROR: Split info not found at {split_path}. Train the model first.")
        sys.exit(1)
    with open(split_path) as f:
        split_info = json.load(f)
    split_idx = split_info["n_train_val"]

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Build sequences
    sequences, targets = [], []
    for i in range(window_size, len(df)):
        sequences.append(df.iloc[i - window_size : i][FEATURE_COLUMNS].values)
        targets.append(df.iloc[i]["label"])
    X = np.array(sequences)
    y = np.array(targets) + 1

    # Use SAVED split indices (not config — immune to post-training changes)
    if split_idx >= len(X):
        print(f"ERROR: Saved split index ({split_idx}) >= total sequences ({len(X)}). Data changed since training?")
        sys.exit(1)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    if len(X_test) == 0:
        print("ERROR: No test data after split.")
        sys.exit(1)

    # Load scaler saved during training (CRITICAL FIX)
    if not scaler_path.exists():
        print(f"ERROR: Scaler not found at {scaler_path}. Train the model first.")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Load model
    model = LSTMModel(
        input_size=X_test.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=3,
        dropout_rate=dropout,
    )
    if not model_path.exists():
        return {"error": f"Model not found at {model_path}"}
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test_scaled))
        _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.numpy()

    return _compute_metrics(y_test, y_pred, "LSTM", reports_dir, history_path)


def evaluate_xgboost(cfg: Config) -> dict:
    """Evaluate saved XGBoost model using the saved scaler."""
    data_path = cfg.resolve(cfg.get("files.final_processed"))
    model_path = cfg.resolve(cfg.get("files.xgboost_model"))
    scaler_path = cfg.resolve(cfg.get("files.xgboost_scaler"))
    split_path = cfg.resolve(cfg.get("files.xgboost_split"))
    reports_dir = cfg.reports_dir
    window_size = cfg.get("xgboost.window_size")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load split indices saved during training
    if not split_path.exists():
        print(f"ERROR: Split info not found at {split_path}. Train the model first.")
        sys.exit(1)
    with open(split_path) as f:
        split_info = json.load(f)
    split_idx = split_info["n_train"]

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    features, targets = [], []
    for i in range(window_size, len(df)):
        features.append(df.iloc[i - window_size : i][FEATURE_COLUMNS].values.flatten())
        targets.append(df.iloc[i]["label"])
    X = np.array(features)
    y = np.array(targets) + 1

    # Use SAVED split indices (not config)
    if split_idx >= len(X):
        print(f"ERROR: Saved split index ({split_idx}) >= total sequences ({len(X)}). Data changed since training?")
        sys.exit(1)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    if len(X_test) == 0:
        print("ERROR: No test data after split.")
        sys.exit(1)

    if not scaler_path.exists():
        print(f"ERROR: Scaler not found at {scaler_path}. Train the model first.")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_scaled)

    return _compute_metrics(y_test, y_pred, "XGBoost", reports_dir)


# ── Shared metrics ─────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, model_name, reports_dir, history_path=None):
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Save text report
    txt_path = reports_dir / f"classification_metrics_{model_name.lower()}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}\nAccuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print(f"  Report: {txt_path}")

    # Plots
    _plot_confusion_matrix(y_true, y_pred, model_name, reports_dir / f"confusion_matrix_{model_name.lower()}.png")
    _plot_classification_report(report_dict, model_name, reports_dir / f"classification_report_{model_name.lower()}.png")
    _plot_summary_metrics(report_dict, model_name, reports_dir / f"summary_metrics_{model_name.lower()}.png")

    # Training history
    if history_path and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        _plot_lstm_history(history, model_name, reports_dir / f"training_history_{model_name.lower()}.png")

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(report_dict["macro avg"]["precision"], 4),
        "recall_macro": round(report_dict["macro avg"]["recall"], 4),
        "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
    }


def evaluate_all(cfg: Config) -> dict:
    """Evaluate all trained models and write metrics.json."""
    results = {}
    try:
        lstm_result = evaluate_lstm(cfg)
        results["lstm"] = lstm_result
    except (FileNotFoundError, SystemExit) as e:
        print(f"LSTM eval skipped: {e}")

    try:
        xgb_result = evaluate_xgboost(cfg)
        results["xgboost"] = xgb_result
    except (FileNotFoundError, SystemExit) as e:
        print(f"XGBoost eval skipped: {e}")

    if results:
        metrics_path = cfg.resolve(cfg.get("files.metrics"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved: {metrics_path}")
    else:
        print("No models evaluated.")
        sys.exit(1)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", choices=["lstm", "xgboost", "all"], default="all")
    args = parser.parse_args()

    if args.model == "lstm":
        evaluate_lstm(cfg)
    elif args.model == "xgboost":
        evaluate_xgboost(cfg)
    else:
        evaluate_all(cfg)
