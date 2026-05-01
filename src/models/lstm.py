"""
LSTM model — sequence-based forecasting with configurable architecture.

Key refactoring: scaler is now SAVED after training and LOADED during evaluation,
eliminating data leakage in the evaluation script.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

from src.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def train_lstm(cfg: Config) -> dict:
    """Train LSTM model using parameters from config. Returns training metrics."""
    data_path = cfg.resolve(cfg.get("files.final_processed"))
    model_save_path = cfg.resolve(cfg.get("files.lstm_model"))
    scaler_save_path = cfg.resolve(cfg.get("files.lstm_scaler"))
    split_save_path = cfg.resolve(cfg.get("files.lstm_split"))
    history_save_path = cfg.resolve(cfg.get("files.lstm_history"))

    # Hyperparams
    window_size = cfg.get("lstm.window_size")
    hidden_size = cfg.get("lstm.hidden_size")
    num_layers = cfg.get("lstm.num_layers")
    dropout_rate = cfg.get("lstm.dropout")
    epochs = cfg.get("lstm.epochs")
    batch_size = cfg.get("lstm.batch_size")
    learning_rate = cfg.get("lstm.learning_rate")
    early_stopping_patience = cfg.get("lstm.early_stopping_patience")
    weight_decay = cfg.get("lstm.weight_decay")
    gradient_clip_value = cfg.get("lstm.gradient_clip_value")
    feature_columns = cfg.feature_columns

    # Ensure dirs
    cfg.ensure_dirs()
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data not found at {data_path}")
        return {"error": "data_not_found"}
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna().reset_index(drop=True)

    # Create sequences
    sequences, targets = [], []
    for i in range(window_size, len(df)):
        sequences.append(df.iloc[i - window_size : i][feature_columns].values)
        targets.append(df.iloc[i]["label"])

    X = np.array(sequences)
    y = np.array(targets) + 1  # -1,0,1 → 0,1,2

    # --- Temporal split ---
    train_val_ratio = cfg.get("split.train_val_ratio", 0.85)
    split_idx = int(len(X) * train_val_ratio)
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]

    from sklearn.model_selection import train_test_split
    val_ratio = cfg.get("split.test_ratio", 0.15) / train_val_ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, shuffle=False
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --- Scale (fit ONLY on train) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_scaled = scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)

    # Save scaler for later evaluation
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved: {scaler_save_path}")

    # Save split indices for consistent evaluation
    split_info = {
        "train_val_ratio": train_val_ratio,
        "n_total": len(X),
        "n_train_val": split_idx,
        "n_test": len(X) - split_idx,
        "val_ratio": val_ratio,
    }
    with open(split_save_path, "w") as f:
        json.dump(split_info, f)
    print(f"Split info saved: {split_save_path}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False,
    )

    # --- Model ---
    model = LSTMModel(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(np.unique(y)),
        dropout_rate=dropout_rate,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # --- Training loop ---
    min_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                total_val_loss += criterion(model(batch_X), batch_y).item()

        avg_train = total_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train:.4f} Val: {avg_val:.4f}")

        if avg_val < min_val_loss:
            min_val_loss = avg_val
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # --- Save ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved: {model_save_path}")

    history = {"train_loss": train_losses, "val_loss": val_losses}
    with open(history_save_path, "w") as f:
        json.dump(history, f)
    print(f"History saved: {history_save_path}")

    return {"train_loss": min(train_losses), "val_loss": min(val_losses), "epochs": len(train_losses)}


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    train_lstm(cfg)
