import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Model and Feature Definitions ---
DEFAULT_DATA_PATH = 'data/processed/eurusd_final_processed.csv'
DEFAULT_MODEL_SAVE_PATH = 'results/models/lstm_model.h5'
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.3
DEFAULT_WINDOW_SIZE = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_VALUE = 1.0

FEATURE_COLUMNS = [
    # Base
    'log_return', 'gdelt_sentiment',
    # User's added features
    'sentiment_7d_mean', 'log_return_7d_mean', 'log_return_7d_std',
    'close_30d_ma', 'close_30d_std', 'daily_range', 'open_close_change',
    # New Features
    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'ATRr_14',
    'bb_pos',
    'sentiment_delta', 'sentiment_7d_std',
    'confluence', 'return_x_sentiment'
]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def train_lstm(data_path, model_save_path, epochs, batch_size, learning_rate, hidden_size, num_layers, dropout_rate, window_size, early_stopping_patience, weight_decay, gradient_clip_value):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # --- Data Loading and Preparation ---
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna().reset_index(drop=True)

    sequences = []
    targets = []
    for i in range(window_size, len(df)):
        sequence = df.iloc[i-window_size:i][FEATURE_COLUMNS].values
        sequences.append(sequence)
        targets.append(df.iloc[i]['label'])

    X = np.array(sequences)
    y = np.array(targets) + 1  # Labels to 0, 1, 2

    # --- Train/Validation/Test Split ---
    train_val_ratio = 0.85
    test_ratio = 0.15
    split_idx_test = int(len(X) * train_val_ratio)

    X_train_val, X_test = X[:split_idx_test], X[split_idx_test:]
    y_train_val, y_test = y[:split_idx_test], y[split_idx_test:]

    val_relative_ratio = test_ratio / train_val_ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, shuffle=False
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    # Note: Test set is not used here, but scaled in evaluate_model.py using the same logic.

    # --- PyTorch DataLoader ---
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Training ---
    model = LSTMModel(input_size=X_train.shape[-1], hidden_size=hidden_size, num_layers=num_layers, num_classes=len(np.unique(y)), dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    min_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                total_val_loss += loss_val.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}.')
            break

    # --- Save Model and History ---
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Restored best model weights.")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    history = {'train_loss': train_losses, 'val_loss': val_losses}
    with open(model_save_path.replace('.h5', '_history.json'), 'w') as f:
        json.dump(history, f)
    print("Training history saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Model for Forex Prediction")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data')
    parser.add_argument('--model_save_path', type=str, default=DEFAULT_MODEL_SAVE_PATH, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--dropout_rate', type=float, default=DEFAULT_DROPOUT)
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument('--early_stopping_patience', type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--gradient_clip_value', type=float, default=DEFAULT_GRADIENT_CLIP_VALUE)

    args = parser.parse_args()

    train_lstm(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        window_size=args.window_size,
        early_stopping_patience=args.early_stopping_patience,
        weight_decay=args.weight_decay,
        gradient_clip_value=args.gradient_clip_value
    )
