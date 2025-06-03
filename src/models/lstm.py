import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse

# Define constants for default values
DEFAULT_DATA_PATH = 'data/processed/eurusd_final_processed.csv'
DEFAULT_MODEL_SAVE_PATH = 'results/models/lstm_model.h5'
DEFAULT_EPOCHS = 100 # Reduced default for quicker runs, can be overridden
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_WINDOW_SIZE = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 15

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=DEFAULT_HIDDEN_SIZE, num_layers=DEFAULT_NUM_LAYERS, num_classes=3, dropout_rate=DEFAULT_DROPOUT):
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

def train_lstm(data_path, model_save_path, epochs, batch_size, learning_rate, hidden_size, num_layers, dropout_rate, window_size, early_stopping_patience):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    sequences = []
    targets = []

    if len(df) <= window_size:
        print(f"Error: DataFrame has insufficient data (rows: {len(df)}) to create sequences with window size {window_size}.")
        return None

    for i in range(window_size, len(df)):
        window = df.iloc[i-window_size:i]
        sequence = window[['log_return', 'gdelt_sentiment']].values
        sequences.append(sequence)
        targets.append(df.iloc[i]['label'])

    if not sequences:
        print("Error: No sequences were created. Check window_size and data length.")
        return None

    X = np.array(sequences)
    y = np.array(targets)
    y = y + 1

    train_val_ratio = 0.85
    test_ratio = 0.15
    split_idx_test = int(len(X) * (1 - test_ratio))
    X_train_val, X_test = X[:split_idx_test], X[split_idx_test:]
    y_train_val, y_test = y[:split_idx_test], y[split_idx_test:]

    if len(X_train_val) == 0 or len(X_test) == 0:
        print("Error: Not enough data for initial train_val/test split.")
        return None

    val_relative_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, shuffle=False
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("Error: One of the data splits is empty. Adjust ratios or check data size.")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[-1], hidden_size=hidden_size, num_layers=num_layers, num_classes=len(np.unique(y)), dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    min_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                total_val_loss += loss_val.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {early_stopping_patience} epochs.')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("Restored best model weights.")
            break

    if best_model_state is not None :
        model.load_state_dict(best_model_state)
        print("Using best model weights found during training.")
    else:
        print("Warning: No best model state was found or saved. Using last model state.")

    # Save the trained model
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nLSTM Training finished.") # Updated message
    # Evaluation on test set is now handled by evaluate_model.py
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Model for Forex Prediction")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to the processed data CSV file')
    parser.add_argument('--model_save_path', type=str, default=DEFAULT_MODEL_SAVE_PATH, help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for Adam optimizer')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE, help='Number of features in the hidden state h')
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS, help='Number of recurrent layers')
    parser.add_argument('--dropout_rate', type=float, default=DEFAULT_DROPOUT, help='Dropout rate for LSTM and FC layers')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE, help='Size of the lookback window for sequences')
    parser.add_argument('--early_stopping_patience', type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help='Patience for early stopping')

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
        early_stopping_patience=args.early_stopping_patience
    )
