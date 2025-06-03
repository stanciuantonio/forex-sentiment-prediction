import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def train_lstm():
    # Load data
    df = pd.read_csv('../../data/processed/eurusd_final_processed.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create sequences (30-day windows)
    window_size = 30
    sequences = []
    targets = []

    for i in range(window_size, len(df)):
        # Get 30-day window
        window = df.iloc[i-window_size:i]

        # Features: [log_return, gdelt_sentiment] for each day
        sequence = window[['log_return', 'gdelt_sentiment']].values

        sequences.append(sequence)
        targets.append(df.iloc[i]['label'])

    X = np.array(sequences)  # Shape: (samples, 30, 2)
    y = np.array(targets)

    # Convert labels from -1,0,1 to 0,1,2 for PyTorch
    y = y + 1

    # Temporal split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_test_scaled = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_scaled)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 500
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.numpy()

    print("\nLSTM Results:")
    # Print the accuracy
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

    return model

if __name__ == "__main__":
    train_lstm()
