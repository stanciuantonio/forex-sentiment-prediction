"""
Modelo LSTM simplificado para predicción EUR/USD
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configurar semilla
torch.manual_seed(42)
np.random.seed(42)

class LSTMDataset(Dataset):
    """Dataset para secuencias LSTM"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y + 1)  # Convertir -1,0,1 a 0,1,2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    """Red LSTM simple para clasificación"""
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length=30, input_size=2)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Tomar la última salida temporal
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Clasificar
        output = self.classifier(last_output)
        return output

def create_sequences(df, lookback_window=30):
    """
    Crea secuencias temporales para LSTM

    El CSV tiene: date, open, high, low, close, gdelt_sentiment, log_return, fwd_return, label
    Queremos: secuencias 3D (samples, 30, 2) -> label
    """
    print(f"📊 Creando secuencias temporales con ventana de {lookback_window} días...")

    feature_cols = ['log_return', 'gdelt_sentiment']

    X_sequences = []
    y_labels = []
    dates = []

    # Crear ventanas deslizantes
    for i in range(len(df) - lookback_window):
        start_idx = i
        end_idx = i + lookback_window

        # Ventana de 30 días x 2 features
        window_data = df[feature_cols].iloc[start_idx:end_idx].values

        # Etiqueta del día siguiente
        label = df['label'].iloc[end_idx]

        if not pd.isna(label):
            X_sequences.append(window_data)
            y_labels.append(int(label))
            dates.append(df.index[end_idx])

    X_sequences = np.array(X_sequences)
    y_labels = np.array(y_labels)

    print(f"✅ Secuencias creadas: {X_sequences.shape} (samples, timesteps, features)")
    print(f"📅 Balance de clases: {np.unique(y_labels, return_counts=True)}")

    return X_sequences, y_labels, dates

def normalize_sequences(X_sequences):
    """Normaliza las secuencias manteniendo estructura temporal"""
    original_shape = X_sequences.shape

    # Reshape para normalizar
    X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])

    # Normalizar
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)

    # Volver a forma original
    X_sequences_norm = X_normalized.reshape(original_shape)

    print("🔧 Secuencias normalizadas")
    return X_sequences_norm, scaler

def train_lstm(model, train_loader, epochs=100, lr=0.001):
    """Entrena el modelo LSTM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"🎯 Entrenando LSTM en {device} por {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_lstm(model, test_loader):
    """Evalúa el modelo LSTM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_true = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

    # Convertir de vuelta a -1,0,1
    y_pred = np.array(all_predictions) - 1
    y_true = np.array(all_true) - 1

    return y_pred, y_true

def main_lstm():
    """Entrenar modelo LSTM"""
    print("🚀 ENTRENANDO LSTM")

    # 1. Cargar datos
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data/processed/eurusd_final_processed.csv"

    print(f"📁 Cargando datos desde: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    print(f"✅ Datos cargados: {len(df)} filas")
    print(f"📅 Período: {df.index.min().date()} a {df.index.max().date()}")

    # 2. Crear secuencias temporales
    X_sequences, y_labels, dates = create_sequences(df, lookback_window=30)

    # 3. Normalizar secuencias
    X_sequences_norm, scaler = normalize_sequences(X_sequences)

    # 4. División temporal (NO aleatoria)
    split_idx = int(len(X_sequences_norm) * 0.8)

    X_train = X_sequences_norm[:split_idx]
    X_test = X_sequences_norm[split_idx:]
    y_train = y_labels[:split_idx]
    y_test = y_labels[split_idx:]
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]

    print(f"📊 División temporal:")
    print(f"  Train: {X_train.shape} ({dates_train[0].date()} a {dates_train[-1].date()})")
    print(f"  Test:  {X_test.shape} ({dates_test[0].date()} a {dates_test[-1].date()})")

    # 5. Crear datasets y dataloaders
    train_dataset = LSTMDataset(X_train, y_train)
    test_dataset = LSTMDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Crear y entrenar modelo
    model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=2, num_classes=3)
    train_lstm(model, train_loader, epochs=100, lr=0.001)

    # 7. Evaluar
    print("📊 Evaluando modelo...")
    y_pred, y_true = evaluate_lstm(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                 target_names=['SELL', 'HOLD', 'BUY'],
                                 zero_division=0)

    print(f"\n🎯 RESULTADOS LSTM")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")

    # 8. Guardar modelo
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'accuracy': accuracy,
        'predictions': y_pred,
        'test_dates': dates_test
    }

    torch.save(model_data, results_dir / "lstm_model.pth")
    print(f"💾 Modelo guardado en: {results_dir / 'lstm_model.pth'}")

    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'predictions': y_pred,
        'test_dates': dates_test,
        'classification_report': report
    }

if __name__ == '__main__':
    results = main_lstm()
