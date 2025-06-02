"""
Modelo Baseline XGBoost simplificado para predicción EUR/USD
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

def create_flat_features(df, lookback_window=30):
    """
    Crea features aplanados para XGBoost a partir del CSV final

    El CSV tiene: date, open, high, low, close, gdelt_sentiment, log_return, fwd_return, label
    Queremos: [ret1, ret2, ..., ret30, sent1, sent2, ..., sent30] -> label
    """
    print(f"📊 Creando features planos con ventana de {lookback_window} días...")

    feature_cols = ['log_return', 'gdelt_sentiment']

    X_flat = []
    y_labels = []
    dates = []

    # Crear ventanas deslizantes
    for i in range(len(df) - lookback_window):
        start_idx = i
        end_idx = i + lookback_window

        # Ventana de 30 días
        window_data = df[feature_cols].iloc[start_idx:end_idx]

        # Aplanar: [ret1,...,ret30, sent1,...,sent30]
        flat_row = window_data.values.flatten()

        # Etiqueta del ultimo día de la ventana
        label = df['label'].iloc[end_idx]

        if not pd.isna(label):
            X_flat.append(flat_row)
            y_labels.append(int(label))
            dates.append(df.index[end_idx])

    X_flat = np.array(X_flat)
    y_labels = np.array(y_labels)

    print(f"✅ Features creados: {X_flat.shape} (samples, features)")
    print(f"📅 Balance de clases: {np.unique(y_labels, return_counts=True)}")

    return X_flat, y_labels, dates

def main_baseline():
    """Entrenar modelo baseline XGBoost"""
    print("🚀 ENTRENANDO BASELINE XGBoost")

    # 1. Cargar datos
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data/processed/eurusd_final_processed.csv"

    print(f"📁 Cargando datos desde: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index()  # Orden temporal

    print(f"✅ Datos cargados: {len(df)} filas")
    print(f"📅 Período: {df.index.min().date()} a {df.index.max().date()}")

    # 2. Crear features aplanados
    X_flat, y_labels, dates = create_flat_features(df, lookback_window=30)

    # 3. Normalizar features
    scaler = StandardScaler()
    X_flat_norm = scaler.fit_transform(X_flat)

    # 4. División temporal (NO aleatoria)
    # 80% más antiguo para train, 20% más reciente para test
    split_idx = int(len(X_flat_norm) * 0.8)

    X_train = X_flat_norm[:split_idx]
    X_test = X_flat_norm[split_idx:]
    y_train = y_labels[:split_idx]
    y_test = y_labels[split_idx:]
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]

    print(f"📊 División temporal:")
    print(f"  Train: {X_train.shape} ({dates_train[0].date()} a {dates_train[-1].date()})")
    print(f"  Test:  {X_test.shape} ({dates_test[0].date()} a {dates_test[-1].date()})")

    # 5. Entrenar XGBoost
    print("🎯 Entrenando XGBoost...")

    # Convertir etiquetas -1,0,1 a 0,1,2 para XGBoost
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train + 1)  # Convertir a 0,1,2

    # 6. Evaluar
    print("📊 Evaluando modelo...")

    y_pred_xgb = model.predict(X_test)
    y_pred = y_pred_xgb - 1  # Convertir de vuelta a -1,0,1

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                 target_names=['SELL', 'HOLD', 'BUY'],
                                 zero_division=0)

    print(f"\n🎯 RESULTADOS BASELINE")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")

    # 7. Guardar modelo
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'predictions': y_pred,
        'test_dates': dates_test
    }
    joblib.dump(model_data, results_dir / "baseline_xgboost.pkl")
    print(f"💾 Modelo guardado en: {results_dir / 'baseline_xgboost.pkl'}")

    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'predictions': y_pred,
        'test_dates': dates_test,
        'classification_report': report
    }

if __name__ == '__main__':
    results = main_baseline()