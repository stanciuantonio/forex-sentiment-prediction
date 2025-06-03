"""
Modelo Baseline XGBoost simplificado para predicción EUR/USD
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb

def train_baseline():
    # Load data
    df = pd.read_csv('../../data/processed/eurusd_final_processed.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create features from rolling windows (30 days)
    window_size = 30
    features = []
    targets = []

    for i in range(window_size, len(df)):
        # Get 30-day window
        window = df.iloc[i-window_size:i]

        # Features: log_return and gdelt_sentiment for 30 days
        feature_row = []
        for j in range(len(window)):
            feature_row.extend([window.iloc[j]['log_return'], window.iloc[j]['gdelt_sentiment']])

        features.append(feature_row)
        targets.append(df.iloc[i]['label'])

    X = np.array(features)
    y = np.array(targets)

    # Convert labels from -1,0,1 to 0,1,2 for XGBoost
    y = y + 1

    # Temporal split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("XGBoost Baseline Results:")
    # Print the accuracy
    print(f"Accuracy: {model.score(X_test, y_test)}")
    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

    return model

if __name__ == "__main__":
    train_baseline()
