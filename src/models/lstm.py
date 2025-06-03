import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import keras_tuner as kt

# Load and preprocess data
df = pd.read_csv('data/processed/eurusd_final_processed.csv')
df['date'] = pd.to_datetime(df['date'])

features = ['gdelt_sentiment', 'log_return', 'ma_5', 'ma_10', 'momentum_5', 'volatility_5']
print(f'Features length: {len(features)}')

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

sequence_length = 10

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][-1])  # Label column
    return np.array(X), np.array(y)

X, y = create_sequences(df[features + ['label']].values, sequence_length)

# Exclude the label from features
X = X[:, :, :-1]  # Remove the last column (label) from features
y = y + 1  # Convert labels from -1,0,1 to 0,1,2 for LSTM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')

def build_model(hp):
    model = Sequential([
        LSTM(hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=(sequence_length, len(features))),
        Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
        LSTM(hp.Int('units_2', min_value=32, max_value=256, step=32)),
        Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuning',
    project_name='lstm_hyperparameter'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best LSTM Units: {best_hps.get('units')}, Best Dropout: {best_hps.get('dropout')}, Best Optimizer: {best_hps.get('optimizer')}")

model = Sequential([
    LSTM(best_hps.get('units'), return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(best_hps.get('dropout')),
    LSTM(best_hps.get('units_2')),
    Dropout(best_hps.get('dropout_2')),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=best_hps.get('optimizer'), metrics=['accuracy'])

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weights)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

pred = model.predict(X_test)
predicted_labels = np.argmax(pred, axis=1) - 1  # Map back to [-1, 0, 1]

print(classification_report(y_test - 1, predicted_labels))