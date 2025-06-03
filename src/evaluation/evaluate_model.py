import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset # Required for LSTM data loading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Import LSTMModel class from the training script
# Assuming lstm.py is in ../models relative to this script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from lstm import LSTMModel # This imports the class definition

DEFAULT_DATA_PATH = '../../data/processed/eurusd_final_processed.csv'
DEFAULT_REPORTS_DIR = '../../results/reports'
DEFAULT_WINDOW_SIZE = 30 # Should match the window size used during training for LSTM

def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def plot_classification_report(report_dict, model_name, save_path):
    plt.figure(figsize=(10, 6))
    # Transpose for better plotting
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T # Exclude support row for heatmap clarity
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Classification Report - {model_name}')
    plt.savefig(save_path)
    plt.close()
    print(f"Classification report plot saved to {save_path}")

def evaluate_model(model_path, model_type, data_path, reports_dir, window_size):
    os.makedirs(reports_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- Data Preprocessing --- (This needs to be aligned with how training data was split and scaled)
    # For simplicity, we'll re-split and scale here. Ideally, test set and scaler would be saved from training.

    # Labels for classification report
    class_names = ['SELL', 'HOLD', 'BUY']
    y_true_original_labels = df.iloc[window_size:]['label'].values # Store original -1, 0, 1 labels if needed for direct comparison or specific metrics
    y_pytorch = df.iloc[window_size:]['label'].values + 1 # For PyTorch/XGBoost 0,1,2 labels

    if model_type == 'lstm':
        # Create sequences for LSTM
        sequences = []
        if len(df) <= window_size:
            print(f"Error: DataFrame has insufficient data for LSTM window size {window_size}.")
            return
        for i in range(window_size, len(df)):
            sequence = df.iloc[i-window_size:i][['log_return', 'gdelt_sentiment']].values
            sequences.append(sequence)
        X_eval = np.array(sequences)

        if X_eval.size == 0:
            print("Error: No sequences created for LSTM evaluation.")
            return

        # Temporal split to get a consistent test set portion (e.g., last 15% as in training)
        # This assumes the evaluation is on the same "test set" portion as seen in training
        # A more robust way would be to save the test set indices or the test set itself from training
        test_ratio = 0.15
        split_idx_test_eval = int(len(X_eval) * (1 - test_ratio))
        X_eval_test_portion = X_eval[split_idx_test_eval:]
        y_eval_test_portion = y_pytorch[split_idx_test_eval:]

        if X_eval_test_portion.size == 0:
            print("Error: Test portion for LSTM is empty after split.")
            return

        # Scale features - IMPORTANT: Use a scaler fitted on TRAINING data.
        # For this standalone script, we refit on the train_val part of this data for consistency demonstration.
        # Ideally, the scaler from the training phase should be saved and loaded.
        scaler = StandardScaler()
        # Fit scaler on the training part of the *current* full dataset to simulate unseen test data.
        # Create a temporary training set from the df to fit the scaler correctly
        temp_train_sequences = []
        for i in range(window_size, split_idx_test_eval + window_size): # up to the start of the test portion
             temp_sequence = df.iloc[i-window_size:i][['log_return', 'gdelt_sentiment']].values
             temp_train_sequences.append(temp_sequence)

        if not temp_train_sequences:
            print("Error: Not enough data to fit scaler for LSTM.")
            return

        X_temp_train_val = np.array(temp_train_sequences)
        scaler.fit(X_temp_train_val.reshape(-1, X_temp_train_val.shape[-1]))

        X_eval_scaled = scaler.transform(X_eval_test_portion.reshape(-1, X_eval_test_portion.shape[-1]))
        X_eval_scaled = X_eval_scaled.reshape(X_eval_test_portion.shape)
        X_eval_tensor = torch.FloatTensor(X_eval_scaled)

        # Load LSTM model
        # Determine input_size, hidden_size, num_layers, num_classes from saved model or script defaults
        # This is a simplification; a more robust way is to save model config with the model
        model = LSTMModel(input_size=X_eval_tensor.shape[-1], num_classes=len(class_names)) # Use defaults from lstm.py for hidden_size, num_layers for now
        try:
            model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"Error: LSTM model file not found at {model_path}")
            return
        except Exception as e:
            print(f"Error loading LSTM model state: {e}")
            return
        model.eval()
        with torch.no_grad():
            outputs = model(X_eval_tensor)
            _, y_pred = torch.max(outputs.data, 1)
            y_pred = y_pred.numpy()
        y_true = y_eval_test_portion

    elif model_type == 'xgboost':
        # Create features for XGBoost
        features_eval = []
        if len(df) <= window_size:
            print(f"Error: DataFrame has insufficient data for XGBoost window size {window_size}.")
            return
        for i in range(window_size, len(df)):
            feature_row = df.iloc[i-window_size:i][['log_return', 'gdelt_sentiment']].values.flatten()
            features_eval.append(feature_row)
        X_eval = np.array(features_eval)

        if X_eval.size == 0:
            print("Error: No features created for XGBoost evaluation.")
            return

        # Temporal split (e.g., last 20% as in XGBoost training)
        split_idx_eval = int(0.8 * len(X_eval))
        X_eval_test_portion = X_eval[split_idx_eval:]
        y_eval_test_portion = y_pytorch[split_idx_eval:]

        if X_eval_test_portion.size == 0:
            print("Error: Test portion for XGBoost is empty after split.")
            return

        # Scale features (similar to LSTM, ideally load scaler from training)
        scaler = StandardScaler()
        X_temp_train = X_eval[:split_idx_eval] # Use the train part of this eval data to fit scaler
        if X_temp_train.size == 0:
            print("Error: Not enough data to fit scaler for XGBoost.")
            return

        scaler.fit(X_temp_train)
        X_eval_scaled = scaler.transform(X_eval_test_portion)

        # Load XGBoost model
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: XGBoost model file not found at {model_path}")
            return
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            return
        y_pred = model.predict(X_eval_scaled)
        y_true = y_eval_test_portion
    else:
        print(f"Error: Unknown model_type '{model_type}'. Choose 'lstm' or 'xgboost'.")
        return

    # --- Metrics and Reporting --- #
    accuracy = accuracy_score(y_true, y_pred)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    print(f"\n--- Evaluation Results for {model_type.upper()} ({os.path.basename(model_path)}) ---")
    print(f"Data Source: {data_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Text):")
    print(report_str)

    # Save text report
    report_txt_path = os.path.join(reports_dir, f"classification_metrics_{model_type}.txt")
    with open(report_txt_path, 'w') as f:
        f.write(f"Model: {model_type.upper()} ({os.path.basename(model_path)})\n")
        f.write(f"Data Source: {data_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report_str)
    print(f"Text classification report saved to {report_txt_path}")

    # Save plots
    cm_plot_path = os.path.join(reports_dir, f"confusion_matrix_{model_type}.png")
    plot_confusion_matrix(y_true, y_pred, class_names, model_type.upper(), cm_plot_path)

    report_plot_path = os.path.join(reports_dir, f"classification_report_{model_type}.png")
    plot_classification_report(report_dict, model_type.upper(), report_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Forex prediction model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.h5 for LSTM, .joblib for XGBoost)')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'xgboost'], help='Type of the model to evaluate')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help=f'Path to the evaluation data CSV file (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--reports_dir', type=str, default=DEFAULT_REPORTS_DIR, help=f'Directory to save evaluation reports and plots (default: {DEFAULT_REPORTS_DIR})')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE, help=f'Lookback window size used during training (default: {DEFAULT_WINDOW_SIZE})')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        window_size=args.window_size
    )
