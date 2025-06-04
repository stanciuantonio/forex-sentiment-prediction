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
import json # Added for loading history

# Import LSTMModel class from the training script
# Assuming lstm.py is in ../models relative to this script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from lstm import LSTMModel, DEFAULT_HIDDEN_SIZE as LSTM_DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS as LSTM_DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT as LSTM_DEFAULT_DROPOUT # Import defaults

DEFAULT_DATA_PATH = 'data/processed/eurusd_final_processed.csv'
DEFAULT_REPORTS_DIR = 'results/reports'
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

def plot_classification_report(report_dict, class_names, model_name, save_path):
    # Construct a DataFrame for plotting only per-class precision, recall, and f1-score
    plot_data = {}
    for cls_name in class_names:
        if cls_name in report_dict and isinstance(report_dict[cls_name], dict):
            plot_data[cls_name] = {
                'precision': report_dict[cls_name].get('precision', 0),
                'recall': report_dict[cls_name].get('recall', 0),
                'f1-score': report_dict[cls_name].get('f1-score', 0)
            }

    if not plot_data:
        print("Warning: No class-specific data found in classification report for plotting heatmap.")
        # Still try to save a blank or minimal plot if needed, or just return
        # For now, let's return if no data to prevent error with pd.DataFrame(plot_data)
        return

    report_df = pd.DataFrame(plot_data).T # Transpose to have classes as rows

    plt.figure(figsize=(8, max(3, len(class_names) * 0.8))) # Adjusted figure size for clarity
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=1) # Set vmin/vmax for consistency
    plt.title(f'Classification Report (Per-Class Metrics) - {model_name}')
    # Explicitly set y-axis labels if needed, though .T should handle it with DataFrame index
    # plt.yticks(ticks=np.arange(len(report_df.index)) + 0.5, labels=report_df.index, rotation=0)
    plt.tight_layout() # Adjust layout
    plt.savefig(save_path)
    plt.close()
    print(f"Per-class classification report plot saved to {save_path}")

def plot_summary_classification_metrics(report_dict, model_name, save_path):
    metrics_to_plot = {}
    if 'accuracy' in report_dict:
        metrics_to_plot['Accuracy'] = report_dict['accuracy']

    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in report_dict[avg_type]:
                    metrics_to_plot[f'{avg_type.replace(" ", "_").capitalize()}_{metric.capitalize()}'] = report_dict[avg_type][metric]

    if not metrics_to_plot:
        print("Warning: No summary metrics found in classification report for plotting.")
        return

    metrics_df = pd.DataFrame(list(metrics_to_plot.items()), columns=['Metric', 'Score'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Score', y='Metric', data=metrics_df, palette='mako', orient='h')
    plt.title(f'Summary Classification Metrics - {model_name}')
    plt.xlabel('Score')
    plt.ylabel('')
    plt.xlim(0, 1) # Scores are between 0 and 1

    # Add text annotations to the bars
    for i, v in enumerate(metrics_df['Score']):
        ax.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Summary classification metrics plot saved to {save_path}")

def plot_lstm_training_history(history_data, model_name, save_path):
    epochs_trained = history_data.get('epochs_trained', len(history_data.get('train_loss', [])))
    epochs_range = range(1, epochs_trained + 1)

    plt.figure(figsize=(10, 6))
    if 'train_loss' in history_data:
        plt.plot(epochs_range, history_data['train_loss'][:epochs_trained], label='Training Loss', marker='o', linestyle='-')
    if 'val_loss' in history_data:
        plt.plot(epochs_range, history_data['val_loss'][:epochs_trained], label='Validation Loss', marker='x', linestyle='--')

    plt.title(f'LSTM Training and Validation Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"LSTM training history plot saved to {save_path}")

def plot_xgboost_training_history(history_data, model_name, save_path):
    # XGBoost history is typically like: {'validation_0': {'mlogloss': [values]}}
    # Assuming one eval set, typically named 'validation_0' or similar by default.
    eval_set_key = next(iter(history_data.keys()), None) # Get the first (and likely only) eval set key
    if not eval_set_key or not isinstance(history_data[eval_set_key], dict):
        print("Could not find evaluation metric data in XGBoost history.")
        return

    metric_key = next(iter(history_data[eval_set_key].keys()), None) # Get the first metric (e.g., 'mlogloss')
    if not metric_key:
        print(f"Could not find metric data under {eval_set_key} in XGBoost history.")
        return

    metric_values = history_data[eval_set_key][metric_key]
    iterations = range(1, len(metric_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metric_values, label=f'{metric_key} ({eval_set_key})', marker='o', linestyle='-')
    plt.title(f'XGBoost Training History ({metric_key}) - {model_name}')
    plt.xlabel('Boosting Round')
    plt.ylabel(metric_key.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"XGBoost training history plot saved to {save_path}")

def evaluate_model(model_path, model_type, data_path, reports_dir, window_size, lstm_hidden_size, lstm_num_layers, lstm_dropout_rate, training_history_path=None):
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
        model = LSTMModel(
            input_size=X_eval_tensor.shape[-1],
            num_classes=len(class_names),
            hidden_size=lstm_hidden_size,      # Use passed argument
            num_layers=lstm_num_layers,        # Use passed argument
            dropout_rate=lstm_dropout_rate     # Use passed argument
        )
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Added map_location for CPU loading
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
    plot_classification_report(report_dict, class_names, model_type.upper(), report_plot_path)

    # --- Plot Summary Metrics ---
    summary_metrics_plot_path = os.path.join(reports_dir, f"summary_metrics_plot_{model_type}.png")
    plot_summary_classification_metrics(report_dict, model_type.upper(), summary_metrics_plot_path)

    # --- Plot Training History (if path provided) ---
    if training_history_path:
        try:
            with open(training_history_path, 'r') as f:
                history_data = json.load(f)

            model_name_for_plot = f"{model_type.upper()} ({os.path.basename(model_path)})"
            history_plot_save_path = os.path.join(reports_dir, f"training_history_{model_type}.png")

            if model_type == 'lstm':
                plot_lstm_training_history(history_data, model_name_for_plot, history_plot_save_path)
            elif model_type == 'xgboost':
                plot_xgboost_training_history(history_data, model_name_for_plot, history_plot_save_path)

        except FileNotFoundError:
            print(f"Warning: Training history file not found at {training_history_path}. Skipping history plot.")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from training history file {training_history_path}. Skipping history plot.")
        except Exception as e:
            print(f"Warning: Could not plot training history due to an error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Forex prediction model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.h5 for LSTM, .joblib for XGBoost)')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'xgboost'], help='Type of the model to evaluate')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help=f'Path to the evaluation data CSV file (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--reports_dir', type=str, default=DEFAULT_REPORTS_DIR, help=f'Directory to save evaluation reports and plots (default: {DEFAULT_REPORTS_DIR})')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE, help=f'Lookback window size used during training (default: {DEFAULT_WINDOW_SIZE})')
    # LSTM-specific architecture arguments
    parser.add_argument('--lstm_hidden_size', type=int, default=LSTM_DEFAULT_HIDDEN_SIZE, help='Hidden size of LSTM layers (used if model_type is lstm)')
    parser.add_argument('--lstm_num_layers', type=int, default=LSTM_DEFAULT_NUM_LAYERS, help='Number of LSTM layers (used if model_type is lstm)')
    parser.add_argument('--lstm_dropout_rate', type=float, default=LSTM_DEFAULT_DROPOUT, help='Dropout rate for LSTM (used if model_type is lstm)')
    parser.add_argument('--training_history_path', type=str, default=None, help='Optional path to the training history JSON file (e.g., loss_history.json)')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        window_size=args.window_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout_rate=args.lstm_dropout_rate,
        training_history_path=args.training_history_path
    )
