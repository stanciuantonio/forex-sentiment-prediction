import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path

def process_final_dataset():
    """
    Processes gdelt_eurusd_with_sentiment.csv and adds necessary columns
    """
    # Load your current file
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "processed" / "gdelt_eurusd_with_sentiment.csv"
    output_file = project_root / "data" / "processed" / "eurusd_final_processed.csv"

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Calculate log_return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # --- Base Features (from user) ---
    df['sentiment_7d_mean'] = df['gdelt_sentiment'].rolling(window=7).mean()
    df['log_return_7d_mean'] = df['log_return'].rolling(window=7).mean()
    df['log_return_7d_std'] = df['log_return'].rolling(window=7).std()
    df['close_30d_ma'] = df['close'].rolling(window=30).mean()
    df['close_30d_std'] = df['close'].rolling(window=30).std()
    df['daily_range'] = df['high'] - df['low']
    df['open_close_change'] = df['close'] - df['open']

    # --- New Feature Engineering ---
    # 1. Momentum and Oscillators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)

    # 2. Volatility and Ranges
    df.ta.atr(length=14, append=True)
    df['bb_pos'] = (df['close'] - df['close_30d_ma']) / df['close_30d_std']

    # 3. Advanced Sentiment Features
    df['sentiment_delta'] = df['gdelt_sentiment'] - df['sentiment_7d_mean']
    df['sentiment_7d_std'] = df['gdelt_sentiment'].rolling(window=7).std()

    # 4. Price and Sentiment Interactions
    return_sign = np.sign(df['log_return'].rolling(window=30).mean())
    sentiment_sign = np.sign(df['gdelt_sentiment'].rolling(window=30).mean())
    df['confluence'] = (return_sign == sentiment_sign).astype(int)
    df['return_x_sentiment'] = df['log_return'] * df['gdelt_sentiment']

    # Calculate fwd_return (for labels)
    df['fwd_return'] = np.log(df['close'].shift(-1) / df['close'])

    # Create BUY/SELL/HOLD labels
    THRESH = 0.002  # 0.2%
    conditions = [
        df['fwd_return'] > THRESH,   # BUY = 1
        df['fwd_return'] < -THRESH   # SELL = -1
    ]
    df['label'] = np.select(conditions, [1, -1], default=0)  # HOLD = 0

    # Label distribution
    label_dist = df['label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    print(f"SELL (-1): {label_dist.get(-1, 0)} ({label_dist.get(-1, 0)/len(df)*100:.1f}%)")
    print(f"HOLD ( 0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df)*100:.1f}%)")
    print(f"BUY  (+1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df)*100:.1f}%)")

    # Clean NaNs - this will also remove rows from the start where rolling features couldn't be computed
    df_clean = df.dropna()

    # Save final file
    df_clean.to_csv(output_file)
    print(f"\n Processed file with new features saved in: {output_file}")

    return df_clean

if __name__ == '__main__':
    df_final = process_final_dataset()
