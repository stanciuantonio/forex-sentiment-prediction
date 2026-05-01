"""
Feature engineering — compute technical indicators, sentiment features, and target labels.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path

from src.config import Config


def build_features(
    input_path: str | Path,
    output_path: str | Path,
    label_threshold: float = 0.002,
) -> pd.DataFrame:
    """
    Load merged price+sentiment data, compute all features and labels.

    Features:
      - Base: log_return, gdelt_sentiment
      - Rolling: sentiment_7d_mean, log_return_7d_{mean,std}
      - Price: close_30d_{ma,std}, daily_range, open_close_change
      - Technical: RSI_14, MACD_12_26_9, ATRr_14, bb_pos
      - Sentiment: delta, 7d_std, confluence, return_x_sentiment

    Labels (3-class): SELL=-1, HOLD=0, BUY=1  (based on fwd_return threshold)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Base rolling features
    df["sentiment_7d_mean"] = df["gdelt_sentiment"].rolling(window=7).mean()
    df["log_return_7d_mean"] = df["log_return"].rolling(window=7).mean()
    df["log_return_7d_std"] = df["log_return"].rolling(window=7).std()
    df["close_30d_ma"] = df["close"].rolling(window=30).mean()
    df["close_30d_std"] = df["close"].rolling(window=30).std()
    df["daily_range"] = df["high"] - df["low"]
    df["open_close_change"] = df["close"] - df["open"]

    # Technical indicators (via pandas_ta)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.atr(length=14, append=True)
    df["bb_pos"] = (df["close"] - df["close_30d_ma"]) / df["close_30d_std"]

    # Advanced sentiment
    df["sentiment_delta"] = df["gdelt_sentiment"] - df["sentiment_7d_mean"]
    df["sentiment_7d_std"] = df["gdelt_sentiment"].rolling(window=7).std()

    # Price × sentiment interactions
    return_sign = np.sign(df["log_return"].rolling(window=30).mean())
    sentiment_sign = np.sign(df["gdelt_sentiment"].rolling(window=30).mean())
    df["confluence"] = (return_sign == sentiment_sign).astype(int)
    df["return_x_sentiment"] = df["log_return"] * df["gdelt_sentiment"]

    # --- Target labels ---
    df["fwd_return"] = np.log(df["close"].shift(-1) / df["close"])
    conditions = [
        df["fwd_return"] > label_threshold,    # BUY
        df["fwd_return"] < -label_threshold,   # SELL
    ]
    df["label"] = np.select(conditions, [1, -1], default=0)  # HOLD

    # Report distribution
    dist = df["label"].value_counts().sort_index()
    total = len(df)
    print(f"\nLabel distribution:")
    print(f"  SELL (-1): {dist.get(-1, 0):>5} ({dist.get(-1, 0) / total * 100:.1f}%)")
    print(f"  HOLD  (0): {dist.get(0, 0):>5} ({dist.get(0, 0) / total * 100:.1f}%)")
    print(f"  BUY  (+1): {dist.get(1, 0):>5} ({dist.get(1, 0) / total * 100:.1f}%)")

    # Drop rows with NaN from rolling windows
    df_clean = df.dropna()
    df_clean.to_csv(output_path)
    print(f"\nFeatures saved: {output_path} ({len(df_clean):,} rows)")
    return df_clean


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Build features and labels")
    parser.add_argument("--input", default=str(cfg.resolve(cfg.get("files.sentiment_merged"))))
    parser.add_argument("--output", default=str(cfg.resolve(cfg.get("files.final_processed"))))
    parser.add_argument("--threshold", type=float, default=cfg.get("data.label_threshold"))
    args = parser.parse_args()

    build_features(args.input, args.output, args.threshold)
