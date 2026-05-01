"""
Sentiment analyzer — score financial news with FinBERT, aggregate daily, merge with price data.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.config import Config

FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 512
SENTIMENT_BATCH_SIZE = 16


def calculate_finbert_sentiment_scores(
    texts: list[str],
    batch_size: int = SENTIMENT_BATCH_SIZE,
    model_name: str = FINBERT_MODEL_NAME,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[float]:
    """Calculate sentiment scores (positive - negative) using FinBERT."""
    if not texts:
        print("No texts provided for sentiment analysis")
        return []

    print(f"Loading FinBERT model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        print("FinBERT model loaded successfully")
    except Exception as e:
        print(f"Failed to load FinBERT model: {e}")
        raise

    sentiments = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiments"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]

        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            return_tensors="pt", max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT: [positive, negative, neutral] → score = positive - negative
        batch_sentiments = (predictions[:, 0] - predictions[:, 1]).cpu().tolist()
        sentiments.extend(batch_sentiments)

    print(f"Calculated {len(sentiments)} sentiment scores")
    return sentiments


def load_and_validate_gdelt_data(file_path: str | Path) -> pd.DataFrame:
    """Load and validate GDELT news data."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GDELT data not found: {file_path}")

    print(f"Loading GDELT data from {file_path}")
    df = pd.read_csv(file_path)

    for col in ["text", "date"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"Loaded {len(df)} articles")
    return df


def aggregate_daily_sentiment(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by date."""
    if "sentiment_score" not in gdelt_df.columns or gdelt_df["sentiment_score"].isna().all():
        print("No valid sentiment scores found")
        return pd.DataFrame(columns=["date", "gdelt_sentiment"])

    daily = (
        gdelt_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "gdelt_sentiment"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    print(f"Aggregated {len(daily)} days of sentiment")
    return daily


def load_and_prepare_price_data(file_path: str | Path) -> pd.DataFrame:
    """Load and prepare forex OHLC price data."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Price data not found: {file_path}")

    print(f"Loading price data from {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Price data is empty: {file_path}")

    df.rename(columns={df.columns[0]: "date_time"}, inplace=True)
    df["date"] = pd.to_datetime(df["date_time"]).dt.date
    df.set_index("date", inplace=True)

    ohlc = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    print(f"Loaded {len(df)} days, columns: {ohlc}")
    return df[ohlc]


def merge_sentiment_and_price_data(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sentiment and price data on date."""
    if sentiment_df.empty:
        print("Sentiment empty — using zero sentiment")
        final = price_df.copy()
        final["gdelt_sentiment"] = 0.0
    else:
        sentiment_df = sentiment_df.set_index("date")
        final = price_df.join(sentiment_df, how="inner")
        final["gdelt_sentiment"] = final["gdelt_sentiment"].fillna(0.0)

    print(f"Merged dataset: {len(final)} trading days")
    return final


def run_sentiment_pipeline(
    articles_path: str | Path,
    price_path: str | Path,
    output_path: str | Path,
    finbert_model: str = FINBERT_MODEL_NAME,
    batch_size: int = SENTIMENT_BATCH_SIZE,
) -> pd.DataFrame:
    """Run the full sentiment pipeline: load → score → aggregate → merge → save."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdelt_df = load_and_validate_gdelt_data(articles_path)

    if not gdelt_df.empty:
        scores = calculate_finbert_sentiment_scores(
            gdelt_df["text"].tolist(), batch_size, finbert_model
        )
        gdelt_df["sentiment_score"] = scores
    else:
        gdelt_df["sentiment_score"] = np.nan

    daily_sentiment = aggregate_daily_sentiment(gdelt_df)
    price_df = load_and_prepare_price_data(price_path)
    final_df = merge_sentiment_and_price_data(price_df, daily_sentiment)

    final_df.reset_index(inplace=True)
    final_df.to_csv(output_path, index=False)
    print(f"Final dataset saved: {output_path} ({final_df.shape})")
    return final_df


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Run sentiment analysis pipeline")
    parser.add_argument("--articles", default=str(cfg.resolve(cfg.get("files.cleaned_articles"))))
    parser.add_argument("--forex-price", default=str(cfg.resolve(cfg.get("files.forex_daily"))))
    parser.add_argument("--output", default=str(cfg.resolve(cfg.get("files.sentiment_merged"))))
    parser.add_argument("--finbert-model", default=FINBERT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=SENTIMENT_BATCH_SIZE)
    args = parser.parse_args()

    run_sentiment_pipeline(
        articles_path=args.articles,
        price_path=args.forex_price,
        output_path=args.output,
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
    )
