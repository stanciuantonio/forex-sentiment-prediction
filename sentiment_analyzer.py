"""
Sentiment analyzer module for financial news sentiment scoring and data merging.

This module processes cleaned GDELT news articles using FinBERT to calculate
sentiment scores, aggregates them daily, and merges with forex price data
to create the final dataset for machine learning.
"""

import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse

# Configuration constants
PROCESSED_DATA_DIR = "data/processed"
RAW_DATA_DIR = "data/raw"
CLEANED_ARTICLES_FILE = "data/processed/gdelt_articles_text_cleaned.csv"
FOREX_PRICE_FILE = "data/raw/eurusd_daily.csv"
FINAL_SENTIMENT_FILE = "data/processed/gdelt_eurusd_with_sentiment.csv"
FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 512
SENTIMENT_BATCH_SIZE = 16

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def calculate_finbert_sentiment_scores(texts, batch_size=SENTIMENT_BATCH_SIZE):
    """
    Calculate sentiment scores for a list of texts using FinBERT.

    Args:
        texts: List of text strings to analyze
        batch_size (int): Batch size for processing

    Returns:
        List[float]: List of sentiment scores (positive - negative)
    """
    if not texts:
        print("No texts provided for sentiment analysis")
        return []

    # Load FinBERT model and tokenizer
    print(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        model.eval()  # Set to evaluation mode
        print("FinBERT model loaded successfully")
    except Exception as e:
        print(f"Failed to load FinBERT model: {e}")
        raise

    sentiments = []

    print(f"Calculating sentiment scores for {len(texts)} texts")

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiments"):
        batch_texts = texts[i:i+batch_size]

        # Ensure all texts are strings and handle NaN values
        batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SEQUENCE_LENGTH
        )

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        # Calculate sentiment score as: positive - negative
        batch_sentiments = (predictions[:, 0] - predictions[:, 1]).cpu().tolist()
        sentiments.extend(batch_sentiments)

    print(f"Calculated {len(sentiments)} sentiment scores")
    return sentiments

def load_and_validate_gdelt_data(file_path: str):
    """
    Load and validate GDELT news data.

    Args:
        file_path (str): Path to cleaned GDELT CSV file

    Returns:
        pd.DataFrame: Validated GDELT data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GDELT data file not found at {file_path}")

    print(f"Loading GDELT data from {file_path}")
    gdelt_df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = ['text', 'date']
    missing_columns = [col for col in required_columns if col not in gdelt_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Clean and prepare data
    gdelt_df = gdelt_df.dropna(subset=['text'])
    gdelt_df['text'] = gdelt_df['text'].astype(str)
    gdelt_df['date'] = pd.to_datetime(gdelt_df['date']).dt.date

    print(f"Loaded and validated {len(gdelt_df)} GDELT articles")
    return gdelt_df

def aggregate_daily_sentiment(gdelt_df):
    """
    Aggregate sentiment scores by date.

    Args:
        gdelt_df (pd.DataFrame): GDELT data with sentiment scores

    Returns:
        pd.DataFrame: Daily aggregated sentiment data
    """
    if 'sentiment_score' not in gdelt_df.columns or gdelt_df['sentiment_score'].isna().all():
        print("No valid sentiment scores found for aggregation")
        return pd.DataFrame(columns=['date', 'gdelt_sentiment'])

    print("Aggregating sentiment scores by date")
    daily_sentiment = (gdelt_df.groupby('date')['sentiment_score']
                              .mean()
                              .reset_index())
    daily_sentiment.rename(columns={'sentiment_score': 'gdelt_sentiment'}, inplace=True)

    # Ensure date is in correct format for merging
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date

    print(f"Aggregated sentiment data for {len(daily_sentiment)} days")
    return daily_sentiment

def load_and_prepare_price_data(file_path: str):
    """
    Load and prepare forex price data.

    Args:
        file_path (str): Path to forex price CSV file

    Returns:
        pd.DataFrame: Prepared price data indexed by date
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Price data file not found at {file_path}")

    print(f"Loading price data from {file_path}")
    price_df = pd.read_csv(file_path)

    if price_df.empty:
        raise ValueError(f"Price data file at {file_path} is empty")

    # Standardize date column name and format
    price_df.rename(columns={price_df.columns[0]: 'date_time'}, inplace=True)
    price_df['date'] = pd.to_datetime(price_df['date_time']).dt.date
    price_df.set_index('date', inplace=True)

    # Keep only OHLC columns
    ohlc_columns = ['open', 'high', 'low', 'close']
    available_ohlc = [col for col in ohlc_columns if col in price_df.columns]
    price_df = price_df[available_ohlc]

    print(f"Loaded price data for {len(price_df)} days with columns: {available_ohlc}")
    return price_df

def merge_sentiment_and_price_data(price_df, sentiment_df):
    """
    Merge sentiment and price data on date.

    Args:
        price_df (pd.DataFrame): Price data indexed by date
        sentiment_df (pd.DataFrame): Daily sentiment data

    Returns:
        pd.DataFrame: Merged dataset with price and sentiment data
    """
    print("Merging price data with sentiment scores")

    if sentiment_df.empty:
        print("Sentiment data is empty. Creating dataset with zero sentiment")
        final_df = price_df.copy()
        final_df['gdelt_sentiment'] = 0.0
    else:
        # Set date as index for sentiment data
        sentiment_df.set_index('date', inplace=True)

        # Perform inner join to keep only overlapping dates
        final_df = price_df.join(sentiment_df, how='inner')

        # Fill any remaining NaN sentiment values with 0
        if 'gdelt_sentiment' in final_df.columns:
            final_df['gdelt_sentiment'].fillna(0.0, inplace=True)

    print(f"Merged dataset contains {len(final_df)} trading days")
    return final_df

def save_final_dataset(df, output_path: str):
    """
    Save final merged dataset to CSV.

    Args:
        df (pd.DataFrame): Final dataset to save
        output_path (str): Path to save the CSV file
    """
    if df.empty:
        print("Final DataFrame is empty. No data to save.")
        return

    # Reset index to save date as column
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)

    print(f"Final dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Display sentiment statistics if available
    if 'gdelt_sentiment' in df.columns:
        sentiment_stats = df['gdelt_sentiment'].describe()
        non_zero_days = len(df[df['gdelt_sentiment'] != 0.0])

        print("=== SENTIMENT STATISTICS ===")
        print(f"Sentiment range: {sentiment_stats['min']:.4f} to {sentiment_stats['max']:.4f}")
        print(f"Mean sentiment: {sentiment_stats['mean']:.4f}")
        print(f"Days with non-zero sentiment: {non_zero_days}/{len(df)}")

# ------------- run -------------
if __name__ == '__main__':
    print("Starting sentiment analysis pipeline")

    try:
        # Load and process GDELT data
        gdelt_df = load_and_validate_gdelt_data(CLEANED_ARTICLES_FILE)

        # Calculate sentiment scores
        if not gdelt_df.empty and 'text' in gdelt_df.columns:
            texts_to_score = gdelt_df['text'].tolist()
            sentiment_scores = calculate_finbert_sentiment_scores(texts_to_score)
            gdelt_df['sentiment_score'] = sentiment_scores
        else:
            print("No text data available for sentiment analysis")
            gdelt_df['sentiment_score'] = np.nan

        # Aggregate daily sentiment
        daily_sentiment = aggregate_daily_sentiment(gdelt_df)

        # Load price data
        price_df = load_and_prepare_price_data(FOREX_PRICE_FILE)

        # Merge sentiment and price data
        final_df = merge_sentiment_and_price_data(price_df, daily_sentiment)

        # Save final dataset
        save_final_dataset(final_df, FINAL_SENTIMENT_FILE)

        print("Sentiment analysis pipeline completed successfully")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise
