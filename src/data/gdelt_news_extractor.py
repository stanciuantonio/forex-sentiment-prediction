"""
GDELT news extractor module for forex-related news article retrieval.

Extracts and filters news articles from GDELT relevant to EUR/USD forex trading,
using parallel processing for efficiency.
"""

import gdelt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import gc
import logging
import argparse
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.config import Config

# Default configuration (overridable via Config)
GDELT_TABLE_NAME = "gkg"
MAX_ARTICLES_PER_DAY = 5
MAX_PARALLEL_WORKERS = 8

FOREX_KEYWORDS = [
    'EUR USD', 'EURUSD', 'euro dollar', 'european central bank',
    'ECB', 'federal reserve', 'interest rates', 'monetary policy',
    'inflation', 'unemployment', 'GDP', 'economic growth'
]

RELEVANT_COUNTRIES = ['US', 'GM', 'FR', 'IT', 'SP', 'EU']

gdelt_client = gdelt.gdelt(version=1)


def extract_single_day_news(date_str: str, max_articles: int = MAX_ARTICLES_PER_DAY):
    """Extract and filter GDELT news data for a single day."""
    search_query = ' '.join(FOREX_KEYWORDS)

    try:
        results = gdelt_client.Search(
            [date_str], table=GDELT_TABLE_NAME, coverage=True, translation=True
        )
    except Exception as e:
        return f"GDELT search error: {e}"

    if results is None:
        return None

    df = pd.DataFrame(results)

    if df.empty:
        return None

    required_columns = ['url', 'urlarticletitle', 'urltone']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return f"Missing columns: {missing_cols}"

    # Filter by relevant countries
    if 'countrycode' in df.columns:
        df = df[df['countrycode'].isin(RELEVANT_COUNTRIES)]

    # Filter by keywords in title
    title_mask = df['urlarticletitle'].str.contains(
        '|'.join(FOREX_KEYWORDS), case=False, na=False
    )
    df = df[title_mask]

    if df.empty:
        return None

    df = df.head(max_articles)
    return df[['url', 'urlarticletitle', 'urltone', 'date']]


def extract_news_range(
    start_date: str,
    end_date: str,
    max_articles_per_day: int = MAX_ARTICLES_PER_DAY,
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> pd.DataFrame:
    """Extract GDELT news across a date range using parallel workers."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = []
    current = start
    while current <= end:
        date_range.append(current.strftime("%Y %b %d"))
        current += timedelta(days=1)

    logging.info(f"Processing {len(date_range)} days with {max_workers} workers...")

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {
            executor.submit(extract_single_day_news, date_str, max_articles_per_day): date_str
            for date_str in date_range
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_date),
            total=len(date_range),
            desc="Extracting GDELT news",
        ):
            date_str = future_to_date[future]
            try:
                result = future.result()
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_results.append(result)
                elif isinstance(result, str):
                    logging.warning(f"{date_str}: {result}")
            except Exception as e:
                logging.error(f"{date_str}: Exception: {e}")

    if not all_results:
        logging.warning("No articles found for the specified date range.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logging.info(f"Total articles extracted: {len(combined)}")
    return combined


def save_news_data(
    df: pd.DataFrame,
    output_path: str | Path = "data/raw/gdelt_forex_news.csv",
) -> None:
    """Save extracted GDELT news data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"News data saved: {output_path} ({len(df):,} articles)")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description="Extract GDELT news for forex analysis")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--max-per-day", type=int, default=MAX_ARTICLES_PER_DAY)
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL_WORKERS)
    parser.add_argument("--output", default=str(cfg.resolve("data/raw/gdelt_forex_news.csv")))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    df = extract_news_range(args.start_date, args.end_date, args.max_per_day, args.workers)
    if not df.empty:
        save_news_data(df, args.output)
    else:
        print("No articles extracted.")
        sys.exit(1)
