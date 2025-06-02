"""
Forex data fetcher module for retrieving currency pair data from Alpha Vantage API.

This module provides functionality to fetch daily OHLC data for forex currency pairs
and save them to CSV files for further analysis.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Configuration constants
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
RAW_DATA_DIR = "../../../data/raw"
REQUEST_TIMEOUT_SECONDS = 30
RATE_LIMIT_DELAY_SECONDS = 12  # For Alpha Vantage 5 req/min limit
DEFAULT_FOREX_PAIRS = ["EUR/USD"]

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def fetch_forex_daily_data(currency_pair: str, output_size: str = "full") -> pd.DataFrame:
    """
    Download daily OHLC series for a currency pair from Alpha Vantage API.

    Args:
        currency_pair (str): Currency pair in format 'EUR/USD'
        output_size (str): Data size to fetch - 'full' or 'compact'

    Returns:
        pd.DataFrame: DataFrame indexed by date with columns: open, high, low, close
    """
    from_symbol, to_symbol = currency_pair.upper().split("/")

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": output_size,
        "datatype": "json",
        "apikey": API_KEY,
    }

    response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    # Extract time series data from Alpha Vantage response
    time_series_key = [k for k in payload.keys() if "Time Series" in k][0]

    df = (pd.DataFrame(payload[time_series_key])
          .T.rename(columns=lambda c: c.split(". ")[1])  # Remove "1. open" prefix
          .astype(float)
          .sort_index())

    # Convert index to timezone-aware datetime
    df.index = pd.to_datetime(df.index)
    df = df.tz_localize("UTC")

    return df

def save_forex_pair_data(currency_pair: str, output_folder: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Fetch and save forex pair data to CSV file.

    Args:
        currency_pair (str): Currency pair in format 'EUR/USD'
        output_folder (str): Directory to save CSV file

    Returns:
        pd.DataFrame: The fetched and saved data
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    df = fetch_forex_daily_data(currency_pair)
    output_file = Path(output_folder) / f"{currency_pair.replace('/', '_')}_daily.csv"

    df.to_csv(output_file)
    print(f"{currency_pair}: {len(df):,} rows written → {output_file}")

    return df

# ------------- run -------------
if __name__ == "__main__":
    if not API_KEY:
        print("Error: ALPHAVANTAGE_API_KEY not found in environment variables")
        print("Please add your API key to the .env file")
        exit(1)

    for i, pair in enumerate(DEFAULT_FOREX_PAIRS, start=1):
        df = save_forex_pair_data(pair)
        if i < len(DEFAULT_FOREX_PAIRS):  # Don't sleep after last pair
            time.sleep(RATE_LIMIT_DELAY_SECONDS)  # Respect 5 req/min cap
