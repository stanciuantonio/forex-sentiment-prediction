"""
Forex data fetcher — retrieve currency pair OHLC data from Alpha Vantage API.

Usage:
    python -m src.data.forex_data_fetcher [--pair EUR/USD]
"""

import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import Config


def fetch_forex_daily(
    currency_pair: str = "EUR/USD",
    output_size: str = "full",
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download daily OHLC series for a currency pair from Alpha Vantage.

    Args:
        currency_pair: e.g. "EUR/USD"
        output_size: "full" or "compact"
        api_key: Alpha Vantage API key (falls back to env ALPHAVANTAGE_API_KEY)
        timeout: request timeout in seconds

    Returns:
        DataFrame indexed by date with columns: open, high, low, close
    """
    if api_key is None:
        import os
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key required — set ALPHAVANTAGE_API_KEY env var")

    from_symbol, to_symbol = currency_pair.upper().split("/")

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": output_size,
        "datatype": "json",
        "apikey": api_key,
    }

    response = requests.get(
        "https://www.alphavantage.co/query",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    time_series_key = [k for k in payload.keys() if "Time Series" in k][0]
    df = (
        pd.DataFrame(payload[time_series_key])
        .T.rename(columns=lambda c: c.split(". ")[1])
        .astype(float)
        .sort_index()
    )
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
    return df


def save_forex_data(
    currency_pair: str = "EUR/USD",
    output_path: str | Path = "data/raw/eurusd_daily.csv",
    api_key: Optional[str] = None,
    output_size: str = "full",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch and save forex pair data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_forex_daily(currency_pair, output_size, api_key, timeout)
    df.to_csv(output_path)
    print(f"{currency_pair}: {len(df):,} rows → {output_path}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Fetch forex data from Alpha Vantage")
    parser.add_argument("--pair", default=cfg.get("data.forex_pairs")[0])
    parser.add_argument("--output", default=cfg.resolve(cfg.get("files.forex_daily")))
    parser.add_argument("--output-size", default=cfg.get("data.output_size"))
    args = parser.parse_args()

    import os
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("Error: ALPHAVANTAGE_API_KEY not set in environment")
        exit(1)

    save_forex_data(
        currency_pair=args.pair,
        output_path=args.output,
        api_key=api_key,
        output_size=args.output_size,
        timeout=cfg.get("api.request_timeout", 30),
    )
