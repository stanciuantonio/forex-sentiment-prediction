"""
GDELT news extractor module for forex-related news article retrieval.

This module provides functionality to extract and filter news articles from GDELT
that are relevant to EUR/USD forex trading, using parallel processing for efficiency.
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

# Configuration constants
RAW_DATA_DIR = "data/raw"
GDELT_TABLE_NAME = "gkg"
MAX_ARTICLES_PER_DAY = 5
MAX_PARALLEL_WORKERS = 8

# Forex keywords for filtering GDELT news
FOREX_KEYWORDS = [
    'EUR USD', 'EURUSD', 'euro dollar', 'european central bank',
    'ECB', 'federal reserve', 'interest rates', 'monetary policy',
    'inflation', 'unemployment', 'GDP', 'economic growth'
]

# Relevant countries for news filtering (GDELT country codes)
RELEVANT_COUNTRIES = ['US', 'GM', 'FR', 'IT', 'SP', 'EU']

# Initialize GDELT client
gdelt_client = gdelt.gdelt(version=1)

def extract_single_day_news(date_str: str, max_articles: int = MAX_ARTICLES_PER_DAY):
    """
    Extract and filter GDELT news data for a single day.

    Args:
        date_str (str): Date string in format "YYYY Mon DD" (e.g., "2023 Jan 15")
        max_articles (int): Maximum number of articles to return per day

    Returns:
        pd.DataFrame or str or None: Filtered news data, error message, or None if no data
    """
    try:
        # Search GDELT data for the specified date
        daily_data = gdelt_client.Search(date_str, table=GDELT_TABLE_NAME)

        if daily_data.empty:
            return None

        # Filter by relevant countries using location data
        location_columns = ['V2Locations', 'LOCATIONS']
        for col in location_columns:
            if col in daily_data.columns:
                daily_data = daily_data[
                    daily_data[col].astype(str).str.contains(
                        '|'.join(RELEVANT_COUNTRIES), na=False
                    )
                ]
                break

        # Filter by forex-related keywords in themes
        theme_columns = ['V2Themes', 'THEMES']
        for col in theme_columns:
            if col in daily_data.columns:
                themes_filter = daily_data[col].astype(str).str.lower()
                keyword_pattern = '|'.join([k.lower() for k in FOREX_KEYWORDS])
                daily_data = daily_data[themes_filter.str.contains(keyword_pattern, na=False)]
                break

        if daily_data.empty:
            return None

        # Limit number of articles per day, prioritizing by tone if available
        if len(daily_data) > max_articles:
            tone_columns = ['TONE', 'AvgTone', 'tone', 'avgtone']
            for tone_col in tone_columns:
                if tone_col in daily_data.columns:
                    try:
                        # Extract average tone (first part of TONE string)
                        daily_data_copy = daily_data.copy()
                        daily_data_copy[tone_col] = daily_data_copy[tone_col].astype(str).str.split(',').str[0]
                        daily_data_copy[tone_col] = pd.to_numeric(daily_data_copy[tone_col], errors='coerce')
                        daily_data_copy = daily_data_copy.dropna(subset=[tone_col])

                        if not daily_data_copy.empty:
                            # Sort by absolute tone value, descending
                            sorted_df = daily_data_copy.reindex(
                                daily_data_copy[tone_col].abs().sort_values(ascending=False).index
                            )
                            daily_data = sorted_df.head(max_articles)
                            break
                    except Exception:
                        continue
            else:
                # Fallback to first N articles if tone sorting fails
                daily_data = daily_data.head(max_articles)

        return daily_data

    except Exception as e:
        return f"Error processing {date_str}: {str(e)}"

def extract_gdelt_news_data(start_date: str, end_date: str, output_dir: str = RAW_DATA_DIR):
    """
    Extract GDELT news data for EUR/USD sentiment analysis with parallel processing.

    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        output_dir (str): Directory to save output CSV

    Returns:
        pd.DataFrame: Extracted and filtered news data
    """
    # Generate business days only
    try:
        business_dates = pd.bdate_range(start=start_date, end=end_date)
    except ValueError as e:
        print(f"Error generating business day range: {e}")
        return pd.DataFrame()

    # Format dates for GDELT API
    formatted_dates = [f"{d.year} {d.strftime('%b')} {d.day}" for d in business_dates]

    if not formatted_dates:
        print(f"No business days found in range: {start_date} to {end_date}")
        return pd.DataFrame()

    print(f"Processing {len(formatted_dates)} business days from {start_date} to {end_date}")

    # Process dates in parallel
    all_data = []
    errors = []

    with tqdm(total=len(formatted_dates), desc="Extracting GDELT data", unit="day") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_date = {
                executor.submit(extract_single_day_news, date): date
                for date in formatted_dates
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    if isinstance(result, str):  # Error message
                        errors.append(result)
                    elif result is not None and not result.empty:
                        all_data.append(result)
                except Exception as exc:
                    errors.append(f"Error processing {date}: {exc}")

                pbar.update(1)

                # Periodic memory cleanup
                if len(all_data) % 50 == 0:
                    gc.collect()

    # Report errors
    if errors:
        print(f"Found {len(errors)} errors during processing")
        for i, error in enumerate(errors[:3]):  # Show first 3 errors
            print(f"  {i+1}. {error}")

    # Combine results
    if all_data:
        print(f"Combining {len(all_data)} data chunks...")
        final_data = pd.concat(all_data, ignore_index=True)

        # Process and save data
        if 'DATE' in final_data.columns:
            final_data['DATE'] = pd.to_datetime(final_data['DATE'].astype(str), format='%Y%m%d', errors='coerce')
            final_data = final_data.sort_values(by='DATE')
            final_data['DATE'] = final_data['DATE'].dt.strftime('%Y%m%d')

        # Save data
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "gdelt_eurusd_news.csv")
        final_data.to_csv(output_file, index=False)

        print(f"Data successfully saved to {output_file}")
        return final_data
    else:
        print("No data extracted.")
        return pd.DataFrame()

# ------------- run -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract GDELT data for EUR/USD sentiment analysis")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default=RAW_DATA_DIR, help="Output directory for CSV file")
    parser.add_argument("--test", action="store_true", help="Run mini test with sample data")
    args = parser.parse_args()

    # Mini test mode
    if args.test:
        print("=== MINI TEST MODE ===")
        print("Testing GDELT extraction with sample date...")

        # Test single day extraction
        test_date = "2024 Jan 01"
        print(f"Testing single day extraction for: {test_date}")

        try:
            result = extract_single_day_news(test_date, max_articles=2)

            if isinstance(result, str):  # Error message
                print(f"Error result: {result}")
            elif result is not None and not result.empty:
                print(f"Success! Extracted {len(result)} articles")
                print("Sample columns:", list(result.columns)[:5])
                print("Sample data:")
                print(result.head(2))
            else:
                print("No data found for test date")

        except Exception as e:
            print(f"Test failed: {e}")

        print("\n=== Testing with date range ===")
        # Test small date range
        small_data = extract_gdelt_news_data("2024-01-01", "2024-01-16", "data/test")
        if not small_data.empty:
            print(f"Range test success: {len(small_data)} articles")
            print("Columns:", list(small_data.columns))
        else:
            print("Range test: No data extracted")

        exit(0)

    start_time = time.time()

    print("Starting GDELT data extraction...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {args.output_dir}")

    news_data = extract_gdelt_news_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir
    )

    if not news_data.empty:
        print("=== EXTRACTION SUMMARY ===")
        print(f"Extracted {len(news_data)} news articles")

        # Display date range from extracted data
        if 'DATE' in news_data.columns:
            try:
                news_data['DATE'] = pd.to_datetime(news_data['DATE'].astype(str), errors='coerce')
                if not news_data['DATE'].isna().all():
                    min_date = news_data['DATE'].min().strftime('%Y-%m-%d')
                    max_date = news_data['DATE'].max().strftime('%Y-%m-%d')
                    print(f"Period: {min_date} to {max_date}")
            except Exception as e:
                print(f"Could not determine date range: {e}")

        execution_time = (time.time() - start_time) / 60
        print(f"Total execution time: {execution_time:.2f} minutes")

        print("First 5 rows of extracted data:")
        print(news_data.head())
    else:
        print("No news data was extracted")
