"""
GDELT data cleaner module for processing and standardizing extracted news data.

This module cleans and standardizes GDELT news data by handling date formatting,
managing missing values, exploding multi-value fields, and preparing article-level
data for sentiment analysis.
"""

import pandas as pd
import os
from pathlib import Path

# Configuration constants
PROCESSED_DATA_DIR = "../../../data/processed"
ARTICLES_TEXT_FILE = "../../../data/processed/gdelt_articles_text.csv"
CLEANED_ARTICLES_FILE = "../../../data/processed/gdelt_articles_text_cleaned.csv"
FINAL_COLUMNS = ["date", "url", "title", "text"]

def clean_and_process_gdelt_articles(input_path: str, output_path: str):
    """
    Clean and process GDELT news articles data for sentiment analysis.

    This function performs the following operations:
    1. Loads and validates input CSV data
    2. Selects and renames essential columns (date, url, title, text)
    3. Handles missing column scenarios by creating empty columns
    4. Parses and formats date column to YYYY-MM-DD format
    5. Manages missing values (drops rows with missing URLs, fills empty titles/text)
    6. Explodes multi-value fields separated by newlines into individual rows
    7. Ensures consistent text formatting and column ordering
    8. Saves processed data for downstream sentiment analysis

    Args:
        input_path (str): Path to input GDELT CSV file
        output_path (str): Path to save cleaned CSV file

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please ensure the GDELT data has been extracted to this location")
        return False

    # Load raw data
    try:
        raw_df = pd.read_csv(input_path, low_memory=False)
        print(f"Successfully loaded {input_path}. Shape: {raw_df.shape}")
    except Exception as e:
        print(f"Error reading CSV file {input_path}: {e}")
        return False

    # Initialize processed DataFrame
    processed_df = pd.DataFrame()

    # Validate and process date column
    if 'DATE' not in raw_df.columns:
        print(f"Error: Date column 'DATE' not found in input CSV")
        print(f"Available columns: {', '.join(raw_df.columns)}")
        return False

    processed_df['date'] = raw_df['DATE']

    # Validate and process URL column
    if 'url' not in raw_df.columns:
        print(f"Error: URL column 'url' not found in input CSV")
        print(f"Available columns: {', '.join(raw_df.columns)}")
        return False

    processed_df['url'] = raw_df['url']

    # Process title column (optional)
    if 'title' in raw_df.columns:
        processed_df['title'] = raw_df['title']
    else:
        print(f"Warning: Title column 'title' not found. Creating empty title field")
        processed_df['title'] = ''

    # Process text column (optional)
    if 'text' in raw_df.columns:
        processed_df['text'] = raw_df['text']
    else:
        print(f"Warning: Text column 'text' not found. Creating empty text field")
        processed_df['text'] = ''

    print("Essential columns prepared for processing")

    # Parse and format dates
    original_rows = len(processed_df)

    # Determine date parsing strategy based on data type
    if pd.api.types.is_integer_dtype(processed_df['date']):
        try:
            # Handle YYYYMMDD integer format
            processed_df['date'] = pd.to_datetime(
                processed_df['date'].astype(str),
                format='%Y%m%d',
                errors='coerce'
            )
        except ValueError:
            print("Warning: Some integer dates could not be parsed with YYYYMMDD format")
    else:
        # Handle string or other formats
        processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

    # Remove rows with unparseable dates
    processed_df = processed_df.dropna(subset=['date'])
    rows_dropped = original_rows - len(processed_df)

    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to unparseable dates")

    # Format dates to string
    if not processed_df.empty:
        processed_df['date'] = processed_df['date'].dt.strftime('%Y-%m-%d')
        print("Date column formatted to YYYY-MM-DD")

    if processed_df.empty:
        print("DataFrame is empty after date parsing. No data to process.")
        return False

    # Handle missing values
    # Drop rows with missing URLs (essential for article identification)
    processed_df = processed_df.dropna(subset=['url'])

    # Fill missing title and text with empty strings
    processed_df['title'] = processed_df['title'].fillna('').astype(str)
    processed_df['text'] = processed_df['text'].fillna('').astype(str)

    print("Missing values handled for url, title, text")

    # Explode multi-value fields
    print("Checking for and exploding multi-value fields (url, title, text)...")

    exploded_rows = []

    if not processed_df.empty:
        for _, row in processed_df.iterrows():
            current_date = row['date']

            # Split multi-value fields by newlines
            urls = str(row['url']).split('\n')
            titles = str(row['title']).split('\n')
            texts = str(row['text']).split('\n')

            # Use URL count as primary determinant for number of rows
            max_length = len(urls)

            # Pad shorter lists to match URL count
            titles = titles + [''] * (max_length - len(titles))
            texts = texts + [''] * (max_length - len(texts))

            # Create individual rows for each URL
            for i in range(max_length):
                exploded_rows.append({
                    'date': current_date,
                    'url': urls[i].strip(),
                    'title': titles[i].strip(),
                    'text': texts[i].strip()
                })

    if exploded_rows:
        processed_df = pd.DataFrame(exploded_rows)
        print(f"Exploded multi-value fields. New shape: {processed_df.shape}")
    else:
        # No explosion needed, just ensure text fields are stripped
        if not processed_df.empty:
            processed_df['url'] = processed_df['url'].astype(str).str.strip()
            processed_df['title'] = processed_df['title'].astype(str).str.strip()
            processed_df['text'] = processed_df['text'].astype(str).str.strip()
            print("No multi-value fields found for explosion")
        else:
            processed_df = pd.DataFrame(columns=FINAL_COLUMNS)
            print("Input DataFrame was empty before explosion step")

    # Prepare final output
    if processed_df.empty:
        print("DataFrame is empty after cleaning steps. No data to save.")
        final_df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        # Validate required columns exist
        missing_columns = [col for col in FINAL_COLUMNS if col not in processed_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(processed_df.columns)}")
            return False

        # Select and order final columns
        final_df = processed_df[FINAL_COLUMNS].copy()
        print(f"Data prepared for saving. Final shape: {final_df.shape}")

    # Save processed data
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Save to CSV
        final_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Successfully processed data saved to {output_path}")
        print(f"Final dataset shape: {final_df.shape}")
        return True

    except Exception as e:
        print(f"Error saving CSV file to {output_path}: {e}")
        return False

# ------------- run -------------
if __name__ == '__main__':
    print("Starting GDELT article data cleaning process...")

    input_file = ARTICLES_TEXT_FILE
    output_file = CLEANED_ARTICLES_FILE

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    success = clean_and_process_gdelt_articles(input_file, output_file)

    if success:
        print("GDELT data cleaning completed successfully")
    else:
        print("GDELT data cleaning failed")
        exit(1)
