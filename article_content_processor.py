"""
Article content processor module for extracting text from news URLs.

This module processes GDELT news data by extracting article content from URLs,
handling URL explosion for multi-URL entries, and using parallel processing
for efficient content extraction.
"""

import pandas as pd
import os
import time
import argparse
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# Configuration constants
PROCESSED_DATA_DIR = "data/processed"
MAX_PARALLEL_WORKERS =8
ARTICLE_PROCESSING_BATCH_SIZE = 10
REQUEST_TIMEOUT_SECONDS = 10
ARTICLE_FETCH_DELAY_SECONDS = 0.2
MAX_ARTICLE_TEXT_LENGTH = 1000

def extract_article_text_newspaper(url: str):
    """
    Extract article title and text using newspaper3k library.

    Args:
        url (str): Article URL to extract content from

    Returns:
        dict or None: Dictionary with title, text, and success status, or None if failed
    """
    try:
        article = Article(url)
        article.download()
        time.sleep(ARTICLE_FETCH_DELAY_SECONDS)  # Rate limiting
        article.parse()

        title = article.title.strip()
        text = article.text.strip()

        # Handle cases where title or text extraction failed
        if not title and not text:
            return None

        # Use first part of text as title if title is missing
        if not title and text:
            title = text[:100] + "..."

        # Use title as text if text is missing
        if title and not text:
            text = title

        return {
            'title': title,
            'text': text,
            'success': True
        }
    except Exception as e:
        return {
            'title': "",
            'text': "",
            'success': False,
            'error': str(e)
        }

def extract_article_text_beautifulsoup(url: str):
    """
    Extract article text using BeautifulSoup as fallback method.

    Args:
        url (str): Article URL to extract content from

    Returns:
        dict or None: Dictionary with title, text, and success status, or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title from HTML title tag
        title = ""
        if soup.title:
            title = soup.title.text.strip()

        # Extract text from first few paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.text.strip() for p in paragraphs[:5]])  # First 5 paragraphs

        if not title and not text:
            return None

        return {
            'title': title,
            'text': text[:MAX_ARTICLE_TEXT_LENGTH],  # Limit text length
            'success': True
        }
    except Exception as e:
        return {
            'title': "",
            'text': "",
            'success': False,
            'error': str(e)
        }

def extract_article_full_content(url: str):
    """
    Extract article content using newspaper3k first, falling back to BeautifulSoup.

    Args:
        url (str): Article URL to extract content from

    Returns:
        dict: Dictionary with extracted content and metadata
    """
    # Try newspaper3k first
    result = extract_article_text_newspaper(url)

    # Fall back to BeautifulSoup if newspaper3k fails
    if not result or not result['success'] or not result['text']:
        result = extract_article_text_beautifulsoup(url)

    return result if result else {
        'title': "",
        'text': "",
        'success': False,
        'error': "Both extraction methods failed"
    }

def process_article_url_batch(urls):
    """
    Process a batch of URLs and extract their content.

    Args:
        urls: List of URLs to process

    Returns:
        List[dict]: List of extraction results
    """
    results = []

    for url in urls:
        # Validate URL
        if not url or not isinstance(url, str) or not url.startswith('http'):
            results.append({
                'url': url,
                'title': "",
                'text': "",
                'success': False,
                'error': "Invalid URL"
            })
            continue

        # Extract content
        result = extract_article_full_content(url)
        if result:
            result['url'] = url
            results.append(result)
        else:
            results.append({
                'url': url,
                'title': "",
                'text': "",
                'success': False,
                'error': "Content extraction failed"
            })

    return results

def explode_urls_and_extract_content(gdelt_csv_path: str, output_csv_path: str, sample_size: int = 0):
    """
    Main processing function that explodes GDELT data and extracts article content.

    This function:
    1. Loads GDELT CSV data
    2. Explodes data to article level (one row per URL)
    3. Extracts textual content from each URL using parallel processing
    4. Saves results to CSV

    Args:
        gdelt_csv_path (str): Path to input GDELT CSV file
        output_csv_path (str): Path to save output CSV file
        sample_size (int): Number of articles to sample (0 for all)

    Returns:
        pd.DataFrame or None: Processed DataFrame with article content
    """
    print(f"Starting processing for file: {gdelt_csv_path}")

    # Load GDELT CSV data
    try:
        df = pd.read_csv(gdelt_csv_path)
        print(f"CSV loaded successfully. {len(df)} rows found.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Convert DATE column to datetime
    df['date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')

    # Explode SOURCEURLS column for article-level processing
    print("Exploding data to article level...")

    if 'SOURCEURLS' not in df.columns:
        print("Column 'SOURCEURLS' not found in CSV")
        return None

    # Process URLs: split by semicolon and explode
    df['SOURCEURLS'] = df['SOURCEURLS'].astype(str)
    df = df.assign(SOURCEURLS=df['SOURCEURLS'].str.split(';')).explode('SOURCEURLS')
    df = df.rename(columns={'SOURCEURLS': 'url'})

    # Clean and deduplicate URLs
    df = df.dropna(subset=['url'])
    df = df[df['url'].str.strip() != '']
    df = df.drop_duplicates(subset=['url'])

    # Apply sampling if requested
    if sample_size > 0:
        print(f"Sampling {sample_size} articles for processing")
        df = df.head(sample_size).copy()

    print(f"After exploding and sampling: {len(df)} unique articles to process")

    # Extract content from URLs using parallel processing
    print(f"Extracting content from {len(df)} URLs using {MAX_PARALLEL_WORKERS} workers...")

    urls = df['url'].tolist()
    url_batches = [urls[i:i+ARTICLE_PROCESSING_BATCH_SIZE] for i in range(0, len(urls), ARTICLE_PROCESSING_BATCH_SIZE)]

    results = []
    with tqdm(total=len(url_batches), desc="Extracting content") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_batch = {
                executor.submit(process_article_url_batch, batch): i
                for i, batch in enumerate(url_batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                pbar.update(1)

    # Create DataFrame with extraction results
    content_df = pd.DataFrame(results)

    # Merge with original DataFrame
    merged_df = pd.merge(df, content_df, on='url', how='left')

    # Select and organize final columns
    columns_to_keep = [
        'date', 'url', 'title', 'text', 'success',
        'DATE', 'NUMARTS', 'THEMES', 'LOCATIONS', 'TONE'  # Original GDELT columns
    ]

    # Keep only existing columns
    columns_to_keep = [col for col in columns_to_keep if col in merged_df.columns]
    final_df = merged_df[columns_to_keep].copy()

    # Sort by date and save results
    final_df = final_df.sort_values('date')

    print(f"Saving {len(final_df)} processed articles to {output_csv_path}")
    final_df.to_csv(output_csv_path, index=False)

    # Display processing statistics
    success_count = final_df['success'].sum()
    success_rate = success_count / len(final_df) * 100

    print("=== EXTRACTION STATISTICS ===")
    print(f"Total articles: {len(final_df)}")
    print(f"Successful extractions: {success_count} ({success_rate:.1f}%)")
    print(f"Period: {final_df['date'].min()} to {final_df['date'].max()}")

    return final_df

# ------------- run -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GDELT Article Content Processor: Extract text from news URLs'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/gdelt_eurusd_news.csv',
        help='Path to input GDELT news CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/gdelt_articles_text.csv',
        help='Path to save processed CSV with article text'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=0,
        help="Number of articles to sample for processing (0 to process all)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Execute processing
    start_time = time.time()

    final_df = explode_urls_and_extract_content(
        gdelt_csv_path=args.input,
        output_csv_path=args.output,
        sample_size=args.sample_size
    )

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time/60:.2f} minutes")
