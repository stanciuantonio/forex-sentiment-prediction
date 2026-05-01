"""
Article content processor — extract text from news article URLs using newspaper3k.
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
from pathlib import Path
from typing import Optional

from src.config import Config

# Defaults
MAX_PARALLEL_WORKERS = 8
ARTICLE_PROCESSING_BATCH_SIZE = 10
REQUEST_TIMEOUT_SECONDS = 10
ARTICLE_FETCH_DELAY_SECONDS = 0.2
MAX_ARTICLE_TEXT_LENGTH = 1000


def extract_article_text_newspaper(url: str) -> Optional[dict]:
    """Extract article title and text using newspaper3k library."""
    try:
        article = Article(url)
        article.download()
        time.sleep(ARTICLE_FETCH_DELAY_SECONDS)
        article.parse()

        title = article.title.strip()
        text = article.text.strip()

        if not title and not text:
            return None

        return {
            "url": url,
            "title": title,
            "text": text[:MAX_ARTICLE_TEXT_LENGTH] if len(text) > MAX_ARTICLE_TEXT_LENGTH else text,
            "success": True,
        }
    except Exception as e:
        return None


def extract_article_text_fallback(url: str) -> Optional[dict]:
    """Fallback: extract text using BeautifulSoup if newspaper3k fails."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""

        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = text[:MAX_ARTICLE_TEXT_LENGTH]

        if not title and not text:
            return None

        return {
            "url": url,
            "title": title,
            "text": text,
            "success": True,
        }
    except Exception as e:
        return None


def process_article_batch(
    urls: list,
    batch_size: int = ARTICLE_PROCESSING_BATCH_SIZE,
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> pd.DataFrame:
    """Process a batch of article URLs with parallel workers."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(extract_article_text_newspaper, url): url
            for url in urls
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url),
            total=len(urls),
            desc="Extracting articles",
        ):
            try:
                result = future.result()
                if result and result.get("success"):
                    results.append(result)
                elif result is None:
                    # Fallback to BeautifulSoup
                    url = future_to_url[future]
                    fallback_result = extract_article_text_fallback(url)
                    if fallback_result:
                        results.append(fallback_result)
            except Exception as e:
                pass

    return pd.DataFrame(results) if results else pd.DataFrame()


def process_gdelt_articles(
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = ARTICLE_PROCESSING_BATCH_SIZE,
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> pd.DataFrame:
    """Extract article content for all URLs in a GDELT CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    if "url" not in df.columns:
        print(f"Error: 'url' column not found in {input_path}")
        return pd.DataFrame()

    urls = df["url"].dropna().unique().tolist()
    print(f"Processing {len(urls)} unique URLs...\n")

    articles_df = process_article_batch(urls, batch_size, max_workers)
    articles_df.to_csv(output_path, index=False)
    print(f"\nArticles saved to {output_path} ({len(articles_df):,} articles)")
    return articles_df


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description="Extract article text from GDELT URLs")
    parser.add_argument("--input", default=str(cfg.resolve("data/raw/gdelt_forex_news.csv")))
    parser.add_argument("--output", default=str(cfg.resolve(cfg.get("files.cleaned_articles"))))
    parser.add_argument("--batch-size", type=int, default=ARTICLE_PROCESSING_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL_WORKERS)
    args = parser.parse_args()

    process_gdelt_articles(args.input, args.output, args.batch_size, args.workers)
