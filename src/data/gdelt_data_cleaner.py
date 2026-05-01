"""
GDELT data cleaner — standardize and clean extracted news data for sentiment analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import Config

FINAL_COLUMNS = ["date", "url", "title", "text"]


def clean_and_process_gdelt_articles(
    input_path: str | Path,
    output_path: str | Path,
) -> bool:
    """
    Clean and process GDELT news articles for sentiment analysis.

    - Validates and loads input CSV
    - Selects/renames essential columns (date, url, title, text)
    - Parses dates to YYYY-MM-DD
    - Drops rows with missing URLs, fills empty titles/text
    - Explodes multi-value fields into individual rows
    - Saves cleaned data
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return False

    try:
        raw_df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {input_path}. Shape: {raw_df.shape}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # --- Column selection ---
    column_map = {
        "date": "date",
        "url": "url",
        "urlarticletitle": "title",
        "title": "title",
        "text": "text",
        "article": "text",
    }

    available = {col: new for col, new in column_map.items() if col in raw_df.columns}
    df = raw_df[list(available.keys())].rename(columns=available)

    # Ensure all FINAL_COLUMNS exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[FINAL_COLUMNS]

    # --- Date parsing ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["date"] = df["date"].fillna("1970-01-01")

    # --- Clean missing ---
    before = len(df)
    df = df.dropna(subset=["url"])
    df = df[df["url"] != ""]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing URLs")

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # --- Explode multi-value fields ---
    for col in ["url", "title", "text"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split("\n")
    df = df.explode(["url", "title", "text"], ignore_index=True)
    df = df[df["url"] != ""].reset_index(drop=True)

    print(f"Cleaned data shape: {df.shape}")

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned articles saved to {output_path}")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Clean GDELT article data")
    parser.add_argument("--input", default=str(cfg.resolve(cfg.get("files.cleaned_articles"))))
    parser.add_argument("--output", default=str(cfg.resolve(cfg.get("files.cleaned_articles"))))
    args = parser.parse_args()

    success = clean_and_process_gdelt_articles(args.input, args.output)
    if not success:
        exit(1)
