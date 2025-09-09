"""
Email body cleaner

- Separates quoted/forwarded text from the original message body.
- Normalizes case, strips whitespace, removes non-ASCII characters.
- Saves cleaned data to CSV.

Usage:
    Run as a script (expects `config` and `data_loader` modules to provide paths and loading).
"""

from pathlib import Path
import logging
import re
from typing import Tuple, Union

import pandas as pd

import config
import data_loader

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Heuristic pattern to detect quoted/forwarded message separators.
# Matches lines starting with ">" (email quoting), common separator lines like "-----Original Message-----",
# or "On <date> <person> wrote:" constructs.
QUOTE_PATTERN = re.compile(r"^(?:>+|[-_]{2,}|on .+ wrote:)", re.IGNORECASE)


def extract_quoted_text(text: str) -> Tuple[str, str]:
    """
    Split `text` into (clean_body, quoted_text).

    Heuristics:
      - Any line that starts with '>' is treated as quoted.
      - Lines matching QUOTE_PATTERN (e.g., "On ... wrote:" or separators) are treated as quoted.
      - Preserves relative ordering and blank lines in the "clean" portion.
    Returns empty strings for non-string or empty inputs.
    """
    if not text or not isinstance(text, str):
        return "", ""

    lines = text.splitlines()
    clean_lines = []
    quoted_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # preserve blank lines in the cleaned body
            clean_lines.append("")
            continue

        # If the line looks like quoted content, collect it in quoted_lines
        if stripped.startswith(">") or QUOTE_PATTERN.match(stripped):
            quoted_lines.append(line)
        else:
            clean_lines.append(line)

    clean_body = "\n".join(clean_lines).strip()
    quoted_text = "\n".join(quoted_lines).strip()
    return clean_body, quoted_text


def clean_email_data(
    dataframe: pd.DataFrame, body_column: Union[str, None] = None
) -> pd.DataFrame:
    """
    Clean an email DataFrame in-place and return a cleaned copy.

    Steps:
      - Validate that `body_column` exists (defaults to config.BODY_COLUMN).
      - Count and drop missing bodies.
      - Convert to string, strip whitespace, lowercase.
      - Remove empty bodies.
      - Remove non-ASCII characters (vectorized regex).
      - Extract quoted text into 'clean_body' and 'quoted_text' columns.

    Returns:
      A new DataFrame (copy) with added columns 'clean_body' and 'quoted_text'.
    """
    if body_column is None:
        body_column = config.BODY_COLUMN

    if body_column not in dataframe.columns:
        raise KeyError(f"Expected column '{body_column}' in the input DataFrame.")

    cleaned_df = dataframe.copy()
    logging.info("Starting data cleaning. Initial shape: %s", cleaned_df.shape)

    # Count missing values before any casting/coercion
    initial_missing = int(cleaned_df[body_column].isna().sum())
    if initial_missing:
        logging.info("Found %d rows with missing '%s'. Dropping them.", initial_missing, body_column)

    # Drop rows where the body is missing
    cleaned_df = cleaned_df.dropna(subset=[body_column]).copy()

    # Normalize: cast to str, strip whitespace, convert to lowercase
    cleaned_df[body_column] = (
        cleaned_df[body_column].astype(str).str.strip().str.lower()
    )

    # Remove rows that are empty after stripping
    empty_mask = cleaned_df[body_column] == ""
    empty_count = int(empty_mask.sum())
    if empty_count:
        logging.info("Dropping %d rows with empty '%s' after strip.", empty_count, body_column)
    cleaned_df = cleaned_df.loc[~empty_mask].copy()

    logging.info("Removed %d invalid rows; new shape: %s", initial_missing + empty_count, cleaned_df.shape)

    # Remove non-ASCII characters using a vectorized regex replace (fast & avoids Python-level loops)
    cleaned_df[body_column] = cleaned_df[body_column].str.replace(r"[^\x00-\x7F]+", "", regex=True)
    logging.info("Removed non-ASCII characters from '%s'.", body_column)

    # Extract quoted text (vectorized-ish: map then assign columns)
    # Using list comprehension to avoid applying two Series operations.
    extracted = [extract_quoted_text(text) for text in cleaned_df[body_column].tolist()]
    if extracted:
        clean_bodies, quoted_texts = zip(*extracted)
        cleaned_df["clean_body"] = list(clean_bodies)
        cleaned_df["quoted_text"] = list(quoted_texts)
    else:
        cleaned_df["clean_body"] = []
        cleaned_df["quoted_text"] = []

    logging.info("Extraction complete: added 'clean_body' and 'quoted_text' columns.")
    return cleaned_df


def save_cleaned_data(dataframe: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """
    Save dataframe to CSV. Ensures output directory exists.

    Returns the Path to the saved file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(out_path, index=False)
    logging.info("Cleaned data saved to %s", str(out_path))
    return out_path


if __name__ == "__main__":
    logging.info("Testing data cleaner module...")

    try:
        logging.info("Loading raw data from %s", config.RAW_DATA_PATH)
        raw_df = data_loader.load_email_data(config.RAW_DATA_PATH)
    except Exception:
        logging.exception("Failed to load raw data. Ensure data_loader.load_email_data works and the path is correct.")
        raise

    logging.info("Cleaning the data...")
    cleaned_df = clean_email_data(raw_df)

    # Build output path robustly (supports config.PROCESSED_DATA_DIR being Path or str)
    output_path = Path(config.PROCESSED_DATA_DIR) / "cleaned_emails.csv"
    save_cleaned_data(cleaned_df, output_path)

    # Final preview — make sure we reference existing columns names
    preview_cols = [c for c in ("clean_body", "quoted_text") if c in cleaned_df.columns]
    if preview_cols:
        logging.info("Final preview of cleaned data:\n%s", cleaned_df[preview_cols].head().to_string())
    else:
        logging.info("No preview columns found in cleaned data.")

    logging.info("Data cleaning complete! Final shape: %s", cleaned_df.shape)
