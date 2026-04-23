"""
Data filtering, scraping, cleaning, preprocessing, and train/test splitting.

This module covers the full data lifecycle:
  1. Filtering raw exports into article-only DataFrames
  2. Scraping article text from URLs (resumable with checkpoints)
  3. Combining paragraph-level scrape output into one row per article
  4. Saving EDA and ML-ready datasets
  5. Text preprocessing (two representations):
       - clean_combined : lowercased, lemmatized, stopword-removed, title-weighted.
                          Best for TF-IDF and classical sklearn classifiers.
       - raw_combined   : minimally processed (title [SEP] body), body capped at
                          ~512 words.  Best for transformer models (SBERT, SetFit,
                          Longformer) that have their own tokenizers.
  6. Stratified train/test splitting with label encoding
"""

import datetime
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    CUSTOM_STOPWORDS,
    DATA_PATH,
    LABEL_COLUMN,
    MAX_CHARS,
    RAW_BODY_CAP,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TITLE_COLUMN,
    TITLE_WEIGHT,
    UNLABELED_VALUE,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RAW DATA FILTERING
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_user_need(tags: str) -> str:
    """Return the first user-need value found in a comma-separated tag string."""
    for tag in tags.split(","):
        tag = tag.strip()
        if tag.startswith("user_need: "):
            return tag[len("user_need: "):]
    return "none"


def filter_articles(
    raw_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the raw dataframe to only include rows that correspond to articles.

    Parameters
    ----------
    raw_df : pd.DataFrame
        The raw dataframe containing all data.
    start_date : str, optional
        Filter articles published on or after this date (YYYY-MM-DD).
    end_date : str, optional
        Filter articles published before this date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        A filtered dataframe containing only article data.
    """
    art_df = raw_df[raw_df["Post id"].notna()].copy()
    logger.info("%d articles from %d total rows.", len(art_df), len(raw_df))

    art_df["Publish_date"] = art_df["Publish date"].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M")
    )

    if start_date:
        art_df = art_df[
            art_df["Publish_date"] >= datetime.datetime.strptime(start_date, "%Y-%m-%d")
        ]
    if end_date:
        art_df = art_df[
            art_df["Publish_date"] < datetime.datetime.strptime(end_date, "%Y-%m-%d")
        ]

    logger.info(
        "Filtered to %d articles between %s and %s.",
        len(art_df), start_date, end_date,
    )

    art_df["User_Needs"] = art_df["Tags"].apply(_extract_user_need)
    art_df = art_df[art_df["User_Needs"] != "other-not-news"].reset_index(drop=True)

    logger.info(
        "User Needs distribution:\n%s",
        art_df["User_Needs"].value_counts(normalize=True),
    )

    return art_df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ARTICLE SCRAPING
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_single_article(
    url: str,
    cookies: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 15,
) -> dict:
    """Fetch and parse a single article page.

    Returns
    -------
    dict
        Keys: ``url``, ``title``, ``paragraphs`` (list[str]).
    """
    headers = headers or _DEFAULT_HEADERS
    response = requests.get(url, cookies=cookies, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    article_tag = soup.find("article")
    if not article_tag:
        raise ValueError("No <article> tag found.")

    entry_content = article_tag.find("div", class_="entry-content")
    if not entry_content:
        raise ValueError("No div.entry-content found inside <article>.")

    paragraphs = [
        p.get_text(strip=True)
        for p in entry_content.find_all("p")
        if p.get_text(strip=True)
    ]
    if not paragraphs:
        raise ValueError("No <p> text found.")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    return {"url": url, "title": title, "paragraphs": paragraphs}


class _GracefulInterrupt:
    """Context manager that converts SIGINT/SIGTERM into a flag."""

    def __init__(self):
        self.interrupted = False

    def __enter__(self):
        self._prev_int = signal.getsignal(signal.SIGINT)
        self._prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        return self

    def __exit__(self, *_):
        signal.signal(signal.SIGINT, self._prev_int)
        signal.signal(signal.SIGTERM, self._prev_term)

    def _handle(self, _signum, _frame):
        logger.warning("Interrupt received — finishing current article then saving…")
        self.interrupted = True


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_checkpoint(path: Path, url: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(url + "\n")


def _flush_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def scrape_articles(
    input_csv: str,
    output_csv: str = "article_output.csv",
    failures_csv: str = "scrape_failures.csv",
    checkpoint_file: str = "scrape_checkpoint.txt",
    cookies: Optional[dict] = None,
    batch_size: int = 10,
    request_delay: float = 1.0,
) -> None:
    """Scrape article text from a CSV of URLs, saving results incrementally.

    Resumable: tracks completed URLs in a checkpoint file and skips on re-run.
    """
    input_df = pd.read_csv(input_csv)
    urls = input_df["URL"].dropna().unique().tolist()
    logger.info("Total URLs in input: %d", len(urls))

    checkpoint_path = Path(checkpoint_file)
    output_path = Path(output_csv)
    failures_path = Path(failures_csv)

    done = _load_checkpoint(checkpoint_path)
    remaining = [u for u in urls if u not in done]
    logger.info("Already completed: %d | Remaining: %d", len(done), len(remaining))

    pending_rows: list[dict] = []
    pending_failures: list[dict] = []
    success_count = 0
    fail_count = 0

    with _GracefulInterrupt() as gi:
        for batch_start in range(0, len(remaining), batch_size):
            if gi.interrupted:
                break

            batch = remaining[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(remaining) + batch_size - 1) // batch_size
            logger.info("Batch %d/%d (%d URLs)", batch_num, total_batches, len(batch))

            for url in batch:
                if gi.interrupted:
                    break

                try:
                    result = scrape_single_article(url, cookies=cookies)
                    for para_num, text in enumerate(result["paragraphs"], 1):
                        pending_rows.append({
                            "url": result["url"],
                            "title": result["title"],
                            "paragraph": para_num,
                            "text": text,
                        })
                    _append_checkpoint(checkpoint_path, url)
                    success_count += 1
                    logger.info(
                        "%d paragraphs from %s", len(result["paragraphs"]), url
                    )
                except Exception as exc:
                    fail_count += 1
                    logger.error("FAILED %s: %s", url, exc)
                    pending_failures.append({"url": url, "error": str(exc)})

                time.sleep(request_delay)

            _flush_rows(pending_rows, output_path)
            _flush_rows(pending_failures, failures_path)
            pending_rows.clear()
            pending_failures.clear()

    logger.info(
        "Scraping %s. Succeeded: %d | Failed: %d",
        "interrupted" if gi.interrupted else "complete",
        success_count, fail_count,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARAGRAPH COMBINING & DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════


def combine_paragraphs(input_csv: str, output_csv: str) -> None:
    """Combine paragraph-level scrape output into one row per article."""
    df = pd.read_csv(input_csv)
    combined = df.groupby("url", as_index=False).agg(
        title=("title", "first"),
        text=("text", lambda s: " ".join(s.dropna().astype(str))),
    )
    combined.to_csv(output_csv, index=False)
    logger.info("Combined %d articles → %s", len(combined), output_csv)


_EDA_COLUMNS = [
    "Apikey", "URL", "Title", "text", "Publish_date", "Authors",
    "Section", "User_Needs", "Views", "Avg. views", "Engaged minutes",
    "Avg. minutes", "Desktop views", "Mobile views", "Tablet views",
]

_ML_COLUMNS = ["URL", "Title", "text", "User_Needs"]


def _merge_article_text(raw_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join raw metadata with scraped article text."""
    merged = raw_df.merge(text_df, left_on="URL", right_on="url", how="inner")
    return merged.drop(columns=["url"])


def save_eda_dataset(
    raw_df: pd.DataFrame,
    text_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Merge, clean, and save a dataset suitable for exploratory analysis."""
    merged = _merge_article_text(raw_df, text_df)
    eda_df = merged[_EDA_COLUMNS].copy()
    eda_df["Tablet views"] = eda_df["Tablet views"].fillna(0)
    eda_df = eda_df.dropna(ignore_index=True)

    eda_df.to_csv(output_path, index=False)
    logger.info("Saved EDA dataset (%d rows) → %s", len(eda_df), output_path)
    return eda_df


def save_ml_datasets(
    raw_df: pd.DataFrame,
    text_df: pd.DataFrame,
    tagged_path: str,
    untagged_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge, split by label presence, and save ML-ready datasets."""
    merged = _merge_article_text(raw_df, text_df)
    ml_df = merged[_ML_COLUMNS].dropna(ignore_index=True)

    tagged_df = ml_df[ml_df["User_Needs"] != "none"].reset_index(drop=True)
    untagged_df = ml_df[ml_df["User_Needs"] == "none"].reset_index(drop=True)

    tagged_df.to_csv(tagged_path, index=False)
    untagged_df.to_csv(untagged_path, index=False)

    logger.info("Saved %d tagged rows: %s", len(tagged_df), tagged_path)
    logger.info("Saved %d untagged rows: %s", len(untagged_df), untagged_path)
    return tagged_df, untagged_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_nltk():
    """Download required NLTK data if missing."""
    import nltk
    for resource in ("punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if "punkt" in resource
                else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk()
_lemmatizer = WordNetLemmatizer()
_STOP = set(stopwords.words("english")) | CUSTOM_STOPWORDS


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation, lemmatize, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"<[^>]+>", "", text)             # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # non-alpha
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens
              if t not in _STOP and len(t) > 2]
    return " ".join(tokens)


def build_combined_clean(row, title_weight: int = TITLE_WEIGHT) -> str:
    """Cleaned text with title repeated for TF-IDF / sklearn classifiers."""
    title_clean = clean_text(row[TITLE_COLUMN])
    body_clean = clean_text(row[TEXT_COLUMN])
    return " ".join([title_clean] * title_weight + [body_clean])


def build_combined_raw(row) -> str:
    """Minimally processed text for transformer models (title [SEP] body)."""
    title = str(row[TITLE_COLUMN]).strip()
    body = str(row[TEXT_COLUMN]).strip()
    body_words = body.split()[:RAW_BODY_CAP]
    return f"{title} [SEP] {' '.join(body_words)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOADING & ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════


def load_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the EDA CSV and add all text columns needed downstream."""
    df = pd.read_csv(path)

    # Legacy columns (kept for backward compatibility)
    df["combined_text"] = df[TITLE_COLUMN] + ". " + df[TEXT_COLUMN].fillna("")
    df["combined_short"] = (
        df[TITLE_COLUMN] + ". " + df[TEXT_COLUMN].fillna("").str[:MAX_CHARS]
    )

    # Preprocessed columns
    print("Preprocessing all articles...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    df["clean_combined"] = df.apply(build_combined_clean, axis=1)
    df["raw_combined"] = df.apply(build_combined_raw, axis=1)
    print(f"  clean_combined sample: {df['clean_combined'].iloc[0][:120]}...")
    print(f"  raw_combined sample:   {df['raw_combined'].iloc[0][:120]}...")

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded dataset."""
    print(f"Loaded {len(df):,} articles  |  columns: {list(df.columns)}")
    print(f"Title null: {df[TITLE_COLUMN].isna().sum()}, "
          f"text null: {df[TEXT_COLUMN].isna().sum()}")
    lengths = df[TEXT_COLUMN].str.len()
    print(f"Text length — mean: {lengths.mean():.0f}, "
          f"median: {lengths.median():.0f}, max: {lengths.max():.0f}")
    print(f"\n{LABEL_COLUMN} distribution:\n{df[LABEL_COLUMN].value_counts()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN / TEST SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════


def split_labeled_unlabeled(df: pd.DataFrame):
    """Return (df_labeled, df_unlabeled) based on UNLABELED_VALUE."""
    mask = df[LABEL_COLUMN] != UNLABELED_VALUE
    return df[mask].copy(), df[~mask].copy()


def encode_labels(df_labeled: pd.DataFrame):
    """Fit a LabelEncoder on the labeled subset; return (encoder, y_array)."""
    le = LabelEncoder()
    y = le.fit_transform(df_labeled[LABEL_COLUMN])
    return le, y


def stratified_split(X, y, indices, test_size=TEST_SIZE,
                     random_state=RANDOM_STATE):
    """Stratified train/test split returning X_train, X_test, y_train,
    y_test, train_idx, test_idx."""
    return train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_supervised_data(df: pd.DataFrame):
    """
    One-stop function: split labeled/unlabeled, encode labels, build
    train/test splits for every text variant.

    Returns a dict with all arrays needed downstream:
      - X_train_clean / X_test_clean  : for TF-IDF + sklearn
      - X_train_raw / X_test_raw      : for transformers (SetFit, Longformer)
      - X_train_text / X_test_text    : legacy combined_text
      - X_train_short / X_test_short  : legacy combined_short
    """
    df_labeled, df_unlabeled = split_labeled_unlabeled(df)
    le, y = encode_labels(df_labeled)
    label_names = le.classes_

    labeled_idx = df_labeled.index.values

    text_variants = {
        "text": "combined_text",
        "short": "combined_short",
        "clean": "clean_combined",
        "raw": "raw_combined",
    }
    arrays = {}
    for key, col in text_variants.items():
        arrays[key] = np.asarray(df_labeled[col], dtype=object)

    (X_train_text, X_test_text,
     y_train, y_test,
     train_idx, test_idx) = stratified_split(
        arrays["text"], y, labeled_idx)

    train_mask = np.isin(labeled_idx, train_idx)
    test_mask = np.isin(labeled_idx, test_idx)

    result = {
        "df_labeled": df_labeled,
        "df_unlabeled": df_unlabeled,
        "label_encoder": le,
        "label_names": label_names,
        "y_train": y_train,
        "y_test": y_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "labeled_idx": labeled_idx,
        "unlabeled_idx": df_unlabeled.index.values,
        # Legacy
        "X_train_text": X_train_text,
        "X_test_text": X_test_text,
        "X_train_short": arrays["short"][train_mask],
        "X_test_short": arrays["short"][test_mask],
        # Cleaned (for TF-IDF / sklearn)
        "X_train_clean": arrays["clean"][train_mask],
        "X_test_clean": arrays["clean"][test_mask],
        "X_clean_unlabeled": np.asarray(
            df_unlabeled["clean_combined"], dtype=object),
        # Raw (for transformers)
        "X_train_raw": arrays["raw"][train_mask],
        "X_test_raw": arrays["raw"][test_mask],
        "X_raw_unlabeled": np.asarray(
            df_unlabeled["raw_combined"], dtype=object),
        # Legacy unlabeled
        "X_text_unlabeled": np.asarray(
            df_unlabeled["combined_text"], dtype=object),
        "X_short_unlabeled": np.asarray(
            df_unlabeled["combined_short"], dtype=object),
    }
    return result
