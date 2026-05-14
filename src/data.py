"""End-to-end data pipeline: filter → scrape → merge → preprocess → split.

Library-only. Called by `scripts/prepare_data.py` and the data-prep notebook.
Text preprocessing produces two representations: `clean_combined` (TF-IDF /
sklearn) and `raw_combined` (transformer-friendly `title [SEP] body`).

Inputs:
    - Raw Chartbeat export CSV (path via `RAW_DATA_PATH`)
    - Scraped paragraphs CSV (produced by `scrape_articles`)

Outputs (CSVs written by the `save_*` / `preprocess_*` helpers):
    - data/EDA_data-FULL.csv          — raw merge (metadata + text + label)
    - data/EDA_data-PREPROCESSED.csv  — adds derived text columns
    - data/ML_tagged_data-FULL.csv    — labelled subset
    - data/ML_untagged_data-FULL.csv  — unlabelled subset
    - scrape checkpoint TXT + failures CSV during scraping
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
    EDA_PREPROCESSED_DATA_PATH,
    LABEL_COLUMN,
    MAX_CHARS,
    RAW_BODY_CAP,
    RANDOM_STATE,
    SECTION_TITLE_MAX_CHARS,
    TEST_SIZE,
    TEXT_COLUMN,
    TITLE_COLUMN,
    TITLE_WEIGHT,
    UNLABELED_VALUE,
)

logger = logging.getLogger(__name__)


def _extract_user_need(tags: str) -> str:
    """Pull the `user_need: …` value out of a comma-separated tag string, else 'none'.

    In:  comma-separated tag string (e.g. 'foo, user_need: educate-me').
    Out: the user-need label or the string 'none'.
    """
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
    """Keep article rows only, optionally narrow to a date window, and attach `User_Needs`.

    In:  full raw export df; optional ISO date bounds (YYYY-MM-DD).
    Out: a fresh df (input untouched) with `Publish_date` + `User_Needs` columns
         and `other-not-news` rows removed.
    """
    art_df = raw_df[raw_df["Post id"].notna()].copy()
    logger.info("%d articles from %d total rows.", len(art_df), len(raw_df))

    art_df["Publish_date"] = art_df["Publish date"].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M")
    )

    if start_date:
        art_df = art_df.loc[
            art_df["Publish_date"] >= datetime.datetime.strptime(start_date, "%Y-%m-%d")
        ].copy()
    if end_date:
        art_df = art_df.loc[
            art_df["Publish_date"] < datetime.datetime.strptime(end_date, "%Y-%m-%d")
        ].copy()

    logger.info(
        "Filtered to %d articles between %s and %s.",
        len(art_df), start_date, end_date,
    )

    # `Tags` may be NaN — coerce to empty string before splitting.
    art_df["User_Needs"] = art_df["Tags"].fillna("").apply(_extract_user_need)
    art_df = art_df.loc[art_df["User_Needs"] != "other-not-news"].reset_index(drop=True)

    logger.info(
        "User Needs distribution:\n%s",
        art_df["User_Needs"].value_counts(normalize=True),
    )

    return art_df


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
    """Fetch one article URL and extract its title + paragraph texts.

    In:  url; optional auth cookies; HTTP headers; request timeout (seconds).
    Out: dict with keys `url`, `title`, `paragraphs` (list of paragraph strings).
    Raises ValueError when the expected DOM structure isn't found.
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
    """Context manager that turns SIGINT/SIGTERM into an `interrupted` flag.

    Used during scraping so Ctrl-C finishes the current article and flushes
    pending rows instead of dropping them.
    """

    def __init__(self):
        """Initialize with `interrupted=False`."""
        self.interrupted = False

    def __enter__(self):
        """Install SIGINT/SIGTERM handlers and return self."""
        self._prev_int = signal.getsignal(signal.SIGINT)
        self._prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        return self

    def __exit__(self, *_):
        """Restore the previous signal handlers."""
        signal.signal(signal.SIGINT, self._prev_int)
        signal.signal(signal.SIGTERM, self._prev_term)

    def _handle(self, _signum, _frame):
        """Set the `interrupted` flag and log a warning."""
        logger.warning("Interrupt received — finishing current article then saving…")
        self.interrupted = True


def _load_checkpoint(path: Path) -> set[str]:
    """Read scrape-checkpoint URLs from disk; return empty set if file missing."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_checkpoint(path: Path, url: str) -> None:
    """Append a successfully-scraped URL to the checkpoint file."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(url + "\n")


def _flush_rows(rows: list[dict], path: Path) -> None:
    """Append a list of dict rows to a CSV, writing a header only when new."""
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
    """Scrape article text in batches, with checkpointing and graceful interrupt.

    In:  CSV of URLs (`URL` column); output/failure/checkpoint paths; optional
         session cookies; batch size; per-request sleep.
    Out: writes paragraph rows to `output_csv`, failures to `failures_csv`, and
         a one-URL-per-line checkpoint to `checkpoint_file`. Resumable: a
         re-run skips URLs already in the checkpoint.
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


def combine_paragraphs(input_csv: str, output_csv: str) -> None:
    """Group paragraph-level scrape output into one row per article URL.

    In:  paragraph CSV (`url`, `title`, `text`) from `scrape_articles`.
    Out: writes a combined CSV with one row per `url`, paragraphs concatenated.
    """
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
    """Inner-join raw metadata with scraped article text. Inputs are not mutated.

    In:  filtered article df (with `URL` column); text df (with `url` column).
    Out: merged df with the duplicate `url` column dropped.
    """
    merged = raw_df.merge(text_df, left_on="URL", right_on="url", how="inner")
    return merged.drop(columns=["url"])


def save_eda_dataset(
    raw_df: pd.DataFrame,
    text_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Merge → drop nulls → write the EDA CSV (raw merge). Inputs are not mutated.

    In:  filtered article df; scraped-text df; output CSV path.
    Out: returns the cleaned df and writes it to `output_path`.
    """
    merged = _merge_article_text(raw_df, text_df)
    eda_df = merged[_EDA_COLUMNS].copy()
    eda_df["Tablet views"] = eda_df["Tablet views"].fillna(0)
    eda_df = eda_df.dropna().reset_index(drop=True)

    eda_df.to_csv(output_path, index=False)
    logger.info("Saved EDA dataset (%d rows) → %s", len(eda_df), output_path)
    return eda_df


def save_ml_datasets(
    raw_df: pd.DataFrame,
    text_df: pd.DataFrame,
    tagged_path: str,
    untagged_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge → split by label presence → write tagged + untagged ML CSVs.

    In:  filtered article df; scraped-text df; output paths for tagged/untagged.
    Out: (tagged_df, untagged_df); writes both CSVs as a side effect.
    """
    merged = _merge_article_text(raw_df, text_df)
    ml_df = merged[_ML_COLUMNS].dropna().reset_index(drop=True)

    tagged_df = ml_df.loc[ml_df["User_Needs"] != "none"].reset_index(drop=True).copy()
    untagged_df = ml_df.loc[ml_df["User_Needs"] == "none"].reset_index(drop=True).copy()

    tagged_df.to_csv(tagged_path, index=False)
    untagged_df.to_csv(untagged_path, index=False)

    logger.info("Saved %d tagged rows: %s", len(tagged_df), tagged_path)
    logger.info("Saved %d untagged rows: %s", len(untagged_df), untagged_path)
    return tagged_df, untagged_df


def _ensure_nltk():
    """Download required NLTK corpora the first time they're needed."""
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
    """Lowercase, strip HTML/URLs/punctuation, lemmatize, and drop stopwords.

    In:  raw text.
    Out: cleaned text — space-joined lemmatized tokens (length > 2, non-stopword).
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens
              if t not in _STOP and len(t) > 2]
    return " ".join(tokens)


def build_combined_clean(row, title_weight: int = TITLE_WEIGHT) -> str:
    """Build a cleaned-text input with the title repeated for TF-IDF emphasis.

    In:  DataFrame row with `Title` + `text`; how many times to repeat title.
    Out: single space-joined cleaned string.
    """
    title_clean = clean_text(row[TITLE_COLUMN])
    body_clean = clean_text(row[TEXT_COLUMN])
    return " ".join([title_clean] * title_weight + [body_clean])


def build_combined_raw(row) -> str:
    """Build a minimally-processed `title [SEP] body` input for transformer models.

    In:  DataFrame row with `Title` + `text`. Body is truncated to RAW_BODY_CAP words.
    Out: single string `"title [SEP] body words..."`.
    """
    title = str(row[TITLE_COLUMN]).strip()
    body = str(row[TEXT_COLUMN]).strip()
    body_words = body.split()[:RAW_BODY_CAP]
    return f"{title} [SEP] {' '.join(body_words)}"


def build_section_title_text(row, max_chars: int = SECTION_TITLE_MAX_CHARS) -> str:
    """Build a `section | title | body` input for RoBERTa-style fine-tuning.

    In:  DataFrame row with `Section`, `Title`, `text`; max output chars.
    Out: char-capped string `"section | title | body"`.
    """
    section = "" if pd.isna(row.get("Section")) else str(row["Section"])
    title = "" if pd.isna(row.get(TITLE_COLUMN)) else str(row[TITLE_COLUMN])
    body = "" if pd.isna(row.get(TEXT_COLUMN)) else str(row[TEXT_COLUMN])
    return f"{section} | {title} | {body}"[:max_chars]


_PREPROCESSED_COLUMNS = ("combined_text", "combined_short", "clean_combined", "raw_combined")


def preprocess_eda_dataset(
    df: pd.DataFrame,
    save_path: Optional[str] = EDA_PREPROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Add derived text columns on a copy and (optionally) save the result.

    In:  raw-merge df; output path (default = EDA_PREPROCESSED_DATA_PATH).
    Out: new df with `combined_text`, `combined_short`, `clean_combined`,
         `raw_combined` columns added. Writes to `save_path` unless None.
         The input `df` is never modified, and the FULL CSV on disk is never
         overwritten — outputs always go to a distinct filename.
    """
    out = df.copy()
    out[TEXT_COLUMN] = out[TEXT_COLUMN].fillna("")
    out["combined_text"] = out[TITLE_COLUMN].astype(str) + ". " + out[TEXT_COLUMN]
    out["combined_short"] = (
        out[TITLE_COLUMN].astype(str) + ". " + out[TEXT_COLUMN].str[:MAX_CHARS]
    )

    logger.info("Preprocessing %d articles (clean + raw combined)...", len(out))
    out["clean_combined"] = out.apply(build_combined_clean, axis=1)
    out["raw_combined"] = out.apply(build_combined_raw, axis=1)

    if save_path:
        out.to_csv(save_path, index=False)
        logger.info("Saved preprocessed dataset (%d rows) → %s", len(out), save_path)
    return out


def load_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed EDA CSV, computing text columns in memory if absent.

    In:  CSV path (default = EDA_data-PREPROCESSED.csv).
    Out: DataFrame guaranteed to have the four preprocessed text columns. If
         the file is the un-preprocessed FULL CSV, preprocessing runs in
         memory and is NOT persisted (the FULL file on disk stays intact).
    """
    df = pd.read_csv(path)
    if not all(col in df.columns for col in _PREPROCESSED_COLUMNS):
        df = preprocess_eda_dataset(df, save_path=None)
    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print row count, columns, null counts, text-length stats, and label distribution.

    In:  loaded DataFrame (typically the EDA preprocessed CSV).
    Out: stdout summary; no return value.
    """
    print(f"Loaded {len(df):,} articles  |  columns: {list(df.columns)}")
    print(f"Title null: {df[TITLE_COLUMN].isna().sum()}, "
          f"text null: {df[TEXT_COLUMN].isna().sum()}")
    lengths = df[TEXT_COLUMN].str.len()
    print(f"Text length — mean: {lengths.mean():.0f}, "
          f"median: {lengths.median():.0f}, max: {lengths.max():.0f}")
    print(f"\n{LABEL_COLUMN} distribution:\n{df[LABEL_COLUMN].value_counts()}")


def split_labeled_unlabeled(df: pd.DataFrame):
    """Partition rows into labelled vs unlabelled (label == UNLABELED_VALUE).

    In:  DataFrame with the project label column.
    Out: (df_labeled, df_unlabeled) — both copies so the input is untouched.
    """
    mask = df[LABEL_COLUMN] != UNLABELED_VALUE
    return df[mask].copy(), df[~mask].copy()


def encode_labels(df_labeled: pd.DataFrame):
    """Fit a sklearn LabelEncoder on the labelled rows.

    In:  labelled DataFrame.
    Out: (fitted encoder, integer label array `y`).
    """
    le = LabelEncoder()
    y = le.fit_transform(df_labeled[LABEL_COLUMN])
    return le, y


def stratified_split(X, y, indices, test_size=TEST_SIZE,
                     random_state=RANDOM_STATE):
    """Stratified train/test split that also carries an integer index array through.

    In:  feature array `X`, label array `y`, index array; split ratio + seed.
    Out: (X_train, X_test, y_train, y_test, train_idx, test_idx).
    """
    return train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_supervised_data(df: pd.DataFrame):
    """Split labelled / unlabelled, encode labels, build train/test for each text variant.

    In:  preprocessed DataFrame with the four text columns + the label column.
    Out: dict containing df_labeled, df_unlabeled, label_encoder, label_names,
         y_train / y_test, train_idx / test_idx / labeled_idx / unlabeled_idx,
         and X_*_(text|short|clean|raw) arrays for both splits and unlabelled.
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
