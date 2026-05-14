"""Top-K predicted-tag helpers shared across all classifier families.

Library-only. Three entry points cover every model the project trains:
  - sklearn-style (LR/SVC/NB/RF/GBM, SetFit, CatBoost/XGBoost/LightGBM)
  - raw probability matrix (HF transformers, ensembles)
  - PyTorch classifier returning logits (FFNN, RNN)

Inputs:
    Trained models or precomputed probability arrays + label-name list.

Outputs:
    pandas DataFrames with columns `tag_1, confidence_1, tag_2, ...`. None of
    these helpers write to disk; callers persist the results.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from config import TOP_K


def top_k_from_proba(
    proba: np.ndarray,
    label_names: List[str],
    k: int = TOP_K,
) -> pd.DataFrame:
    """Convert a (N, C) probability matrix into a top-K-tags DataFrame.

    In:  proba — 2D array of class probabilities; label_names — class names.
    Out: DataFrame with tag_1/confidence_1 ... tag_k/confidence_k per row.
    """
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape {proba.shape}")
    k = min(k, proba.shape[1])

    top_idx = np.argsort(-proba, axis=1)[:, :k]
    rows = []
    for i, idxs in enumerate(top_idx):
        row = {}
        for rank, idx in enumerate(idxs, 1):
            row[f"tag_{rank}"] = label_names[idx]
            row[f"confidence_{rank}"] = round(float(proba[i, idx]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_top_k(model, X, label_names: List[str], k: int = TOP_K) -> pd.DataFrame:
    """Top-K for any estimator exposing `.predict_proba(X)`.

    In:  fitted sklearn-style model; feature matrix or text list; class names.
    Out: top-K DataFrame as in `top_k_from_proba`.
    """
    return top_k_from_proba(np.asarray(model.predict_proba(X)), label_names, k)


def predict_top_k_torch(
    model: torch.nn.Module,
    X: np.ndarray,
    label_names: List[str],
    k: int = TOP_K,
    *,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Top-K for a PyTorch classifier returning logits (FFNN / RNN).

    In:  torch model; dense embeddings (N, D); class names; optional device.
    Out: top-K DataFrame after softmax over the logits.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)

    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            logits = model(X_t[start : start + batch_size].to(device))
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return top_k_from_proba(np.vstack(chunks), label_names, k)


def attach_predictions_to_df(
    df_unlabeled: pd.DataFrame,
    df_top_k: pd.DataFrame,
    title_col: str = "Title",
) -> pd.DataFrame:
    """Align top-K DataFrame index with the source df and prepend the article title.

    In:  unlabeled-articles df; top-K df (same length).
    Out: top-K df indexed like the source, with `Title` as the first column.
    """
    df_top_k = df_top_k.copy()
    df_top_k.index = df_unlabeled.index
    df_top_k.insert(0, title_col, df_unlabeled[title_col].values)
    return df_top_k


def print_confidence_stats(df_top_k: pd.DataFrame) -> None:
    """Print mean/median/min/max of the top-1 confidence column.

    In:  top-K df with a `confidence_1` column.
    Out: stdout summary; no return value.
    """
    conf = df_top_k["confidence_1"]
    print(
        f"Top-1 confidence — mean: {conf.mean():.4f}, "
        f"median: {conf.median():.4f}, "
        f"min: {conf.min():.4f}, max: {conf.max():.4f}"
    )
