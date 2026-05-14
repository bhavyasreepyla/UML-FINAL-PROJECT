"""Compute, cache, and load document embeddings.

Library-only. Supports SBERT (MiniLM, MPNet), Longformer CLS pooling, and
Google EmbeddingGemma. Each computed array is cached under
`data/embeddings/<model>.npy` so reruns load instantly.

Inputs:
    Texts (list[str]) and an optional `force_recompute` flag. Reads cached
    .npy files from `data/embeddings/`.

Outputs:
    np.ndarray of shape (N, dim). Writes the same .npy back as a side effect
    when the cache misses or `force_recompute=True`. `load_gemma_embeddings_h5`
    additionally reads `data/embeddings/semantic_embeddings.h5`.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import (
    DATA_DIR,
    EMBEDDING_MODELS,
    EMBEDDINGS_GEMMA_CLS_PATH,
    EMBEDDINGS_GEMMA_CLUSTER_PATH,
    EMBEDDINGS_LONGFORMER_PATH,
    EMBEDDINGS_MINI_PATH,
    EMBEDDINGS_MPNET_PATH,
    GEMMA_TASKS,
    RANDOM_STATE,
    SEMANTIC_EMBEDDINGS_H5_PATH,
)


_SBERT_CACHE_PATHS = {
    "mini": EMBEDDINGS_MINI_PATH,
    "mpnet": EMBEDDINGS_MPNET_PATH,
    "longformer": EMBEDDINGS_LONGFORMER_PATH,
}

_GEMMA_CACHE_PATHS = {
    "classification": EMBEDDINGS_GEMMA_CLS_PATH,
    "clustering": EMBEDDINGS_GEMMA_CLUSTER_PATH,
}


def cache_path(model_key: str, task: Optional[str] = None) -> Optional[str]:
    """Resolve the on-disk cache path for a given embedding model.

    In:  model_key ('mini' | 'mpnet' | 'longformer' | 'gemma'); `task` is
         required for gemma ('classification' | 'clustering').
    Out: absolute path or None if the key is unknown.
    """
    if model_key == "gemma":
        if task not in GEMMA_TASKS:
            raise ValueError(f"gemma task must be one of {GEMMA_TASKS}, got {task!r}")
        return _GEMMA_CACHE_PATHS[task]
    return _SBERT_CACHE_PATHS.get(model_key)


def _load_cached(path: Optional[str], force_recompute: bool) -> Optional[np.ndarray]:
    """Return cached embeddings if the file exists and we're not forcing recompute."""
    if path and os.path.isfile(path) and not force_recompute:
        print(f"Loading cached embeddings from {path}")
        return np.load(path)
    return None


def _save(path: Optional[str], embeddings: np.ndarray) -> None:
    """Persist embeddings to `path`, creating parent dirs as needed."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or DATA_DIR, exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved to {path}")


def compute_sbert_embeddings(
    texts: List[str],
    model_key: str = "mini",
    force_recompute: bool = False,
) -> np.ndarray:
    """Encode texts with a sentence-transformers SBERT model, with caching.

    In:  list of texts; `model_key` selects MiniLM/MPNet/Longformer config.
    Out: (N, dim) embeddings; also saved to `data/embeddings/<key>.npy`.
    """
    cached = _load_cached(cache_path(model_key), force_recompute)
    if cached is not None:
        return cached

    cfg = EMBEDDING_MODELS[model_key]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model_id)
    if cfg.max_seq_length:
        model.max_seq_length = cfg.max_seq_length

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=cfg.batch_size)
    print(f"{cfg.name} embeddings shape: {embeddings.shape}")
    _save(cache_path(model_key), embeddings)
    return embeddings


def compute_longformer_embeddings(
    texts: List[str],
    max_length: int = 4096,
    batch_size: int = 8,
    force_recompute: bool = False,
) -> np.ndarray:
    """Encode long documents with Longformer CLS pooling (no fine-tuning).

    In:  list of texts; max sequence length; batch size.
    Out: (N, 768) CLS-pooled embeddings; cached to `embeddings_longformer.npy`.
    """
    path = cache_path("longformer")
    cached = _load_cached(path, force_recompute)
    if cached is not None:
        return cached

    from transformers import LongformerModel, LongformerTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        # Longformer needs an explicit global-attention mask on the [CLS] token.
        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            outputs = model(**encoded, global_attention_mask=global_attention_mask)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        if (start // batch_size) % 20 == 0:
            print(
                f"  Longformer batch {start // batch_size + 1}/"
                f"{(len(texts) - 1) // batch_size + 1}"
            )

    embeddings = np.vstack(all_embeddings)
    print(f"Longformer embeddings shape: {embeddings.shape}")
    _save(path, embeddings)
    return embeddings


def _gemma_prompt(title: str, body: str, task: str) -> str:
    """Build the task-tagged prompt string expected by EmbeddingGemma."""
    return f'title: {{title | "{title}"}} | task: {task} | text: {{{body}}}'


def compute_gemma_embeddings(
    texts: List[str],
    titles: Optional[List[str]] = None,
    task: str = "classification",
    force_recompute: bool = False,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Encode texts with google/embeddinggemma-300m using its task-tagged prompts.

    In:  texts; matching titles (optional); task='classification' | 'clustering'.
    Out: (N, 768) embeddings; cached per-task in `data/embeddings/`.
    """
    if task not in GEMMA_TASKS:
        raise ValueError(f"task must be one of {GEMMA_TASKS}, got {task!r}")

    path = cache_path("gemma", task)
    cached = _load_cached(path, force_recompute)
    if cached is not None:
        return cached

    cfg = EMBEDDING_MODELS["gemma"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model_id)
    if cfg.max_seq_length:
        model.max_seq_length = cfg.max_seq_length

    titles = titles or [""] * len(texts)
    documents = [_gemma_prompt(t, x, task) for t, x in zip(titles, texts)]

    encode = getattr(model, "encode_document", model.encode)
    embeddings = np.asarray(
        encode(
            documents,
            show_progress_bar=True,
            batch_size=batch_size or cfg.batch_size,
        )
    )
    print(f"EmbeddingGemma ({task}) shape: {embeddings.shape}")
    _save(path, embeddings)
    return embeddings


def load_gemma_embeddings_h5(
    path: str = SEMANTIC_EMBEDDINGS_H5_PATH,
    task: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """Read EmbeddingGemma vectors from the legacy `semantic_embeddings.h5`.

    In:  h5 path; task selects which dataset ('classification' | 'clustering').
    Out: (embeddings, urls) — embeddings is (N, 768), urls is (N,) of strings.
    """
    if task not in GEMMA_TASKS:
        raise ValueError(f"task must be one of {GEMMA_TASKS}, got {task!r}")
    import h5py

    key = "classification_embedding" if task == "classification" else "cluster_embedding"
    with h5py.File(path, "r") as f:
        embeddings = f[key][:]
        urls = np.asarray([u.decode("utf-8") for u in f["URL"][:]])
    return embeddings, urls


def split_embeddings(
    embeddings: np.ndarray,
    train_idx,
    test_idx,
    unlabeled_idx=None,
) -> dict:
    """Slice a full embedding matrix into train / test / (optional) unlabeled.

    In:  full (N, D) array and integer index arrays from `prepare_supervised_data`.
    Out: dict with keys 'train', 'test', and (if provided) 'unlabeled'.
    """
    splits = {"train": embeddings[train_idx], "test": embeddings[test_idx]}
    if unlabeled_idx is not None:
        splits["unlabeled"] = embeddings[unlabeled_idx]
    return splits
