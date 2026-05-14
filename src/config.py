"""Project-wide paths, hyperparameters, and model registry.

Library-only. Imported by every other module to keep filenames, label sets,
and per-model knobs in one place.

Inputs:
    Optional env vars: DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR,
    EDA_OUTPUT. Each overrides its default directory.

Outputs:
    Path constants (e.g. EDA_DATA_PATH), label/text constants, training
    hyperparameters, EMBEDDING_MODELS registry, CLASS_COLORS for plotting.
"""

import os
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(PROJECT_ROOT, "models", "artifacts")
)
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, "models", "checkpoints")
)
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
EDA_OUTPUT = os.environ.get("EDA_OUTPUT", os.path.join(OUTPUT_DIR, "eda"))

# Two canonical EDA CSVs (FULL = raw merge, PREPROCESSED = + text columns).
# DATA_PATH aliases the preprocessed file — used as default load target.
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "posts-export-by-page-views-Feb-01-2025-Mar-05-2026-Masthead-Maine.csv",
)
EDA_DATA_PATH = os.path.join(DATA_DIR, "EDA_data-FULL.csv")
EDA_PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "EDA_data-PREPROCESSED.csv")
ML_TAGGED_DATA_PATH = os.path.join(DATA_DIR, "ML_tagged_data-FULL.csv")
ML_UNTAGGED_DATA_PATH = os.path.join(DATA_DIR, "ML_untagged_data-FULL.csv")
DATA_PATH = EDA_PREPROCESSED_DATA_PATH

EMBEDDINGS_MINI_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_mini.npy")
EMBEDDINGS_MPNET_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_mpnet.npy")
EMBEDDINGS_LONGFORMER_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_longformer.npy")
EMBEDDINGS_GEMMA_CLS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_gemma_classification.npy")
EMBEDDINGS_GEMMA_CLUSTER_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_gemma_cluster.npy")
SEMANTIC_EMBEDDINGS_H5_PATH = os.path.join(EMBEDDINGS_DIR, "semantic_embeddings.h5")

USER_NEEDS_LABELS = [
    "update-me",
    "educate-me",
    "give-me-perspective",
    "help-me",
    "connect-me",
    "inspire-me",
]

LABEL_COLUMN = "User_Needs"
TEXT_COLUMN = "text"
TITLE_COLUMN = "Title"
UNLABELED_VALUE = "none"

MAX_CHARS = 2000
TITLE_WEIGHT = 3
RAW_BODY_CAP = 512
SECTION_TITLE_MAX_CHARS = 1500

RANDOM_STATE = 42
TOP_K = 3

CUSTOM_STOPWORDS = frozenset(
    {
        "maine",
        "said",
        "one",
        "also",
        "people",
        "state",
        "year",
        "portland",
        "time",
        "like",
        "would",
        "get",
        "new",
    }
)

TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95

TFIDF_CLF_MAX_FEATURES = 15_000
TFIDF_CLF_MIN_DF = 2

N_TOPICS_NMF = 8

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST_CLUSTER = 0.0
UMAP_MIN_DIST_VIZ = 0.1
UMAP_METRIC = "cosine"
UMAP_N_COMPONENTS_CLUSTER = 5
UMAP_N_COMPONENTS_VIZ = 2

HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_SELECTION_METHOD = "eom"

KMEANS_K_RANGE = range(3, 16)

LR_MAX_ITER = 2000
LR_C = 1.0
SVC_MAX_ITER = 5000
GBM_N_ESTIMATORS = 200
GBM_MAX_DEPTH = 5
GBM_LEARNING_RATE = 0.1
RF_N_ESTIMATORS = 300
CV_N_SPLITS = 5

SETFIT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SETFIT_FEW_SHOT_PER_CLASS = 64
SETFIT_BATCH_SIZE_FEW = 16
SETFIT_BATCH_SIZE_FULL = 32
SETFIT_NUM_EPOCHS = 3
SETFIT_NUM_ITERATIONS = 20

# Optional GBDT classifiers (catboost / xgboost / lightgbm packages)
CATBOOST_PARAMS = dict(
    iterations=1000,
    depth=8,
    learning_rate=0.04,
    random_seed=RANDOM_STATE,
    verbose=0,
    auto_class_weights="Balanced",
    bootstrap_type="MVS",
)
XGBOOST_PARAMS = dict(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
LIGHTGBM_PARAMS = dict(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)

FFNN_HIDDEN_DIMS = (512, 256, 128)
FFNN_DROPOUT = 0.35
FFNN_LR = 1e-3
FFNN_WEIGHT_DECAY = 1e-5
FFNN_NUM_EPOCHS = 30
FFNN_BATCH_SIZE = 64

# RNN ensemble: reshape (N, 768) embeddings into (N, seq_len, feat_dim)
RNN_SEQ_LEN = 12
RNN_FEAT_DIM = 64  # seq_len * feat_dim must equal embedding dim (768)
RNN_HIDDEN = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3
RNN_LR = 1e-3
RNN_NUM_EPOCHS = 25
RNN_BATCH_SIZE = 64

ROBERTA_MODEL_ID = "roberta-base"
ROBERTA_MAX_LENGTH = 512
ROBERTA_TRAIN_BATCH_SIZE = 16
ROBERTA_EVAL_BATCH_SIZE = 16
ROBERTA_LEARNING_RATE = 2e-5
ROBERTA_NUM_EPOCHS = 10
ROBERTA_WARMUP_STEPS = 300
ROBERTA_WEIGHT_DECAY = 0.01
ROBERTA_GRADIENT_ACCUMULATION_STEPS = 2
ROBERTA_FP16 = True
ROBERTA_SAVE_TOTAL_LIMIT = 3
ROBERTA_LOGGING_STEPS = 50
ROBERTA_LR_SCHEDULER = "cosine"

LONGFORMER_MODEL_ID = "allenai/longformer-base-4096"
LONGFORMER_MAX_LENGTH = 4096
LONGFORMER_TRAIN_BATCH_SIZE = 2
LONGFORMER_EVAL_BATCH_SIZE = 4
LONGFORMER_LEARNING_RATE = 2e-5
LONGFORMER_NUM_EPOCHS = 5
LONGFORMER_WARMUP_RATIO = 0.1
LONGFORMER_WEIGHT_DECAY = 0.01
LONGFORMER_GRADIENT_ACCUMULATION_STEPS = 8
LONGFORMER_FP16 = True
LONGFORMER_SAVE_STEPS = 200
LONGFORMER_EVAL_STEPS = 200
LONGFORMER_LOGGING_STEPS = 50


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Per-model encode settings: HF id, dimension, batch size, max seq length."""

    name: str
    model_id: str
    dim: int
    batch_size: int
    max_seq_length: Optional[int] = None


EMBEDDING_MODELS = {
    "mini": EmbeddingModelConfig(
        name="MiniLM",
        model_id="all-MiniLM-L6-v2",
        dim=384,
        batch_size=64,
        max_seq_length=256,
    ),
    "mpnet": EmbeddingModelConfig(
        name="MPNet",
        model_id="all-mpnet-base-v2",
        dim=768,
        batch_size=32,
        max_seq_length=384,
    ),
    "longformer": EmbeddingModelConfig(
        name="Longformer",
        model_id="allenai/longformer-base-4096",
        dim=768,
        batch_size=8,
        max_seq_length=4096,
    ),
    "gemma": EmbeddingModelConfig(
        name="EmbeddingGemma",
        model_id="google/embeddinggemma-300m",
        dim=768,
        batch_size=32,
        max_seq_length=2048,
    ),
}

GEMMA_TASKS = ("classification", "clustering")

CLASS_COLORS = {
    "update-me": "#8a00cfff",
    "give-me-perspective": "#0fb400ff",
    "educate-me": "#e0a100ff",
    "connect-me": "#c31000ff",
    "inspire-me": "#0038c7ff",
    "help-me": "#00cbcfff",
    "none": "#818181",
}
LABELED_CLASSES = [c for c in CLASS_COLORS if c != "none"]
