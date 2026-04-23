"""Project-wide configuration constants and model definitions."""

import os
from dataclasses import dataclass
from typing import Optional

# Directory paths (overridable via environment variables)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "models", "artifacts"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, "models", "checkpoints"))
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
EDA_OUTPUT = os.environ.get("EDA_OUTPUT", os.path.join(OUTPUT_DIR, "eda"))

# Data file paths

DATA_PATH = os.path.join(DATA_DIR, "EDA_data-FULL.csv")
RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "posts-export-by-page-views-Feb-01-2025-Mar-05-2026-Masthead-Maine.csv",
)
EMBEDDINGS_MINI_PATH = os.path.join(DATA_DIR, "embeddings_mini.npy")
EMBEDDINGS_MPNET_PATH = os.path.join(DATA_DIR, "embeddings_mpnet.npy")
EMBEDDINGS_LONGFORMER_PATH = os.path.join(DATA_DIR, "embeddings_longformer.npy")

# Label / text processing constants

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
TITLE_WEIGHT = 3          # repeat title N times in clean_combined for TF-IDF
RAW_BODY_CAP = 512        # word-level cap for raw_combined (transformer input)

RANDOM_STATE = 42
TOP_K = 3

# Domain-specific stopwords to add on top of NLTK english defaults
CUSTOM_STOPWORDS = frozenset({
    "maine", "said", "one", "also", "people", "state", "year",
    "portland", "time", "like", "would", "get", "new",
})

TEST_SIZE = 0.2 # Train / test split

# TF-IDF (unsupervised / EDA)
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95

# TF-IDF (supervised classifiers)
TFIDF_CLF_MAX_FEATURES = 15_000
TFIDF_CLF_MIN_DF = 2

# NMF
N_TOPICS_NMF = 8

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST_CLUSTER = 0.0
UMAP_MIN_DIST_VIZ = 0.1
UMAP_METRIC = "cosine"
UMAP_N_COMPONENTS_CLUSTER = 5
UMAP_N_COMPONENTS_VIZ = 2

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_SELECTION_METHOD = "eom"

# K-Means
KMEANS_K_RANGE = range(3, 16)

# Supervised classifiers
LR_MAX_ITER = 2000
LR_C = 1.0
SVC_MAX_ITER = 5000
GBM_N_ESTIMATORS = 200
GBM_MAX_DEPTH = 5
GBM_LEARNING_RATE = 0.1
RF_N_ESTIMATORS = 300
CV_N_SPLITS = 5

# SetFit
SETFIT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SETFIT_FEW_SHOT_PER_CLASS = 64
SETFIT_BATCH_SIZE_FEW = 16
SETFIT_BATCH_SIZE_FULL = 32
SETFIT_NUM_EPOCHS = 3
SETFIT_NUM_ITERATIONS = 20

# Longformer
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

# Embedding model definitions
@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Configuration for a sentence-embedding model."""
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
}

# EDA visualization colors 
CLASS_COLORS = {
    "update-me": "#9962b4ff",
    "give-me-perspective": "#83b37fff",
    "educate-me": "#e3c375ff",
    "connect-me": "#bb6861ff",
    "inspire-me": "#7493e3ff",
    "help-me": "#87c0c1ff",
    "none": "#818181",
}
LABELED_CLASSES = [c for c in CLASS_COLORS if c != "none"]
