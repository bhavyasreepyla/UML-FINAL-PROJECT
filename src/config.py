import os
from dataclases import dataclass
from typing import Optional

DATA_PATH = "../data/EDA_data-FULL.csv"
MAX_CHARS = 2000

USER_NEEDS_LABELS = [
    "update-me",
    "educate-me",
    "give-me-perspective",
    "help-me",
    "connect-me",
    "inspire-me",
]

RANDOM_STATE = 42

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models/artifacts")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "models/checkpoints")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

DATA_PATH = os.path.join(DATA_DIR, "EDA_data-FULL.csv")
EMBEDDINGS_MINI_PATH = os.path.join(DATA_DIR, "embeddings_mini.npy")
EMBEDDINGS_MPNET_PATH = os.path.join(DATA_DIR, "embeddings_mpnet.npy")
EMBEDDINGS_LONGFORMER_PATH = os.path.join(DATA_DIR, "embeddings_longformer.npy")

MAX_CHARS = 2000
LABEL_COLUMN = "User_Needs"
TEXT_COLUMN = "text"
TITLE_COLUMN = "Title"
UNLABELED_VALUE = "none"
TITLE_WEIGHT = 3  # repeat title N times in clean_combined for TF-IDF / sklearn
RAW_BODY_CAP = 512  # word-level cap for raw_combined (transformer input)

# Domain-specific stopwords to add on top of NLTK english defaults
CUSTOM_STOPWORDS = {
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


@dataclass
class EmbeddingModelConfig:
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
