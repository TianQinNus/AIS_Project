import math
from functools import lru_cache
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "multilingual-minilm"
)

EMBEDDING_DIMENSION = 384

_model_lock = Lock()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Embedding model not found: {MODEL_PATH}. "
            "Download the model before starting FleetCare."
        )

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if model.get_sentence_embedding_dimension() != 384:
        raise RuntimeError(
            "Embedding model dimension mismatch"
        )

    return model


def embed_query(query: str) -> list[float]:

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    model = get_embedding_model()

    with _model_lock:
        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    vector = embedding.tolist()

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            "Invalid embedding dimension"
        )

    if not all(math.isfinite(x) for x in vector):
        raise ValueError(
            "Embedding contains invalid values"
        )

    return vector