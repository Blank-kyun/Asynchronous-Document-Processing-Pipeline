from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # matches the model above


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model once per process.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """
    Embed a list of strings into dense vectors.
    """
    model = _load_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors).tolist()
