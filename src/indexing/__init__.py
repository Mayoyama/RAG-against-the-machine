"""Public indexing API."""

from .indexing import index_corpus
from .embedding import EMBEDDING_MODEL

__all__ = [
    "index_corpus",
    "EMBEDDING_MODEL"
]
