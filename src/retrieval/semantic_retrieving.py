"""Semantic (embedding) retrieval over the saved index."""

from .retrieval_base import RetrievalBase
from sentence_transformers import SentenceTransformer
from src import MinimalSource, EMBEDDING_MODEL
from ..utils import output_progress_bar
from pathlib import Path
import numpy as np
import bm25s


class SemanticRetriever(RetrievalBase):
    """Retrieves chunks by cosine similarity over a saved embedding index."""

    def __init__(self, save_dir: str) -> None:
        """Load the embedding matrix, embedding model, and chunk metadata.

        Args:
            save_dir: Directory the semantic and BM25 indexes were
                saved under (same directory for both).
        """

        pathname = Path(save_dir) / "semantic_index.npy"
        try:
            self.matrix = np.load(pathname)
        except (IOError, ValueError) as e:
            raise FileNotFoundError("Valid semantic embeddings not found or "
                                    "not created. Run index() with bonus "
                                    "flag and try again") from e

        def sem_idx() -> SentenceTransformer:
            """Load the embedding model."""

            return SentenceTransformer(EMBEDDING_MODEL)

        self.model = output_progress_bar(sem_idx, "Building semantic index")
        print(flush=True)

        self.retriever = bm25s.BM25.load(save_dir, load_corpus=True)

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Find the k most semantically similar chunks to a query.

        Args:
            query: The search query.
            k: Number of top results to return.

        Returns:
            The top-k matching sources, ranked by cosine similarity.
        """

        if k > len(self.retriever.corpus):
            raise ValueError("SemanticRetriever-search: invalid value [k]")

        query_vector = self.model.encode([query])[0]
        dot_products = self.matrix @ query_vector
        matrix_norms = np.linalg.norm(self.matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        similarities = dot_products / (matrix_norms * query_norm)

        top_k = np.argsort(-similarities)[:k]

        min_srcs: list[MinimalSource] = []

        for i in top_k:
            src = self.retriever.corpus[i]['source']
            min_srcs.append(MinimalSource(**src))

        return min_srcs
