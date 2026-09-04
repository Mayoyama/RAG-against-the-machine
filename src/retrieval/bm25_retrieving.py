"""Keyword (BM25) retrieval over the saved index."""

from .retrieval_base import RetrievalBase
from src import MinimalSource, expand_snake_case
import bm25s


class Bm25Retriever(RetrievalBase):
    """Retrieves chunks by BM25 keyword score over a saved index."""

    def __init__(self, save_dir: str) -> None:
        """Load the saved BM25 index and its chunk corpus.

        Args:
            save_dir: Directory the BM25 index was saved under.
        """

        self.retriever = bm25s.BM25.load(save_dir, load_corpus=True)

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Find the k best BM25 matches for a query.

        Args:
            query: The search query.
            k: Number of top results to return.

        Returns:
            The top-k matching sources, ranked by BM25 score.
        """

        if k > len(self.retriever.corpus):
            raise ValueError("Bm25Retriever-search: invalid value [k]")

        min_srcs: list[MinimalSource] = []

        query_tokens = bm25s.tokenize(expand_snake_case(query))
        results, _ = self.retriever.retrieve(query_tokens, k=k)

        for res in results[0]:
            min_srcs.append(MinimalSource(**res['source']))

        return min_srcs
