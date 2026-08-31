from .retrieval_base import RetrievalBase
from .bm25_retrieving import Bm25Retriever
from .semantic_retrieving import SemanticRetriever
from src import MinimalSource


class HybridRetriever(RetrievalBase):
    def __init__(self, save_dir: str) -> None:
        """Load the BM25 and semantic retrievers to fuse results from.

        Args:
            save_dir: Directory the BM25 and semantic indexes were
                saved under (same directory for both).
        """

        self.bm25 = Bm25Retriever(save_dir)
        self.semantic = SemanticRetriever(save_dir)

    @staticmethod
    def build_results_dict(
        results: list[MinimalSource]
    ) -> dict[tuple[str, int], tuple[float, MinimalSource]]:
        """Map each result to its RRF rank contribution and itself.

        Args:
            results: A single retriever's ranked results.

        Returns:
            A dict keyed by (file_path, first_character_index), each
            value a (rank contribution, source) pair.
        """

        dic: dict[tuple[str, int], tuple[float, MinimalSource]] = {}
        for i, res in enumerate(results):
            dic[(res.file_path, res.first_character_index)] = \
                (1 / (60 + i + 1), res)

        return dic

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Retrieve chunks by fusing BM25 and semantic rankings with RRF.

        Args:
            query: The search query.
            k: Number of top results to return.

        Returns:
            The top-k matching sources, ranked by combined RRF score.
        """

        if k > len(self.bm25.retriever.corpus):
            raise ValueError("HybridRetriever-search: invalid value [k]")

        pool_size = min(k * 4, len(self.bm25.retriever.corpus))

        bm25_res = self.bm25.search(query, pool_size)
        semantic_res = self.semantic.search(query, pool_size)

        bm25_res_d = self.build_results_dict(bm25_res)
        semantic_res_d = self.build_results_dict(semantic_res)

        combined: dict[tuple[str, int], tuple[float, MinimalSource]] = {}

        for d in (bm25_res_d, semantic_res_d):
            for key, (score, src) in d.items():
                old_score = combined.get(key, (0, src))[0]
                combined[key] = (old_score + score, src)

        sorted_res = sorted(
            combined.values(), key=lambda pair: pair[0], reverse=True)[:k]

        return [src for _, src in sorted_res]
