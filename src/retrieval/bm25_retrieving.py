from .retrieval_base import RetrievalBase
from src import MinimalSource, expand_snake_case
import bm25s


class Bm25Retriever(RetrievalBase):
    def __init__(self, save_dir: str) -> None:
        self.retriever = bm25s.BM25.load(save_dir, load_corpus=True)

    def search(self, query: str, k: int) -> list[MinimalSource]:

        if k > len(self.retriever.corpus):
            raise ValueError("Bm25Retriever-search: invalid value [k]")

        min_srcs: list[MinimalSource] = []

        query_tokens = bm25s.tokenize(expand_snake_case(query))
        results, _ = self.retriever.retrieve(query_tokens, k=k)

        for res in results[0]:
            min_srcs.append(MinimalSource(**res['source']))

        return min_srcs
