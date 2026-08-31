"""One-off script: compare Bm25Retriever vs SemanticRetriever vs
HybridRetriever output for the same query. Not part of the graded
codebase — safe to delete after use.

Usage:
    uv run python compare_retrievers.py "your query here" 5
"""

import sys
from src import MinimalSource
from src.retrieval import Bm25Retriever
from src.retrieval.semantic_retrieving import SemanticRetriever
from src.retrieval.hybrid_retrieving import HybridRetriever

SAVE_DIR = "data/processed"


def print_results(label: str, results: list[MinimalSource]) -> None:
    print(f"--- {label} ---")
    for src in results:
        print(f"{src.file_path} "
              f"[{src.first_character_index}:{src.last_character_index}]")


def as_set(results: list[MinimalSource]) -> set[tuple[str, int]]:
    return {(s.file_path, s.first_character_index) for s in results}


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "how does attention work"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Query: {query!r}, k={k}\n")

    bm25_results = Bm25Retriever(SAVE_DIR).search(query, k)
    print_results("BM25", bm25_results)

    semantic_results = SemanticRetriever(SAVE_DIR).search(query, k)
    print_results("Semantic", semantic_results)

    hybrid_results = HybridRetriever(SAVE_DIR).search(query, k)
    print_results("Hybrid", hybrid_results)

    bm25_set = as_set(bm25_results)
    semantic_set = as_set(semantic_results)
    hybrid_set = as_set(hybrid_results)

    print(f"\nOverlap BM25 <-> Semantic: {len(bm25_set & semantic_set)}/{k}")
    print(f"Overlap BM25 <-> Hybrid:   {len(bm25_set & hybrid_set)}/{k}")
    print(f"Overlap Semantic <-> Hybrid: "
          f"{len(semantic_set & hybrid_set)}/{k}")


if __name__ == "__main__":
    main()
