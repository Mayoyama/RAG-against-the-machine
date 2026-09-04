"""Common interface all retrievers implement."""

from abc import ABC, abstractmethod
from src import MinimalSource


class RetrievalBase(ABC):
    """Shared interface for BM25, semantic, and hybrid retrievers."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k matching sources for a query."""

        pass
