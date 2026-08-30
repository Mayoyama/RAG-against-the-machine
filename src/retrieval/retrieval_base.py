from abc import ABC, abstractmethod
from src import MinimalSource


class RetrievalBase(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[MinimalSource]:
        pass
