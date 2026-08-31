from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src import Chunk
from ..utils import output_progress_bar


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_semantic_index(chunks: list[Chunk]) -> np.ndarray:
    """Embed every chunk's content into a dense vector.

    Args:
        chunks: Chunks to embed. Must be the same list, in the same
            order, as the one passed to build_bm25_index — row i here
            has to line up with chunk i in the saved BM25 corpus.

    Returns:
        An array of shape (len(chunks), embedding_dim).
    """

    def sem_idx() -> SentenceTransformer:
        return SentenceTransformer(EMBEDDING_MODEL)

    model = output_progress_bar(sem_idx, "Building semantic index")
    matrix = model.encode([chunk.content for chunk in chunks])

    return matrix


def save_semantic_index(vectors: np.ndarray, save_dir: str) -> None:
    """Persist embeddings to disk, alongside the BM25 index.

    Args:
        vectors: Embeddings from build_semantic_index.
        save_dir: Directory to save under (same as the BM25 index).
    """

    pathname = Path(save_dir) / "semantic_index.npy"
    np.save(pathname, vectors)
