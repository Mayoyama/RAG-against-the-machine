"""Building, saving, and loading the semantic (embedding) index."""

from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src import Chunk
from ..utils import output_progress_bar
from sys import stderr


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

    def load_model() -> SentenceTransformer:
        """Load the embedding model."""

        return SentenceTransformer(EMBEDDING_MODEL)

    def run_encoding() -> np.ndarray:
        """Encode every chunk's content with the loaded model."""

        return model.encode([chunk.content for chunk in chunks])

    model = output_progress_bar(load_model, "Loading embedding model")
    print(flush=True)
    matrix = output_progress_bar(run_encoding, "Embedding chunks")
    print(flush=True)

    return matrix


def save_semantic_index(vectors: np.ndarray, save_dir: str) -> None:
    """Persist embeddings to disk, alongside the BM25 index.

    Args:
        vectors: Embeddings from build_semantic_index.
        save_dir: Directory to save under (same as the BM25 index).
    """

    try:
        pathname = Path(save_dir) / "semantic_index.npy"
        np.save(pathname, vectors)
    except IOError as e:
        print(f"save_semantic_index: Error saving file: {e}",
              file=stderr)


def load_semantic_index(save_dir: str) -> np.ndarray | None:
    """Load the previous run's embedding matrix from disk, if any.

    Args:
        save_dir: Directory the semantic index was saved under (same
            directory as the BM25 index and manifest).

    Returns:
        The saved embedding matrix, or None if no previous run exists
        yet - first run, after a clean, or if the saved file is
        missing or corrupted.
    """

    try:
        pathname = Path(save_dir) / "semantic_index.npy"
        content: np.ndarray = np.load(pathname)
        return content
    except (IOError, ValueError) as e:
        print(f"load_semantic_index: Error loading file: {e}",
              file=stderr)
        return None
