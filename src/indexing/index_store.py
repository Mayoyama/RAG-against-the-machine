from pathlib import Path
from .chunking import chunk_file
from src import Chunk
from tqdm import tqdm
import bm25s


ALLOWED_EXTENSIONS = (".py", ".md", ".txt", ".json")


def build_pathlist(root_dir: str) -> list[str]:
    """Find all indexable files under a directory, recursively.

    Args:
        root_dir: Root directory to search.

    Returns:
        Pathnames of every file whose extension is in ALLOWED_EXTENSIONS.
    """

    raw = Path(root_dir).rglob('*')

    files = [str(item) for item in raw
             if item.suffix in ALLOWED_EXTENSIONS
             and item.is_file()]

    if not files:
        raise ValueError("build_pathlist: No files found")

    return files


def build_chunks(paths: list[str], chunk_size_limit: int) -> list[Chunk]:
    """Chunk every file in paths into a single flat list of Chunks.

    Args:
        paths: Pathnames to chunk.
        chunk_size_limit: Maximum length of each chunk's content.

    Returns:
        All Chunks from every file, in file order.
    """

    all_chunks: list[Chunk] = []

    for path in tqdm(paths, desc="Indexing files"):
        all_chunks.extend(chunk_file(path, chunk_size_limit))

    return all_chunks


def build_bm25_index(chunks: list[Chunk]) -> bm25s.BM25:
    """Build a BM25 index over a list of Chunks' content.

    Args:
        chunks: Chunks to index, in a fixed order.

    Returns:
        An indexed BM25 retriever, ready to save or query.
    """

    bm25_strings = [chunk.content for chunk in chunks]
    bm25_tokens = bm25s.tokenize(bm25_strings)
    retriever = bm25s.BM25()
    retriever.index(bm25_tokens)

    return retriever


def save_index(retriever: bm25s.BM25, chunks: list[Chunk],
               save_dir: str) -> None:
    """Persist a BM25 index and its source chunks to disk.

    Args:
        retriever: An indexed BM25 retriever.
        chunks: The Chunks the retriever was built from, same order.
        save_dir: Directory to save the index and chunks under.
    """

    chunk_json = [chunk.model_dump() for chunk in chunks]
    retriever.save(save_dir, corpus=chunk_json)


def index_corpus(root_dir: str, chunk_size_limit: int, save_dir: str) -> None:
    """Chunk a directory's files and persist a BM25 index for them.

    Args:
        root_dir: Root directory to index.
        chunk_size_limit: Maximum length of each chunk's content.
        save_dir: Directory to save the resulting index under.
    """

    paths = build_pathlist(root_dir)
    chunks = build_chunks(paths, chunk_size_limit)
    retriever = build_bm25_index(chunks)
    save_index(retriever, chunks, save_dir)
