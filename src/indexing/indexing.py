from pathlib import Path
from .chunking import chunk_file
from .embedding import (save_semantic_index, build_semantic_index,
                        load_semantic_index)
from .manifesting import (load_manifest, save_manifest, file_has_changed,
                          build_manifest_entries)
from src import Chunk, ManifestItem
from ..utils import expand_snake_case
from tqdm import tqdm
import numpy as np
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
    print(flush=True)
    
    return all_chunks


def build_bm25_index(chunks: list[Chunk]) -> bm25s.BM25:
    """Build a BM25 index over a list of Chunks' content.

    Args:
        chunks: Chunks to index, in a fixed order.

    Returns:
        An indexed BM25 retriever, ready to save or query.
    """

    bm25_strings = [expand_snake_case(chunk.content) for chunk in chunks]
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


def index_corpus(root_dir: str, chunk_size_limit: int, save_dir: str,
                 apply_bonuses: bool = False) -> None:
    """Chunk a directory's files and persist a BM25 index, optionally
    with semantic (embedding) indexing and incremental reuse.

    By default, only chunks the corpus and builds a BM25 index - no
    embeddings, no manifest. When apply_bonuses is True, also builds
    a semantic index, and skips re-chunking/re-embedding files whose
    modification time matches a previous run, reusing their cached
    chunks and embedding vectors instead. Changed, new, and deleted
    files are detected via the manifest saved under save_dir - never
    touched directly, only through load_manifest/save_manifest.

    Args:
        root_dir: Root directory to index.
        chunk_size_limit: Maximum length of each chunk's content.
        save_dir: Directory to save the resulting index under.
        apply_bonuses: If True, also builds a semantic index and
            enables incremental reuse of unchanged files' chunks and
            embeddings across runs.
    """

    paths = build_pathlist(root_dir)
    
    if apply_bonuses:
        manifest = load_manifest(save_dir)
        old_sem_index = load_semantic_index(save_dir)

        changed_paths: list[str] = []
        unchanged_chunks: list[Chunk] = []
        unchanged_vectors: list[np.ndarray] = []
        mod_times: dict[str, float] = {}

        for path in tqdm(paths):
            changed, mod_time = file_has_changed(path, manifest)
            if changed:
                changed_paths.append(path)
            else:
                item = manifest.files[path]
                if old_sem_index is not None:
                    unchanged_chunks.extend(item.chunks)
                    unchanged_vectors.extend(old_sem_index[
                        item.embedding_row_start: item.embedding_row_end])
            mod_times[path] = mod_time
        print(flush=True)
        
        new_chunks = build_chunks(changed_paths, chunk_size_limit)
        new_vectors = build_semantic_index(new_chunks)

        final_chunks = unchanged_chunks + new_chunks
        matrix = np.vstack(unchanged_vectors + list(new_vectors))

        manifest.files = build_manifest_entries(final_chunks, mod_times)
        manifest.max_chunk_size = chunk_size_limit

        for path in paths:
            if path not in manifest.files:
                manifest.files[path] = ManifestItem(
                    mod_time=mod_times[path],
                    chunks=[],
                    embedding_row_start=len(matrix),
                    embedding_row_end=len(matrix)
                )

        save_semantic_index(matrix, save_dir)
        save_manifest(manifest, save_dir)

    # Non-bonus version:
    else:
        final_chunks = build_chunks(paths, chunk_size_limit)

    retriever = build_bm25_index(final_chunks)
    save_index(retriever, final_chunks, save_dir)

    if apply_bonuses:
        print(f"Re-indexation: {len(changed_paths)} files changed.\n"
            f"{len(paths) - len(changed_paths)} files unchanged and reused")
