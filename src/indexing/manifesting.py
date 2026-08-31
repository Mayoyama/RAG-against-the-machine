from json import load, dump, JSONDecodeError
from pathlib import Path
from sys import stderr
from src import Chunk


def load_manifest(save_dir: str) -> dict:
    """Load the incremental-indexing manifest from disk.

    Args:
        save_dir: Directory the manifest was saved under (same
            directory as the BM25/semantic indexes).

    Returns:
        A dict keyed by file path. Each value holds at least "mtime"
        (float) and "chunks" (list[Chunk], already reconstructed —
        not raw dicts). Empty dict if no manifest exists yet (e.g.
        first run, or after a fresh clean).
    """

    # TODO: same try/except (IOError, JSONDecodeError) pattern as
    # get_cached in caching.py — build the manifest's path the same
    # way save_semantic_index does (Path(save_dir) / "manifest.json"),
    # return {} on any read failure.
    #
    # Once loaded, each file entry's "chunks" list will be raw dicts
    # (that's all JSON can hold) — reconstruct them into real Chunk
    # objects before returning, same idea as get_cached doing
    # `MinimalSource(**data)` for each cached result.
    ...


def save_manifest(manifest: dict, save_dir: str) -> None:
    """Persist the incremental-indexing manifest to disk.

    Args:
        manifest: The full manifest to write — already rebuilt for
            this index() run, not merged with whatever was on disk
            before. Each file entry's "chunks" should be actual
            Chunk objects at this point, same as what load_manifest
            returns.
        save_dir: Directory to save the manifest under.
    """

    # TODO: same try/except (IOError, TypeError) pattern as
    # set_cached in caching.py, printing to stderr on failure.
    #
    # Chunk objects aren't JSON-serializable directly — convert each
    # one via .model_dump() before writing, same idea as set_cached
    # doing `src.model_dump()` for each MinimalSource. Unlike
    # set_cached, there's no read-and-merge step here: index_corpus
    # already builds the complete manifest before calling this, so
    # it's a straight overwrite, not a merge into existing data.
    ...
